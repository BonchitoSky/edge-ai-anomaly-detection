/*
 * Edge AI Anomaly Detection — ESP32 firmware
 *
 * Modes (set via INFERENCE_MODE in config.h):
 *   0 — data-collection: stream CSV rows for Python training (Phase 1)
 *   1 — inference:       run TFLite Micro autoencoder, report anomaly score (Phase 3)
 *
 * When CLASSIFIER_ENABLED is also set (Phase 2.5), a second small TFLite
 * model classifies the fault type whenever an anomaly is flagged.
 *
 * Serial output (inference mode):
 *   {"ts":<ms>,"err":<mse>,"anomaly":<0|1>,"severity":<0|1|2>,"burst":<0|1>,"ax":<>,"ay":<>,"az":<>,"gx":<>,"gy":<>,"gz":<>[,"fault":<name>]}
 *   "fault" is present only when CLASSIFIER_ENABLED is 1 — "none" when no anomaly, else the predicted fault-type name.
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "config.h"

#if INFERENCE_MODE
#include <Chirale_TensorFlowLite.h>  // must precede the tensorflow/lite includes
#include "model_data.h"
#include "model_meta.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#if CLASSIFIER_ENABLED
#include "classifier_data.h"
#include "classifier_meta.h"
#endif
#endif

// ── Globals ───────────────────────────────────────────────────────────────────

Adafruit_MPU6050 mpu;
unsigned long lastSampleTime = 0;

#if INFERENCE_MODE
static bool          inBurst    = false;
static unsigned long burstUntil = 0;
#endif

#if INFERENCE_MODE

static float windowBuf[kWindowSize * kNumFeatures];
static int   windowIdx = 0;

// TFLite Micro arena — 60 KB should fit the quantized LSTM autoencoder
constexpr int kTensorArenaSize = 60 * 1024;
static uint8_t tensorArena[kTensorArenaSize];

static tflite::MicroMutableOpResolver<8> resolver;
static const tflite::Model* tflModel = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* inputTensor  = nullptr;
static TfLiteTensor* outputTensor = nullptr;

void setupTFLite() {
    resolver.AddFullyConnected();
    resolver.AddUnidirectionalSequenceLSTM();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddMul();
    // TFLite's LSTM fusion emits these around the encoder/decoder boundary:
    // STRIDED_SLICE picks the encoder's last timestep (return_sequences=False),
    // BROADCAST_TO implements the decoder's RepeatLatent (see ml/train.py —
    // chosen over RepeatVector because this TFLM build has no TILE kernel).
    resolver.AddStridedSlice();
    resolver.AddBroadcastTo();

    tflModel = tflite::GetModel(g_model_data);
    if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: TFLite schema version mismatch.");
        while (1) delay(100);
    }

    static tflite::MicroInterpreter staticInterpreter(
        tflModel, resolver, tensorArena, kTensorArenaSize);
    interpreter = &staticInterpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() failed.");
        while (1) delay(100);
    }

    inputTensor  = interpreter->input(0);
    outputTensor = interpreter->output(0);
    Serial.println("TFLite Micro ready.");
}

inline float scaleFeature(float x, int featureIdx) {
    return (x - kScalerMean[featureIdx]) / kScalerScale[featureIdx];
}

float runInference() {
    for (int i = 0; i < kWindowSize * kNumFeatures; i++) {
        inputTensor->data.f[i] = windowBuf[i];
    }
    if (interpreter->Invoke() != kTfLiteOk) return -1.0f;

    float mse = 0.0f;
    for (int i = 0; i < kWindowSize * kNumFeatures; i++) {
        float diff = windowBuf[i] - outputTensor->data.f[i];
        mse += diff * diff;
    }
    return mse / (kWindowSize * kNumFeatures);
}

#endif  // INFERENCE_MODE

#if INFERENCE_MODE && CLASSIFIER_ENABLED

// Fault classifier — small Dense NN over per-window statistical features.
// Runs only when the VAE above already flagged an anomaly (see loop()).

constexpr int kClassifierArenaSize = 8 * 1024;
static uint8_t classifierArena[kClassifierArenaSize];

static tflite::MicroMutableOpResolver<3> classifierResolver;
static const tflite::Model* classifierModel = nullptr;
static tflite::MicroInterpreter* classifierInterpreter = nullptr;
static TfLiteTensor* classifierInput  = nullptr;
static TfLiteTensor* classifierOutput = nullptr;

void setupClassifier() {
    classifierResolver.AddFullyConnected();
    classifierResolver.AddSoftmax();
    classifierResolver.AddRelu();

    classifierModel = tflite::GetModel(g_classifier_data);
    if (classifierModel->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Classifier TFLite schema version mismatch.");
        while (1) delay(100);
    }

    static tflite::MicroInterpreter staticClassifierInterpreter(
        classifierModel, classifierResolver, classifierArena, kClassifierArenaSize);
    classifierInterpreter = &staticClassifierInterpreter;

    if (classifierInterpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: Classifier AllocateTensors() failed.");
        while (1) delay(100);
    }

    classifierInput  = classifierInterpreter->input(0);
    classifierOutput = classifierInterpreter->output(0);
    Serial.println("Fault classifier ready.");
}

// Per-feature [mean, std, min, max, ptp] over the scaled window — order must
// match ml/train_classifier.py's extract_features().
void extractFeatures(const float* buf, float* outFeatures) {
    for (int f = 0; f < kNumFeatures; f++) {
        float sum   = 0.0f;
        float sumSq = 0.0f;
        float mn    = buf[f];
        float mx    = buf[f];
        for (int i = 0; i < kWindowSize; i++) {
            float v = buf[i * kNumFeatures + f];
            sum   += v;
            sumSq += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        float mean     = sum / kWindowSize;
        float variance = (sumSq / kWindowSize) - (mean * mean);
        float stddev   = variance > 0.0f ? sqrtf(variance) : 0.0f;

        int base = f * 5;
        outFeatures[base + 0] = mean;
        outFeatures[base + 1] = stddev;
        outFeatures[base + 2] = mn;
        outFeatures[base + 3] = mx;
        outFeatures[base + 4] = mx - mn;
    }
}

// Returns the predicted class index, or -1 on inference failure.
int runClassifier() {
    float features[kClassifierNumFeatures];
    extractFeatures(windowBuf, features);
    for (int i = 0; i < kClassifierNumFeatures; i++) {
        classifierInput->data.f[i] = features[i];
    }
    if (classifierInterpreter->Invoke() != kTfLiteOk) return -1;

    int   bestIdx = 0;
    float bestVal = classifierOutput->data.f[0];
    for (int i = 1; i < kNumClasses; i++) {
        if (classifierOutput->data.f[i] > bestVal) {
            bestVal = classifierOutput->data.f[i];
            bestIdx = i;
        }
    }
    return bestIdx;
}

#endif  // INFERENCE_MODE && CLASSIFIER_ENABLED

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) delay(10);
    Wire.begin(SDA_PIN, SCL_PIN);

    if (!mpu.begin()) {
        Serial.println("ERROR: MPU6050 not found. Check wiring.");
        while (1) delay(100);
    }
    mpu.setAccelerometerRange(ACCEL_RANGE);
    mpu.setGyroRange(GYRO_RANGE);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

#if INFERENCE_MODE
    pinMode(LED_PIN, OUTPUT);
    setupTFLite();
#if CLASSIFIER_ENABLED
    setupClassifier();
#endif
    Serial.println("{\"status\":\"ready\",\"mode\":\"inference\"}");
#else
    Serial.println("timestamp_ms,ax,ay,az,gx,gy,gz,temp");
#endif
}

// ── Loop ──────────────────────────────────────────────────────────────────────

void loop() {
    unsigned long now = millis();

#if INFERENCE_MODE
    // Revert to normal rate once the burst window expires
    if (inBurst && now >= burstUntil) {
        inBurst = false;
    }
    unsigned int interval = inBurst ? BURST_INTERVAL_MS : SAMPLE_INTERVAL_MS;
#else
    unsigned int interval = SAMPLE_INTERVAL_MS;
#endif

    if (now - lastSampleTime < interval) return;
    lastSampleTime = now;

    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);

#if INFERENCE_MODE
    float raw[kNumFeatures] = {
        accel.acceleration.x,
        accel.acceleration.y,
        accel.acceleration.z,
        gyro.gyro.x,
        gyro.gyro.y,
        gyro.gyro.z,
        temp.temperature,
    };

    int base = windowIdx * kNumFeatures;
    for (int i = 0; i < kNumFeatures; i++) {
        windowBuf[base + i] = scaleFeature(raw[i], i);
    }
    windowIdx++;

    if (windowIdx >= kWindowSize) {
        windowIdx = 0;
        float err      = runInference();
        // 0=normal, 1=warning (1x–2x threshold), 2=critical (>2x threshold)
        int   severity = (err >= 2.0f * kThreshold) ? 2
                       : (err >= kThreshold)         ? 1
                       :                               0;
        int   anomaly  = (severity > 0) ? 1 : 0;

#if CLASSIFIER_ENABLED
        int faultIdx = anomaly ? runClassifier() : -1;
#endif

        // Enter burst mode on any anomaly; extend window if already bursting
        if (anomaly && (!inBurst || now + BURST_DURATION_MS > burstUntil)) {
            inBurst    = true;
            burstUntil = now + BURST_DURATION_MS;
        }

        digitalWrite(LED_PIN, anomaly ? HIGH : LOW);

        Serial.print("{\"ts\":");
        Serial.print(now);
        Serial.print(",\"err\":");
        Serial.print(err, 6);
        Serial.print(",\"anomaly\":");
        Serial.print(anomaly);
        Serial.print(",\"severity\":");
        Serial.print(severity);
        Serial.print(",\"burst\":");
        Serial.print(inBurst ? 1 : 0);
        Serial.print(",\"ax\":");
        Serial.print(accel.acceleration.x, 3);
        Serial.print(",\"ay\":");
        Serial.print(accel.acceleration.y, 3);
        Serial.print(",\"az\":");
        Serial.print(accel.acceleration.z, 3);
        Serial.print(",\"gx\":");
        Serial.print(gyro.gyro.x, 4);
        Serial.print(",\"gy\":");
        Serial.print(gyro.gyro.y, 4);
        Serial.print(",\"gz\":");
        Serial.print(gyro.gyro.z, 4);
#if CLASSIFIER_ENABLED
        Serial.print(",\"fault\":\"");
        Serial.print((faultIdx >= 0 && faultIdx < kNumClasses) ? kClassNames[faultIdx] : "none");
        Serial.print("\"");
#endif
        Serial.println("}");
    }
#else
    Serial.print(now);                              Serial.print(",");
    Serial.print(accel.acceleration.x, 4);          Serial.print(",");
    Serial.print(accel.acceleration.y, 4);          Serial.print(",");
    Serial.print(accel.acceleration.z, 4);          Serial.print(",");
    Serial.print(gyro.gyro.x, 4);                   Serial.print(",");
    Serial.print(gyro.gyro.y, 4);                   Serial.print(",");
    Serial.print(gyro.gyro.z, 4);                   Serial.print(",");
    Serial.println(temp.temperature, 2);
#endif
}
