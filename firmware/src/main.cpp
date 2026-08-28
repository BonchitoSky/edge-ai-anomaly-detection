/*
 * Edge AI anomaly detection — ESP32 sensor-node firmware.
 *
 * The node is selected at build time by SENSOR_PROFILE (see sensor_profile.h);
 * there is one PlatformIO environment per node so the firmware always matches the
 * hardware it is flashed to. Everything profile-specific lives in sampler.cpp, so
 * this file only deals with mode, timing, and output.
 *
 * Modes (INFERENCE_MODE in config.h):
 *   0 — collection: stream one CSV row per slot for training (Phase 1)
 *   1 — inference:  run the TFLite Micro model over a sliding window (Phase 3)
 *
 * Collection output is PROFILE_CSV_HEADER followed by one row per second.
 */

#include <Arduino.h>

#include "config.h"
#include "sampler.h"
#include "sensor_profile.h"

#if INFERENCE_MODE
#if SENSOR_PROFILE != PROFILE_IMU_LEGACY
#error "On-device inference for the home-sensor profiles arrives with the ESP-NOW work. \
Build these profiles with INFERENCE_MODE 0 for data collection."
#endif
#include <Chirale_TensorFlowLite.h> // must precede the tensorflow/lite includes
#include "model_data.h"
#include "model_meta.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#if CLASSIFIER_ENABLED
#include "classifier_data.h"
#include "classifier_meta.h"
#endif

// The generated header and the compiled-in profile must agree, or the scaler
// constants line up against the wrong sensors and the model is quietly wrong.
static_assert(kNumFeatures == PROFILE_NUM_FEATURES,
              "model_meta.h was generated for a different sensor profile than this build.");
#endif

// ── Fatal-error handling ──────────────────────────────────────────────────────

// Report an unrecoverable setup failure, then restart.
//
// These nodes run headless with no console attached, so parking in a delay loop
// leaves a device that is silently dead and indistinguishable from one that lost
// power. Instead: blink the LED in a recognisable pattern and repeat the reason on
// serial (so a console attached late still sees it), then reboot. A transient
// fault — brownout, I2C glitch on a cold start — recovers on its own; a permanent
// one becomes a visible reboot loop, which is diagnosable.
[[noreturn]] static void fatal(const char* reason) {
    pinMode(LED_PIN, OUTPUT);

    for (int cycle = 0; cycle < FATAL_REPORT_CYCLES; cycle++) {
        Serial.print("FATAL: ");
        Serial.println(reason);
        for (int blink = 0; blink < FATAL_BLINK_COUNT; blink++) {
            digitalWrite(LED_PIN, HIGH);
            delay(FATAL_BLINK_ON_MS);
            digitalWrite(LED_PIN, LOW);
            delay(FATAL_BLINK_OFF_MS);
        }
        delay(FATAL_CYCLE_GAP_MS);
    }

    Serial.println("FATAL: restarting.");
    Serial.flush();
    ESP.restart();
    while (true) {
    } // ESP.restart() never returns; keeps the compiler happy about [[noreturn]]
}

// ── Inference state ───────────────────────────────────────────────────────────

#if INFERENCE_MODE

// Sliding window, kept as a ring so successive windows can overlap. Overlap
// matters because an event straddling a boundary would otherwise be split across
// two windows and diluted in both.
static float windowRing[kWindowSize * kNumFeatures];
static int   ringHead        = 0; // next slot to write
static int   ringFilled      = 0; // slots written, capped at kWindowSize
static int   slotsSinceInfer = 0;

// Linearised copy handed to the interpreter, oldest slot first.
static float windowBuf[kWindowSize * kNumFeatures];

static bool          inBurst    = false;
static unsigned long burstUntil = 0;

// TFLite Micro arena — 60 KB should fit the quantized LSTM autoencoder
constexpr int  kTensorArenaSize = 60 * 1024;
static uint8_t tensorArena[kTensorArenaSize];

static tflite::MicroMutableOpResolver<8> resolver;
static const tflite::Model*              tflModel     = nullptr;
static tflite::MicroInterpreter*         interpreter  = nullptr;
static TfLiteTensor*                     inputTensor  = nullptr;
static TfLiteTensor*                     outputTensor = nullptr;

void setupTFLite() {
    resolver.AddFullyConnected();
    resolver.AddUnidirectionalSequenceLSTM();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddMul();
    // The LSTM fusion pass in TFLite emits these around the encoder/decoder
    // boundary: STRIDED_SLICE picks the last encoder timestep
    // (return_sequences=False), BROADCAST_TO implements the RepeatLatent layer in
    // the decoder (see ml/train.py — chosen over RepeatVector because this TFLM
    // build ships no TILE kernel).
    resolver.AddStridedSlice();
    resolver.AddBroadcastTo();

    tflModel = tflite::GetModel(g_model_data);
    if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
        fatal("TFLite schema version mismatch — regenerate model_data.h with "
              "ml/convert_tflite.py against this TFLM version.");
    }

    static tflite::MicroInterpreter staticInterpreter(tflModel, resolver, tensorArena,
                                                      kTensorArenaSize);
    interpreter = &staticInterpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        fatal("AllocateTensors() failed — kTensorArenaSize is too small for this model.");
    }

    inputTensor  = interpreter->input(0);
    outputTensor = interpreter->output(0);
    Serial.println("TFLite Micro ready.");
}

inline float scaleFeature(float x, int featureIdx) {
    return (x - kScalerMean[featureIdx]) / kScalerScale[featureIdx];
}

// Copies the ring into windowBuf in chronological order, oldest slot first.
static void lineariseWindow() {
    const int start = (ringHead - ringFilled + kWindowSize) % kWindowSize;
    for (int i = 0; i < kWindowSize; i++) {
        const int src = ((start + i) % kWindowSize) * kNumFeatures;
        const int dst = i * kNumFeatures;
        for (int f = 0; f < kNumFeatures; f++) {
            windowBuf[dst + f] = windowRing[src + f];
        }
    }
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

#endif // INFERENCE_MODE

#if INFERENCE_MODE && CLASSIFIER_ENABLED

// Fault classifier — small Dense NN over per-window statistical features.
// Runs only when the VAE above already flagged an anomaly (see loop()).

constexpr int  kClassifierArenaSize = 8 * 1024;
static uint8_t classifierArena[kClassifierArenaSize];

static tflite::MicroMutableOpResolver<3> classifierResolver;
static const tflite::Model*              classifierModel       = nullptr;
static tflite::MicroInterpreter*         classifierInterpreter = nullptr;
static TfLiteTensor*                     classifierInput       = nullptr;
static TfLiteTensor*                     classifierOutput      = nullptr;

void setupClassifier() {
    classifierResolver.AddFullyConnected();
    classifierResolver.AddSoftmax();
    classifierResolver.AddRelu();

    classifierModel = tflite::GetModel(g_classifier_data);
    if (classifierModel->version() != TFLITE_SCHEMA_VERSION) {
        fatal("Classifier TFLite schema version mismatch — regenerate classifier_data.h.");
    }

    static tflite::MicroInterpreter staticClassifierInterpreter(
        classifierModel, classifierResolver, classifierArena, kClassifierArenaSize);
    classifierInterpreter = &staticClassifierInterpreter;

    if (classifierInterpreter->AllocateTensors() != kTfLiteOk) {
        fatal("Classifier AllocateTensors() failed — kClassifierArenaSize is too small.");
    }

    classifierInput  = classifierInterpreter->input(0);
    classifierOutput = classifierInterpreter->output(0);
    Serial.println("Fault classifier ready.");
}

// Per-feature [mean, std, min, max, ptp] over the scaled window — order must
// match extract_features() in ml/train_classifier.py.
void extractFeatures(const float* buf, float* outFeatures) {
    for (int f = 0; f < kNumFeatures; f++) {
        float sum   = 0.0f;
        float sumSq = 0.0f;
        float mn    = buf[f];
        float mx    = buf[f];
        for (int i = 0; i < kWindowSize; i++) {
            float v = buf[i * kNumFeatures + f];
            sum += v;
            sumSq += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        float mean     = sum / kWindowSize;
        float variance = (sumSq / kWindowSize) - (mean * mean);
        float stddev   = variance > 0.0f ? sqrtf(variance) : 0.0f;

        int base              = f * 5;
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

#endif // INFERENCE_MODE && CLASSIFIER_ENABLED

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(SERIAL_BAUD);
    // Bounded — a headless node has no USB host and must boot anyway.
    for (unsigned long t0 = millis(); !Serial && millis() - t0 < SERIAL_WAIT_MS;) {
        delay(10);
    }

    pinMode(LED_PIN, OUTPUT);
#if PROFILE_HAS_MQ135
    pinMode(PIN_BUZZER, OUTPUT);
    digitalWrite(PIN_BUZZER, LOW);
#endif
#ifdef PIN_LED_ALERT
    pinMode(PIN_LED_ALERT, OUTPUT);
    digitalWrite(PIN_LED_ALERT, LOW);
#endif
#ifdef PIN_LED_STATUS
    pinMode(PIN_LED_STATUS, OUTPUT);
    digitalWrite(PIN_LED_STATUS, LOW);
#endif

    if (const char* err = sampler::begin()) {
        fatal(err);
    }

#if INFERENCE_MODE
    setupTFLite();
#if CLASSIFIER_ENABLED
    setupClassifier();
#endif
    Serial.print("{\"status\":\"ready\",\"mode\":\"inference\",\"profile\":\"");
    Serial.print(PROFILE_NAME);
    Serial.println("\"}");
#else
    Serial.println(sampler::csvHeader());
#endif
}

// ── Loop ──────────────────────────────────────────────────────────────────────

void loop() {
    const unsigned long now = millis();

    // Sub-slot polling: this is what turns PIR and the sound module into
    // continuous features rather than a single bit sampled once a second.
    static unsigned long lastFastPoll = 0;
    if (now - lastFastPoll >= FAST_POLL_INTERVAL_MS) {
        lastFastPoll = now;
        sampler::poll(now);
    }

    if (!sampler::slotReady(now)) return;

    sampler::Slot slot;
    sampler::readSlot(now, slot);

    // SAFETY: the rule-based gas alert drives the buzzer directly, in this loop,
    // with no dependence on the model. See the safety invariant in CLAUDE.md.
#if PROFILE_HAS_MQ135
    digitalWrite(PIN_BUZZER, slot.ruleAlert ? HIGH : LOW);
    digitalWrite(PIN_LED_ALERT, slot.ruleAlert ? HIGH : LOW);
#endif

#if !INFERENCE_MODE
    sampler::printCsvRow(now, slot);
    // Heartbeat so a node recording unattended for an hour is visibly alive.
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
#ifdef PIN_LED_STATUS
    digitalWrite(PIN_LED_STATUS, slot.dhtStale ? LOW : HIGH);
#endif

#else
    const int base = ringHead * kNumFeatures;
    for (int i = 0; i < kNumFeatures; i++) {
        windowRing[base + i] = scaleFeature(slot.features[i], i);
    }
    ringHead = (ringHead + 1) % kWindowSize;
    if (ringFilled < kWindowSize) ringFilled++;
    slotsSinceInfer++;

    if (ringFilled < kWindowSize || slotsSinceInfer < kStride) return;
    slotsSinceInfer = 0;

    lineariseWindow();
    const float err = runInference();
    // 0=normal, 1=warning (1x–2x threshold), 2=critical (>2x threshold)
    const int severity = (err >= 2.0f * kThreshold) ? 2 : (err >= kThreshold) ? 1 : 0;
    const int anomaly  = (severity > 0) ? 1 : 0;

#if CLASSIFIER_ENABLED
    const int faultIdx = anomaly ? runClassifier() : -1;
#endif

    if (anomaly && (!inBurst || now + BURST_DURATION_MS > burstUntil)) {
        inBurst    = true;
        burstUntil = now + BURST_DURATION_MS;
    } else if (inBurst && now >= burstUntil) {
        inBurst = false;
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
    for (int i = 0; i < kNumFeatures; i++) {
        Serial.print(",\"f");
        Serial.print(i);
        Serial.print("\":");
        Serial.print(slot.features[i], 4);
    }
#if CLASSIFIER_ENABLED
    Serial.print(",\"fault\":\"");
    Serial.print((faultIdx >= 0 && faultIdx < kNumClasses) ? kClassNames[faultIdx] : "none");
    Serial.print("\"");
#endif
    Serial.println("}");
#endif
}
