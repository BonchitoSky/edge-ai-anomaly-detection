#include "sampler.h"

#include "config.h"

#if SENSOR_PROFILE == PROFILE_IMU_LEGACY
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#else
#include "sensors/dht11_sensor.h"
#if PROFILE_HAS_MQ135
#include "sensors/mq135_sensor.h"
#endif
#if PROFILE_HAS_PIR
#include "sensors/pir_sensor.h"
#endif
#if PROFILE_HAS_SOUND
#include "sensors/sound_sensor.h"
#endif
#endif

namespace sampler {
namespace {

unsigned long slotStart = 0;

#if SENSOR_PROFILE == PROFILE_IMU_LEGACY
Adafruit_MPU6050 mpu;
#else
Dht11Sensor dht;
#if PROFILE_HAS_MQ135
Mq135Sensor mq135;
#endif
#if PROFILE_HAS_PIR
PirSensor pir;
#endif
#if PROFILE_HAS_SOUND
SoundSensor sound;
#endif
#endif

} // namespace

const char* csvHeader() {
    return PROFILE_CSV_HEADER;
}

const char* begin() {
    slotStart = millis();

#if SENSOR_PROFILE == PROFILE_IMU_LEGACY
    Wire.begin(SDA_PIN, SCL_PIN);
    if (!mpu.begin()) {
        return "MPU6050 not found - check SDA/SCL wiring and that VCC is on 3V3, not 5V.";
    }
    mpu.setAccelerometerRange(ACCEL_RANGE);
    mpu.setGyroRange(GYRO_RANGE);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
#else
    dht.begin();
#if PROFILE_HAS_MQ135
    mq135.begin();
#endif
#if PROFILE_HAS_PIR
    pir.begin();
#endif
#if PROFILE_HAS_SOUND
    sound.begin();
#endif
    // The DHT11 needs a moment after power-up before its first read succeeds, and
    // the MQ-135 needs far longer to settle (24-48 h for a stable baseline, per its
    // datasheet). Neither is a startup failure, so there is nothing to report here:
    // early rows are simply flagged stale and the operator warms the gas sensor up
    // before recording. See CLAUDE.md, Sensor specifications.
#endif

    return nullptr;
}

void poll(unsigned long now) {
#if PROFILE_HAS_DHT
    dht.update(now);
#endif
#if PROFILE_HAS_PIR
    pir.poll(now);
#endif
#if PROFILE_HAS_SOUND
    sound.poll(now);
#endif
    (void) now;
}

bool slotReady(unsigned long now) {
    if (now - slotStart < SLOT_INTERVAL_MS) return false;
    slotStart = now;
    return true;
}

void readSlot(unsigned long now, Slot& out) {
#if SENSOR_PROFILE == PROFILE_IMU_LEGACY
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);
    out.features[0] = accel.acceleration.x;
    out.features[1] = accel.acceleration.y;
    out.features[2] = accel.acceleration.z;
    out.features[3] = gyro.gyro.x;
    out.features[4] = gyro.gyro.y;
    out.features[5] = gyro.gyro.z;
    out.features[6] = temp.temperature;
    out.dhtStale    = false;
    out.ruleAlert   = false;

#elif SENSOR_PROFILE == PROFILE_ENV_SAFETY
    mq135.update();
    out.features[0] = dht.temperature();
    out.features[1] = dht.humidity();
    out.features[2] = (float) mq135.raw();
    out.dhtStale    = dht.isStale(now);
    out.ruleAlert   = mq135.alert();

#elif SENSOR_PROFILE == PROFILE_KITCHEN
    out.features[0] = dht.temperature();
    out.features[1] = dht.humidity();
    out.features[2] = pir.consumeDuty(now);
    out.features[3] = sound.consumeEvents();
    out.dhtStale    = dht.isStale(now);
    out.ruleAlert   = false;
#endif
}

void printCsvRow(unsigned long now, const Slot& slot) {
    Serial.print(now);
    for (int i = 0; i < PROFILE_NUM_FEATURES; i++) {
        Serial.print(',');
        Serial.print(slot.features[i], 4);
    }
#if PROFILE_HAS_DHT
    // Not a model feature - it marks rows that reuse a held DHT11 reading so a
    // recording can be audited later. usecols= in the ML loaders ignores it.
    Serial.print(',');
    Serial.print(slot.dhtStale ? 1 : 0);
#endif
    Serial.println();
}

} // namespace sampler
