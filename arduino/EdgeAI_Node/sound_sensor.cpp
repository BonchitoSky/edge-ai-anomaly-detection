#include "sensor_profile.h"

#if PROFILE_HAS_SOUND

#include "sound_sensor.h"

void SoundSensor::begin() {
    pinMode(PIN_SOUND, INPUT);
}

void SoundSensor::poll(unsigned long now) {
    const bool level = digitalRead(PIN_SOUND) == HIGH;

    // Rising edge only. The debounce is checked against the last accepted event,
    // not the last edge, so a burst of chatter counts once rather than repeatedly
    // re-arming itself.
    if (level && !lastLevel_) {
        if (!everTriggered_ || now - lastEventTime_ >= SOUND_DEBOUNCE_MS) {
            events_++;
            lastEventTime_ = now;
            everTriggered_ = true;
        }
    }
    lastLevel_ = level;
}

float SoundSensor::consumeEvents() {
    const uint16_t n = events_;
    events_          = 0;
    return (float) n;
}

#endif // PROFILE_HAS_SOUND
