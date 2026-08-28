#include "sensor_profile.h"

#if PROFILE_HAS_PIR

#include "pir_sensor.h"

void PirSensor::begin() {
    pinMode(PIN_PIR, INPUT);
    lastPoll_  = millis();
    slotStart_ = lastPoll_;
}

void PirSensor::poll(unsigned long now) {
    // Credit the interval that just elapsed to whatever level was held during it,
    // rather than counting samples. This keeps the duty correct even if the loop
    // stalls briefly — a missed poll widens one interval instead of losing time.
    const unsigned long elapsed = now - lastPoll_;
    if (lastLevel_) highAccumMs_ += elapsed;
    lastPoll_  = now;
    lastLevel_ = digitalRead(PIN_PIR) == HIGH;
}

float PirSensor::consumeDuty(unsigned long now) {
    const unsigned long span = now - slotStart_;
    slotStart_               = now;

    const unsigned long high = highAccumMs_;
    highAccumMs_             = 0;

    if (span == 0) return 0.0f;
    const float duty = (float) high / (float) span;
    return duty > 1.0f ? 1.0f : duty; // guard against poll jitter overshooting
}

#endif // PROFILE_HAS_PIR
