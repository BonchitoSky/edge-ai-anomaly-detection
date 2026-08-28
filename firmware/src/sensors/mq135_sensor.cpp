#include "sensor_profile.h"

#if PROFILE_HAS_MQ135

#include "mq135_sensor.h"

void Mq135Sensor::begin() {
    // GPIO 34-39 are input-only and have no internal pull-ups; pinMode is still
    // required so the pad is configured before the first analogRead.
    pinMode(PIN_MQ135_ANALOG, INPUT);
    pinMode(PIN_MQ135_DIGITAL, INPUT);
    analogReadResolution(12);                            // 0-4095, matching the reported ADC range
    analogSetPinAttenuation(PIN_MQ135_ANALOG, ADC_11db); // full-scale ~3.3 V
}

void Mq135Sensor::update() {
    raw_           = analogRead(PIN_MQ135_ANALOG);
    moduleDigital_ = digitalRead(PIN_MQ135_DIGITAL) == HIGH;

    // Schmitt trigger: rise above ON to assert, fall below OFF to clear. Without
    // the gap a reading hovering at the threshold toggles the buzzer every sample.
    if (!alert_ && raw_ >= GAS_ALERT_ON_ADC) {
        alert_ = true;
    } else if (alert_ && raw_ < GAS_ALERT_OFF_ADC) {
        alert_ = false;
    }
}

#endif // PROFILE_HAS_MQ135
