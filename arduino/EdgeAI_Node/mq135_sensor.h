#pragma once

#include <Arduino.h>

/*
 * MQ-135 air-quality sensor.
 *
 * Two outputs are used: the analog level (0-4095 on the ESP32's 12-bit ADC), which
 * becomes a model feature, and the rule-based alert derived from it.
 *
 * The alert is deliberately computed here from the analog value with hysteresis
 * rather than taken from the module's digital pin, because the module's onboard
 * potentiometer drifts and gives no hysteresis at all — a level sitting on the
 * threshold chatters the buzzer. On/off levels come from the profile.
 *
 * SAFETY: this alert path must stay independent of the anomaly model. See the
 * safety invariant in CLAUDE.md — gas alerting has to work with the Master Node
 * off and must never wait on a 60-second inference window.
 */
class Mq135Sensor {
public:
    void begin();

    // Samples the ADC and updates the latched alert state. Call once per slot.
    void update();

    int  raw() const { return raw_; }
    bool alert() const { return alert_; }

    // The module's own comparator output, recorded for comparison but not used to
    // drive the buzzer.
    bool moduleDigital() const { return moduleDigital_; }

private:
    int  raw_           = 0;
    bool alert_         = false;
    bool moduleDigital_ = false;
};
