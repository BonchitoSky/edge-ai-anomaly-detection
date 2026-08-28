#pragma once

#include <Arduino.h>

/*
 * PIR (HC-SR501) with duty-cycle aggregation.
 *
 * A raw PIR read is a single bit, which carries almost nothing for a sequence
 * model: sampled once per second it is mostly 0 with occasional 1s, and the model
 * can only learn its marginal rate. Instead the pin is polled at 100 Hz and the
 * fraction of the slot spent HIGH is reported, turning it into a continuous 0..1
 * feature that distinguishes someone passing through from someone staying put.
 *
 * The HC-SR501 holds its output HIGH for ~3 s after a trigger (hardware, adjustable
 * by the onboard potentiometer), so consecutive slots during sustained activity
 * saturate at 1.0 — that saturation is real signal, not a defect.
 */
class PirSensor {
public:
    void begin();

    // Accumulates HIGH time. Call at the fast poll rate, not once per slot.
    void poll(unsigned long now);

    // Fraction of the elapsed slot spent HIGH (0..1). Resets the accumulator.
    float consumeDuty(unsigned long now);

    bool isHigh() const { return lastLevel_; }

private:
    bool          lastLevel_   = false;
    unsigned long lastPoll_    = 0;
    unsigned long slotStart_   = 0;
    unsigned long highAccumMs_ = 0;
};
