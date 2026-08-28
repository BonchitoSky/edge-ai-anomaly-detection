#pragma once

#include <Arduino.h>

/*
 * Sound module with debounced event counting.
 *
 * The module fires a digital pulse per loud event. Counting debounced rising edges
 * per slot rather than sampling the level distinguishes a single bang from
 * sustained noise — one is a spike, the other a run of high counts — which a level
 * sample at 1 Hz cannot express at all.
 *
 * The 500 ms debounce matches the deployed firmware and the figure quoted in the
 * project reports, so recordings stay comparable with earlier results. It also caps
 * the count at two events per second, which is the intended behaviour: the feature
 * measures "how often something loud happened", not the audio waveform.
 */
class SoundSensor {
public:
    void begin();

    // Detects rising edges. Call at the fast poll rate, not once per slot.
    void poll(unsigned long now);

    // Debounced events since the last call. Resets the counter.
    float consumeEvents();

    bool isHigh() const { return lastLevel_; }

private:
    bool          lastLevel_     = false;
    unsigned long lastEventTime_ = 0;
    bool          everTriggered_ = false;
    uint16_t      events_        = 0;
};
