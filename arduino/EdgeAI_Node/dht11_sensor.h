#pragma once

#include <Arduino.h>

/*
 * DHT11 wrapper with explicit staleness.
 *
 * The DHT11 cannot be read faster than every 2 s, but the sampler emits a feature
 * row every second, so roughly half of all rows reuse the previous reading. That
 * is unavoidable; what matters is that a held value is never mistaken for a fresh
 * one. update() enforces the minimum interval whether or not the last read
 * succeeded, the last good reading is retained across checksum failures (which are
 * common and transient on this part), and isStale() reports when the data has aged
 * past the point where it should be trusted.
 *
 * One DHT11 per node — the pin comes from the active profile.
 */
class Dht11Sensor {
public:
    void begin();

    // Attempts a read if the minimum interval has elapsed. Cheap to call every loop.
    void update(unsigned long now);

    float temperature() const { return temperature_; }
    float humidity() const { return humidity_; }

    bool hasReading() const { return hasReading_; }
    bool isStale(unsigned long now) const;

    unsigned long failedReads() const { return failedReads_; }

private:
    float         temperature_   = NAN;
    float         humidity_      = NAN;
    bool          hasReading_    = false;
    bool          everAttempted_ = false;
    unsigned long lastAttempt_   = 0;
    unsigned long lastGoodRead_  = 0;
    unsigned long failedReads_   = 0;
};
