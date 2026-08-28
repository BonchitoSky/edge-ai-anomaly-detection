#pragma once

#include <Arduino.h>

#include "sensor_profile.h"

/*
 * Per-profile sampling.
 *
 * Everything profile-specific lives behind this interface so main.cpp never
 * branches on which node it is running on. The sampler owns the sensors, runs the
 * fast poll that turns PIR and sound into continuous features, and emits one
 * feature row per slot.
 *
 * Feature order is fixed by PROFILE_CSV_HEADER and must match ml/profiles.py.
 */
namespace sampler {

struct Slot {
    float features[PROFILE_NUM_FEATURES];
    bool  dhtStale  = true;  // temperature/humidity are held, not fresh
    bool  ruleAlert = false; // rule-based gas threshold (env_safety only)
};

// Initialises the profile's sensors. Returns nullptr on success, or a
// human-readable reason on failure so the caller can report it and restart.
const char* begin();

// Fast poll for sub-slot aggregation. Call every loop; cheap for profiles with no
// binary sensors.
void poll(unsigned long now);

// True once a full slot has elapsed. Advances the slot boundary as a side effect
// when it returns true, so call it once per loop.
bool slotReady(unsigned long now);

// Fills the feature row for the slot that just ended.
void readSlot(unsigned long now, Slot& out);

// CSV header for collection mode, matching the feature order exactly.
const char* csvHeader();

// Prints one collection-mode CSV row: timestamp followed by the features.
void printCsvRow(unsigned long now, const Slot& slot);

} // namespace sampler
