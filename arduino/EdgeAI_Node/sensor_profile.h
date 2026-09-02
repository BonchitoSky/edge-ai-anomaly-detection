#pragma once

#include "node_select.h" // <-- set your node number in this file

/*
 * Sensor profile selection.
 *
 * Node 1 and Node 2 carry different sensors, so they get different feature
 * vectors and therefore different models. The profile is chosen at compile time
 * by -DSENSOR_PROFILE in platformio.ini; there is one PlatformIO environment per
 * node so the right firmware is always paired with the right hardware.
 *
 * Feature order is load-bearing. It must match ml/profiles.py exactly — the
 * scaler constants and the model input are both positional. Changing one side
 * without the other produces a build that runs and is quietly wrong.
 */

#define PROFILE_IMU_LEGACY 0 // MPU-6050 bench rig — kept buildable, not deployed
#define PROFILE_ENV_SAFETY 1 // Node 1: DHT11 + MQ-135 + buzzer/LEDs
#define PROFILE_KITCHEN 2    // Node 2: DHT11 + PIR + sound + LEDs

#ifndef SENSOR_PROFILE
#error "SENSOR_PROFILE is not defined. Build through a PlatformIO env, not bare pio run."
#endif

// ── Slot timing (shared by every profile) ─────────────────────────────────────
//
// One feature row per second, matching the project's existing 1 Hz ESP-NOW
// transmit interval. PIR and the sound module are polled far faster than that and
// aggregated into the slot, which turns two binary sensors into continuous-valued
// features the model can actually learn from.

#define SLOT_INTERVAL_MS 1000
#define FAST_POLL_INTERVAL_MS 10 // 100 Hz sub-sampling inside the slot

// DHT11 cannot be read faster than every 2 s. Values are held between reads and
// flagged stale so a held reading is never mistaken for a fresh one.
#define DHT_MIN_INTERVAL_MS 2000
#define DHT_STALE_AFTER_MS 5000 // two missed reads → treat as stale

// Sound module emits a pulse per loud event; debounce matches the 500 ms used by
// the deployed firmware and quoted in the project reports.
#define SOUND_DEBOUNCE_MS 500

// ── Per-profile pin map and feature layout ────────────────────────────────────
//
// !! CONFIRM THESE AGAINST THE EasyEDA SCHEMATICS BEFORE FLASHING !!
// The reports describe the connections but not the GPIO numbers, so these are
// chosen to be electrically safe rather than copied from the boards.
//
// Two ESP32 constraints drove the analog choices:
//   * ADC2 (GPIO 0, 2, 4, 12-15, 25-27) is UNUSABLE while the WiFi radio is
//     active, and ESP-NOW keeps the radio active. Any analog sensor must sit on
//     ADC1 (GPIO 32-39) or analogRead() returns garbage once ESP-NOW starts.
//   * GPIO 34-39 are input-only with no internal pull-ups — ideal for sensor
//     inputs, unusable for LEDs or buzzers.

#if SENSOR_PROFILE == PROFILE_ENV_SAFETY

#define PROFILE_NAME "env_safety"
#define PROFILE_NUM_FEATURES 3
#define PROFILE_CSV_HEADER "timestamp_ms,temp_c,humidity,gas_adc,dht_stale"
#define PROFILE_HAS_DHT 1
#define PROFILE_HAS_MQ135 1
#define PROFILE_HAS_PIR 0
#define PROFILE_HAS_SOUND 0

#define PIN_DHT_DATA 4
#define PIN_MQ135_ANALOG 34 // ADC1_CH6 — must be ADC1, see note above
#define PIN_MQ135_DIGITAL 35
#define PIN_BUZZER 25
#define PIN_LED_ALERT 26  // red
#define PIN_LED_STATUS 27 // blue

// Rule-based gas alert. Independent of the model and never gated by it — see the
// safety invariant in CLAUDE.md. Hysteresis prevents buzzer chatter at the edge.
#define GAS_ALERT_ON_ADC 3000
#define GAS_ALERT_OFF_ADC 2400

#elif SENSOR_PROFILE == PROFILE_KITCHEN

#define PROFILE_NAME "kitchen"
#define PROFILE_NUM_FEATURES 4
#define PROFILE_CSV_HEADER "timestamp_ms,temp_c,humidity,motion_duty,sound_events,dht_stale"
#define PROFILE_HAS_DHT 1
#define PROFILE_HAS_MQ135 0
#define PROFILE_HAS_PIR 1
#define PROFILE_HAS_SOUND 1

#define PIN_DHT_DATA 4
#define PIN_PIR 32
#define PIN_SOUND 33
#define PIN_LED_ALERT 26  // red
#define PIN_LED_STATUS 27 // blue

#elif SENSOR_PROFILE == PROFILE_IMU_LEGACY

#define PROFILE_NAME "imu_legacy"
#define PROFILE_NUM_FEATURES 7
#define PROFILE_CSV_HEADER "timestamp_ms,ax,ay,az,gx,gy,gz,temp"
#define PROFILE_HAS_DHT 0
#define PROFILE_HAS_MQ135 0
#define PROFILE_HAS_PIR 0
#define PROFILE_HAS_SOUND 0

#else
#error "Unknown SENSOR_PROFILE. Expected PROFILE_IMU_LEGACY, PROFILE_ENV_SAFETY or PROFILE_KITCHEN."
#endif
