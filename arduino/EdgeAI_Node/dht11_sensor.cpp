#include "sensor_profile.h"

#if PROFILE_HAS_DHT

#include "dht11_sensor.h"

#include <DHT.h>

namespace {
// Fixed instance, no dynamic allocation — the pin is a compile-time constant from
// the active profile, so there is nothing to configure at runtime.
DHT dht(PIN_DHT_DATA, DHT11);
} // namespace

void Dht11Sensor::begin() {
    dht.begin();
}

void Dht11Sensor::update(unsigned long now) {
    // The interval is enforced whether or not the previous attempt succeeded. Only
    // gating on "no reading yet" would let a failing sensor be polled every loop,
    // far faster than the part can respond, so it would never recover.
    if (everAttempted_ && now - lastAttempt_ < DHT_MIN_INTERVAL_MS) return;
    everAttempted_ = true;
    lastAttempt_   = now;

    const float h = dht.readHumidity();
    const float t = dht.readTemperature();

    // NaN means a checksum failure — common and transient. Keep the previous value
    // rather than poisoning the feature row; staleness is reported separately.
    if (isnan(h) || isnan(t)) {
        failedReads_++;
        return;
    }

    humidity_     = h;
    temperature_  = t;
    hasReading_   = true;
    lastGoodRead_ = now;
}

bool Dht11Sensor::isStale(unsigned long now) const {
    if (!hasReading_) return true;
    return now - lastGoodRead_ > DHT_STALE_AFTER_MS;
}

#endif // PROFILE_HAS_DHT
