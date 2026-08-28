"""
Single source of truth for the sensor profiles the ML pipeline trains on.

Node 1 and Node 2 carry different sensors, so they get different feature vectors
and therefore different models — never one padded vector. This module defines the
feature list, numeric id, and shared windowing for each profile in one place;
train.py / evaluate.py / train_classifier.py / convert_tflite.py all import from
here so they can never drift apart.

Feature order is load-bearing. It must match, exactly and positionally:
  * firmware/include/sensor_profile.h   (PROFILE_* feature layout + PROFILE_NUM_FEATURES)
  * data_collection/serial_listener.py  (PROFILE_HEADERS column order)
The scaler constants and the model input are both positional; changing the order
on one side without the others produces a build that runs and is quietly wrong.

Windowing is shared by every profile: 1 Hz sampling, a 60-second window, and a
30-second stride (50 % overlap → a verdict every 30 s). The overlap also stops an
event that straddles a window boundary from being split across two inferences.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    """A named sensor profile: its numeric id and ordered feature columns."""

    id: int
    name: str
    features: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return len(self.features)


# id values match SENSOR_PROFILE in firmware/include/sensor_profile.h.
PROFILES: dict[str, Profile] = {
    "imu_legacy": Profile(0, "imu_legacy", ("ax", "ay", "az", "gx", "gy", "gz", "temp")),
    "env_safety": Profile(1, "env_safety", ("temp_c", "humidity", "gas_adc")),
    "kitchen": Profile(2, "kitchen", ("temp_c", "humidity", "motion_duty", "sound_events")),
}

# Shared windowing — see module docstring. These are the defaults every script
# uses; train.py records the values it actually trained with into config.json so
# the rest of the pipeline reads them back rather than re-assuming a default.
WINDOW = 60
STRIDE = 30

# The two labels that are never fault classes: 'normal' is the detector's training
# data, and 'anomaly' is a reserved generic positive. Kept here so evaluate.py and
# train_classifier.py agree on what counts as a fault class.
RESERVED_LABELS = frozenset({"normal", "anomaly"})


def get_profile(name: str) -> Profile:
    """Look up a profile by name, with a helpful error listing the valid names."""
    try:
        return PROFILES[name]
    except KeyError:
        valid = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {name!r}. Expected one of: {valid}") from None


def profile_names() -> list[str]:
    """Sorted profile names, for argparse `choices=`."""
    return sorted(PROFILES)
