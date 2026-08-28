"""
Reads CSV sensor data from an ESP32 node over serial and saves it to a timestamped
file under the recording profile it belongs to.

Usage:
    python serial_listener.py --port COM3 --duration 3600 --label normal
    python serial_listener.py --port COM3 --duration 300  --label gasleak

'normal' is reserved for detector training (ml/train.py). Any other label is both
an evaluation positive (ml/evaluate.py) and a fault class (ml/train_classifier.py).
Prefer a descriptive label such as 'gasleak' over the generic 'anomaly', which is
reserved and never becomes a fault class.

Profiles
--------
Node 1 and Node 2 carry different sensors, so their recordings have different
columns and must never be blended into one model. The profile is read from the
header line the firmware prints, and recordings are filed under raw/<profile>/
accordingly. Pass --profile to assert an expected profile and refuse to record if
the node connected is not the one you meant to record.
"""

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path

import serial

DATA_DIR = Path(__file__).parent / "raw"
BAUD_RATE = 115200
LABEL_RE = re.compile(r"^[a-zA-Z0-9_]+$")

# Header line each firmware profile prints on boot, keyed by profile name.
# Must stay in step with PROFILE_CSV_HEADER in firmware/include/sensor_profile.h.
PROFILE_HEADERS = {
    "env_safety": ["timestamp_ms", "temp_c", "humidity", "gas_adc", "dht_stale"],
    "kitchen": ["timestamp_ms", "temp_c", "humidity", "motion_duty", "sound_events", "dht_stale"],
    "imu_legacy": ["timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz", "temp"],
}
_HEADER_TO_PROFILE = {",".join(cols): name for name, cols in PROFILE_HEADERS.items()}


class ProfileMismatch(RuntimeError):
    pass


def identify_profile(header_line: str, expected: str | None) -> tuple[str, list[str]]:
    """Map a firmware header line to a profile name and its columns.

    Identifying the node from what it actually prints, rather than from a flag,
    removes the possibility of recording Node 2 data into a Node 1 dataset — a
    mistake that is invisible until training fails or, worse, does not.
    """
    profile = _HEADER_TO_PROFILE.get(header_line.strip())
    if profile is None:
        raise ProfileMismatch(
            f"Unrecognised header from the node: {header_line!r}\nExpected one of:\n  "
            + "\n  ".join(f"{n}: {','.join(c)}" for n, c in PROFILE_HEADERS.items())
            + "\nIs the node running collection firmware (INFERENCE_MODE 0)?"
        )
    if expected is not None and profile != expected:
        raise ProfileMismatch(
            f"Connected node is profile {profile!r}, but --profile said {expected!r}. "
            "Refusing to record so the datasets do not get mixed."
        )
    return profile, PROFILE_HEADERS[profile]


def _read_header(ser: serial.Serial, attempts: int = 20) -> str:
    """Wait for the firmware header line, skipping boot chatter.

    The ESP32 bootloader prints its own banner at reset, and a node that was
    already running may be mid-row when the port opens, so the first readable line
    is often not the header. Scan a bounded number of lines for one that matches a
    known profile instead of assuming line one.
    """
    for _ in range(attempts):
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line in _HEADER_TO_PROFILE:
            return line
    raise ProfileMismatch(
        f"No recognisable header line within {attempts} lines. If the node has been "
        "running a while, press its reset button so it reprints the header."
    )


def collect(port: str, duration: int, label: str, expected_profile: str | None):
    print(f"Connecting to {port} at {BAUD_RATE} baud...")
    with serial.Serial(port, BAUD_RATE, timeout=2) as ser:
        ser.reset_input_buffer()
        time.sleep(0.5)

        profile, header = identify_profile(_read_header(ser), expected_profile)
        n_cols = len(header)
        print(f"Node identified as profile {profile!r} ({n_cols} columns).")

        out_dir = DATA_DIR / profile
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            print(f"Collecting {label!r} for {duration}s -> {out_file}")
            deadline = time.time() + duration
            count = 0
            malformed = 0

            while time.time() < deadline:
                raw = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue
                parts = raw.split(",")
                if len(parts) != n_cols:
                    malformed += 1
                    continue
                writer.writerow(parts)
                count += 1
                if count % 60 == 0:
                    print(f"  {count} rows, {int(deadline - time.time())}s remaining...")

    print(f"Done. {count} rows saved to {out_file}")
    if malformed:
        # Dropped rows are normal in small numbers (a line split across a read),
        # but a high rate means the wrong baud, a loose wire, or a resetting node.
        rate = malformed / (count + malformed)
        note = "  <- investigate before trusting this recording" if rate > 0.02 else ""
        print(f"Dropped {malformed} malformed rows ({rate:.1%}).{note}")
    if count == 0:
        raise SystemExit("No rows captured. Check the node is powered and streaming.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM3 or /dev/ttyUSB0")
    parser.add_argument("--duration", type=int, default=60, help="Collection duration in seconds")
    parser.add_argument(
        "--label",
        default="normal",
        help="Label for this recording, e.g. normal, gasleak, overheat, intrusion "
        "(letters/digits/underscore only)",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_HEADERS),
        default=None,
        help="Assert the connected node is this profile and refuse to record if not. "
        "Detected automatically when omitted.",
    )
    args = parser.parse_args()
    if not LABEL_RE.match(args.label):
        parser.error(f"--label {args.label!r} must match {LABEL_RE.pattern}")
    try:
        collect(args.port, args.duration, args.label, args.profile)
    except ProfileMismatch as exc:
        raise SystemExit(f"ERROR: {exc}")
