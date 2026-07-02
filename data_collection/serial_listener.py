"""
Reads CSV sensor data from ESP32 over serial and saves to a timestamped file.

Usage:
    python serial_listener.py --port COM3 --duration 60 --label normal
    python serial_listener.py --port COM3 --duration 60 --label drop
    python serial_listener.py --port COM3 --duration 60 --label shake
    python serial_listener.py --port COM3 --duration 60 --label imbalance

'normal' is reserved for VAE training (ml/train.py). Any other label is
treated as a distinct fault type for the classifier (ml/train_classifier.py) —
collect at least two different fault labels to train it.
"""

import argparse
import csv
import re
import serial
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "raw"
BAUD_RATE = 115200
HEADER = ["timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz", "temp"]
LABEL_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def collect(port: str, duration: int, label: str):
    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = DATA_DIR / f"{label}_{timestamp}.csv"

    print(f"Connecting to {port} at {BAUD_RATE} baud...")
    with serial.Serial(port, BAUD_RATE, timeout=2) as ser, open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        # flush any stale data
        ser.reset_input_buffer()
        time.sleep(0.5)

        # skip the firmware header line
        first = ser.readline().decode("utf-8", errors="ignore").strip()
        if first != ",".join(HEADER):
            print(f"Skipping unexpected first line: {first!r}")

        print(f"Collecting '{label}' data for {duration}s → {out_file}")
        deadline = time.time() + duration
        count = 0

        while time.time() < deadline:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) != len(HEADER):
                continue
            writer.writerow(parts)
            count += 1
            if count % 100 == 0:
                remaining = int(deadline - time.time())
                print(f"  {count} samples collected, {remaining}s remaining...")

    print(f"Done. {count} samples saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM3 or /dev/ttyUSB0")
    parser.add_argument("--duration", type=int, default=60, help="Collection duration in seconds")
    parser.add_argument("--label", default="normal",
                        help="Label for this recording session, e.g. normal, drop, "
                             "shake, imbalance (letters/digits/underscore only)")
    args = parser.parse_args()
    if not LABEL_RE.match(args.label):
        parser.error(f"--label {args.label!r} must match {LABEL_RE.pattern}")
    collect(args.port, args.duration, args.label)
