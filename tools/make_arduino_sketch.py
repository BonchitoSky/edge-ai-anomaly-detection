"""
Generate an Arduino IDE sketch from the PlatformIO firmware sources.

The PlatformIO build selects the node with -DSENSOR_PROFILE, which Arduino IDE
cannot do. This script flattens firmware/ into a sketch folder and replaces that
build flag with a one-line header the operator edits instead.

The sketch is GENERATED, never edited by hand. firmware/ stays the single source
of truth; re-run this after any firmware change:

    python tools/make_arduino_sketch.py

Output: arduino/EdgeAI_Node/
"""

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIRMWARE = ROOT / "firmware"
OUT = ROOT / "arduino" / "EdgeAI_Node"

# Arduino compiles every .cpp in the sketch folder, but only one .ino carries
# setup()/loop(). main.cpp becomes that .ino; everything else is copied flat.
ENTRY = FIRMWARE / "src" / "main.cpp"
FLAT_SOURCES = [
    FIRMWARE / "include" / "config.h",
    FIRMWARE / "include" / "sensor_profile.h",
    FIRMWARE / "src" / "sampler.h",
    FIRMWARE / "src" / "sampler.cpp",
    *sorted((FIRMWARE / "src" / "sensors").glob("*.h")),
    *sorted((FIRMWARE / "src" / "sensors").glob("*.cpp")),
]

NODE_SELECT = """#pragma once

/*
 * ===========================================================================
 *  EDIT THIS ONE LINE, THEN UPLOAD.
 * ===========================================================================
 *
 *  Which node is this board?
 *
 *      1  = Node 1, Environment Safety   (MQ-135 + DHT11 + buzzer + LEDs)
 *      2  = Node 2, Kitchen / Occupancy  (DHT11 + PIR + sound sensor + LEDs)
 *      0  = MPU-6050 bench rig           (not used for the home system)
 *
 *  Change the number below, upload, and the board will stream the right
 *  columns for that node. Everything else is handled for you.
 */

#define SENSOR_PROFILE 1
"""

README = """# EdgeAI_Node - Arduino IDE sketch

**This folder is generated. Do not edit it by hand.**
It is produced from `firmware/` by `python tools/make_arduino_sketch.py`.
Edits here are lost the next time it is regenerated; change `firmware/` instead.

## Using it

1. Open `EdgeAI_Node.ino` in Arduino IDE.
2. Open the `node_select.h` tab and set `SENSOR_PROFILE` to the node you are
   flashing (1 = Environment Safety, 2 = Kitchen/Occupancy).
3. Board: **ESP32 Dev Module**. Upload speed 921600.
4. Install these libraries via Library Manager:
   - DHT sensor library (Adafruit)
   - Adafruit Unified Sensor
5. Upload, then open Serial Monitor at **115200 baud**.

The board prints a header line and then one CSV row per second.
"""


def rewrite(text: str) -> str:
    """Flatten include paths and pull in the node selector."""
    # sensors/ live alongside everything else in a sketch folder
    text = re.sub(r'#include "sensors/([\w.]+)"', r'#include "\1"', text)
    # SENSOR_PROFILE arrives from node_select.h instead of a -D build flag
    text = text.replace(
        "#pragma once\n",
        '#pragma once\n\n#include "node_select.h" // <-- set your node number in this file\n',
        1,
    )
    return text


def main(check: bool) -> int:
    if not ENTRY.exists():
        raise SystemExit(f"Missing {ENTRY}. Run this from a checkout of the repo.")

    staged: dict[str, str] = {"node_select.h": NODE_SELECT, "README.md": README}
    staged["EdgeAI_Node.ino"] = ENTRY.read_text(encoding="utf-8")

    for src in FLAT_SOURCES:
        text = src.read_text(encoding="utf-8")
        # Only sensor_profile.h needs the node selector injected; the rest just
        # need their include paths flattened.
        if src.name == "sensor_profile.h":
            text = rewrite(text)
        else:
            text = re.sub(r'#include "sensors/([\w.]+)"', r'#include "\1"', text)
        staged[src.name] = text

    if check:
        stale = [
            name
            for name, text in staged.items()
            if not (OUT / name).exists() or (OUT / name).read_text(encoding="utf-8") != text
        ]
        if stale:
            print("Arduino sketch is out of date with firmware/:")
            for name in sorted(stale):
                print(f"  {name}")
            print("Regenerate with: python tools/make_arduino_sketch.py")
            return 1
        print(f"Arduino sketch matches firmware/ ({len(staged)} files).")
        return 0

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name, text in staged.items():
        (OUT / name).write_text(text, encoding="utf-8", newline="\n")

    print(f"Generated {OUT.relative_to(ROOT)} ({len(staged)} files):")
    for name in sorted(staged):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the sketch matches firmware/ without writing. Exit 1 if stale.",
    )
    raise SystemExit(main(parser.parse_args().check))
