# Complete Setup Guide: Hardware → Running Project

Step-by-step guide from buying parts to a live anomaly-detection dashboard,
including the optional fault-type classifier (Phase 2.5).

## 1. What to Buy (~₹600–1,000 / ~$10–15 total)

| Item | What to look for | Approx. cost (India) |
|---|---|---|
| **ESP32 dev board** | "ESP32 DevKit V1" / ESP32-WROOM-32, 30 or 38 pin | ₹300–500 ($5–8) |
| **MPU-6050 module** | Sold as "GY-521" breakout board | ₹80–150 ($2–3) |
| **Jumper wires** | Female-to-female, pack of 20/40 | ₹60–100 ($1–2) |
| **Micro-USB cable** | Must be a **data** cable, not charge-only — this is the #1 cause of "board not detected" | ₹50–150 ($2) |
| *(Optional)* Breadboard | Only if your GY-521 comes without soldered header pins | ₹70–100 |

All available on Amazon.in, Robu.in, or any local electronics shop. If the
GY-521 comes with loose header pins, you'll need to solder them (or buy one
pre-soldered).

## 2. Wiring (4 wires only)

```
GY-521 (MPU-6050)          ESP32 DevKit
┌─────────────┐            ┌─────────────┐
│ VCC ────────┼────────────┤ 3V3         │
│ GND ────────┼────────────┤ GND         │
│ SCL ────────┼────────────┤ GPIO 22     │
│ SDA ────────┼────────────┤ GPIO 21     │
│ (XDA, XCL,  │            │             │
│  AD0, INT — │            │             │
│  leave      │            │             │
│  unconnected)            └─────────────┘
└─────────────┘
```

Then plug the ESP32 into your laptop via the micro-USB cable.

## 3. One-Time PC Setup

**a) USB driver.** After plugging in, open **Device Manager → Ports (COM &
LPT)**. You should see "Silicon Labs CP210x" or "USB-SERIAL CH340". If it
shows an unknown device instead:

- CP2102 boards → install the [Silicon Labs CP210x driver](https://www.silabs.com/developer-tools/usb-to-uart-bridge-vcp-drivers)
- CH340 boards → search "CH340 driver Windows"

**Note the COM number** (e.g. COM5) — you'll use it everywhere below.
Bluetooth COM ports don't count; it must be the USB-UART one that appears
when you plug the board in.

**b) Software.**

- **PlatformIO** — install the PlatformIO extension in VS Code, or add
  `%USERPROFILE%\.platformio\penv\Scripts` to PATH to use `pio` in a terminal.
- **Python 3.10+** with the per-folder requirements (installed in the phases
  below via `pip install -r requirements.txt`).

## 4. Phase 1 — Flash Collection Firmware & Gather Data

```bash
cd firmware
pio run --target upload        # first run downloads the ESP32 toolchain (~5–10 min)
pio device monitor             # you should see CSV rows streaming: timestamp,ax,ay,az,...
```

`config.h` already defaults to `INFERENCE_MODE 0` (collection mode), so no
changes needed. If the monitor prints `ERROR: MPU6050 not found` → recheck
the 4 wires (SDA/SCL swapped is the usual culprit). Press Ctrl+C to exit the
monitor **before** running the collectors below (only one program can hold
the port).

```bash
cd ../data_collection
pip install -r requirements.txt

# Normal data — keep the sensor in its "healthy" state (still, or attached to
# a machine running normally). Collect 5+ minutes total across a few sessions:
python serial_listener.py --port COM5 --duration 120 --label normal
python serial_listener.py --port COM5 --duration 120 --label normal
python serial_listener.py --port COM5 --duration 60  --label normal

# Fault-labeled data — perform each motion DURING its recording (~60s each, 2+ sessions per type):
python serial_listener.py --port COM5 --duration 60 --label drop       # lift & drop / sharp taps
python serial_listener.py --port COM5 --duration 60 --label shake      # vigorous rapid shaking
python serial_listener.py --port COM5 --duration 60 --label imbalance  # steady rhythmic wobble/rotation
```

## 5. Phase 2 — Train Both Models

```bash
cd ../ml
pip install -r requirements.txt

python train.py                  # VAE anomaly detector (normal data only)
python evaluate.py               # check separation: aim for ROC-AUC > 0.9
python train_classifier.py      # fault-type classifier (drop/shake/imbalance)
```

Check `ml/models/classifier_confusion.png` — you want most weight on the
diagonal. If two fault types confuse each other, collect more distinct
recordings of them.

## 6. Phase 3 — Deploy to the ESP32

```bash
python convert_tflite.py         # exports VAE (int8) + classifier → firmware/include/*.h
```

Then two edits:

**`firmware/include/config.h`:**
```c
#define INFERENCE_MODE 1        // was 0
#define CLASSIFIER_ENABLED 1    // was 0 (needs the classifier trained above)
```

**`firmware/platformio.ini`** — uncomment the TFLite library line:
```ini
spaziochirale/Chirale_TensorFLowLite@^2.0.0
```

Flash again:
```bash
cd ../firmware
pio run --target upload
pio device monitor    # now JSON: {"ts":...,"err":...,"anomaly":0,"severity":0,...,"fault":"none"}
```

**Test it live:** shake the sensor → the onboard LED lights, `"anomaly":1`
appears, and `"fault":"shake"` should show. The sampling rate also bursts to
200 Hz for 2 s on each anomaly.

## 7. Phase 4 — Dashboard

Close the serial monitor first, then:

```bash
cd ../dashboard
pip install -r requirements.txt
python app.py --port COM5
```

Open **http://127.0.0.1:5000/** — you get live charts, severity KPIs, the
color-coded fault badge, an anomaly event log with dominant fault type per
event, and CSV export. (You can preview all of this without hardware:
`python app.py --demo`.)

## Troubleshooting Quick Reference

| Symptom | Fix |
|---|---|
| Upload fails / "no serial port" | Wrong/charge-only cable; driver missing; or hold the **BOOT** button on the ESP32 during upload |
| `MPU6050 not found` | Swap SDA↔SCL; check 3V3 not 5V pin; reseat jumpers |
| Too many false positives | More normal training data; or lower sensitivity: `python train.py --threshold-pct 97` |
| Model too big / boot crash in inference mode | Reduce LSTM units 64→32 in `train.py`, or raise `kTensorArenaSize` in `main.cpp` |
| Classifier always predicts one class | Fault recordings too similar — make the physical motions more distinct, collect more sessions |
| Dashboard shows nothing | Serial monitor still open elsewhere; or firmware still in `INFERENCE_MODE 0` |

**Bottom line:** buy the 4 items (~₹600–1,000), wire 4 jumpers, and the
software path is: flash → collect (~15 min of recordings) → run 4 Python
scripts → edit 2 config lines → flash again → dashboard.
