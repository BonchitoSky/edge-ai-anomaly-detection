# CLAUDE.md — edge-ai-anomaly-detection

Project rules for this repository. These are **project-specific** and sit on top of the user's global
`~/.claude/CLAUDE.md` (atomic commits, Prettier, never force-push, no AI attribution in commits) — this file
adds only what is specific to this project and does not restate the global rules.

---

## 1. Repository and merge workflow

**Remote:** `https://github.com/BonchitoSky/edge-ai-anomaly-detection.git` — all commits go here.

Every change follows this sequence, without exception:

1. Branch from an up-to-date `master` (`git checkout master && git pull origin master`).
2. Commit the work in atomic commits (rule 2).
3. Push the branch and **open a pull request first** — never push directly to `master`.
4. **Check the PR for merge conflicts.** If any exist, fix them on the branch before going further.
5. **Merge only after explicit confirmation from the user.** Never self-merge.

Never force-push (`--force` / `--force-with-lease`). To undo a pushed commit use `git revert`. To bring an
already-pushed branch up to date with `master`, use `git merge origin/master` — a merge commit is acceptable,
rewriting pushed history is not.

## 2. Atomic commits

One logical change per commit. Never bundle the whole task into a single commit — the history has to make it
obvious which commit did what.

- Separate commits for: a new module, a bug fix, a config change, a docs update, a formatting pass.
- **Formatting never shares a commit with logic.** Bulk reformatting gets its own commit, verified with
  `git diff -w` showing no non-whitespace change.
- Commit messages: short, imperative, specific. "Add MQ-135 driver with threshold latch", not "updates".

## 3. Always check for anomalies and errors

Before every commit, actively look for problems rather than assuming the change is clean:

- Compile warnings, not just errors. Treat a new warning as a defect.
- Silent failure paths — swallowed exceptions, ignored return codes, fallbacks that hide a real fault.
- Values that can be `NaN`, `-1`, stale, or uninitialised, and whether downstream code handles them.
- Mismatches between what the training pipeline computes and what the firmware computes. This class of bug
  has already bitten this repo once (see _Known defects_ below) and is invisible at compile time.

When a problem is found, fix it — in its own commit — rather than noting it and moving on.

## 4. Must match the deployed hardware

This repo exists to serve the physical major project. Any change must be consistent with the hardware that is
actually built and tested. The full spec is in _Hardware reference_ below; do not design against sensors,
protocols, or assumptions the project does not have.

Hard constraints:

- **ESP-NOW only.** No Wi-Fi station/AP association, no router, no MQTT broker, no internet, no cloud
  service. The project's entire thesis is fully-offline decentralised operation.
- **ESP32-WROOM-32** (dual-core, 240 MHz, 520 KB SRAM) for all three nodes; ESP32-S with 4 MB PSRAM for the
  ESP32-CAM. Memory budgets are real — the TFLite arena competes with the ESP-NOW stack and sensor drivers.
- **Only the sensors listed below.** Do not introduce a sensor the project does not own.
- The ESP32-CAM face-recognition model is a **separate Edge Impulse artefact** and is out of scope for this
  repo. Do not modify or duplicate it here.

## 5. Clean, formatted, optimised code

| Language                                         | Formatter      | Config           |
| ------------------------------------------------ | -------------- | ---------------- |
| C / C++ (`firmware/`)                            | `clang-format` | `.clang-format`  |
| Python (`ml/`, `data_collection/`, `dashboard/`) | `black`        | `pyproject.toml` |
| JS / HTML / CSS / JSON / MD / YAML               | `prettier`     | `.prettierrc`    |

- Format only files the change already touches. Do not reformat unrelated files.
- Optimise for the target: the firmware runs on a microcontroller. Prefer fixed-size buffers over dynamic
  allocation, avoid `String`, keep ISR and callback bodies short, and keep the sampling loop non-blocking.
- No dead code, no leftover dependencies. If a library stops being used, remove it from `platformio.ini`.

## 6. Verify every commit before pushing

Nothing gets pushed unverified. Minimum gate per commit, for whatever the commit touched:

```bash
~/.platformio/penv/Scripts/pio.exe run -d firmware
```

Bare `pio run` builds every environment, so the gate stays correct as environments
are added. Never verify a single `-e` and call the firmware checked.

```bash
python -m black --check ml data_collection dashboard && python -m compileall -q ml data_collection dashboard
```

```bash
npx prettier@3 --check "dashboard/**/*.{js,html,css}" "*.md"
```

Firmware changes that affect runtime behaviour also need an on-hardware check before the PR is marked ready —
flash, watch the serial output, confirm the OLED and the alert paths still behave.

---

## Safety invariant — do not violate

**The MQ-135 threshold → buzzer path on Node 1 must remain in the fast loop, unchanged, and completely
independent of the ML model.**

The project's central safety claim, measured and published in both reports, is gas-hazard alerting within
500 ms that works even when the Master Node is powered off or out of range. The anomaly model operates on a
60-second window; it must never gate, delay, suppress, or replace that rule-based path.

- Rule-based alert → buzzer + red LED, immediately, locally.
- ML anomaly → status LED, Master OLED page, dashboard. **Not** the buzzer.
- The two are reported as separate flag bits in the ESP-NOW packet so they are never conflated downstream.

---

## Hardware reference

Transcribed from the project reports so this repo is self-contained — the PDFs are not needed to work here.

### Nodes

| Node                            | Role                                  | MCU                     | Sensors                                            | Actuators                                               | Power                                           |
| ------------------------------- | ------------------------------------- | ----------------------- | -------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------- |
| **Node 1** — Environment Safety | Air quality + thermal, local alerting | ESP32-WROOM-32 (38-pin) | MQ-135 (analog ADC + digital threshold), DHT11     | Active buzzer, red LED (220 Ω), blue status LED (10 kΩ) | 5 V USB-C                                       |
| **Node 2** — Kitchen/Occupancy  | Motion, sound, thermal                | ESP32-WROOM-32 (38-pin) | DHT11, PIR HC-SR501, sound sensor module (digital) | Red LED (220 Ω), blue status LED (10 kΩ)                | 5 V USB-C                                       |
| **Master** — Central            | Aggregation + display                 | ESP32-WROOM-32 (38-pin) | —                                                  | SSD1306 OLED 128×64 I²C (SDA IO21, SCL IO22)            | TP4056 + boost → 5 V, PS-5850A-6PL slide switch |
| **ESP32-CAM**                   | Face-recognition access control       | ESP32-S, 4 MB PSRAM     | OV2640 2 MP                                        | Relay-controlled door lock                              | 5 V                                             |

### Communication — ESP-NOW

| Property          | Value                                           |
| ----------------- | ----------------------------------------------- |
| Latency           | < 1 ms spec; **< 2 ms measured** peer-to-peer   |
| Packet delivery   | > 99 % measured at ~5 m indoor                  |
| Data rate         | 250 Kbps                                        |
| Range             | ~50 m indoor, up to 200 m line of sight         |
| Max peers         | 20 uni-directional / 10 bi-directional          |
| Router / internet | **Not required and not used**                   |
| Topology          | Star — Node 1 and Node 2 transmit to Master MAC |
| Transmit interval | 1 second                                        |

### Sensor specifications

| Sensor           | Key limits                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------ |
| **DHT11**        | 0–50 °C ±2 °C; 20–80 % RH ±5 % RH; single-wire digital; **minimum 2 s between reads**                        |
| **MQ-135**       | NH₃, NOx, alcohol, benzene, smoke, CO₂; analog 0–4095 (12-bit ADC) + digital threshold; 24–48 h warm-up; 5 V |
| **PIR HC-SR501** | Up to 7 m, 120°; digital HIGH on motion; ~3 s hardware hold time; 5 V, 3.3 V-safe output                     |
| **Sound sensor** | Digital output; **500 ms firmware debounce**                                                                 |
| **SSD1306 OLED** | 128×64, I²C, Master only — sensor nodes use no I²C, avoiding bus conflicts                                   |

Measured baselines worth knowing: MQ-135 reads ~320 ADC in ambient air, ~1850 under mild exposure, ~3200
under strong exposure (alert fires, < 500 ms response), and returns to ~450 after ventilation.

### Master OLED behaviour

Startup splash `SYSTEM STARTING... WAITING FOR DATA`, transitions to live data on the first received packet
(~1 s). Pages alternate every **3 seconds**.

---

## Sensor profiles

Node 1 and Node 2 have different sensors, so they get **separate models** — never one padded feature vector.

| Profile      | id  | Node   | Features (order is load-bearing)                                      |
| ------------ | --- | ------ | --------------------------------------------------------------------- |
| `env_safety` | 1   | Node 1 | `temp_c`, `humidity`, `gas_adc`                                       |
| `kitchen`    | 2   | Node 2 | `temp_c`, `humidity`, `motion_duty`, `sound_events`                   |
| `imu_legacy` | 0   | —      | `ax`, `ay`, `az`, `gx`, `gy`, `gz`, `temp` (MPU-6050; kept buildable) |

Derived features — PIR and the sound module are polled at 100 Hz inside each 1 s slot and aggregated, which
gives the model continuous-valued signal instead of a bare 0/1:

- `motion_duty` — fraction of the second the PIR held HIGH (0..1)
- `sound_events` — count of debounced rising edges in the second (500 ms debounce preserved)

`temp_c` and `humidity` are sampled on DHT11's own 2 s cadence and the last-good value is held for
intervening slots; a `dhtStale` flag marks held values so this is visible, never silent.

**Windowing:** 1 Hz sampling, window **60** (60 s), stride **30** (50 % overlap → a verdict every 30 s). The
overlap also prevents an event straddling a boundary from being split across two inferences.

`ml/profiles.py` is the single source of truth for feature lists, window, and stride. Firmware mirrors it via
generated `model_meta.h`. **If you change one, change both** — a silent mismatch here produces a model that
appears to work and is quietly wrong. `main.cpp` carries a `static_assert` that the generated `kNumFeatures`
matches the compiled-in profile, and `data_collection/serial_listener.py` carries the CSV headers each
profile prints; both are contract checks, not documentation, and must be updated together.

### Build environments

One PlatformIO environment per physical node, so firmware can never be flashed to the wrong hardware:

| Environment        | Node   | `-DSENSOR_PROFILE` |
| ------------------ | ------ | ------------------ |
| `node1_env_safety` | Node 1 | 1                  |
| `node2_kitchen`    | Node 2 | 2                  |
| `legacy_imu`       | bench  | 0                  |

```bash
~/.platformio/penv/Scripts/pio.exe run -d firmware -e node2_kitchen -t upload
```

### Pin map

GPIO assignments live in `firmware/include/sensor_profile.h`. **They were chosen to be electrically safe,
not read off the boards — confirm them against the EasyEDA schematics before flashing.** Two ESP32
constraints are non-negotiable regardless of what the schematics say:

- **Analog sensors must sit on ADC1 (GPIO 32–39).** ADC2 (GPIO 0, 2, 4, 12–15, 25–27) is unusable while the
  WiFi radio is active, and ESP-NOW keeps it active — `analogRead()` on ADC2 returns garbage once ESP-NOW
  starts. The MQ-135 analog line is therefore on GPIO 34.
- **GPIO 34–39 are input-only** with no internal pull-ups: fine for sensor inputs, unusable for LEDs or
  buzzers.

### Recording layout

Recordings are filed per profile, because Node 1 and Node 2 both produce `normal_*.csv` with different
columns and must never be blended into one model:

```
data_collection/raw/<profile>/<label>_<YYYYMMDD_HHMMSS>.csv
```

`serial_listener.py` identifies the profile from the header the node prints rather than trusting a flag, so
recording Node 2 data into a Node 1 dataset is not possible by accident.

---

## Toolchain

| Tool       | Location / invocation                                |
| ---------- | ---------------------------------------------------- |
| PlatformIO | `~/.platformio/penv/Scripts/pio.exe` (not on `PATH`) |
| Python     | 3.13, system install                                 |
| Node       | v24 — Prettier via `npx prettier@3`                  |
| GitHub CLI | `gh`, authenticated as `BonchitoSky`                 |

Serial is 115200 baud. Firmware build/lib cache lives in `firmware/.pio/` and is gitignored.

Generated artefacts are **gitignored and must be regenerated** — they are not in the repo: `ml/models/`,
`data_collection/raw/`, and `firmware/include/{model_data,model_meta,classifier_data,classifier_meta}.h`.
This means **inference mode does not build on a fresh clone** until the ML pipeline has been run.

---

## Known defects and gotchas

Carried forward so they are not reintroduced:

- **Score/threshold mismatch.** `train.py` calibrated the threshold on `recon_MSE + β·KL` while the firmware
  compares against plain reconstruction MSE — the KL term is structurally unavailable on-device because the
  exported model has a single output. This biased the device toward _missing_ anomalies. Fixed by scoring on
  reconstruction MSE by default and recording the score mode in `config.json` / `model_meta.h`. Any change to
  scoring must keep host and device provably in agreement.
- **`convert_tflite.py` window source.** The window must come from `config.json`, not a CLI default, or a
  model trained at one window silently gets a header declaring another.
- **Dashboard demo-mode fallback.** Omitting `--port` must be an error, not a silent fall-back to synthetic
  data — fake data that looks live is worse than a crash.
- **Blocking error paths.** `while (1) delay(100)` in firmware means a headless node hangs forever with no
  console attached. Failure paths must log and reset via the watchdog.
- **Classifier recall is capped by the detector** — it only runs once the VAE has flagged an anomaly, so it
  cannot recover a missed detection. Known and accepted; do not present classifier accuracy as end-to-end.
- **No time-of-day context.** With no RTC and no NTP (deliberately offline), the model learns "normal
  overall", not "normal at 3 AM". Documented limitation, not a bug.
