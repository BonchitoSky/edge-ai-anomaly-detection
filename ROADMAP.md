# Project roadmap

Single source of truth for what is done and what is left. **Tick items here as they land**
(`CLAUDE.md` rule 7) — a PR description disappears when the PR merges, this file does not.

**Current stage:** dashboard complete and demo-ready on simulated data. Blocked on real
recordings from the hardware before any model can be trained.

Legend: `[x]` done · `[ ]` outstanding

---

## Phase 1 — Foundations `[x]` (PR #1)

- [x] Project rules, hardware reference and safety invariant in `CLAUDE.md`
- [x] Formatter configs — clang-format, black, Prettier, `.gitattributes`
- [x] Bulk format pass, verified AST-identical on every Python file
- [x] Fix threshold calibrated on `recon + β·KL` while firmware computes recon MSE only
- [x] Fix conversion window read from CLI default instead of `config.json`
- [x] Fix dashboard silently serving synthetic data when `--port` omitted
- [x] Replace 5 blocking `while (1) delay(100)` paths with `fatal()` + watchdog restart
- [x] Fix `evaluate.py` skipping ROC-AUC by globbing only `anomaly_*.csv`

## Phase 2 — Firmware on the real sensor suite `[x]` (PR #2)

- [x] `sensor_profile.h` — profiles, ADC1-safe pin map, feature layout
- [x] `Dht11Sensor` — 2 s cadence, held values, `dhtStale` flag
- [x] `Mq135Sensor` — analog read + Schmitt-trigger rule alert (3000 on / 2400 off)
- [x] `PirSensor` — motion duty cycle aggregated at 100 Hz
- [x] `SoundSensor` — debounced event counting, 500 ms
- [x] `sampler.cpp` — per-profile 1 Hz feature row assembly
- [x] Ring buffer with overlapping windows in `main.cpp`
- [x] One PlatformIO environment per node
- [x] `serial_listener.py` detects profile from the header, files per profile
- [x] `static_assert` on profile/model agreement — verified to actually fail on mismatch

## Phase 3 — ML pipeline on per-node profiles `[x]` (PR #3)

- [x] `ml/profiles.py` as single source of truth for features, window, stride
- [x] `train.py` per profile, overlapping windows, `--score recon` default
- [x] `evaluate.py` reads profile and window from `config.json`
- [x] `train_classifier.py` inherits the detector profile and windowing
- [x] `convert_tflite.py` emits per-profile headers and `kStride`

## Phase 4 — Arduino IDE support `[x]` (PR #4)

- [x] `tools/make_arduino_sketch.py` generator with `--check` drift guard
- [x] Generated sketch compiles for both nodes with the ESP32 toolchain
- [x] `node_select.h` — one line to pick the node

## Phase 5 — Operations dashboard `[x]` (PR #5)

- [x] Vendor Chart.js and fonts — no CDN, renders with no internet
- [x] `nodes.py` — per-node state, event runs, link accounting
- [x] `scenario.py` — scripted incident timeline, deterministic, triggerable
- [x] `/nodes`, `/scenario`, `/api/simulate`, `/qr.svg` endpoints
- [x] Three-node console UI: node cards, unit-aware tiles, alert feed, event log
- [x] Rule-based buzzer shown separately from ML anomaly
- [x] Responsive to phone width; table collapses to cards
- [x] LAN serving with on-screen QR for phone access
- [x] Reproducible screenshot capture for slides
- [x] Sun/moon theme toggle that shows the mode it switches to
- [x] User-editable node roster — add and remove rooms from the UI
- [x] Per-room sensor selection from a catalogue of parts the project owns
- [x] User-defined alert limits per feature, evaluated every sample with hysteresis
- [x] Threshold breaches rendered separately from the model verdict

---

## Phase 6 — Real data `[ ]` **← blocking everything below**

- [ ] Confirm GPIO pin map against the EasyEDA schematics
- [ ] Verify MQ-135 analog line sits on ADC1 (GPIO 32–39)
- [ ] Warm the MQ-135 for 24–48 h before recording
- [ ] Node 1: 3–4 × 1 h `normal`, spread across times of day
- [ ] Node 1: 3 × 5 min `gasleak`, 3 × 5 min `overheat`
- [ ] Node 2: 3–4 × 1 h `normal`
- [ ] Node 2: 3 × 5 min `intrusion`
- [ ] Sanity-check distributions before training

## Phase 7 — Train and validate `[ ]`

- [ ] `train.py --profile env_safety` and `--profile kitchen`
- [ ] ROC-AUC > 0.9 on held-out fault recordings, both nodes
- [ ] Tune thresholds against measured false-positive rate on real `normal` data
- [ ] `train_classifier.py` for both profiles
- [ ] `convert_tflite.py` — generate `model_data.h` / `model_meta.h` per node
- [ ] Confirm arena size fits alongside the ESP-NOW stack

## Phase 8 — ESP-NOW and on-device inference `[ ]`

- [ ] `protocol.h` — `NodePacket` POD with `static_assert` on size
- [ ] `espnow_link.{h,cpp}` — peer registration, send callback, delivery accounting
- [ ] `node_config.h` — Master MAC
- [ ] Remove the `#error` gating inference for home profiles in `main.cpp`
- [ ] Wire verdicts into `NodePacket`; keep the buzzer path untouched
- [ ] `master_main.cpp` — RX, per-node latest, seq-gap loss, OLED third page
- [ ] Master serial bridge emitting the keyed-feature frames the dashboard expects
- [ ] Firmware emits named features rather than `f0…fN`

## Phase 9 — On-hardware validation `[ ]`

- [ ] Flash all three nodes; OLED splash → live data within ~2 s
- [ ] 10-minute soak, no I²C lockups, pages alternate at 3 s
- [ ] **Safety regression: alcohol vapour → buzzer < 500 ms with Master powered OFF**
- [ ] Slow thermal rise → ML severity climbs, buzzer stays silent
- [ ] ESP-NOW latency and delivery rate over 1,000 packets per node
- [ ] Dashboard in live mode (`--port`) against the Master

## Phase 10 — Report and submission `[ ]`

- [ ] Update both reports with measured anomaly-detection results
- [ ] Document limitations: no time-of-day context, classifier recall capped by detector
- [ ] Final screenshots from live hardware, replacing the simulated ones

---

## Known limitations, carried deliberately

- **No time-of-day context.** No RTC and no NTP by design, so the model learns "normal
  overall", not "normal at 3 AM".
- **Classifier recall is capped by the detector** — it only runs once the VAE flags an
  anomaly, so it cannot recover a missed detection. Do not quote it as end-to-end accuracy.
- **Dashboard live mode is untested against real firmware** — the firmware still emits
  `f0…fN` rather than named features. Closed in Phase 8.
