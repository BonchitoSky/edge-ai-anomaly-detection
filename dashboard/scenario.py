"""
Scripted demo scenario.

The hardware is not ready, so the dashboard is driven by a written storyline
rather than random noise. That is a deliberate choice for a live presentation:

  * It is deterministic (fixed seed), so a screenshot can be reproduced exactly.
  * It can be jumped to any phase on demand, so "what happens during a gas leak?"
    is answered by pressing a button rather than waiting.
  * The numbers follow the measured baselines published in the project reports —
    MQ-135 reads ~320 ADC in ambient air, ~1850 under mild exposure, ~3200 under
    strong exposure, and ~450 after ventilation — so the shapes on screen match
    the real instrument.

Two beats matter for the viva and are written in deliberately:

  1. GAS LEAK — the rule-based buzzer fires the moment the ADC crosses 3000,
     while the ML verdict only lands on the next 30-second stride. The dashboard
     shows the buzzer first, the model second.
  2. OVERHEAT — the model flags an anomaly and the buzzer stays SILENT. That is
     the safety invariant in CLAUDE.md made visible: the rule path and the model
     are independent, and the model never drives the buzzer.

Timing. Real cadence is 1 Hz sensor sampling with a verdict every 30 s (window 60,
stride 30 — see ml/profiles.py). The scenario keeps that cadence in simulated time
and speeds the clock up so a full loop is watchable; the UI shows the multiplier so
nobody mistakes it for real time.
"""

from __future__ import annotations

import itertools
import random
import threading
import time
from dataclasses import dataclass

from nodes import NODE_DEFS

# Sampling cadence in simulated seconds. Mirrors ml/profiles.py WINDOW/STRIDE.
SIM_TICK_S = 1.0
VERDICT_STRIDE_S = 30.0
WINDOW_S = 60.0

LOOP_SECONDS = 480.0

# MQ-135 rule thresholds — identical to GAS_ALERT_ON_ADC / GAS_ALERT_OFF_ADC in
# firmware/include/sensor_profile.h. Duplicated deliberately: if the firmware
# changes, the demo should visibly disagree rather than silently drift.
GAS_ALERT_ON = 3000
GAS_ALERT_OFF = 2400


@dataclass(frozen=True)
class Phase:
    key: str
    label: str
    start: float
    end: float
    subject: int | None  # node this phase is about, None for ambient
    detail: str


SCENARIO: tuple[Phase, ...] = (
    Phase("baseline", "Baseline", 0, 90, None, "All nodes nominal"),
    Phase("occupancy", "Kitchen occupancy", 90, 150, 2, "Normal activity — no alert"),
    Phase("gasleak", "Gas leak", 150, 260, 1, "Buzzer fires on threshold, model follows"),
    Phase("ventilation", "Ventilation", 260, 320, 1, "Gas clearing, alert auto-clears"),
    Phase("overheat", "Thermal rise", 320, 380, 1, "Model flags it — buzzer stays silent"),
    Phase("intrusion", "Unusual occupancy", 380, 440, 2, "Motion and sound out of pattern"),
    Phase("recovery", "Recovery", 440, 480, None, "Returning to baseline"),
)

TRIGGERABLE = {p.key: p for p in SCENARIO if p.key not in ("baseline", "recovery")}
TRIGGERABLE["clear"] = SCENARIO[0]


def _smoothstep(x: float) -> float:
    """Ease 0..1 so ramps look physical rather than linear."""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _ramp(t: float, t0: float, t1: float, v0: float, v1: float) -> float:
    if t1 <= t0:
        return v1
    return v0 + (v1 - v0) * _smoothstep((t - t0) / (t1 - t0))


class Scenario:
    """Generates the frame stream for every node from the scripted timeline."""

    def __init__(self, speed: float = 4.0, seed: int = 7):
        self.speed = max(0.25, speed)
        self.rng = random.Random(seed)
        self._lock = threading.Lock()
        self._sim_t = 0.0
        # One counter PER NODE. A shared counter makes every node see its sequence
        # jump by the node count each tick, which the link tracker correctly reads
        # as dropped packets - reporting ~66% loss on a perfect link.
        self._seq = {nid: itertools.count(1) for nid in NODE_DEFS}

        # Latched state that must persist between ticks.
        self._gas_latched = False
        self._verdicts: dict[int, dict] = {
            1: {"err": 0.006, "severity": 0, "fault": "none", "at": 0.0},
            2: {"err": 0.005, "severity": 0, "fault": "none", "at": 0.0},
        }
        self._last_verdict_t = 0.0

    # ── clock ────────────────────────────────────────────────────────────────

    @property
    def sim_t(self) -> float:
        return self._sim_t

    def phase(self, t: float | None = None) -> Phase:
        t = self._sim_t if t is None else t
        t %= LOOP_SECONDS
        for p in SCENARIO:
            if p.start <= t < p.end:
                return p
        return SCENARIO[-1]

    def trigger(self, event: str) -> Phase | None:
        """Jump the clock to a phase so an incident happens on demand."""
        phase = TRIGGERABLE.get(event)
        if phase is None:
            return None
        with self._lock:
            # Land a little inside the phase so the effect is immediate rather
            # than starting from the very beginning of a slow ramp.
            offset = 0.0 if phase.key == "baseline" else (phase.end - phase.start) * 0.45
            self._sim_t = phase.start + offset
            self._last_verdict_t = self._sim_t - VERDICT_STRIDE_S  # force a fresh verdict
        return phase

    # ── signals ──────────────────────────────────────────────────────────────

    def _node1_features(self, t: float) -> dict:
        """Environment Safety: temperature, humidity, MQ-135 air quality."""
        n = self.rng.gauss
        temp = 27.0 + n(0, 0.12)
        hum = 62.0 + n(0, 0.6)
        gas = 320.0 + n(0, 12)

        p = self.phase(t)
        tl = t % LOOP_SECONDS
        if p.key == "gasleak":
            # 320 -> 1850 (mild) over the first half, then -> 3200 (strong).
            mid = p.start + (p.end - p.start) * 0.45
            if tl < mid:
                gas = _ramp(tl, p.start, mid, 320, 1850) + n(0, 25)
            else:
                gas = _ramp(tl, mid, p.end - 10, 1850, 3200) + n(0, 40)
        elif p.key == "ventilation":
            gas = _ramp(tl, p.start, p.end - 15, 3200, 450) + n(0, 20)
        elif p.key == "overheat":
            # Thermal event only — the gas line stays at ambient, which is what
            # keeps the buzzer silent and proves the two paths are independent.
            temp = _ramp(tl, p.start, p.end - 10, 27.0, 41.5) + n(0, 0.2)
            hum = _ramp(tl, p.start, p.end - 10, 62.0, 38.0) + n(0, 0.8)
        elif p.key == "recovery":
            gas = _ramp(tl, p.start, p.end, 450, 330) + n(0, 12)
            temp = _ramp(tl, p.start, p.end, 30.0, 27.2) + n(0, 0.15)

        return {
            "temp_c": round(temp, 1),
            "humidity": round(max(0.0, hum), 0),
            "gas_adc": round(max(0.0, min(4095.0, gas)), 0),
        }

    def _node2_features(self, t: float) -> dict:
        """Kitchen / Occupancy: temperature, humidity, motion duty, sound events."""
        n = self.rng.gauss
        temp = 27.5 + n(0, 0.12)
        hum = 65.0 + n(0, 0.6)
        motion = max(0.0, n(0.02, 0.02))
        sound = 0.0 if self.rng.random() > 0.06 else 1.0

        p = self.phase(t)
        tl = t % LOOP_SECONDS
        if p.key == "occupancy":
            motion = max(0.0, min(1.0, _ramp(tl, p.start, p.start + 20, 0.05, 0.55) + n(0, 0.08)))
            sound = float(self.rng.choice([0, 1, 1, 2, 2, 3]))
        elif p.key == "intrusion":
            # Sustained high duty and repeated sound events — the pattern the
            # model has never seen in the normal recordings.
            motion = max(0.0, min(1.0, _ramp(tl, p.start, p.start + 12, 0.1, 0.92) + n(0, 0.05)))
            sound = float(self.rng.choice([3, 4, 5, 6, 7, 8]))
        elif p.key == "recovery":
            motion = max(0.0, _ramp(tl, p.start, p.end, 0.4, 0.03) + n(0, 0.03))

        return {
            "temp_c": round(temp, 1),
            "humidity": round(max(0.0, hum), 0),
            "motion_duty": round(motion, 2),
            "sound_events": sound,
        }

    # ── verdicts ─────────────────────────────────────────────────────────────

    def _recompute_verdicts(self, t: float, f1: dict, f2: dict) -> None:
        """Refresh the model verdict for both nodes.

        Called once per stride, never per sample — the real model consumes a
        60-second window and emits a verdict every 30 seconds, and the dashboard
        should show that cadence rather than pretending inference is instant.
        """
        p = self.phase(t)
        n = self.rng.gauss

        err1, fault1 = 0.006 + abs(n(0, 0.001)), "none"
        if p.key == "gasleak":
            err1 = 0.018 + (f1["gas_adc"] / 3200.0) * 0.045 + abs(n(0, 0.002))
            fault1 = "gasleak"
        elif p.key == "ventilation":
            err1 = max(0.006, 0.045 * (f1["gas_adc"] / 3200.0)) + abs(n(0, 0.002))
            fault1 = "gasleak" if err1 >= 0.02 else "none"
        elif p.key == "overheat":
            err1 = 0.014 + ((f1["temp_c"] - 27.0) / 14.5) * 0.038 + abs(n(0, 0.002))
            fault1 = "overheat"

        err2, fault2 = 0.005 + abs(n(0, 0.001)), "none"
        if p.key == "occupancy":
            # Ordinary activity: the model has seen plenty of this, so it stays
            # below threshold. Showing a non-event is worth as much as an alert.
            err2 = 0.009 + abs(n(0, 0.0015))
        elif p.key == "intrusion":
            err2 = 0.021 + f2["motion_duty"] * 0.03 + (f2["sound_events"] / 8.0) * 0.012
            fault2 = "intrusion"

        for node_id, err, fault in ((1, err1, fault1), (2, err2, fault2)):
            thr = 0.02
            severity = 2 if err >= 2 * thr else (1 if err >= thr else 0)
            self._verdicts[node_id] = {
                "err": round(err, 6),
                "severity": severity,
                "fault": fault if severity > 0 else "none",
                "at": t,
            }

    def _rule_alert(self, gas_adc: float) -> int:
        """MQ-135 threshold path with hysteresis, exactly as the firmware does it.

        This is the safety path: it reacts on the sample, not on a 60-second
        window, and nothing about the model can gate it.
        """
        if not self._gas_latched and gas_adc >= GAS_ALERT_ON:
            self._gas_latched = True
        elif self._gas_latched and gas_adc < GAS_ALERT_OFF:
            self._gas_latched = False
        return 1 if self._gas_latched else 0

    # ── frame stream ─────────────────────────────────────────────────────────

    def tick(self) -> list[dict]:
        """Advance one simulated second and return one frame per node."""
        with self._lock:
            self._sim_t += SIM_TICK_S
            t = self._sim_t

        f1 = self._node1_features(t)
        f2 = self._node2_features(t)

        if t - self._last_verdict_t >= VERDICT_STRIDE_S:
            self._recompute_verdicts(t, f1, f2)
            self._last_verdict_t = t

        phase = self.phase(t)
        ts_ms = int(t * 1000)
        rule = self._rule_alert(f1["gas_adc"])
        next_verdict_in = max(0.0, VERDICT_STRIDE_S - (t - self._last_verdict_t))

        frames = []
        for node_id, feats in ((1, f1), (2, f2)):
            v = self._verdicts[node_id]
            frames.append(
                {
                    "node": node_id,
                    "profile": NODE_DEFS[node_id]["profile"],
                    "ts": ts_ms,
                    "seq": next(self._seq[node_id]),
                    "features": feats,
                    "err": v["err"],
                    "severity": v["severity"],
                    "anomaly": 1 if v["severity"] > 0 else 0,
                    "fault": v["fault"],
                    "rule_alert": rule if node_id == 1 else 0,
                    "dht_stale": 1 if int(t) % 2 else 0,
                    "model_ready": 1,
                    "verdict_age_s": round(t - v["at"], 1),
                    "next_verdict_s": round(next_verdict_in, 1),
                    "phase": phase.key,
                    "phase_label": phase.label,
                }
            )

        # The Master carries no sensors; it reports that it is alive and relaying.
        frames.append(
            {
                "node": 3,
                "profile": None,
                "ts": ts_ms,
                "seq": next(self._seq[3]),
                "features": {},
                "err": None,
                "severity": 0,
                "anomaly": 0,
                "fault": "none",
                "rule_alert": 0,
                "model_ready": 0,
                "relayed": 2,
                "phase": phase.key,
                "phase_label": phase.label,
            }
        )
        return frames

    def run(self, emit) -> None:
        """Drive the scenario forever, calling emit(frame) for every frame."""
        interval = SIM_TICK_S / self.speed
        while True:
            for frame in self.tick():
                emit(frame)
            time.sleep(interval)

    def status(self) -> dict:
        p = self.phase()
        return {
            "sim_t": round(self._sim_t % LOOP_SECONDS, 1),
            "loop_seconds": LOOP_SECONDS,
            "speed": self.speed,
            "phase": p.key,
            "phase_label": p.label,
            "phase_detail": p.detail,
            "phase_subject": p.subject,
            "window_s": WINDOW_S,
            "stride_s": VERDICT_STRIDE_S,
            "phases": [
                {"key": ph.key, "label": ph.label, "start": ph.start, "end": ph.end}
                for ph in SCENARIO
            ],
            "triggerable": sorted(TRIGGERABLE),
        }
