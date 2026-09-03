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

  1. GAS LEAK — the gas threshold trips the buzzer the moment the ADC crosses the
     limit, while the model verdict only lands on the next 30-second stride. The
     dashboard shows the buzzer first, the model second.
  2. OVERHEAT — the model flags an anomaly and the buzzer stays SILENT. That is
     the safety invariant in CLAUDE.md made visible: the threshold path and the
     model are independent, and the model never drives the buzzer.

Only the two built-in sensor nodes have a story. A room the user adds gets calm,
plausible readings for whatever sensors it has — inventing an incident for a node
that does not physically exist would be dressing up an empty room.

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

from node_config import features_for

# Sampling cadence in simulated seconds. Mirrors ml/profiles.py WINDOW/STRIDE.
SIM_TICK_S = 1.0
VERDICT_STRIDE_S = 30.0
WINDOW_S = 60.0

LOOP_SECONDS = 480.0


@dataclass(frozen=True)
class Phase:
    key: str
    label: str
    start: float
    end: float
    subject: str | None  # profile the phase is about, None for ambient
    detail: str


SCENARIO: tuple[Phase, ...] = (
    Phase("baseline", "Baseline", 0, 90, None, "All nodes nominal"),
    Phase("occupancy", "Kitchen occupancy", 90, 150, "kitchen", "Normal activity — no alert"),
    Phase("gasleak", "Gas leak", 150, 260, "env_safety", "Buzzer on threshold, model follows"),
    Phase("ventilation", "Ventilation", 260, 320, "env_safety", "Gas clearing, alert auto-clears"),
    Phase("overheat", "Thermal rise", 320, 380, "env_safety", "Model flags it — buzzer silent"),
    Phase("intrusion", "Unusual occupancy", 380, 440, "kitchen", "Motion and sound out of pattern"),
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
    """Generates the frame stream for every configured node."""

    def __init__(self, get_nodes, speed: float = 4.0, seed: int = 7):
        """get_nodes() returns the current node configs, so rooms added at
        runtime start streaming without restarting the server."""
        self.get_nodes = get_nodes
        self.speed = max(0.25, speed)
        self.rng = random.Random(seed)
        self._lock = threading.Lock()
        self._sim_t = 0.0
        # One counter per node. A shared counter makes every node see its sequence
        # jump by the node count each tick, which the link tracker correctly reads
        # as dropped packets — reporting huge loss on a perfect link.
        self._seq: dict[int, itertools.count] = {}
        self._verdicts: dict[int, dict] = {}
        self._last_verdict_t = 0.0

    # ── clock ────────────────────────────────────────────────────────────────

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

    def _feature(self, t: float, feat: str, subject: str | None) -> float:
        """One reading. `subject` is the node's profile, or None if user-added.

        Only the two built-in profiles take part in the storyline; every other
        room stays at a calm baseline.
        """
        n = self.rng.gauss
        p = self.phase(t)
        tl = t % LOOP_SECONDS
        acting = subject is not None and p.subject == subject

        if feat == "temp_c":
            if acting and p.key == "overheat":
                return round(_ramp(tl, p.start, p.end - 10, 27.0, 41.5) + n(0, 0.2), 1)
            if acting and p.key == "recovery":
                return round(_ramp(tl, p.start, p.end, 30.0, 27.2) + n(0, 0.15), 1)
            return round(27.2 + n(0, 0.12), 1)

        if feat == "humidity":
            if acting and p.key == "overheat":
                return round(_ramp(tl, p.start, p.end - 10, 62.0, 38.0) + n(0, 0.8), 0)
            return round(max(0.0, 63.0 + n(0, 0.6)), 0)

        if feat == "gas_adc":
            if acting and p.key == "gasleak":
                mid = p.start + (p.end - p.start) * 0.45
                if tl < mid:
                    val = _ramp(tl, p.start, mid, 320, 1850) + n(0, 25)
                else:
                    val = _ramp(tl, mid, p.end - 10, 1850, 3200) + n(0, 40)
            elif acting and p.key == "ventilation":
                val = _ramp(tl, p.start, p.end - 15, 3200, 450) + n(0, 20)
            elif acting and p.key == "recovery":
                val = _ramp(tl, p.start, p.end, 450, 330) + n(0, 12)
            else:
                val = 320.0 + n(0, 12)
            return round(max(0.0, min(4095.0, val)), 0)

        if feat == "motion_duty":
            if acting and p.key == "occupancy":
                val = _ramp(tl, p.start, p.start + 20, 0.05, 0.55) + n(0, 0.08)
            elif acting and p.key == "intrusion":
                val = _ramp(tl, p.start, p.start + 12, 0.1, 0.92) + n(0, 0.05)
            elif acting and p.key == "recovery":
                val = _ramp(tl, p.start, p.end, 0.4, 0.03) + n(0, 0.03)
            else:
                val = n(0.02, 0.02)
            return round(max(0.0, min(1.0, val)), 2)

        if feat == "sound_events":
            if acting and p.key == "occupancy":
                return float(self.rng.choice([0, 1, 1, 2, 2, 3]))
            if acting and p.key == "intrusion":
                return float(self.rng.choice([3, 4, 5, 6, 7, 8]))
            return 0.0 if self.rng.random() > 0.06 else 1.0

        return 0.0

    # ── verdicts ─────────────────────────────────────────────────────────────

    def _recompute_verdict(self, t: float, node_id: int, subject: str | None, feats: dict) -> None:
        """Refresh one node's model verdict.

        Called once per stride, never per sample — the real model consumes a
        60-second window and emits a verdict every 30 seconds, and the dashboard
        should show that cadence rather than pretending inference is instant.
        """
        p = self.phase(t)
        n = self.rng.gauss
        acting = subject is not None and p.subject == subject

        err, fault = 0.006 + abs(n(0, 0.001)), "none"
        if acting and p.key == "gasleak":
            err = 0.018 + (feats.get("gas_adc", 0) / 3200.0) * 0.045 + abs(n(0, 0.002))
            fault = "gasleak"
        elif acting and p.key == "ventilation":
            err = max(0.006, 0.045 * (feats.get("gas_adc", 0) / 3200.0)) + abs(n(0, 0.002))
            fault = "gasleak" if err >= 0.02 else "none"
        elif acting and p.key == "overheat":
            err = 0.014 + ((feats.get("temp_c", 27) - 27.0) / 14.5) * 0.038 + abs(n(0, 0.002))
            fault = "overheat"
        elif acting and p.key == "occupancy":
            # Ordinary activity: the model has seen plenty of this, so it stays
            # below threshold. Showing a non-event is worth as much as an alert.
            err = 0.009 + abs(n(0, 0.0015))
        elif acting and p.key == "intrusion":
            err = (
                0.021
                + feats.get("motion_duty", 0) * 0.03
                + (feats.get("sound_events", 0) / 8.0) * 0.012
            )
            fault = "intrusion"

        thr = 0.02
        severity = 2 if err >= 2 * thr else (1 if err >= thr else 0)
        self._verdicts[node_id] = {
            "err": round(err, 6),
            "severity": severity,
            "fault": fault if severity > 0 else "none",
            "at": t,
        }

    # ── frame stream ─────────────────────────────────────────────────────────

    def tick(self) -> list[dict]:
        """Advance one simulated second and return one frame per node."""
        with self._lock:
            self._sim_t += SIM_TICK_S
            t = self._sim_t

        configs = self.get_nodes()
        phase = self.phase(t)
        ts_ms = int(t * 1000)
        refresh = t - self._last_verdict_t >= VERDICT_STRIDE_S

        frames = []
        for cfg in configs:
            node_id = cfg["id"]
            seq = self._seq.setdefault(node_id, itertools.count(1))

            if cfg.get("role") == "master":
                frames.append(
                    {
                        "node": node_id,
                        "profile": None,
                        "ts": ts_ms,
                        "seq": next(seq),
                        "features": {},
                        "err": None,
                        "severity": 0,
                        "anomaly": 0,
                        "fault": "none",
                        "rule_alert": 0,
                        "model_ready": 0,
                        "relayed": max(0, len(configs) - 1),
                        "phase": phase.key,
                        "phase_label": phase.label,
                    }
                )
                continue

            subject = cfg.get("profile")
            feats = {f: self._feature(t, f, subject) for f in features_for(cfg.get("sensors", []))}

            if refresh or node_id not in self._verdicts:
                self._recompute_verdict(t, node_id, subject, feats)
            v = self._verdicts[node_id]

            frames.append(
                {
                    "node": node_id,
                    "profile": subject,
                    "ts": ts_ms,
                    "seq": next(seq),
                    "features": feats,
                    "err": v["err"],
                    "severity": v["severity"],
                    "anomaly": 1 if v["severity"] > 0 else 0,
                    "fault": v["fault"],
                    # The buzzer is decided by the node's own thresholds in
                    # nodes.py, not here — the user owns that limit.
                    "rule_alert": 0,
                    "dht_stale": 1 if int(t) % 2 else 0,
                    "model_ready": 1,
                    "verdict_age_s": round(t - v["at"], 1),
                    "next_verdict_s": round(
                        max(0.0, VERDICT_STRIDE_S - (t - self._last_verdict_t)), 1
                    ),
                    "phase": phase.key,
                    "phase_label": phase.label,
                }
            )

        if refresh:
            self._last_verdict_t = t
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
