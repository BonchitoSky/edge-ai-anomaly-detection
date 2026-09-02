"""
Per-node dashboard state.

The original dashboard tracked one global threshold, one EWMA and one anomaly
run, because it watched a single MPU-6050. With three ESP-NOW nodes multiplexed
into one stream that model breaks quietly rather than loudly: a severity-0 frame
from Node 2 would close Node 1's active gas alert, and the dominant fault of every
stored event would mix faults across nodes.

Everything that used to be a singleton is therefore keyed by node id here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

# Presentation metadata for every feature any profile can emit. Served through
# /meta so the browser renders correct labels, units and precision without
# hardcoding a sensor list — the same reason ml/profiles.py exists on the ML side.
FEATURE_META: dict[str, dict] = {
    "temp_c": {"label": "Temperature", "unit": "°C", "precision": 1, "group": "climate"},
    "humidity": {"label": "Humidity", "unit": "%RH", "precision": 0, "group": "climate"},
    "gas_adc": {"label": "Air quality", "unit": "ADC", "precision": 0, "group": "gas"},
    "motion_duty": {"label": "Motion", "unit": "duty", "precision": 2, "group": "occupancy"},
    "sound_events": {"label": "Sound", "unit": "ev/s", "precision": 0, "group": "occupancy"},
}

# Node identity. Mirrors the deployed hardware described in CLAUDE.md; the Master
# carries no sensors, so it has no profile and no model.
NODE_DEFS: dict[int, dict] = {
    1: {
        "name": "Environment Safety",
        "short": "Node 1",
        "role": "sensor",
        "profile": "env_safety",
        "location": "Kitchen / high-risk area",
        "sensors": ["MQ-135", "DHT11"],
        "actuators": ["Buzzer", "Red LED", "Blue LED"],
    },
    2: {
        "name": "Kitchen / Occupancy",
        "short": "Node 2",
        "role": "sensor",
        "profile": "kitchen",
        "location": "Kitchen / entry area",
        "sensors": ["PIR HC-SR501", "Sound module", "DHT11"],
        "actuators": ["Red LED", "Blue LED"],
    },
    3: {
        "name": "Master Coordinator",
        "short": "Master",
        "role": "master",
        "profile": None,
        "location": "Living area",
        "sensors": [],
        "actuators": ["SSD1306 OLED"],
    },
}

# A node counts as offline if nothing arrives for this long. The real transmit
# interval is 1 s, so three missed packets is a generous margin.
OFFLINE_AFTER_S = 6.0

_EWMA_ALPHA = 0.05
_EWMA_MULTIPLIER = 2.5

MAX_EVENTS_PER_NODE = 100


@dataclass
class NodeState:
    """Live state for one node: its readings, its threshold, and its own events."""

    node_id: int
    name: str
    short: str
    role: str
    profile: str | None
    location: str
    sensors: list[str]
    actuators: list[str]
    default_threshold: float

    threshold: float = 0.0
    threshold_source: str = "default"
    auto_threshold: bool = False
    ewma_value: float | None = None

    events: list = field(default_factory=list)
    active_run: dict | None = None

    last_frame: dict | None = None
    last_seen: float = 0.0
    frames_seen: int = 0
    next_seq: int | None = None
    packets_lost: int = 0

    def __post_init__(self):
        if self.threshold == 0.0:
            self.threshold = self.default_threshold

    # ── link ─────────────────────────────────────────────────────────────────

    @property
    def online(self) -> bool:
        return self.last_seen > 0 and (time.time() - self.last_seen) < OFFLINE_AFTER_S

    @property
    def loss_pct(self) -> float:
        total = self.frames_seen + self.packets_lost
        return (self.packets_lost / total * 100.0) if total else 0.0

    def _track_sequence(self, seq) -> None:
        """Count gaps in the packet counter as loss.

        The nodes number their packets, so a jump means packets never arrived.
        Without this the dashboard would report a perfect link no matter what.
        """
        if seq is None:
            return
        if self.next_seq is not None and seq > self.next_seq:
            self.packets_lost += seq - self.next_seq
        self.next_seq = seq + 1

    # ── ingest ───────────────────────────────────────────────────────────────

    def ingest(self, frame: dict) -> None:
        """Record one frame: link accounting, adaptive threshold, event runs."""
        self.last_frame = frame
        self.last_seen = time.time()
        self.frames_seen += 1
        self._track_sequence(frame.get("seq"))
        self._update_ewma(frame.get("err"))
        self._track_run(frame)

    def _update_ewma(self, err) -> None:
        if not self.auto_threshold or err is None:
            return
        if self.ewma_value is None:
            self.ewma_value = err
        else:
            self.ewma_value = _EWMA_ALPHA * err + (1 - _EWMA_ALPHA) * self.ewma_value
        self.threshold = round(self.ewma_value * _EWMA_MULTIPLIER, 6)
        self.threshold_source = "auto"

    def _track_run(self, frame: dict) -> None:
        """Open, extend and close anomaly runs for THIS node only."""
        sev = int(frame.get("severity", 0) or 0)
        ts = frame.get("ts", 0)
        err = float(frame.get("err") or 0.0)
        fault = frame.get("fault") or "none"

        if sev > 0:
            if self.active_run is None:
                self.active_run = {
                    "node": self.node_id,
                    "node_short": self.short,
                    "start_ts": ts,
                    "end_ts": ts,
                    "peak_err": err,
                    "peak_severity": sev,
                    "frame_count": 1,
                    "rule_alert": bool(frame.get("rule_alert")),
                    "fault_counts": {fault: 1},
                }
            else:
                run = self.active_run
                run["end_ts"] = ts
                run["frame_count"] += 1
                run["rule_alert"] = run["rule_alert"] or bool(frame.get("rule_alert"))
                run["fault_counts"][fault] = run["fault_counts"].get(fault, 0) + 1
                if err > run["peak_err"]:
                    run["peak_err"] = err
                    run["peak_severity"] = sev
        elif self.active_run is not None:
            run = self.active_run
            counts = run.pop("fault_counts")
            run["dominant_fault"] = max(counts, key=counts.get) if counts else "none"
            run["duration_ms"] = max(0, run["end_ts"] - run["start_ts"])
            self.events.insert(0, run)
            del self.events[MAX_EVENTS_PER_NODE:]
            self.active_run = None

    # ── views ────────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        frame = self.last_frame or {}
        return {
            "node": self.node_id,
            "name": self.name,
            "short": self.short,
            "role": self.role,
            "profile": self.profile,
            "location": self.location,
            "sensors": self.sensors,
            "actuators": self.actuators,
            "online": self.online,
            "loss_pct": round(self.loss_pct, 2),
            "frames_seen": self.frames_seen,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "auto_threshold": self.auto_threshold,
            "features": frame.get("features", {}),
            "err": frame.get("err"),
            "severity": frame.get("severity", 0),
            "anomaly": frame.get("anomaly", 0),
            "rule_alert": frame.get("rule_alert", 0),
            "dht_stale": frame.get("dht_stale", 0),
            "fault": frame.get("fault", "none"),
            "ts": frame.get("ts"),
            "active_run": self.active_run,
            "event_count": len(self.events),
        }


class NodeRegistry:
    """All node states, plus the lock that guards their event lists."""

    def __init__(self, default_threshold: float):
        self.lock = threading.Lock()
        self.nodes: dict[int, NodeState] = {
            nid: NodeState(
                node_id=nid,
                name=d["name"],
                short=d["short"],
                role=d["role"],
                profile=d["profile"],
                location=d["location"],
                sensors=list(d["sensors"]),
                actuators=list(d["actuators"]),
                default_threshold=default_threshold,
            )
            for nid, d in NODE_DEFS.items()
        }

    def get(self, node_id: int) -> NodeState | None:
        return self.nodes.get(node_id)

    def ingest(self, frame: dict) -> None:
        node = self.nodes.get(frame.get("node"))
        if node is None:
            return
        with self.lock:
            node.ingest(frame)

    def snapshots(self) -> list[dict]:
        with self.lock:
            return [n.snapshot() for n in self.nodes.values()]

    def all_events(self) -> list[dict]:
        """Completed runs across every node, newest first."""
        with self.lock:
            merged = [ev for n in self.nodes.values() for ev in n.events]
        merged.sort(key=lambda e: e.get("start_ts", 0), reverse=True)
        return merged

    def active_runs(self) -> list[dict]:
        with self.lock:
            return [n.active_run for n in self.nodes.values() if n.active_run]

    def set_threshold(self, node_id: int, value: float) -> bool:
        node = self.nodes.get(node_id)
        if node is None:
            return False
        with self.lock:
            node.threshold = value
            node.threshold_source = "manual"
            node.auto_threshold = False
            node.ewma_value = None
        return True

    def set_auto(self, node_id: int, enabled: bool) -> bool:
        node = self.nodes.get(node_id)
        if node is None:
            return False
        with self.lock:
            node.auto_threshold = enabled
            if not enabled:
                node.ewma_value = None
                node.threshold = node.default_threshold
                node.threshold_source = "default"
        return True
