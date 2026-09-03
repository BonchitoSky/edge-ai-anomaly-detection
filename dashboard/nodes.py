"""
Per-node dashboard state.

The original dashboard tracked one global threshold, one EWMA and one anomaly
run, because it watched a single MPU-6050. With several nodes multiplexed into one
stream that model breaks quietly rather than loudly: a severity-0 frame from one
node would close another's active alert, and the dominant fault of every stored
event would mix nodes together.

Everything that used to be a singleton is keyed by node id here, and the roster
itself comes from node_config so rooms can be added without editing Python.

Two alert paths, deliberately kept apart:
  * THRESHOLD  — the user's own limits, evaluated on every sample. Immediate.
  * MODEL      — the LSTM-VAE verdict, once per 30 s stride. Slower, but catches
                 patterns no single threshold describes.
They are reported as separate fields and rendered separately. Collapsing them
would hide which one actually fired, which is the whole safety argument.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from node_config import (
    DEFAULT_THRESHOLDS,
    FEATURE_META,
    SENSOR_CATALOG,
    NodeConfigStore,
    features_for,
)

__all__ = [
    "FEATURE_META",
    "SENSOR_CATALOG",
    "DEFAULT_THRESHOLDS",
    "NodeState",
    "NodeRegistry",
    "features_for",
]

# A node counts as offline if nothing arrives for this long. The real transmit
# interval is 1 s, so three missed packets is a generous margin.
OFFLINE_AFTER_S = 6.0

# Fraction of the threshold a reading must fall back through before the alert
# clears. Without hysteresis a value sitting on the limit toggles every sample.
HYSTERESIS = 0.8

_EWMA_ALPHA = 0.05
_EWMA_MULTIPLIER = 2.5

MAX_EVENTS_PER_NODE = 100


@dataclass
class NodeState:
    """Live state for one node: its readings, its limits, and its own events."""

    node_id: int
    name: str
    short: str
    role: str
    room: str
    profile: str | None
    sensors: list[str]
    thresholds: dict[str, dict]
    builtin: bool
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
    latched: set = field(default_factory=set)

    def __post_init__(self):
        if self.threshold == 0.0:
            self.threshold = self.default_threshold

    @property
    def features(self) -> list[str]:
        return features_for(self.sensors)

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

    # ── user thresholds ──────────────────────────────────────────────────────

    def evaluate_thresholds(self, features: dict) -> tuple[list[str], bool]:
        """Check readings against the user's limits.

        Returns the breaching feature names and whether any of them is configured
        to sound the buzzer. Latching with hysteresis stops a reading hovering on
        the limit from flickering the alert on and off every sample.
        """
        breaches: list[str] = []
        buzzer = False

        for feat, value in features.items():
            rule = self.thresholds.get(feat)
            if not rule or not rule.get("enabled", True) or value is None:
                self.latched.discard(feat)
                continue

            limit = float(rule.get("value", 0))
            op = rule.get("op", "gt")
            was = feat in self.latched

            if op == "gt":
                hit = value >= limit if not was else value >= limit * HYSTERESIS
            else:
                # For a lower bound the release band sits above the limit.
                release = limit / HYSTERESIS if limit else limit
                hit = value <= limit if not was else value <= release

            if hit:
                self.latched.add(feat)
                breaches.append(feat)
                buzzer = buzzer or bool(rule.get("buzzer"))
            else:
                self.latched.discard(feat)

        return breaches, buzzer

    # ── ingest ───────────────────────────────────────────────────────────────

    def ingest(self, frame: dict) -> None:
        """Record one frame: link accounting, thresholds, adaptive limit, events."""
        breaches, buzzer = self.evaluate_thresholds(frame.get("features") or {})
        frame["threshold_breaches"] = breaches
        # A frame may already carry rule_alert from the firmware; the user's own
        # limits can raise it but never clear a hardware alert.
        frame["rule_alert"] = 1 if (buzzer or frame.get("rule_alert")) else 0

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
            "room": self.room,
            "profile": self.profile,
            "sensors": self.sensors,
            "features": frame.get("features", {}),
            "feature_names": self.features,
            "thresholds": self.thresholds,
            "builtin": self.builtin,
            "online": self.online,
            "loss_pct": round(self.loss_pct, 2),
            "frames_seen": self.frames_seen,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "auto_threshold": self.auto_threshold,
            "threshold_breaches": frame.get("threshold_breaches", []),
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
    """All node states, rebuilt from the config store when the roster changes."""

    def __init__(self, default_threshold: float, store: NodeConfigStore | None = None):
        self.lock = threading.Lock()
        self.default_threshold = default_threshold
        self.store = store or NodeConfigStore()
        self.nodes: dict[int, NodeState] = {}
        self.sync()

    def _build(self, cfg: dict) -> NodeState:
        return NodeState(
            node_id=cfg["id"],
            name=cfg["name"],
            short=cfg.get("short") or f"Node {cfg['id']}",
            role=cfg.get("role", "sensor"),
            room=cfg.get("room", ""),
            profile=cfg.get("profile"),
            sensors=list(cfg.get("sensors", [])),
            thresholds=dict(cfg.get("thresholds", {})),
            builtin=bool(cfg.get("builtin")),
            default_threshold=self.default_threshold,
        )

    def sync(self) -> None:
        """Reconcile live state with the stored roster.

        Existing nodes keep their history and link counters — re-creating them on
        every settings change would wipe the event log the user is looking at.
        """
        with self.lock:
            configs = {c["id"]: c for c in self.store.nodes()}
            for node_id, cfg in configs.items():
                node = self.nodes.get(node_id)
                if node is None:
                    self.nodes[node_id] = self._build(cfg)
                else:
                    node.name = cfg["name"]
                    node.room = cfg.get("room", "")
                    node.sensors = list(cfg.get("sensors", []))
                    node.thresholds = dict(cfg.get("thresholds", {}))
                    node.latched &= set(node.features)
            for node_id in [n for n in self.nodes if n not in configs]:
                del self.nodes[node_id]

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
            return [n.snapshot() for n in sorted(self.nodes.values(), key=lambda n: n.node_id)]

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
