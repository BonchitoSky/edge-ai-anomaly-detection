"""
User-editable node roster, sensor catalogue and alert thresholds.

The node list used to be a hardcoded dict, which meant a room could only be added
by editing Python. Rooms are the one thing a household actually changes, so the
roster now lives in a JSON file the dashboard can edit at runtime, seeded from the
three nodes that are physically built.

A node is a room plus a set of sensors. Its feature vector is derived from those
sensors rather than declared separately, so a node can never claim a reading it has
no hardware for.

Thresholds are the user's own definition of "too much" — a temperature, a sound
rate, a gas level. They are evaluated on every sample, independently of the ML
model, and are what the UI reacts to immediately.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "user_nodes.json"

# ── Feature presentation ──────────────────────────────────────────────────────
# Served through /meta so the browser renders correct labels, units and precision
# without hardcoding a sensor list.

FEATURE_META: dict[str, dict] = {
    "temp_c": {
        "label": "Temperature",
        "unit": "°C",
        "precision": 1,
        "min": 0,
        "max": 60,
        "step": 0.5,
    },
    "humidity": {
        "label": "Humidity",
        "unit": "%RH",
        "precision": 0,
        "min": 0,
        "max": 100,
        "step": 1,
    },
    "gas_adc": {
        "label": "Air quality",
        "unit": "ADC",
        "precision": 0,
        "min": 0,
        "max": 4095,
        "step": 50,
    },
    "motion_duty": {
        "label": "Motion",
        "unit": "duty",
        "precision": 2,
        "min": 0,
        "max": 1,
        "step": 0.05,
    },
    "sound_events": {
        "label": "Sound",
        "unit": "ev/s",
        "precision": 0,
        "min": 0,
        "max": 20,
        "step": 1,
    },
}

# ── Sensor catalogue ──────────────────────────────────────────────────────────
# Only parts the project actually owns (CLAUDE.md rule 4). Adding a sensor here
# without the hardware would let the UI promise readings that cannot exist.

SENSOR_CATALOG: dict[str, dict] = {
    "dht11": {
        "label": "DHT11",
        "detail": "Temperature + humidity",
        "features": ["temp_c", "humidity"],
    },
    "mq135": {
        "label": "MQ-135",
        "detail": "Air quality / gas",
        "features": ["gas_adc"],
    },
    "pir": {
        "label": "PIR HC-SR501",
        "detail": "Motion",
        "features": ["motion_duty"],
    },
    "sound": {
        "label": "Sound module",
        "detail": "Acoustic events",
        "features": ["sound_events"],
    },
}

# Starting points for a newly enabled sensor. The gas value matches
# GAS_ALERT_ON_ADC in firmware/include/sensor_profile.h so the dashboard and the
# node agree out of the box; the rest are sensible household defaults the user is
# expected to tune.
DEFAULT_THRESHOLDS: dict[str, dict] = {
    "gas_adc": {"op": "gt", "value": 3000, "buzzer": True, "enabled": True},
    "temp_c": {"op": "gt", "value": 40, "buzzer": False, "enabled": True},
    "humidity": {"op": "gt", "value": 85, "buzzer": False, "enabled": False},
    "motion_duty": {"op": "gt", "value": 0.85, "buzzer": False, "enabled": False},
    "sound_events": {"op": "gt", "value": 5, "buzzer": False, "enabled": True},
}

# The three nodes that are physically built. Used to seed the config file on first
# run; after that the file wins, so user edits survive a restart.
DEFAULT_NODES: list[dict] = [
    {
        "id": 1,
        "name": "Environment Safety",
        "short": "Node 1",
        "room": "Kitchen / high-risk area",
        "role": "sensor",
        "profile": "env_safety",
        "sensors": ["dht11", "mq135"],
        "builtin": True,
    },
    {
        "id": 2,
        "name": "Kitchen / Occupancy",
        "short": "Node 2",
        "room": "Kitchen / entry area",
        "role": "sensor",
        "profile": "kitchen",
        "sensors": ["dht11", "pir", "sound"],
        "builtin": True,
    },
    {
        "id": 3,
        "name": "Master Coordinator",
        "short": "Master",
        "room": "Living area",
        "role": "master",
        "profile": None,
        "sensors": [],
        "builtin": True,
    },
]


def features_for(sensors: list[str]) -> list[str]:
    """Ordered, de-duplicated feature list implied by a node's sensors.

    Two sensors can contribute the same feature, so the result is de-duplicated
    while keeping catalogue order stable — feature order is positional everywhere
    else in this project.
    """
    out: list[str] = []
    for key in sensors:
        for feat in SENSOR_CATALOG.get(key, {}).get("features", []):
            if feat not in out:
                out.append(feat)
    return out


def default_thresholds_for(sensors: list[str]) -> dict[str, dict]:
    return {
        feat: dict(DEFAULT_THRESHOLDS[feat])
        for feat in features_for(sensors)
        if feat in DEFAULT_THRESHOLDS
    }


class NodeConfigStore:
    """Reads and writes the node roster, seeding it from the built-in nodes."""

    def __init__(self, path: Path = CONFIG_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data.get("nodes"), list) and data["nodes"]:
                    return self._migrate(data)
            except (json.JSONDecodeError, OSError) as exc:
                # A corrupt file must not take the dashboard down mid-demo; fall
                # back to the built-ins and say so rather than failing silently.
                print(f"[config] {self.path.name} unreadable ({exc}); using built-in nodes.")
        return self._seed()

    def _seed(self) -> dict:
        nodes = []
        for spec in DEFAULT_NODES:
            node = dict(spec)
            node["thresholds"] = default_thresholds_for(node["sensors"])
            nodes.append(node)
        data = {"nodes": nodes, "next_id": max(n["id"] for n in nodes) + 1}
        self._write(data)
        return data

    @staticmethod
    def _migrate(data: dict) -> dict:
        """Fill in anything a hand-edited or older file is missing."""
        for node in data["nodes"]:
            node.setdefault("sensors", [])
            node.setdefault("role", "sensor")
            node.setdefault("room", "")
            node.setdefault("builtin", False)
            node.setdefault("profile", None)
            node.setdefault("short", f"Node {node.get('id', '?')}")
            node.setdefault("thresholds", default_thresholds_for(node["sensors"]))
        data.setdefault("next_id", max((n["id"] for n in data["nodes"]), default=0) + 1)
        return data

    def _write(self, data: dict) -> None:
        try:
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"[config] could not save {self.path.name}: {exc}")

    # ── Reads ────────────────────────────────────────────────────────────────

    def nodes(self) -> list[dict]:
        with self._lock:
            return [dict(n) for n in self._data["nodes"]]

    def get(self, node_id: int) -> dict | None:
        with self._lock:
            for n in self._data["nodes"]:
                if n["id"] == node_id:
                    return dict(n)
        return None

    # ── Writes ───────────────────────────────────────────────────────────────

    def add(self, name: str, room: str, sensors: list[str]) -> dict:
        sensors = [s for s in sensors if s in SENSOR_CATALOG]
        with self._lock:
            node_id = self._data["next_id"]
            self._data["next_id"] = node_id + 1
            node = {
                "id": node_id,
                "name": name.strip() or f"Node {node_id}",
                "short": f"Node {node_id}",
                "room": room.strip(),
                "role": "sensor",
                "profile": None,
                "sensors": sensors,
                "thresholds": default_thresholds_for(sensors),
                "builtin": False,
            }
            self._data["nodes"].append(node)
            self._write(self._data)
            return dict(node)

    def update(self, node_id: int, **fields) -> dict | None:
        with self._lock:
            for node in self._data["nodes"]:
                if node["id"] != node_id:
                    continue
                if "name" in fields and fields["name"]:
                    node["name"] = str(fields["name"]).strip()
                if "room" in fields:
                    node["room"] = str(fields["room"]).strip()
                if "sensors" in fields:
                    sensors = [s for s in fields["sensors"] if s in SENSOR_CATALOG]
                    node["sensors"] = sensors
                    # Keep thresholds for features the node still has and seed
                    # defaults for new ones. Dropping a sensor should not discard a
                    # threshold the user tuned for a different one.
                    wanted = features_for(sensors)
                    kept = {f: t for f, t in node["thresholds"].items() if f in wanted}
                    for feat in wanted:
                        kept.setdefault(feat, dict(DEFAULT_THRESHOLDS.get(feat, {})))
                    node["thresholds"] = {f: t for f, t in kept.items() if t}
                self._write(self._data)
                return dict(node)
        return None

    def set_threshold(self, node_id: int, feature: str, patch: dict) -> dict | None:
        with self._lock:
            for node in self._data["nodes"]:
                if node["id"] != node_id:
                    continue
                if feature not in features_for(node["sensors"]):
                    return None
                thr = node["thresholds"].setdefault(
                    feature, dict(DEFAULT_THRESHOLDS.get(feature, {"op": "gt", "value": 0}))
                )
                if "value" in patch:
                    thr["value"] = float(patch["value"])
                if "op" in patch and patch["op"] in ("gt", "lt"):
                    thr["op"] = patch["op"]
                if "enabled" in patch:
                    thr["enabled"] = bool(patch["enabled"])
                if "buzzer" in patch:
                    thr["buzzer"] = bool(patch["buzzer"])
                self._write(self._data)
                return dict(node)
        return None

    def remove(self, node_id: int) -> bool:
        with self._lock:
            for i, node in enumerate(self._data["nodes"]):
                if node["id"] != node_id:
                    continue
                if node.get("builtin"):
                    return False  # the three real nodes are not removable
                del self._data["nodes"][i]
                self._write(self._data)
                return True
        return False

    def reset(self) -> None:
        with self._lock:
            self._data = self._seed()
