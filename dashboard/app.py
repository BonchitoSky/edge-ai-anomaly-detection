"""
Three-node operations dashboard.

Serves a live view of the ESP-NOW network: Node 1 (Environment Safety), Node 2
(Kitchen / Occupancy) and the Master coordinator. Two data sources, chosen
explicitly on the command line and never inferred:

    python app.py --port COM5      # live, reading the Master over USB serial
    python app.py --demo           # scripted scenario, for presentations

Demo mode is always labelled as simulated in the UI. It is not a fallback: if you
forget --port you get an error, because a dashboard full of fabricated readings
that looks live is worse than one that refuses to start.

Frames are node-keyed dicts with named features (see scenario.py), so nothing here
or in the browser hardcodes a sensor list.
"""

import argparse
import csv
import io
import json
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import serial
from flask import Flask, Response, jsonify, render_template, request

from node_config import SENSOR_CATALOG, NodeConfigStore
from nodes import FEATURE_META, NodeRegistry
from scenario import Scenario

BAUD_RATE = 115200
MAX_Q = 500
DEFAULT_THRESHOLD = 0.02
MODEL_DIR = Path(__file__).parent.parent / "ml" / "models"

app = Flask(__name__)


@dataclass
class AppState:
    demo_mode: bool = False
    scenario: Scenario | None = None
    config: NodeConfigStore = field(default_factory=NodeConfigStore)
    nodes: NodeRegistry | None = None
    model_meta: dict = field(default_factory=dict)
    access_url: str = ""
    started_at: float = field(default_factory=time.time)


state = AppState()
state.nodes = NodeRegistry(DEFAULT_THRESHOLD, state.config)

# ── Client fan-out ────────────────────────────────────────────────────────────

_clients_lock = threading.Lock()
_clients: list[queue.Queue] = []


def _broadcast(obj: dict) -> None:
    payload = json.dumps(obj)
    with _clients_lock:
        for q in _clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                # Drop the oldest rather than the newest: a slow client should
                # fall behind, not miss the alert that just fired.
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except queue.Empty:
                    pass


def _enqueue(frame: dict) -> None:
    state.nodes.ingest(frame)
    _broadcast(frame)


# ── Ingest ────────────────────────────────────────────────────────────────────


def serial_reader(port: str) -> None:
    """Read node frames from the Master over USB, reconnecting on unplug."""
    while True:
        try:
            with serial.Serial(port, BAUD_RATE, timeout=2) as ser:
                print(f"[app] Connected to {port}.")
                while True:
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not raw.startswith("{"):
                        continue
                    try:
                        _enqueue(json.loads(raw))
                    except json.JSONDecodeError:
                        continue
        except serial.SerialException as exc:
            print(f"[app] Serial error on {port}: {exc}. Retrying in 3 s.")
            time.sleep(3)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template(
        "index.html",
        demo=state.demo_mode,
        speed=state.scenario.speed if state.scenario else 1,
        access_url=state.access_url,
    )


@app.route("/stream")
def stream():
    def gen():
        q: queue.Queue = queue.Queue(maxsize=MAX_Q)
        with _clients_lock:
            _clients.append(q)
        try:
            yield "retry: 1000\n\n"
            while True:
                try:
                    yield f"data: {q.get(timeout=5)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _clients_lock:
                if q in _clients:
                    _clients.remove(q)

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/nodes")
def nodes_view():
    return jsonify({"nodes": state.nodes.snapshots()})


@app.route("/events")
def events_view():
    return jsonify({"events": state.nodes.all_events(), "active": state.nodes.active_runs()})


@app.route("/events/export.csv")
def export_events():
    fieldnames = [
        "node",
        "node_short",
        "start_ts",
        "end_ts",
        "duration_ms",
        "peak_err",
        "peak_severity",
        "severity_label",
        "frame_count",
        "dominant_fault",
        "rule_alert",
    ]
    labels = {0: "normal", 1: "warning", 2: "critical"}
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for ev in state.nodes.all_events():
        writer.writerow(
            {
                **ev,
                "severity_label": labels.get(ev.get("peak_severity", 0), "unknown"),
                "rule_alert": int(bool(ev.get("rule_alert"))),
            }
        )
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=anomaly_events.csv"},
    )


@app.route("/threshold", methods=["POST"])
def set_threshold():
    data = request.get_json(silent=True) or {}
    try:
        node_id = int(data["node"])
        value = float(data["threshold"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "expected {node:int, threshold:float}"}), 400
    if not state.nodes.set_threshold(node_id, value):
        return jsonify({"ok": False, "error": f"unknown node {node_id}"}), 404
    return jsonify({"ok": True, "node": node_id, "threshold": value})


@app.route("/threshold/auto", methods=["POST"])
def set_auto_threshold():
    data = request.get_json(silent=True) or {}
    try:
        node_id = int(data["node"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "expected {node:int, enabled:bool}"}), 400
    enabled = bool(data.get("enabled"))
    if not state.nodes.set_auto(node_id, enabled):
        return jsonify({"ok": False, "error": f"unknown node {node_id}"}), 404
    return jsonify({"ok": True, "node": node_id, "auto": enabled})


@app.route("/api/nodes", methods=["POST"])
def add_node():
    """Add a room. Sensors are validated against the catalogue so the UI cannot
    create a node claiming hardware the project does not have."""
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    if not name:
        return jsonify({"ok": False, "error": "a room name is required"}), 400
    sensors = [s for s in data.get("sensors", []) if s in SENSOR_CATALOG]
    if not sensors:
        return jsonify({"ok": False, "error": "pick at least one sensor"}), 400
    node = state.config.add(name, str(data.get("room", "")), sensors)
    state.nodes.sync()
    return jsonify({"ok": True, "node": node}), 201


@app.route("/api/nodes/<int:node_id>", methods=["PATCH"])
def edit_node(node_id: int):
    data = request.get_json(silent=True) or {}
    fields = {k: data[k] for k in ("name", "room", "sensors") if k in data}
    if "sensors" in fields and not [s for s in fields["sensors"] if s in SENSOR_CATALOG]:
        return jsonify({"ok": False, "error": "pick at least one sensor"}), 400
    node = state.config.update(node_id, **fields)
    if node is None:
        return jsonify({"ok": False, "error": f"unknown node {node_id}"}), 404
    state.nodes.sync()
    return jsonify({"ok": True, "node": node})


@app.route("/api/nodes/<int:node_id>", methods=["DELETE"])
def delete_node(node_id: int):
    node = state.config.get(node_id)
    if node is None:
        return jsonify({"ok": False, "error": f"unknown node {node_id}"}), 404
    if not state.config.remove(node_id):
        # The three built-in nodes mirror hardware that physically exists;
        # deleting them from the dashboard would misrepresent the system.
        return jsonify({"ok": False, "error": "built-in nodes cannot be removed"}), 409
    state.nodes.sync()
    return jsonify({"ok": True})


@app.route("/api/nodes/<int:node_id>/threshold", methods=["POST"])
def set_node_threshold(node_id: int):
    """Set the user's own limit for one feature — the value they consider an
    anomaly. Evaluated on every sample, separately from the model."""
    data = request.get_json(silent=True) or {}
    feature = str(data.get("feature", ""))
    if not feature:
        return jsonify({"ok": False, "error": "feature is required"}), 400
    node = state.config.set_threshold(node_id, feature, data)
    if node is None:
        return jsonify({"ok": False, "error": "unknown node or feature not on this node"}), 404
    state.nodes.sync()
    return jsonify({"ok": True, "node": node})


@app.route("/scenario")
def scenario_view():
    if not state.scenario:
        return jsonify({"active": False})
    return jsonify({"active": True, **state.scenario.status()})


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Jump the scenario to a phase so an incident can be shown on demand."""
    if not state.scenario:
        return jsonify({"ok": False, "error": "not running in demo mode"}), 409
    event = (request.get_json(silent=True) or {}).get("event", "")
    phase = state.scenario.trigger(event)
    if phase is None:
        return jsonify({"ok": False, "error": f"unknown event {event!r}"}), 400
    return jsonify({"ok": True, "phase": phase.key, "label": phase.label})


@app.route("/qr.svg")
def qr_svg():
    """QR for the LAN URL, so a phone can open the dashboard by scanning it."""
    svg = _render_qr(state.access_url)
    if svg is None:
        return Response("qr unavailable", status=503, mimetype="text/plain")
    return Response(svg, mimetype="image/svg+xml", headers={"Cache-Control": "no-cache"})


@app.route("/meta")
def meta():
    return jsonify(
        {
            "demo_mode": state.demo_mode,
            "access_url": state.access_url,
            "feature_meta": FEATURE_META,
            "sensor_catalog": SENSOR_CATALOG,
            "uptime_s": round(time.time() - state.started_at, 1),
            **state.model_meta,
        }
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _render_qr(url: str):
    """Render a QR as inline SVG, or None if the optional dep is missing.

    Degrades rather than crashes: the access card falls back to showing the URL
    as text, which is still usable, and startup prints why.
    """
    if not url:
        return None
    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        return None
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue().decode("utf-8")


def _lan_ip() -> str:
    """Best-effort LAN address of this machine.

    Opens a UDP socket toward a public address purely to ask the OS which local
    interface would be used. No packets are sent, so this works with no internet.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _load_model_meta() -> dict:
    """Read whatever the ML pipeline has produced. Absent files are not errors."""
    meta_ = {
        "model_available": False,
        "model_type": None,
        "profile": None,
        "window": None,
        "stride": None,
        "score_mode": None,
        "threshold": None,
        "classifier_available": False,
        "fault_classes": [],
    }
    cfg_path = MODEL_DIR / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            meta_.update(
                model_available=True,
                model_type=cfg.get("model_type"),
                profile=cfg.get("profile"),
                window=cfg.get("window"),
                stride=cfg.get("stride"),
                score_mode=cfg.get("score_mode"),
                threshold=cfg.get("threshold"),
            )
        except (json.JSONDecodeError, OSError):
            pass

    clf_path = MODEL_DIR / "classifier_config.json"
    if clf_path.exists():
        try:
            clf = json.loads(clf_path.read_text())
            meta_.update(classifier_available=True, fault_classes=clf.get("labels", []))
        except (json.JSONDecodeError, OSError):
            pass
    return meta_


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    # The Windows console defaults to cp1252 and would die on the box-drawing
    # characters in the startup banner.
    for stream_ in (sys.stdout, sys.stderr):
        try:
            stream_.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    parser = argparse.ArgumentParser(description="Edge AI three-node dashboard")
    parser.add_argument("--port", default=None, help="Serial port of the Master, e.g. COM5")
    parser.add_argument("--demo", action="store_true", help="Run the scripted demo scenario")
    parser.add_argument("--host", default=None, help="Bind address (default 127.0.0.1)")
    parser.add_argument("--flask-port", type=int, default=5000)
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Bind to all interfaces so phones on the same WiFi can open it",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=4.0,
        help="Demo clock multiplier. Real cadence is 1 Hz with a verdict every 30 s; "
        "the UI always shows this multiplier (default: 4)",
    )
    args = parser.parse_args()

    # A data source must be chosen explicitly. Silently generating synthetic data
    # when --port is missing would produce a dashboard that looks live and is not.
    if args.demo and args.port:
        parser.error("--demo and --port are mutually exclusive; pick one data source")
    if not args.demo and not args.port:
        parser.error("no data source: pass --port <COM5> for live data, or --demo for the scenario")

    host = args.host or ("0.0.0.0" if args.lan else "127.0.0.1")
    shown_host = _lan_ip() if host == "0.0.0.0" else host
    state.access_url = f"http://{shown_host}:{args.flask_port}/"
    state.model_meta = _load_model_meta()
    state.demo_mode = args.demo

    if args.demo:
        state.scenario = Scenario(state.config.nodes, speed=args.speed)
        threading.Thread(target=state.scenario.run, args=(_enqueue,), daemon=True).start()
        source = f"SIMULATED scenario at x{args.speed:g} — readings are not real"
    else:
        threading.Thread(target=serial_reader, args=(args.port,), daemon=True).start()
        source = f"live, reading {args.port}"

    if state.demo_mode and _render_qr(state.access_url) is None:
        print("[app] Optional 'qrcode' package not installed — access card will show a URL only.")

    print(f"\n  Edge AI dashboard  ({source})")
    print(f"  Open: {state.access_url}")
    if args.lan:
        print("  Reachable from any device on this WiFi. Stays local: no cloud, no internet.")
    print()
    app.run(host=host, port=args.flask_port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
