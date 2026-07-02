"""
Phase 4 — Flask dashboard for real-time anomaly visualization.

Usage:
    python app.py --port COM3
    python app.py --port COM3 --host 0.0.0.0   # expose on LAN
    python app.py --demo                        # synthetic data, no ESP32 needed

The ESP32 must be flashed with INFERENCE_MODE=1.
Reads JSON lines from serial:
    {"ts":<ms>,"err":<mse>,"anomaly":<0|1>,"ax":<>,"ay":<>,"az":<>}

Browser connects to /stream via Server-Sent Events.
"""

import argparse
import csv
import io
import json
import math
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import serial
from flask import Flask, Response, render_template, request

BAUD_RATE  = 115200
MAX_Q      = 500
THRESHOLD  = 0.02
MAX_EVENTS = 200  # keep last N completed anomaly runs
MODEL_DIR  = Path(__file__).parent.parent / "ml" / "models"

app = Flask(__name__, template_folder="templates", static_folder="static")


@dataclass
class AppState:
    """Single home for mutable server state, replacing scattered globals."""
    threshold:        float          = THRESHOLD
    threshold_source: str            = "default"   # default | model_file | manual
    demo_mode:        bool           = False
    auto_threshold:   bool           = False
    ewma_value:       float | None   = None
    events:           list           = field(default_factory=list)
    active_run:       dict | None    = None
    model_meta:       dict           = field(default_factory=dict)

    events_lock:         threading.Lock = field(default_factory=threading.Lock)
    auto_threshold_lock: threading.Lock = field(default_factory=threading.Lock)


state = AppState()

# ── Pub-sub broadcaster ────────────────────────────────────────────────────────
# Each /stream client registers its own queue; _broadcast() fans out to all.

_clients_lock: threading.Lock          = threading.Lock()
_clients: list[queue.Queue]            = []


def _broadcast(obj: dict):
    with _clients_lock:
        for q in _clients:
            try:
                q.put_nowait(obj)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                q.put_nowait(obj)

# ── Adaptive (EWMA) threshold ──────────────────────────────────────────────────
# When auto-mode is on, _threshold tracks an EWMA of recent reconstruction
# errors, biased upward by a fixed multiplier so normal variation doesn't
# trigger false positives.

_EWMA_ALPHA          = 0.05   # smoothing factor (lower = slower adaptation)
_EWMA_MULTIPLIER     = 2.5    # threshold = EWMA * multiplier


def _update_ewma(err: float):
    if not state.auto_threshold:
        return
    with state.auto_threshold_lock:
        if state.ewma_value is None:
            state.ewma_value = err
        else:
            state.ewma_value = _EWMA_ALPHA * err + (1 - _EWMA_ALPHA) * state.ewma_value
        state.threshold = round(state.ewma_value * _EWMA_MULTIPLIER, 6)

# ── Anomaly event tracking ─────────────────────────────────────────────────────


def _process_event(obj: dict):
    """Update the anomaly-run tracker from a new data frame."""
    sev   = obj.get("severity", 0)
    ts    = obj.get("ts", 0)
    err   = obj.get("err", 0.0)
    fault = obj.get("fault", "none")

    with state.events_lock:
        if sev > 0:
            if state.active_run is None:
                state.active_run = {
                    "start_ts":      ts,
                    "end_ts":        ts,
                    "peak_err":      err,
                    "peak_severity": sev,
                    "frame_count":   1,
                    "fault_counts":  {fault: 1},
                }
            else:
                state.active_run["end_ts"]        = ts
                state.active_run["frame_count"]  += 1
                state.active_run["fault_counts"][fault] = \
                    state.active_run["fault_counts"].get(fault, 0) + 1
                if err > state.active_run["peak_err"]:
                    state.active_run["peak_err"]      = err
                    state.active_run["peak_severity"] = sev
        else:
            if state.active_run is not None:
                fault_counts = state.active_run.pop("fault_counts")
                state.active_run["dominant_fault"] = max(
                    fault_counts, key=fault_counts.get
                )
                state.events.insert(0, state.active_run)
                if len(state.events) > MAX_EVENTS:
                    state.events.pop()
                state.active_run = None


# ── Serial reader thread ───────────────────────────────────────────────────────

def serial_reader(port: str):
    print(f"[serial] connecting to {port} @ {BAUD_RATE} …")
    while True:
        try:
            with serial.Serial(port, BAUD_RATE, timeout=2) as ser:
                print("[serial] connected.")
                while True:
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not raw.startswith("{"):
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    _enqueue(obj)
        except serial.SerialException as e:
            print(f"[serial] {e} — retrying in 3s …")
            time.sleep(3)


DEMO_FAULT_CLASSES = ["drop", "shake", "imbalance"]


def demo_generator():
    t = 0
    anomaly_phase = False
    phase_counter = 0
    anomaly_run_idx = -1  # increments each time a new anomaly phase starts
    while True:
        time.sleep(0.05)
        phase_counter += 1
        if phase_counter % 80 == 0:
            anomaly_phase = not anomaly_phase
            if anomaly_phase:
                anomaly_run_idx += 1

        noise = random.gauss(0, 0.05)
        base_err = 0.035 if anomaly_phase else 0.008
        err = max(0.0, base_err + random.gauss(0, 0.003))
        severity = 2 if err >= 2 * state.threshold else (1 if err >= state.threshold else 0)
        gyro_noise = random.gauss(0, 0.02 if anomaly_phase else 0.005)
        fault = (
            DEMO_FAULT_CLASSES[anomaly_run_idx % len(DEMO_FAULT_CLASSES)]
            if anomaly_phase and severity > 0 else "none"
        )
        obj = {
            "ts":       int(t * 50),
            "err":      round(err, 6),
            "anomaly":  1 if severity > 0 else 0,
            "severity": severity,
            "burst":    1 if anomaly_phase else 0,
            "fault":    fault,
            "ax":       round(math.sin(t * 0.1) + noise, 3),
            "ay":       round(math.cos(t * 0.15) + noise, 3),
            "az":       round(9.81 + noise * 0.2, 3),
            "gx":       round(math.sin(t * 0.07) * 0.3 + gyro_noise, 4),
            "gy":       round(math.cos(t * 0.11) * 0.3 + gyro_noise, 4),
            "gz":       round(gyro_noise * 0.5, 4),
        }
        t += 1
        _enqueue(obj)


def _enqueue(obj):
    _update_ewma(obj.get("err", 0.0))
    _process_event(obj)
    _broadcast(obj)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", threshold=state.threshold, demo=state.demo_mode)


@app.route("/stream")
def stream():
    client_q: queue.Queue = queue.Queue(maxsize=MAX_Q)
    with _clients_lock:
        _clients.append(client_q)

    def generate():
        try:
            yield "retry: 1000\n\n"
            while True:
                try:
                    obj = client_q.get(timeout=5)
                    yield f"data: {json.dumps(obj)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _clients_lock:
                _clients.remove(client_q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/events")
def get_events():
    with state.events_lock:
        snapshot = list(state.events)
        active   = dict(state.active_run) if state.active_run else None
    return {"events": snapshot, "active": active}


@app.route("/events/export.csv")
def export_events():
    with state.events_lock:
        snapshot = list(state.events)

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "start_ts", "end_ts", "duration_ms",
            "peak_err", "peak_severity", "severity_label", "frame_count",
            "dominant_fault",
        ],
    )
    writer.writeheader()
    sev_labels = {0: "Normal", 1: "Warning", 2: "Critical"}
    for ev in snapshot:
        writer.writerow({
            **ev,
            "duration_ms":    ev["end_ts"] - ev["start_ts"],
            "severity_label": sev_labels.get(ev["peak_severity"], "Normal"),
            "dominant_fault": ev.get("dominant_fault", "none"),
        })
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=anomaly_events.csv"},
    )


@app.route("/threshold", methods=["POST"])
def set_threshold():
    body = request.get_json(silent=True) or {}
    try:
        state.threshold        = float(body["threshold"])
        state.threshold_source = "manual"
        return {"ok": True, "threshold": state.threshold}
    except (KeyError, ValueError):
        return {"ok": False, "error": "invalid threshold"}, 400


@app.route("/threshold/auto", methods=["GET", "POST"])
def auto_threshold():
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        enabled = bool(body.get("enabled", False))
        with state.auto_threshold_lock:
            state.auto_threshold = enabled
            if not enabled:
                state.ewma_value = None   # reset so next enable starts fresh
        return {"ok": True, "auto": state.auto_threshold}
    return {
        "auto":      state.auto_threshold,
        "threshold": state.threshold,
        "ewma":      state.ewma_value,
    }


@app.route("/meta")
def meta():
    return {
        "demo_mode":        state.demo_mode,
        "threshold_source": "auto" if state.auto_threshold else state.threshold_source,
        **state.model_meta,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def _load_model_meta():
    """Pull whatever training metadata is on disk; every field degrades gracefully."""
    meta = {"model_available": False, "model_type": None, "window": None,
            "latent_dim": None, "threshold_pct": None,
            "classifier_available": False, "fault_classes": []}

    threshold = THRESHOLD
    source    = "default"
    cfg_path  = MODEL_DIR / "config.json"
    thr_path  = MODEL_DIR / "threshold.txt"
    clf_cfg_path = MODEL_DIR / "classifier_config.json"

    if thr_path.exists():
        try:
            threshold = float(thr_path.read_text().strip())
            source    = "model_file"
        except ValueError:
            pass

    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            meta.update({
                "model_available": True,
                "model_type":      cfg.get("model_type"),
                "window":          cfg.get("window"),
                "latent_dim":      cfg.get("latent_dim"),
                "threshold_pct":   cfg.get("threshold_pct"),
            })
        except (json.JSONDecodeError, OSError):
            pass

    if clf_cfg_path.exists():
        try:
            clf_cfg = json.loads(clf_cfg_path.read_text())
            meta.update({
                "classifier_available": True,
                "fault_classes":        clf_cfg.get("labels", []),
            })
        except (json.JSONDecodeError, OSError):
            pass

    return threshold, source, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None, help="Serial port, e.g. COM3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--flask-port", type=int, default=5000)
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic data (no ESP32 required)")
    args = parser.parse_args()

    state.threshold, state.threshold_source, state.model_meta = _load_model_meta()
    state.demo_mode = args.demo or (args.port is None)

    if state.demo_mode:
        print("[app] Demo mode — generating synthetic sensor data.")
        threading.Thread(target=demo_generator, daemon=True).start()
    else:
        threading.Thread(target=serial_reader, args=(args.port,), daemon=True).start()

    print(f"[app] Dashboard -> http://{args.host}:{args.flask_port}/")
    app.run(host=args.host, port=args.flask_port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
