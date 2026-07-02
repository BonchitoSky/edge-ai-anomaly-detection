"""
Phase 2.5 — Train a multi-class fault classifier (root-cause hints).

The VAE (train.py) only answers "is this anomalous?". This script trains a
second, supervised classifier that answers "what kind of anomaly is it?",
using whatever fault-type labels you've collected with:

    python ../data_collection/serial_listener.py --port COM3 --label drop
    python ../data_collection/serial_listener.py --port COM3 --label shake
    python ../data_collection/serial_listener.py --port COM3 --label imbalance

Any label other than 'normal' or the legacy generic 'anomaly' is treated as
a distinct fault class. Collect at least two to train a classifier.

Design: instead of a second sequence model, this uses cheap per-window
statistical features (mean/std/min/max/peak-to-peak per axis = 35 inputs)
fed into a small Dense network. Much smaller TFLite footprint than a second
LSTM, and needs far less labeled data per class. Feature extraction is
mirrored exactly in firmware/src/main.cpp's extractFeatures() — the order
here (per-feature: mean, std, min, max, ptp) must match that function.

Reuses the VAE's fitted scaler (models/scaler.pkl) so the classifier sees
the same scaled values the firmware already computes into windowBuf — no
second scaler needs to be shipped to the device.

Usage:
    python train_classifier.py
    python train_classifier.py --window 50 --epochs 50

Reads CSVs from ../data_collection/raw/<label>_*.csv (label != normal/anomaly).
Saves:
    models/classifier.keras          — trained Dense classifier
    models/classifier_labels.json    — ordered class names (index = class id)
    models/classifier_config.json    — window, n_features, labels
    models/classifier_confusion.png  — confusion matrix on the val split
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# Must match train.py/convert_tflite.py — keeps every .keras artifact in this
# pipeline on the same Keras version, set before the first `import tensorflow`.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

RAW_DIR = Path(__file__).parent.parent / "data_collection" / "raw"
MODEL_DIR = Path(__file__).parent / "models"
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz", "temp"]
RESERVED_LABELS = {"normal", "anomaly"}  # not fault classes


# ── Data helpers ─────────────────────────────────────────────────────────────

def discover_fault_labels() -> dict:
    """Map fault label -> sorted list of matching CSV paths."""
    labels = defaultdict(list)
    for p in sorted(RAW_DIR.glob("*.csv")):
        label = p.stem.rsplit("_", 2)[0]  # strip _YYYYMMDD_HHMMSS
        if label in RESERVED_LABELS:
            continue
        labels[label].append(p)
    return dict(labels)


def load_label_csv(paths: list) -> pd.DataFrame:
    frames = [pd.read_csv(p, usecols=FEATURES).dropna() for p in paths]
    return pd.concat(frames, ignore_index=True)


def make_windows(data: np.ndarray, window: int) -> np.ndarray:
    n_windows = len(data) // window
    return data[: n_windows * window].reshape(n_windows, window, data.shape[1])


def extract_features(windows: np.ndarray) -> np.ndarray:
    """windows: (n, window, n_features) -> (n, n_features*5), per-feature
    [mean, std, min, max, ptp], matching firmware extractFeatures() order."""
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    mn = windows.min(axis=1)
    mx = windows.max(axis=1)
    ptp = mx - mn
    n = windows.shape[0]
    nf = windows.shape[2]
    out = np.empty((n, nf * 5), dtype=np.float32)
    for i in range(nf):
        out[:, i * 5 + 0] = mean[:, i]
        out[:, i * 5 + 1] = std[:, i]
        out[:, i * 5 + 2] = mn[:, i]
        out[:, i * 5 + 3] = mx[:, i]
        out[:, i * 5 + 4] = ptp[:, i]
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main(window: int, epochs: int, batch: int):
    MODEL_DIR.mkdir(exist_ok=True)

    scaler_path = MODEL_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"{scaler_path} not found. Run train.py first (it fits the scaler "
            "on normal data that this classifier reuses)."
        )
    scaler = joblib.load(scaler_path)

    fault_csvs = discover_fault_labels()
    if len(fault_csvs) < 2:
        found = sorted(fault_csvs)
        raise FileNotFoundError(
            f"Need at least 2 distinct fault labels, found {found}. Collect more "
            "with: python ../data_collection/serial_listener.py --label <type>"
        )

    labels = sorted(fault_csvs)  # index = class id, saved for firmware + dashboard
    print(f"Fault classes ({len(labels)}): {labels}")

    X_list, y_list = [], []
    for class_id, label in enumerate(labels):
        df = load_label_csv(fault_csvs[label])
        scaled = scaler.transform(df[FEATURES].values)
        windows = make_windows(scaled, window)
        print(f"  {label}: {len(df)} samples -> {len(windows)} windows")
        X_list.append(extract_features(windows))
        y_list.append(np.full(len(windows), class_id))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    n_classes = len(labels)
    n_features = X.shape[1]

    # Stratified 80/20 split
    rng = np.random.default_rng(42)
    train_idx, val_idx = [], []
    for class_id in range(n_classes):
        idx = np.where(y == class_id)[0]
        rng.shuffle(idx)
        split = max(1, int(0.8 * len(idx)))
        train_idx.extend(idx[:split])
        val_idx.extend(idx[split:])
    train_idx, val_idx = np.array(train_idx), np.array(val_idx)
    rng.shuffle(train_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"\nTrain windows: {len(X_train)}  |  Val windows: {len(X_val)}")

    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ], name="fault_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                       restore_best_weights=True),
    ]

    print("\nTraining…")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_DIR / "classifier.keras")
    print("Classifier saved.")

    (MODEL_DIR / "classifier_labels.json").write_text(json.dumps(labels, indent=2))
    cfg = {"window": window, "n_features": n_features, "labels": labels}
    (MODEL_DIR / "classifier_config.json").write_text(json.dumps(cfg, indent=2))

    if len(X_val) > 0:
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        print("\nValidation classification report:")
        print(classification_report(y_val, y_pred, target_names=labels, labels=range(n_classes)))
        _plot_confusion(y_val, y_pred, labels, MODEL_DIR)
    else:
        print("\nNo validation windows available — skipping classification report.")

    print("\nDone. Run convert_tflite.py to export classifier_data.h / classifier_meta.h.")


def _plot_confusion(y_true, y_pred, labels, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Fault Classifier — Validation Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_dir / "classifier_confusion.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {out_dir / 'classifier_confusion.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    main(args.window, args.epochs, args.batch)
