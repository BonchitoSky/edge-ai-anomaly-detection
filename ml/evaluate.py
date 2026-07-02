"""
Phase 2 — Evaluate the trained autoencoder/VAE on normal vs anomaly data.

Usage:
    python evaluate.py
    python evaluate.py --window 50

For a VAE (model_type=vae in config.json), anomaly score =
    reconstruction_MSE + kl_beta · KL_divergence
which is shown decomposed in the output.

Reads models/ produced by train.py, computes scores, prints metrics, saves:
    models/eval_histogram.png   — score distribution plot
    models/roc_curve.png        — ROC curve (AUC)
"""

import argparse
import json
import os
from pathlib import Path

# Must match train.py — loading a model saved under Keras 2 (legacy) requires
# the same legacy mode, set before the first `import tensorflow` in the process.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import tensorflow as tf

RAW_DIR   = Path(__file__).parent.parent / "data_collection" / "raw"
MODEL_DIR = Path(__file__).parent / "models"
FEATURES  = ["ax", "ay", "az", "gx", "gy", "gz", "temp"]


def load_csvs(label: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(p, usecols=FEATURES).dropna()
        for p in sorted(RAW_DIR.glob(f"{label}_*.csv"))
    ]
    if not frames:
        return pd.DataFrame(columns=FEATURES)
    return pd.concat(frames, ignore_index=True)


def make_windows(data: np.ndarray, window: int) -> np.ndarray:
    n = len(data) // window
    return data[: n * window].reshape(n, window, data.shape[1])


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_plain(model, X: np.ndarray):
    """Simple MSE reconstruction error for a plain autoencoder."""
    preds = model.predict(X, verbose=0)
    return np.mean(np.square(X - preds), axis=(1, 2)), None, None


def score_vae(encoder, decoder, X: np.ndarray, kl_beta: float):
    """Combined score + per-component arrays for a VAE."""
    z_mean, z_log_var, _ = encoder.predict(X, verbose=0)
    recon     = decoder.predict(z_mean, verbose=0)
    recon_err = np.mean(np.square(X - recon), axis=(1, 2))
    kl        = -0.5 * np.mean(
        1.0 + z_log_var - z_mean ** 2 - np.exp(z_log_var), axis=1
    )
    return recon_err + kl_beta * kl, recon_err, kl


# ── Main ───────────────────────────────────────────────────────────────────────

def main(window: int):
    cfg_path = MODEL_DIR / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("models/config.json not found. Run train.py first.")

    cfg       = json.loads(cfg_path.read_text())
    is_vae    = cfg.get("model_type") == "vae"
    kl_beta   = cfg.get("kl_beta", 1.0)
    threshold = float((MODEL_DIR / "threshold.txt").read_text())
    scaler    = joblib.load(MODEL_DIR / "scaler.pkl")

    if is_vae:
        print("Detected VAE model — loading encoder + decoder…")
        from train import Sampling, RepeatLatent  # custom layers for deserialization
        encoder = tf.keras.models.load_model(
            MODEL_DIR / "vae_encoder.keras",
            custom_objects={"Sampling": Sampling},
        )
        decoder = tf.keras.models.load_model(
            MODEL_DIR / "vae_decoder.keras",
            custom_objects={"RepeatLatent": RepeatLatent},
        )
        print(f"  kl_beta = {kl_beta}")
    else:
        print("Detected plain autoencoder — loading model…")
        model = tf.keras.models.load_model(MODEL_DIR / "autoencoder.keras")

    normal_df  = load_csvs("normal")
    anomaly_df = load_csvs("anomaly")

    if normal_df.empty:
        raise ValueError("No normal CSV data found.")
    if anomaly_df.empty:
        print("WARNING: No anomaly CSV data found. Only normal evaluation will run.")

    X_normal = make_windows(scaler.transform(normal_df[FEATURES].values), window)

    if is_vae:
        err_normal, recon_n, kl_n = score_vae(encoder, decoder, X_normal, kl_beta)
    else:
        err_normal, recon_n, kl_n = score_plain(model, X_normal)

    sep = "─" * 44
    print(f"\n{sep}")
    print(f"Normal — mean score: {err_normal.mean():.6f}, std: {err_normal.std():.6f}")
    if is_vae and recon_n is not None:
        print(f"  recon_MSE: {recon_n.mean():.6f}  |  "
              f"KL (×{kl_beta}): {(kl_beta * kl_n).mean():.6f}")
    print(f"Threshold: {threshold:.6f}")
    fp_rate = (err_normal > threshold).mean()
    print(f"False-positive rate on normal: {fp_rate:.2%}")

    if not anomaly_df.empty:
        X_anomaly = make_windows(scaler.transform(anomaly_df[FEATURES].values), window)

        if is_vae:
            err_anomaly, recon_a, kl_a = score_vae(encoder, decoder, X_anomaly, kl_beta)
        else:
            err_anomaly, recon_a, kl_a = score_plain(model, X_anomaly)

        print(f"Anomaly — mean score: {err_anomaly.mean():.6f}, std: {err_anomaly.std():.6f}")
        if is_vae and recon_a is not None:
            print(f"  recon_MSE: {recon_a.mean():.6f}  |  "
                  f"KL (×{kl_beta}): {(kl_beta * kl_a).mean():.6f}")

        tp_rate = (err_anomaly > threshold).mean()
        print(f"True-positive rate on anomaly: {tp_rate:.2%}")

        y_true  = np.concatenate([np.zeros(len(err_normal)), np.ones(len(err_anomaly))])
        y_score = np.concatenate([err_normal, err_anomaly])
        auc     = roc_auc_score(y_true, y_score)
        print(f"ROC-AUC: {auc:.4f}")

        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "roc_curve.png", dpi=150)
        plt.close()

        y_pred = (y_score > threshold).astype(int)
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"]))

        score_label = "Combined Score (MSE + β·KL)" if is_vae else "Reconstruction Error (MSE)"
        plt.figure(figsize=(8, 4))
        plt.hist(err_normal,  bins=50, alpha=0.6, label="normal",  color="steelblue")
        plt.hist(err_anomaly, bins=50, alpha=0.6, label="anomaly", color="tomato")
        plt.axvline(threshold, color="black", linestyle="--",
                    label=f"threshold={threshold:.4f}")
        plt.xlabel(score_label)
        plt.ylabel("Count")
        plt.title("Score Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "eval_histogram.png", dpi=150)
        plt.close()
        print(f"\nPlots saved to {MODEL_DIR}/")

    else:
        score_label = "Combined Score" if is_vae else "Reconstruction Error"
        plt.figure(figsize=(8, 4))
        plt.hist(err_normal, bins=50, alpha=0.8, label="normal", color="steelblue")
        plt.axvline(threshold, color="black", linestyle="--",
                    label=f"threshold={threshold:.4f}")
        plt.xlabel(score_label)
        plt.ylabel("Count")
        plt.title("Normal Score Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "eval_histogram.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()
    main(args.window)
