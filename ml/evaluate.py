"""
Phase 2 — Evaluate the trained autoencoder/VAE on normal vs anomaly data.

Usage:
    python evaluate.py
    python evaluate.py --profile env_safety   # optional cross-check against config.json

The profile, window and stride are read from config.json (what train.py actually
trained with), so evaluation always matches the model rather than a CLI default.

For a VAE (model_type=vae in config.json) the anomaly score follows the
score_mode recorded in config.json by train.py — "recon" (reconstruction MSE,
what the firmware computes) or "combined" (recon + kl_beta · KL, offline only).
Scoring with a different metric than the threshold was calibrated on makes every
rate below meaningless, so the mode is read from config, never assumed.
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

from console import enable_utf8_output
from profiles import get_profile, Profile

RAW_DIR = Path(__file__).parent.parent / "data_collection" / "raw"
MODEL_DIR = Path(__file__).parent / "models"


def _concat(paths, features) -> pd.DataFrame:
    cols = list(features)
    frames = [pd.read_csv(p, usecols=cols).dropna() for p in paths]
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def load_csvs(label: str, profile: Profile) -> pd.DataFrame:
    return _concat(sorted((RAW_DIR / profile.name).glob(f"{label}_*.csv")), profile.features)


def load_anomaly_csvs(profile: Profile) -> tuple[pd.DataFrame, list[str]]:
    """Every recording under raw/<profile>/ that is not normal_*.csv.

    A fault recording (gasleak_*, overheat_*, ...) is by definition an anomaly, so
    globbing only anomaly_*.csv silently skipped the fault data and left ROC-AUC
    unreported — the exact metric SETUP.md tells you to check. Any non-normal label
    counts here; train_classifier.py separately decides which are fault *classes*.
    """
    profile_dir = RAW_DIR / profile.name
    paths = sorted(p for p in profile_dir.glob("*.csv") if not p.stem.startswith("normal_"))
    labels = sorted({p.stem.rsplit("_", 2)[0] for p in paths})
    return _concat(paths, profile.features), labels


def make_windows(data: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Overlapping sliding windows, matching train.make_windows (same stride)."""
    if len(data) < window:
        return np.empty((0, window, data.shape[1]), dtype=data.dtype)
    n = (len(data) - window) // stride + 1
    idx = np.arange(window)[None, :] + stride * np.arange(n)[:, None]
    return data[idx]


# ── Scoring ────────────────────────────────────────────────────────────────────


def score_plain(model, X: np.ndarray):
    """Simple MSE reconstruction error for a plain autoencoder."""
    preds = model.predict(X, verbose=0)
    return np.mean(np.square(X - preds), axis=(1, 2)), None, None


def score_vae(encoder, decoder, X: np.ndarray, kl_beta: float, score_mode: str = "recon"):
    """Anomaly score + per-component arrays for a VAE.

    score_mode must match what train.py used to set the threshold.
    """
    z_mean, z_log_var, _ = encoder.predict(X, verbose=0)
    recon = decoder.predict(z_mean, verbose=0)
    recon_err = np.mean(np.square(X - recon), axis=(1, 2))
    kl = -0.5 * np.mean(1.0 + z_log_var - z_mean**2 - np.exp(z_log_var), axis=1)
    score = recon_err if score_mode == "recon" else recon_err + kl_beta * kl
    return score, recon_err, kl


# ── Main ───────────────────────────────────────────────────────────────────────


def main(profile_override: str | None):
    cfg_path = MODEL_DIR / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("models/config.json not found. Run train.py first.")

    cfg = json.loads(cfg_path.read_text())
    is_vae = cfg.get("model_type") == "vae"
    kl_beta = cfg.get("kl_beta", 1.0)
    # Pre-fix models have no score_mode; they were calibrated on the combined score.
    score_mode = cfg.get("score_mode", "combined")

    # Profile/window/stride come from the trained config, never a CLI default —
    # evaluating with a different window than was trained makes every rate below
    # meaningless. Pre-profile configs predate the home sensors and are IMU-legacy.
    profile = get_profile(cfg.get("profile", "imu_legacy"))
    if profile_override is not None and profile_override != profile.name:
        raise SystemExit(
            f"--profile {profile_override!r} contradicts the trained model "
            f"(profile={profile.name!r}). Omit --profile to use the trained value."
        )
    window = cfg["window"]
    stride = cfg.get("stride", window)  # legacy configs had no stride → non-overlapping
    features = list(profile.features)
    print(f"Profile {profile.name!r}: window {window}, stride {stride}, features {features}")

    threshold = float((MODEL_DIR / "threshold.txt").read_text())
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

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
        print(f"  score_mode = {score_mode}")
        if score_mode != "recon":
            print(
                "  WARNING: score_mode is not 'recon' — these numbers do NOT reflect\n"
                "  on-device behaviour. Retrain with --score recon before flashing."
            )
    else:
        print("Detected plain autoencoder — loading model…")
        model = tf.keras.models.load_model(MODEL_DIR / "autoencoder.keras")

    normal_df = load_csvs("normal", profile)
    anomaly_df, anomaly_labels = load_anomaly_csvs(profile)

    if normal_df.empty:
        raise ValueError("No normal CSV data found.")
    if anomaly_df.empty:
        print(
            "WARNING: No non-normal recordings found in data_collection/raw/.\n"
            "  ROC-AUC cannot be computed. Record at least one fault session, e.g.\n"
            "  python data_collection/serial_listener.py --port COM3 --label gasleak"
        )
    else:
        print(f"Anomaly recordings found: {', '.join(anomaly_labels)}")

    X_normal = make_windows(scaler.transform(normal_df[features].values), window, stride)

    if is_vae:
        err_normal, recon_n, kl_n = score_vae(encoder, decoder, X_normal, kl_beta, score_mode)
    else:
        err_normal, recon_n, kl_n = score_plain(model, X_normal)

    sep = "─" * 44
    print(f"\n{sep}")
    print(f"Normal — mean score: {err_normal.mean():.6f}, std: {err_normal.std():.6f}")
    if is_vae and recon_n is not None:
        print(
            f"  recon_MSE: {recon_n.mean():.6f}  |  "
            f"KL (×{kl_beta}): {(kl_beta * kl_n).mean():.6f}"
        )
    print(f"Threshold: {threshold:.6f}")
    fp_rate = (err_normal > threshold).mean()
    print(f"False-positive rate on normal: {fp_rate:.2%}")

    if not anomaly_df.empty:
        X_anomaly = make_windows(scaler.transform(anomaly_df[features].values), window, stride)

        if is_vae:
            err_anomaly, recon_a, kl_a = score_vae(encoder, decoder, X_anomaly, kl_beta, score_mode)
        else:
            err_anomaly, recon_a, kl_a = score_plain(model, X_anomaly)

        print(f"Anomaly — mean score: {err_anomaly.mean():.6f}, std: {err_anomaly.std():.6f}")
        if is_vae and recon_a is not None:
            print(
                f"  recon_MSE: {recon_a.mean():.6f}  |  "
                f"KL (×{kl_beta}): {(kl_beta * kl_a).mean():.6f}"
            )

        tp_rate = (err_anomaly > threshold).mean()
        print(f"True-positive rate on anomaly: {tp_rate:.2%}")

        y_true = np.concatenate([np.zeros(len(err_normal)), np.ones(len(err_anomaly))])
        y_score = np.concatenate([err_normal, err_anomaly])
        auc = roc_auc_score(y_true, y_score)
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
        plt.hist(err_normal, bins=50, alpha=0.6, label="normal", color="steelblue")
        plt.hist(err_anomaly, bins=50, alpha=0.6, label="anomaly", color="tomato")
        plt.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.4f}")
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
        plt.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.4f}")
        plt.xlabel(score_label)
        plt.ylabel("Count")
        plt.title("Normal Score Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "eval_histogram.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    enable_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional cross-check that the trained model is this profile; the "
        "profile, window and stride are otherwise read from config.json.",
    )
    args = parser.parse_args()
    main(args.profile)
