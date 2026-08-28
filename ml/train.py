"""
Phase 2 — Train a VAE (Variational Autoencoder) for anomaly detection.

Architecture:
  Encoder: LSTM(64) → z_mean, z_log_var  (latent_dim each)
  Sampling: z = z_mean + exp(0.5·z_log_var) · ε   (reparameterization trick)
  Decoder: RepeatLatent (broadcast) → LSTM(64) → TimeDistributed Dense

Loss: reconstruction_MSE + kl_beta · KL_divergence

Anomaly score: reconstruction MSE only, by default. This is deliberate. The
exported TFLite model has a single output and never surfaces z_log_var, so the
firmware physically cannot compute the KL term (see firmware/src/main.cpp
runInference). Calibrating the threshold on recon+KL while the device compares
against recon alone makes the on-device threshold systematically too high and
biases it toward MISSING anomalies. Pass --score combined only for offline
analysis, never for a threshold that will be flashed.

Usage:
    python train.py --profile env_safety
    python train.py --profile kitchen --window 60 --stride 30 --epochs 50
    python train.py --profile env_safety --score combined   # offline analysis only

The feature layout and default window/stride come from profiles.py; the profile
chosen here is recorded in config.json so evaluate.py / train_classifier.py /
convert_tflite.py read it back rather than re-assume it.

Reads CSVs from ../data_collection/raw/<profile>/normal_*.csv. Saves:
    models/vae_encoder.keras      — encoder (outputs z_mean, z_log_var, z)
    models/vae_decoder.keras      — decoder (z → reconstruction)
    models/autoencoder.keras      — deterministic inference model (z_mean path, TFLite-ready)
    models/scaler.pkl             — fitted StandardScaler
    models/threshold.txt          — anomaly-score threshold (float), in the units of --score
    models/config.json            — training config
"""

import argparse
import json
import os
from pathlib import Path

# TFLite's LSTM fusion pass (needed for on-device inference — see convert_tflite.py)
# only recognizes the composite op that Keras 2's LSTM tracing produces. Must be
# set before the first `import tensorflow` in the process.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

from console import enable_utf8_output
from profiles import get_profile, profile_names, Profile, WINDOW, STRIDE

RAW_DIR = Path(__file__).parent.parent / "data_collection" / "raw"
MODEL_DIR = Path(__file__).parent / "models"


# ── Data helpers ───────────────────────────────────────────────────────────────


def load_csvs(label: str, profile: Profile) -> pd.DataFrame:
    """Concatenate every raw/<profile>/<label>_*.csv into one frame.

    Recordings are filed per profile because Node 1 and Node 2 both produce
    normal_*.csv with different columns; reading only this profile's directory is
    what stops them being blended into one model.
    """
    profile_dir = RAW_DIR / profile.name
    frames = []
    for p in sorted(profile_dir.glob(f"{label}_*.csv")):
        df = pd.read_csv(p, usecols=list(profile.features))
        df.dropna(inplace=True)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No '{label}_*.csv' files found in {profile_dir}. Run serial_listener.py "
            f"against a {profile.name!r} node first."
        )
    return pd.concat(frames, ignore_index=True)


def make_windows(data: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Overlapping sliding windows: (n, window, n_features).

    stride < window gives overlap (stride == window is the old non-overlapping
    behaviour). The 50 % overlap the pipeline uses also stops an event straddling
    a window boundary from being split across two inferences.
    """
    if len(data) < window:
        return np.empty((0, window, data.shape[1]), dtype=data.dtype)
    n_windows = (len(data) - window) // stride + 1
    idx = np.arange(window)[None, :] + stride * np.arange(n_windows)[:, None]
    return data[idx]


# ── VAE building blocks ────────────────────────────────────────────────────────


class Sampling(keras.layers.Layer):
    """Reparameterization trick: z = z_mean + exp(0.5·z_log_var) · ε."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class RepeatLatent(keras.layers.Layer):
    """RepeatVector replacement that lowers to BROADCAST_TO in TFLite instead
    of TILE — the TFLite Micro build in the firmware's library has no TILE
    kernel, but does ship BROADCAST_TO (registered in main.cpp's resolver)."""

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def call(self, z):
        z = tf.expand_dims(z, 1)  # [batch, 1, latent]
        return tf.broadcast_to(z, [tf.shape(z)[0], self.n, tf.shape(z)[2]])  # [batch, n, latent]

    def get_config(self):
        cfg = super().get_config()
        cfg["n"] = self.n
        return cfg


def build_encoder(window: int, n_features: int, latent_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(window, n_features), name="encoder_input")
    x = keras.layers.LSTM(64, return_sequences=False)(inputs)
    z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z")([z_mean, z_log_var])
    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


def build_decoder(window: int, n_features: int, latent_dim: int) -> keras.Model:
    z_input = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = RepeatLatent(window)(z_input)
    x = keras.layers.LSTM(64, return_sequences=True)(x)
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(n_features))(x)
    return keras.Model(z_input, outputs, name="decoder")


class VAEModel(keras.Model):
    """LSTM-VAE trained with reconstruction + KL loss."""

    def __init__(self, encoder: keras.Model, decoder: keras.Model, kl_beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_beta = kl_beta
        self._total_loss = keras.metrics.Mean(name="loss")
        self._recon_loss = keras.metrics.Mean(name="recon_loss")
        self._kl_loss = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self._total_loss, self._recon_loss, self._kl_loss]

    def _compute_loss(self, x, training: bool):
        z_mean, z_log_var, z = self.encoder(x, training=training)
        reconstruction = self.decoder(z, training=training)
        recon = tf.reduce_mean(tf.square(x - reconstruction))
        kl = -0.5 * tf.reduce_mean(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return recon + self.kl_beta * kl, recon, kl

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            total, recon, kl = self._compute_loss(x, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(total, self.trainable_weights), self.trainable_weights)
        )
        self._total_loss.update_state(total)
        self._recon_loss.update_state(recon)
        self._kl_loss.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        total, recon, kl = self._compute_loss(x, training=False)
        self._total_loss.update_state(total)
        self._recon_loss.update_state(recon)
        self._kl_loss.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x, training=False):
        _, _, z = self.encoder(x, training=training)
        return self.decoder(z, training=training)


def build_inference_model(encoder: keras.Model, decoder: keras.Model) -> keras.Model:
    """Deterministic model using z_mean (no sampling) — single output for TFLite."""
    inputs = encoder.input
    z_mean, _, _ = encoder(inputs)
    reconstruction = decoder(z_mean)
    return keras.Model(inputs, reconstruction, name="inference_model")


# ── Anomaly scoring ────────────────────────────────────────────────────────────


SCORE_MODES = ("recon", "combined")


def anomaly_score(
    encoder: keras.Model,
    decoder: keras.Model,
    X: np.ndarray,
    kl_beta: float,
    mode: str = "recon",
) -> np.ndarray:
    """Per-window anomaly score.

    mode="recon"    — reconstruction MSE. Matches what the firmware computes, so a
                      threshold calibrated here is directly comparable on-device.
    mode="combined" — recon_MSE + kl_beta · KL. Offline analysis only: the device
                      cannot compute the KL term.
    """
    if mode not in SCORE_MODES:
        raise ValueError(f"score mode must be one of {SCORE_MODES}, got {mode!r}")

    z_mean, z_log_var, _ = encoder.predict(X, verbose=0)
    recon = decoder.predict(z_mean, verbose=0)  # deterministic path
    recon_err = np.mean(np.square(X - recon), axis=(1, 2))
    if mode == "recon":
        return recon_err

    kl = -0.5 * np.mean(1.0 + z_log_var - z_mean**2 - np.exp(z_log_var), axis=1)
    return recon_err + kl_beta * kl


# ── Main ───────────────────────────────────────────────────────────────────────


def main(
    profile_name: str,
    window: int,
    stride: int,
    epochs: int,
    batch: int,
    threshold_pct: float,
    latent_dim: int,
    kl_beta: float,
    score_mode: str = "recon",
):
    MODEL_DIR.mkdir(exist_ok=True)

    profile = get_profile(profile_name)
    features = list(profile.features)
    print(f"Profile {profile.name!r} (id {profile.id}): features {features}")

    print("Loading normal training data…")
    normal_df = load_csvs("normal", profile)
    print(f"  {len(normal_df)} normal samples")

    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal_df[features].values)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print("  Scaler saved.")

    X_normal = make_windows(normal_scaled, window, stride)
    print(f"  {len(X_normal)} windows of size {window} (stride {stride})")
    if len(X_normal) < 2:
        raise SystemExit(
            f"Only {len(X_normal)} window(s) from {len(normal_df)} samples at window={window}, "
            f"stride={stride}. Collect more 'normal' data before training."
        )

    split = int(0.8 * len(X_normal))
    X_train = X_normal[:split]
    X_val = X_normal[split:]

    print(f"\nBuilding VAE (latent_dim={latent_dim}, kl_beta={kl_beta})…")
    n_features = profile.n_features
    encoder = build_encoder(window, n_features, latent_dim)
    decoder = build_decoder(window, n_features, latent_dim)
    vae = VAEModel(encoder, decoder, kl_beta=kl_beta, name="lstm_vae")
    vae.compile(optimizer="adam")
    encoder.summary()
    decoder.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5),
    ]

    print("\nTraining…")
    history = vae.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch,
        callbacks=callbacks,
        verbose=1,
    )

    encoder.save(MODEL_DIR / "vae_encoder.keras")
    decoder.save(MODEL_DIR / "vae_decoder.keras")
    print("Encoder/decoder saved.")

    inf_model = build_inference_model(encoder, decoder)
    inf_model.save(MODEL_DIR / "autoencoder.keras")
    print("Inference model saved as autoencoder.keras (TFLite-ready).")

    val_scores = anomaly_score(encoder, decoder, X_val, kl_beta, score_mode)
    threshold = float(np.percentile(val_scores, threshold_pct))
    (MODEL_DIR / "threshold.txt").write_text(str(threshold))
    print(f"Threshold ({threshold_pct}th pct of val {score_mode} scores): {threshold:.6f}")
    if score_mode != "recon":
        print(
            "  WARNING: this threshold is NOT device-compatible. The firmware compares\n"
            "  against reconstruction MSE only. convert_tflite.py will refuse to export it."
        )

    cfg = {
        "model_type": "vae",
        "profile": profile.name,
        "profile_id": profile.id,
        "window": window,
        "stride": stride,
        "features": features,
        "latent_dim": latent_dim,
        "kl_beta": kl_beta,
        "threshold": threshold,
        "threshold_pct": threshold_pct,
        "score_mode": score_mode,
    }
    (MODEL_DIR / "config.json").write_text(json.dumps(cfg, indent=2))

    _plot_loss(history, MODEL_DIR)
    print("\nDone. Run evaluate.py to see anomaly separation.")


def _plot_loss(history, out_dir: Path):
    keys = [
        ("loss", "val_loss", "Total Loss"),
        ("recon_loss", "val_recon_loss", "Reconstruction MSE"),
        ("kl_loss", "val_kl_loss", "KL Divergence"),
    ]
    present = [(tk, vk, title) for tk, vk, title in keys if tk in history.history]
    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4))
    if len(present) == 1:
        axes = [axes]
    for ax, (tk, vk, title) in zip(axes, present):
        ax.plot(history.history[tk], label="train")
        if vk in history.history:
            ax.plot(history.history[vk], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()
    print(f"Loss curves saved to {out_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    enable_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        required=True,
        choices=profile_names(),
        help="Sensor profile to train (selects feature layout and raw/<profile>/ data).",
    )
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument(
        "--stride",
        type=int,
        default=STRIDE,
        help=f"Sliding-window stride (default {STRIDE}; {STRIDE}<window gives overlap).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument(
        "--kl-beta", type=float, default=0.5, help="Weight for KL term in VAE loss (default: 0.5)"
    )
    parser.add_argument("--threshold-pct", type=float, default=95)
    parser.add_argument(
        "--score",
        choices=SCORE_MODES,
        default="recon",
        help="Anomaly-score metric for threshold calibration. 'recon' matches the "
        "firmware and is the only device-compatible choice (default: recon).",
    )
    args = parser.parse_args()
    main(
        args.profile,
        args.window,
        args.stride,
        args.epochs,
        args.batch,
        args.threshold_pct,
        args.latent_dim,
        args.kl_beta,
        args.score,
    )
