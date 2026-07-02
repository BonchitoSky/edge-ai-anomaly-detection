"""
Phase 3 — Convert trained Keras model → quantized TFLite → C header for ESP32.

Usage:
    python convert_tflite.py
    python convert_tflite.py --window 50 --no-quantize

Outputs:
    models/autoencoder.tflite        — float32 TFLite model
    models/autoencoder_int8.tflite   — int8-quantized TFLite model
    ../firmware/include/model_data.h — C array ready to compile into ESP32
    ../firmware/include/model_meta.h — threshold + feature scaling constants

If models/classifier.keras exists (see train_classifier.py), also converts
the fault-type classifier to float32 TFLite (small enough that int8
quantization isn't worth the added complexity) and writes:
    ../firmware/include/classifier_data.h — C array for the classifier
    ../firmware/include/classifier_meta.h — class names + feature count

Requires the 'tf-keras' package (see requirements.txt) — the VAE's LSTM
layers need Keras 2's legacy RNN tracing (TF_USE_LEGACY_KERAS=1, set below)
for TFLite's LSTM fusion pass to produce the fused UnidirectionalSequenceLSTM
op the firmware's op resolver expects, instead of failing to legalize
'tf.TensorListReserve'.
"""

import argparse
import json
import os
from pathlib import Path

# Must match train.py — loading a model saved under Keras 2 (legacy) requires
# the same legacy mode, set before the first `import tensorflow` in the process.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_DIR = Path(__file__).parent / "models"
FIRMWARE_INCLUDE = Path(__file__).parent.parent / "firmware" / "include"
RAW_DIR = Path(__file__).parent.parent / "data_collection" / "raw"
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz", "temp"]


def load_representative_dataset(scaler, window: int, n_samples: int = 200):
    """Yield representative input tensors for int8 calibration."""
    frames = []
    for p in sorted(RAW_DIR.glob("normal_*.csv")):
        df = pd.read_csv(p, usecols=FEATURES).dropna()
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No normal CSV data found for calibration.")

    data = scaler.transform(pd.concat(frames, ignore_index=True)[FEATURES].values)
    n = min(n_samples, len(data) // window)
    windows = data[: n * window].reshape(n, window, len(FEATURES)).astype(np.float32)

    def gen():
        for w in windows:
            yield [w[np.newaxis]]

    return gen


def to_static_batch_concrete_function(model, window: int, n_features: int):
    """TFLite's LSTM fusion pass (which produces the efficient, TFLite-Micro-
    friendly UnidirectionalSequenceLSTM op) requires a static element_shape,
    which needs a fixed batch dimension. TFLiteConverter.from_keras_model()
    traces with a dynamic (None) batch and fails to legalize the VAE decoder's
    LSTM ('tf.TensorListReserve ... requires element_shape to be static').
    Tracing a concrete function with batch=1 first avoids that."""
    return tf.function(lambda x: model(x)).get_concrete_function(
        tf.TensorSpec([1, window, n_features], tf.float32)
    )


def to_c_array(tflite_bytes: bytes, var_name: str = "g_model_data",
                len_name: str = "g_model_len") -> str:
    lines = []
    lines.append("#pragma once")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"const unsigned int {len_name} = {len(tflite_bytes)};")
    lines.append(f"alignas(8) const uint8_t {var_name}[] = {{")
    chunk = 12
    chunks = [
        "  " + ", ".join(f"0x{b:02x}" for b in tflite_bytes[i : i + chunk])
        for i in range(0, len(tflite_bytes), chunk)
    ]
    lines.append(",\n".join(chunks))
    lines.append("};")
    return "\n".join(lines)


def convert_classifier():
    """Convert models/classifier.keras (if present) to float32 TFLite + C headers."""
    clf_path = MODEL_DIR / "classifier.keras"
    if not clf_path.exists():
        print("\nNo classifier.keras found — skipping fault classifier export "
              "(run train_classifier.py first if you want fault-type hints).")
        return

    cfg = json.loads((MODEL_DIR / "classifier_config.json").read_text())
    class_names = json.loads((MODEL_DIR / "classifier_labels.json").read_text())

    model_cfg = json.loads((MODEL_DIR / "config.json").read_text())
    if cfg["window"] != model_cfg["window"]:
        raise ValueError(
            f"Classifier window ({cfg['window']}) != autoencoder window "
            f"({model_cfg['window']}). Retrain one so they match — the firmware "
            "shares a single windowBuf between both models."
        )

    model = tf.keras.models.load_model(clf_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    tflite_path = MODEL_DIR / "classifier.tflite"
    tflite_path.write_bytes(tflite_bytes)
    print(f"\nClassifier float32 TFLite: {tflite_path} ({len(tflite_bytes)/1024:.1f} KB)")

    c_header = to_c_array(tflite_bytes, var_name="g_classifier_data",
                           len_name="g_classifier_len")
    data_h = FIRMWARE_INCLUDE / "classifier_data.h"
    data_h.write_text(c_header)
    print(f"C header:       {data_h}")

    n_features = cfg["n_features"]
    names_decl = ", ".join(f'"{name}"' for name in class_names)
    meta_lines = [
        "#pragma once",
        "",
        f"constexpr int kNumClasses = {len(class_names)};",
        f"constexpr int kClassifierNumFeatures = {n_features};",
        "",
        "// Per-feature [mean, std, min, max, ptp] over ax,ay,az,gx,gy,gz,temp —",
        "// order must match ml/train_classifier.py's extract_features().",
        f"constexpr const char* kClassNames[kNumClasses] = {{{names_decl}}};",
    ]
    meta_h = FIRMWARE_INCLUDE / "classifier_meta.h"
    meta_h.write_text("\n".join(meta_lines))
    print(f"Meta header:    {meta_h}")


def main(window: int, quantize: bool):
    cfg_path = MODEL_DIR / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("models/config.json not found. Run train.py first.")

    cfg       = json.loads(cfg_path.read_text())
    is_vae    = cfg.get("model_type") == "vae"
    threshold = float((MODEL_DIR / "threshold.txt").read_text())
    scaler    = joblib.load(MODEL_DIR / "scaler.pkl")

    # For VAE: autoencoder.keras is the deterministic inference model (z_mean path,
    # single input → single output). For a plain autoencoder the same file is used.
    if is_vae:
        from train import Sampling, RepeatLatent
        model = tf.keras.models.load_model(
            MODEL_DIR / "autoencoder.keras",
            custom_objects={"Sampling": Sampling, "RepeatLatent": RepeatLatent},
        )
        print("VAE detected — using deterministic inference model (z_mean path).")
    else:
        model = tf.keras.models.load_model(MODEL_DIR / "autoencoder.keras")

    concrete_func = to_static_batch_concrete_function(model, window, len(FEATURES))

    # Float32 TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    tflite_fp32 = converter.convert()
    fp32_path = MODEL_DIR / "autoencoder.tflite"
    fp32_path.write_bytes(tflite_fp32)
    print(f"Float32 TFLite: {fp32_path} ({len(tflite_fp32)/1024:.1f} KB)")

    # Int8 quantized TFLite
    if quantize:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = load_representative_dataset(scaler, window)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_int8 = converter.convert()
        int8_path = MODEL_DIR / "autoencoder_int8.tflite"
        int8_path.write_bytes(tflite_int8)
        print(f"Int8 TFLite:    {int8_path} ({len(tflite_int8)/1024:.1f} KB)")
        model_bytes = tflite_int8
    else:
        model_bytes = tflite_fp32

    # C header with model data
    FIRMWARE_INCLUDE.mkdir(exist_ok=True)
    c_header = to_c_array(model_bytes)
    model_h = FIRMWARE_INCLUDE / "model_data.h"
    model_h.write_text(c_header)
    print(f"C header:       {model_h}")

    # Meta header with threshold + scaler params
    mean = scaler.mean_.tolist()
    scale = scaler.scale_.tolist()
    n_features = len(FEATURES)

    meta_lines = [
        "#pragma once",
        "",
        f"constexpr int   kWindowSize  = {window};",
        f"constexpr int   kNumFeatures = {n_features};",
        f"constexpr float kThreshold   = {threshold}f;",
        "",
        "// StandardScaler parameters (fit on normal training data)",
        f"constexpr float kScalerMean[{n_features}]  = {{" + ", ".join(f"{v:.6f}f" for v in mean) + "};",
        f"constexpr float kScalerScale[{n_features}] = {{" + ", ".join(f"{v:.6f}f" for v in scale) + "};",
        "",
        "// Feature order: " + ", ".join(FEATURES),
    ]
    meta_h = FIRMWARE_INCLUDE / "model_meta.h"
    meta_h.write_text("\n".join(meta_lines))
    print(f"Meta header:    {meta_h}")

    convert_classifier()

    print("\nPhase 3 complete. Flash the firmware with 'pio run --target upload'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.set_defaults(quantize=True)
    args = parser.parse_args()
    main(args.window, args.quantize)
