#!/usr/bin/env python
"""
FASE 1 — Exportación del artefacto TFLite entrenable on-device.

Genera el paquete completo que consume el cliente Android:

    auth_fedper.tflite     modelo con firmas de entrenamiento/inferencia/pesos
    model_manifest.json    contrato cliente <-> servidor (formas, tamaños, HPs)
    weight_layout.json     mapa del vector plano -> tensores de Keras
    scaler_stats.json      media/escala del StandardScaler global
    initial_encoder.npz    pesos del encoder horneados en el .tflite
    background_train.bin   (opcional) pool de impostores para entrenar
    background_calib.bin   (opcional) pool DISJUNTO para calibrar el umbral

Uso típico, con los artefactos que deja `mejor.py`:

    python export_tflite_model.py \
        --encoder-weights checkpoints_fl/best_encoder_weights.npz \
        --scaler-stats    scaler_stats.json \
        --background-train  bg_train_scaled.npy \
        --background-calib  bg_calib_scaled.npy \
        --android-assets-dir ../autenticacionContinua/app/src/main/assets

Sin `--encoder-weights` el encoder sale con inicialización aleatoria, lo que
sólo sirve para pruebas de integración: en producción hay que partir del
encoder pre-entrenado de la simulación (`pretrain_rounds=8` de mejor.py) para
no tirar a la basura esa etapa, que on-device sería carísima de reproducir.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import tf_env  # noqa: F401  (debe importarse antes que tensorflow)

import numpy as np
import tensorflow as tf

from app.config import FedPerConfig
from app.models import fedper_model as fp
from app.models.tflite_wrapper import TrainableAuthModel


TFLITE_NAME = "auth_fedper.tflite"
MANIFEST_NAME = "model_manifest.json"
LAYOUT_NAME = "weight_layout.json"
SCALER_NAME = "scaler_stats.json"
INITIAL_ENCODER_NAME = "initial_encoder.npz"
BG_TRAIN_NAME = "background_train.bin"
BG_CALIB_NAME = "background_calib.bin"

# Ficheros que se copian al APK. `initial_encoder.npz` se queda en el backend:
# el dispositivo recibe el encoder por gRPC, no empaquetado.
ANDROID_ASSETS = [
    TFLITE_NAME, MANIFEST_NAME, SCALER_NAME, BG_TRAIN_NAME, BG_CALIB_NAME,
]


# ---------------------------------------------------------------------------
# Carga de artefactos de la simulación
# ---------------------------------------------------------------------------


def load_npz_weights(path: Path) -> List[np.ndarray]:
    """Lee un `np.savez(path, *weights)` respetando el orden arr_0..arr_N."""
    data = np.load(path)
    return [data[f"arr_{i}"] for i in range(len(data.files))]


def load_scaler_stats(path: Optional[Path], cfg: FedPerConfig) -> dict:
    """
    Carga las estadísticas del StandardScaler global de `mejor.py`.

    Sin ellas el cliente alimentaría al modelo con lecturas crudas del sensor,
    que están en otra escala por completo (m/s^2 y rad/s frente a valores
    tipificados). El modelo no fallaría: devolvería puntuaciones sin sentido.
    Por eso, si no se aportan, se emite una identidad y un aviso ruidoso en
    lugar de fallar en silencio.
    """
    if path is None:
        print(
            "  ADVERTENCIA: no se aportó --scaler-stats. Se emite un "
            "normalizador identidad (mean=0, scale=1). El modelo NO producirá "
            "resultados válidos hasta sustituirlo por el scaler real de la "
            "simulación."
        )
        return {
            "mean": [0.0] * cfg.n_features,
            "scale": [1.0] * cfg.n_features,
            "sensor_config": cfg.sensor_config,
            "window_size": cfg.window_size,
            "channel_order": list(cfg.channel_order),
            "is_identity_placeholder": True,
        }

    with open(path) as f:
        stats = json.load(f)

    for key in ("mean", "scale"):
        if key not in stats:
            raise ValueError(f"{path} no contiene la clave '{key}'.")
        if len(stats[key]) != cfg.n_features:
            raise ValueError(
                f"{path}: '{key}' tiene {len(stats[key])} valores, se esperaban "
                f"{cfg.n_features} (uno por canal)."
            )

    stats["channel_order"] = list(cfg.channel_order)
    stats["is_identity_placeholder"] = False
    return stats


def export_background(path: Optional[Path], out_path: Path, cfg: FedPerConfig) -> int:
    """
    Vuelca un array (N, window_size, n_features) YA NORMALIZADO a float32
    crudo, little-endian, en orden C.

    Se usa binario plano y no CSV porque el APK lo mapea directamente a un
    ByteBuffer: una ventana ocupa 128*6*4 = 3 072 bytes, así que un pool de
    600 ventanas son ~1,8 MB.
    """
    if path is None:
        return 0

    arr = np.load(path)
    expected = (cfg.window_size, cfg.n_features)
    if arr.ndim != 3 or arr.shape[1:] != expected:
        raise ValueError(
            f"{path}: se esperaba forma (N, {expected[0]}, {expected[1]}), "
            f"recibido {arr.shape}."
        )
    arr = np.ascontiguousarray(arr, dtype="<f4")
    out_path.write_bytes(arr.tobytes())
    print(f"  {out_path.name}: {arr.shape[0]} ventanas "
          f"({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return int(arr.shape[0])


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------


def build_wrapper(
    cfg: FedPerConfig,
    encoder_weights: Optional[List[np.ndarray]],
    head_weights: Optional[List[np.ndarray]],
) -> TrainableAuthModel:
    tf.keras.backend.clear_session()
    wrapper = TrainableAuthModel(cfg)

    if encoder_weights is not None:
        current = wrapper.encoder.get_weights()
        if len(encoder_weights) != len(current):
            raise ValueError(
                f"El encoder esperaba {len(current)} tensores, el .npz trae "
                f"{len(encoder_weights)}. ¿Coincide la arquitectura con la de "
                "mejor.py?"
            )
        for i, (new, old) in enumerate(zip(encoder_weights, current)):
            if tuple(new.shape) != tuple(old.shape):
                raise ValueError(
                    f"Tensor {i} del encoder: forma {new.shape}, se esperaba "
                    f"{old.shape}."
                )
        wrapper.encoder.set_weights(encoder_weights)
        print(f"  Encoder inicializado desde checkpoint ({len(encoder_weights)} tensores).")
    else:
        print("  Encoder con inicialización aleatoria (sólo para pruebas).")

    if head_weights is not None:
        wrapper.head.set_weights(head_weights)
        print(f"  Cabeza inicializada desde checkpoint ({len(head_weights)} tensores).")

    return wrapper


def convert_to_tflite(saved_dir: Path) -> bytes:
    """
    Convierte el SavedModel a TFLite.

    `SELECT_TF_OPS` es obligatorio: LSTM entrenable, `Dropout` con generación
    de aleatoriedad y las escrituras a variables de recurso no tienen
    equivalente entre los builtins de TFLite. En Android eso exige el
    artefacto `tensorflow-lite-select-tf-ops` y registrar `FlexDelegate`.
    `_experimental_lower_tensor_list_ops = False` reproduce la conversión de
    la Celda 15 de `mejor.py`.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.experimental_enable_resource_variables = True
    converter._experimental_lower_tensor_list_ops = False
    return converter.convert()


def build_manifest(
    cfg: FedPerConfig,
    wrapper: TrainableAuthModel,
    n_bg_train: int,
    n_bg_calib: int,
    scaler_is_placeholder: bool,
) -> dict:
    """
    Contrato explícito entre el .tflite, el cliente Kotlin y el servidor.

    Todo lo que ambos lados tienen que acordar vive aquí, de modo que un
    desajuste se detecta al arrancar y no como un EER inexplicable.
    """
    return {
        "schema_version": 1,
        "model_file": TFLITE_NAME,
        "architecture": "FedPer/SharedEncoder+PersonalHead",
        "source_notebook": "mejor.py",

        "signal": {
            "window_size": cfg.window_size,
            "n_features": cfg.n_features,
            "target_hz": cfg.target_hz,
            "window_step": cfg.window_step,
            "sensor_config": cfg.sensor_config,
            "channel_order": list(cfg.channel_order),
        },

        "weights": {
            # Único número que Python y Kotlin deben acordar para el
            # intercambio FedAvg.
            "encoder_flat_size": wrapper.encoder_layout.total_size,
            "head_flat_size": wrapper.head_layout.total_size,
            "encoder_num_tensors": len(wrapper.encoder_layout.slots),
            "head_num_tensors": len(wrapper.head_layout.slots),
            "federated_block": "encoder",
            "local_only_block": "head",
        },

        "training": {
            "local_epochs": cfg.local_epochs,
            "batch_size": cfg.batch_size,
            "train_genuine_per_batch": cfg.train_genuine_per_batch,
            "train_background_per_batch": cfg.train_background_per_batch,
            "learning_rate": cfg.learning_rate,
            "lr_decay_factor": cfg.lr_decay_factor,
            "lr_decay_at": cfg.lr_decay_at,
            "cls_loss_weight": cfg.cls_loss_weight,
            "background_ratio": cfg.background_ratio,
            "reset_optimizer_each_round": True,
        },

        "evaluation": {
            "infer_batch": cfg.infer_batch,
            "decision_threshold": cfg.decision_threshold,
            "test_ratio": cfg.test_ratio,
            "val_ratio": cfg.val_ratio,
            # El servidor selecciona la mejor ronda con val_auc; el test
            # on-device no se toca durante la federación.
            "model_selection_metric": "val_auc",
        },

        "assets": {
            "scaler_stats": SCALER_NAME,
            "scaler_is_identity_placeholder": scaler_is_placeholder,
            "background_train": BG_TRAIN_NAME if n_bg_train else None,
            "background_train_windows": n_bg_train,
            "background_calib": BG_CALIB_NAME if n_bg_calib else None,
            "background_calib_windows": n_bg_calib,
        },

        "signatures": {
            "initialize": {"inputs": [], "outputs": ["status"]},
            "train_step": {
                "inputs": ["x_genuine", "x_background"],
                "outputs": ["loss", "recon_loss", "cls_loss"],
            },
            "infer": {
                "inputs": ["x", "threshold"],
                "outputs": ["genuine_score", "reconstruction_error", "is_genuine"],
            },
            "infer_batch": {
                "inputs": ["x", "threshold"],
                "outputs": ["genuine_score", "reconstruction_error", "is_genuine"],
            },
            "save_encoder": {"inputs": [], "outputs": ["encoder_flat"]},
            "restore_encoder": {"inputs": ["encoder_flat"], "outputs": ["status"]},
            "save_head": {"inputs": [], "outputs": ["head_flat"]},
            "restore_head": {"inputs": ["head_flat"], "outputs": ["status"]},
            "set_lr": {"inputs": ["lr"], "outputs": ["lr"]},
            "reset_optimizer": {"inputs": [], "outputs": ["status"]},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder-weights", type=Path, default=None,
                        help="best_encoder_weights.npz de la simulación")
    parser.add_argument("--head-weights", type=Path, default=None,
                        help="cabeza inicial opcional (normalmente NO se aporta)")
    parser.add_argument("--scaler-stats", type=Path, default=None,
                        help="scaler_stats.json de la simulación")
    parser.add_argument("--background-train", type=Path, default=None,
                        help=".npy (N,128,6) normalizado del pool de impostores")
    parser.add_argument("--background-calib", type=Path, default=None,
                        help=".npy (N,128,6) normalizado del pool DISJUNTO de calibración")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--saved-model-dir", type=Path, default=None,
                        help="directorio temporal del SavedModel intermedio")
    parser.add_argument("--android-assets-dir", type=Path, default=None,
                        help="si se indica, copia los assets al APK")
    args = parser.parse_args()

    tf_env.assert_legacy_keras()

    cfg = FedPerConfig()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_dir = args.saved_model_dir or (out_dir / "saved_model_fedper")

    print("=" * 70)
    print("  FASE 1 — Exportación del artefacto TFLite FedPer")
    print("=" * 70)
    print(f"TensorFlow {tf.__version__} / Keras {tf_env.keras_version()}")

    print("\n[1/6] Construyendo modelo...")
    encoder_weights = (
        load_npz_weights(args.encoder_weights) if args.encoder_weights else None
    )
    head_weights = load_npz_weights(args.head_weights) if args.head_weights else None
    wrapper = build_wrapper(cfg, encoder_weights, head_weights)

    enc_params = wrapper.encoder.count_params()
    head_params = wrapper.head.count_params()
    print(f"  Encoder compartido : {enc_params:,} parámetros "
          f"-> {wrapper.encoder_layout.total_size:,} floats "
          f"({len(wrapper.encoder_layout.slots)} tensores) VIAJAN por FedAvg")
    print(f"  Cabeza personal    : {head_params:,} parámetros "
          f"-> {wrapper.head_layout.total_size:,} floats "
          f"({len(wrapper.head_layout.slots)} tensores) NUNCA salen del móvil")
    print(f"  Entrenables on-device: {len(wrapper.model.trainable_variables)} tensores "
          f"(encoder + cabeza, como AuthClient.fit)")

    print("\n[2/6] Trazando firmas y calentando variables...")
    signatures = wrapper.build_signatures()
    wrapper.warmup()
    print(f"  {len(signatures)} firmas: {', '.join(sorted(signatures))}")

    print("\n[3/6] Guardando SavedModel...")
    if saved_dir.exists():
        shutil.rmtree(saved_dir)
    tf.saved_model.save(wrapper, str(saved_dir), signatures=signatures)

    print("\n[4/6] Convirtiendo a TFLite (TFLITE_BUILTINS + SELECT_TF_OPS)...")
    tflite_bytes = convert_to_tflite(saved_dir)
    tflite_path = out_dir / TFLITE_NAME
    tflite_path.write_bytes(tflite_bytes)
    print(f"  {tflite_path.name}: {len(tflite_bytes) / 1024 / 1024:.2f} MB")

    print("\n[5/6] Exportando assets acompañantes...")
    scaler = load_scaler_stats(args.scaler_stats, cfg)
    (out_dir / SCALER_NAME).write_text(json.dumps(scaler, indent=2))

    n_bg_train = export_background(args.background_train, out_dir / BG_TRAIN_NAME, cfg)
    n_bg_calib = export_background(args.background_calib, out_dir / BG_CALIB_NAME, cfg)
    if n_bg_train == 0:
        print("  ADVERTENCIA: sin --background-train el cliente no tiene "
              "impostores reales con los que entrenar el clasificador binario.")
    if n_bg_calib == 0:
        print("  ADVERTENCIA: sin --background-calib el cliente no puede "
              "calibrar su umbral sin mirar el test. El pool de calibración "
              "DEBE provenir de sujetos disjuntos de los de --background-train.")

    layout = {
        "encoder": wrapper.encoder_layout.to_dict(),
        "head": wrapper.head_layout.to_dict(),
    }
    (out_dir / LAYOUT_NAME).write_text(json.dumps(layout, indent=2))

    np.savez(out_dir / INITIAL_ENCODER_NAME, *wrapper.encoder.get_weights())

    manifest = build_manifest(
        cfg, wrapper, n_bg_train, n_bg_calib, scaler["is_identity_placeholder"],
    )
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    print("\n[6/6] Copiando assets al APK...")
    if args.android_assets_dir:
        assets_dir: Path = args.android_assets_dir
        assets_dir.mkdir(parents=True, exist_ok=True)
        for name in ANDROID_ASSETS:
            src = out_dir / name
            if src.exists():
                shutil.copy2(src, assets_dir / name)
                print(f"  -> {assets_dir / name}")
    else:
        print("  (omitido: no se indicó --android-assets-dir)")

    print("\n" + "=" * 70)
    print(f"Artefactos en {out_dir}")
    print(f"  encoder_flat_size = {wrapper.encoder_layout.total_size}  "
          f"(este número debe coincidir en servidor y cliente)")
    print("Siguiente paso: python verify_tflite_model.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
