#!/usr/bin/env python
"""
FASE 1 — Criterios de aprobación del artefacto TFLite.

Ejecuta, sobre el .tflite ya convertido (no sobre el modelo de Keras), las
comprobaciones que el plan de refactorización exige antes de pasar a la
Fase 2:

  1. Todas las firmas declaradas en el manifiesto existen en el intérprete.
  2. `save_encoder` devuelve un vector del tamaño acordado y coincide con
     `initial_encoder.npz`.
  3. `restore_encoder` con ceros modifica SÓLO el encoder: la cabeza queda
     intacta (verifica el aislamiento de bloques que exige FedPer).
  4. `train_step` reduce la pérdida sobre un lote sintético separable.
  5. El entrenamiento MUEVE los pesos del encoder. Es la prueba decisiva:
     el wrapper de la Celda 15 de `mejor.py` congela el encoder y fallaría
     aquí, lo que haría que FedAvg agregase pesos que nunca cambian.
  6. `infer` / `infer_batch` devuelven las formas correctas y coherentes
     entre sí.
  7. `set_lr` y `reset_optimizer` responden.
  8. `save_head` / `restore_head` hacen round-trip exacto.

Uso:
    python verify_tflite_model.py [--artifacts-dir artifacts]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import tf_env  # noqa: F401  (debe importarse antes que tensorflow)

import numpy as np
import tensorflow as tf


class CheckFailed(Exception):
    """Defecto del artefacto: lo exportado no cumple el contrato."""


class EnvironmentUnsupported(Exception):
    """
    Limitación del intérprete de Python, no del artefacto.

    Se separa de `CheckFailed` a propósito: confundir "este entorno no puede
    ejecutarlo" con "el modelo está mal" lleva a rehacer una exportación que
    era correcta.
    """


class Report:
    """Acumula resultados para devolver un exit code útil en CI."""

    def __init__(self) -> None:
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.blocked: List[str] = []
        self.skipped: List[str] = []

    def run(
        self,
        title: str,
        fn: Callable[[], str],
        depends_on: str | None = None,
    ) -> None:
        # Una comprobación cuyo requisito no se pudo evaluar no tiene veredicto.
        # Ejecutarla igualmente produce diagnósticos falsos: si `train_step` no
        # llegó a correr, "el encoder no cambió" acusa de congelado a un encoder
        # que simplemente no se entrenó.
        if depends_on is not None and depends_on not in self.passed:
            self.skipped.append(title)
            print(f"  [OMIT] {title}")
            print(f"         sin veredicto: depende de '{depends_on}'")
            return

        try:
            detail = fn()
        except EnvironmentUnsupported as exc:
            self.blocked.append(title)
            print(f"  [ENTORNO] {title}")
            print(f"         {exc}")
        except Exception as exc:  # noqa: BLE001 - se reporta, no se propaga
            self.failed.append(title)
            print(f"  [FALLO] {title}")
            print(f"         {type(exc).__name__}: {exc}")
        else:
            self.passed.append(title)
            print(f"  [OK]   {title}")
            if detail:
                print(f"         {detail}")

    def summary(self) -> int:
        total = (len(self.passed) + len(self.failed)
                 + len(self.blocked) + len(self.skipped))
        print("\n" + "=" * 70)
        print(f"  {len(self.passed)}/{total} comprobaciones superadas")

        for label, names in (
            ("Fallaron (defecto del artefacto)", self.failed),
            ("Bloqueadas por el entorno de Python", self.blocked),
            ("Sin evaluar", self.skipped),
        ):
            if names:
                print(f"  {label}:")
                for name in names:
                    print(f"    - {name}")

        if self.failed:
            print("\n  El artefacto NO sirve: corrige la exportación.")
        elif self.blocked or self.skipped:
            print("\n  El artefacto no está descartado, pero tampoco verificado.")
            print("  Valida lo que falta en el dispositivo con FedPerOnDeviceTest")
            print("  (./gradlew connectedDebugAndroidTest) antes de seguir.")

        print("=" * 70)
        return 1 if (self.failed or self.blocked or self.skipped) else 0


def make_separable_batch(n_gen: int, n_bg: int, window: int, feats: int, seed: int = 0):
    """
    Lote sintético trivialmente separable: el genuino es una senoide de fase
    fija con ruido bajo, el impostor es ruido puro. Si `train_step` está bien
    conectado, la pérdida tiene que bajar en pocos pasos.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4 * np.pi, window, dtype=np.float32)
    base = np.stack([np.sin(t + k) for k in range(feats)], axis=-1)

    x_gen = (base[None] + 0.05 * rng.randn(n_gen, window, feats)).astype(np.float32)
    x_bg = (1.0 * rng.randn(n_bg, window, feats)).astype(np.float32)
    return x_gen, x_bg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--train-steps", type=int, default=30)
    args = parser.parse_args()

    art: Path = args.artifacts_dir
    manifest = json.loads((art / "model_manifest.json").read_text())
    tflite_path = art / manifest["model_file"]

    print("=" * 70)
    print("  FASE 1 — Verificación del artefacto TFLite")
    print("=" * 70)
    print(f"Modelo    : {tflite_path} "
          f"({tflite_path.stat().st_size / 1024 / 1024:.2f} MB)")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    available = set(interpreter.get_signature_list())
    print(f"Firmas    : {', '.join(sorted(available))}\n")

    sig = {name: interpreter.get_signature_runner(name) for name in available}

    ws = manifest["signal"]["window_size"]
    nf = manifest["signal"]["n_features"]
    n_gen = manifest["training"]["train_genuine_per_batch"]
    n_bg = manifest["training"]["train_background_per_batch"]
    infer_batch = manifest["evaluation"]["infer_batch"]
    enc_size = manifest["weights"]["encoder_flat_size"]
    head_size = manifest["weights"]["head_flat_size"]
    threshold = np.float32(manifest["evaluation"]["decision_threshold"])

    report = Report()

    # -- 1. Firmas declaradas -------------------------------------------
    def check_signatures() -> str:
        expected = set(manifest["signatures"])
        missing = expected - available
        if missing:
            raise CheckFailed(f"faltan firmas en el .tflite: {sorted(missing)}")
        return f"{len(expected)} firmas declaradas, todas presentes"

    report.run("Firmas del manifiesto presentes en el intérprete", check_signatures)

    sig["initialize"]()

    # -- 2. Tamaño del vector del encoder --------------------------------
    def check_encoder_size() -> str:
        flat = sig["save_encoder"]()["encoder_flat"]
        if flat.shape != (enc_size,):
            raise CheckFailed(
                f"save_encoder devolvió {flat.shape}, se esperaba ({enc_size},)"
            )
        if flat.dtype != np.float32:
            raise CheckFailed(f"dtype {flat.dtype}, se esperaba float32")

        npz = art / "initial_encoder.npz"
        if npz.exists():
            data = np.load(npz)
            ref = np.concatenate(
                [data[f"arr_{i}"].reshape(-1) for i in range(len(data.files))]
            ).astype(np.float32)
            if not np.allclose(flat, ref, atol=1e-5):
                raise CheckFailed(
                    "los pesos del .tflite no coinciden con initial_encoder.npz"
                )
            return (f"{enc_size} floats, idénticos a initial_encoder.npz "
                    f"({len(data.files)} tensores)")
        return f"{enc_size} floats"

    report.run("save_encoder: tamaño, dtype y contenido", check_encoder_size)

    encoder_original = sig["save_encoder"]()["encoder_flat"].copy()
    head_original = sig["save_head"]()["head_flat"].copy()

    # -- 3. Aislamiento encoder / cabeza ---------------------------------
    def check_isolation() -> str:
        sig["restore_encoder"](encoder_flat=np.zeros(enc_size, dtype=np.float32))
        enc_after = sig["save_encoder"]()["encoder_flat"]
        head_after = sig["save_head"]()["head_flat"]

        if np.any(enc_after != 0.0):
            raise CheckFailed(
                f"restore_encoder(zeros) dejó {int(np.count_nonzero(enc_after))} "
                "valores no nulos en el encoder"
            )
        if not np.array_equal(head_after, head_original):
            raise CheckFailed(
                "restore_encoder alteró la cabeza personal: los bloques no "
                "están aislados y la cabeza se contaminaría con FedAvg"
            )

        sig["restore_encoder"](encoder_flat=encoder_original)
        restored = sig["save_encoder"]()["encoder_flat"]
        if not np.array_equal(restored, encoder_original):
            raise CheckFailed("no se pudo restaurar el encoder original")
        return (f"encoder puesto a cero y restaurado; la cabeza "
                f"({head_size} floats) no se movió")

    report.run("restore_encoder toca SÓLO el encoder", check_isolation)

    # -- 4. La pérdida baja ----------------------------------------------
    x_gen, x_bg = make_separable_batch(n_gen, n_bg, ws, nf)

    def train_step_once():
        """Un paso de entrenamiento, distinguiendo defecto de limitación."""
        try:
            return sig["train_step"](x_genuine=x_gen, x_background=x_bg)
        except RuntimeError as exc:
            message = str(exc)
            if "Flex" not in message and "Select TensorFlow op" not in message:
                raise
            raise EnvironmentUnsupported(
                f"el intérprete de TFLite de este Python (TF {tf.__version__}) "
                "no trae el delegado Flex, así que no puede ejecutar train_step: "
                "el backward de la LSTM usa ops TensorList que viven en "
                "SELECT_TF_OPS. Esto NO dice nada sobre el artefacto — en Android "
                "esas ops las aporta org.tensorflow:tensorflow-lite-select-tf-ops. "
                "Para verificarlo aquí hace falta un TF cuyo intérprete de Python "
                "aún enlace Flex (2.16.x); si no, valida en el dispositivo."
            ) from exc

    def check_loss_decreases() -> str:
        sig["reset_optimizer"]()
        sig["set_lr"](lr=np.float32(manifest["training"]["learning_rate"]))

        losses = []
        for _ in range(args.train_steps):
            losses.append(float(train_step_once()["loss"]))

        first = float(np.mean(losses[:3]))
        last = float(np.mean(losses[-3:]))
        if not np.isfinite(last):
            raise CheckFailed(f"la pérdida divergió a {last}")
        if last >= first:
            raise CheckFailed(
                f"la pérdida no bajó en {args.train_steps} pasos: "
                f"{first:.4f} -> {last:.4f}"
            )
        return (f"{args.train_steps} pasos: loss {first:.4f} -> {last:.4f} "
                f"({100 * (1 - last / first):.1f}% de reducción)")

    report.run("train_step reduce la pérdida", check_loss_decreases)

    # -- 5. El encoder ES entrenable (la prueba decisiva) -----------------
    def check_encoder_is_trainable() -> str:
        enc_now = sig["save_encoder"]()["encoder_flat"]
        delta = float(np.abs(enc_now - encoder_original).max())
        if delta == 0.0:
            raise CheckFailed(
                "el encoder no cambió tras entrenar: está CONGELADO. Es el "
                "comportamiento de la Celda 15 de mejor.py (freeze_encoder="
                "True), válido para personalización pero inservible para FL: "
                "FedAvg agregaría siempre los mismos pesos."
            )
        head_now = sig["save_head"]()["head_flat"]
        head_delta = float(np.abs(head_now - head_original).max())
        if head_delta == 0.0:
            raise CheckFailed("la cabeza personal no cambió tras entrenar")
        return (f"delta máx. encoder = {delta:.3e}, "
                f"delta máx. cabeza = {head_delta:.3e}")

    report.run(
        "El entrenamiento actualiza encoder Y cabeza",
        check_encoder_is_trainable,
        depends_on="train_step reduce la pérdida",
    )

    # -- 6. Inferencia ----------------------------------------------------
    def check_infer() -> str:
        one = sig["infer"](x=x_gen[:1], threshold=threshold)
        if one["genuine_score"].shape != (1,):
            raise CheckFailed(f"infer.genuine_score -> {one['genuine_score'].shape}")

        padded = np.zeros((infer_batch, ws, nf), dtype=np.float32)
        padded[: min(infer_batch, len(x_gen))] = x_gen[:infer_batch]
        many = sig["infer_batch"](x=padded, threshold=threshold)
        for key, expected in (
            ("genuine_score", (infer_batch,)),
            ("reconstruction_error", (infer_batch,)),
            ("is_genuine", (infer_batch,)),
        ):
            if many[key].shape != expected:
                raise CheckFailed(f"infer_batch.{key} -> {many[key].shape}, "
                                  f"se esperaba {expected}")

        # Ambas firmas comparten variables: la misma ventana debe puntuar
        # igual por las dos vías.
        if not np.allclose(one["genuine_score"][0], many["genuine_score"][0], atol=1e-4):
            raise CheckFailed(
                f"infer={one['genuine_score'][0]:.6f} frente a "
                f"infer_batch={many['genuine_score'][0]:.6f}: las firmas no "
                "comparten estado o BatchNorm no está en modo inferencia"
            )
        scores = many["genuine_score"]
        if np.any(scores < 0) or np.any(scores > 1):
            raise CheckFailed("genuine_score fuera de [0,1]")
        return (f"score genuino tras entrenar = {float(one['genuine_score'][0]):.4f}, "
                f"error de reconstrucción = {float(one['reconstruction_error'][0]):.4f}")

    report.run("infer / infer_batch: formas y consistencia", check_infer)

    # -- 7. Hiperparámetros en caliente -----------------------------------
    def check_hyperparams() -> str:
        decayed = np.float32(
            manifest["training"]["learning_rate"]
            * manifest["training"]["lr_decay_factor"]
        )
        out = sig["set_lr"](lr=decayed)
        got = float(np.asarray(out["lr"]).reshape(-1)[0])
        if abs(got - float(decayed)) > 1e-9:
            raise CheckFailed(f"set_lr devolvió {got}, se esperaba {float(decayed)}")
        sig["reset_optimizer"]()
        return f"learning rate conmutado a {got:g} y optimizador reiniciado"

    report.run("set_lr y reset_optimizer", check_hyperparams)

    # -- 8. Round-trip de la cabeza ---------------------------------------
    def check_head_roundtrip() -> str:
        snapshot = sig["save_head"]()["head_flat"].copy()
        sig["restore_head"](head_flat=np.zeros(head_size, dtype=np.float32))
        if np.any(sig["save_head"]()["head_flat"] != 0.0):
            raise CheckFailed("restore_head(zeros) no puso la cabeza a cero")

        enc_check = sig["save_encoder"]()["encoder_flat"]
        sig["restore_head"](head_flat=snapshot)
        if not np.array_equal(sig["save_head"]()["head_flat"], snapshot):
            raise CheckFailed("el round-trip de la cabeza no es exacto")
        if not np.array_equal(sig["save_encoder"]()["encoder_flat"], enc_check):
            raise CheckFailed("restore_head alteró el encoder")
        return (f"{head_size} floats persistidos y restaurados sin pérdida "
                "(así sobrevive la cabeza personal a reinicios de la app)")

    report.run("save_head / restore_head: round-trip exacto", check_head_roundtrip)

    return report.summary()


if __name__ == "__main__":
    raise SystemExit(main())
