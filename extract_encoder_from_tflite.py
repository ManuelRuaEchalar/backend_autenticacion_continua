#!/usr/bin/env python
"""
Recupera los pesos del encoder desde el `.tflite` que produce la Celda 15 de
`mejor.py` (`continuous_auth_autoencoder_fl.tflite`).

Ese artefacto NO sirve como modelo de la app —exporta el encoder congelado y
no expone `restore_encoder`/`save_encoder`, así que no puede participar en
FedAvg— pero sí contiene los pesos de `BEST_ENCODER_WEIGHTS`, es decir, el
resultado de las ~110 rondas de simulación. Este script los rescata para
alimentar `export_tflite_model.py --encoder-weights` sin reentrenar.

Cómo: el wrapper del cuadernillo expone la firma `save`, que llama a
`tf.raw_ops.Save` y escribe un checkpoint de TensorFlow con los nombres
originales de Keras. Se invoca esa firma, se lee el checkpoint y se emparejan
los tensores con los del encoder de referencia.

Uso:
    python extract_encoder_from_tflite.py \
        --tflite ../continuous_auth_autoencoder_fl.tflite \
        --out    artifacts/best_encoder_weights.npz

Lo que este script NO puede recuperar (no está en el modelo):
  · scaler_stats.json  — la normalización ocurre ANTES de entrar al modelo.
  · los pools de background — son datos, no pesos.
Ambos exigen volver a ejecutar las celdas de carga de datos del cuadernillo
(celdas 2, 4, 6 y 8), que no necesitan GPU ni reentrenamiento.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import tf_env  # noqa: F401  (debe importarse antes que tensorflow)

import numpy as np
import tensorflow as tf

from app.config import FedPerConfig
from app.models import fedper_model as fp


def dump_checkpoint(tflite_path: Path, checkpoint_prefix: Path) -> None:
    """Invoca la firma `save` del modelo para volcar un checkpoint de TF."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    signatures = set(interpreter.get_signature_list())
    if "save" not in signatures:
        raise SystemExit(
            f"{tflite_path.name} no expone la firma 'save'. Firmas presentes: "
            f"{sorted(signatures)}. ¿Es realmente el artefacto de la Celda 15?"
        )

    # La firma `initialize` asigna los valores iniciales a las variables de
    # recurso; sin ella el checkpoint podría salir con ceros.
    if "initialize" in signatures:
        interpreter.get_signature_runner("initialize")()

    runner = interpreter.get_signature_runner("save")
    path_tensor = np.array(str(checkpoint_prefix).encode("utf-8"), dtype=object)
    runner(checkpoint_path=path_tensor)


def resolve_weights(
    reference: tf.keras.Model, reader, label: str
) -> List[np.ndarray]:
    """
    Empareja cada peso del modelo de referencia con su tensor del checkpoint.

    Se prueban varias formas del nombre porque el prefijo del submodelo
    (`SharedEncoder/`) aparece o no según la versión de Keras con la que se
    exportó. Como último recurso se empareja por forma, siempre que sea
    inequívoca.
    """
    shapes = reader.get_variable_to_shape_map()
    available: Dict[str, List[int]] = dict(shapes)
    used: set[str] = set()
    out: List[np.ndarray] = []

    for weight in reference.weights:
        clean = weight.name.split(":")[0]
        candidates = [
            clean,
            f"{reference.name}/{clean}",
            clean.split("/", 1)[-1] if "/" in clean else clean,
        ]

        key = next((c for c in candidates if c in available and c not in used), None)

        if key is None:
            target = [int(d) for d in weight.shape]
            matches = [
                name for name, shape in available.items()
                if name not in used and list(shape) == target
            ]
            if len(matches) != 1:
                raise SystemExit(
                    f"No se pudo localizar '{clean}' (forma {target}) en el "
                    f"checkpoint. Candidatos por forma: {matches or 'ninguno'}.\n"
                    f"Tensores disponibles:\n  " + "\n  ".join(sorted(available))
                )
            key = matches[0]
            print(f"  · {clean}: emparejado por forma con '{key}'")

        value = reader.get_tensor(key)
        if list(value.shape) != [int(d) for d in weight.shape]:
            raise SystemExit(
                f"'{key}' tiene forma {value.shape}, el {label} espera "
                f"{tuple(int(d) for d in weight.shape)}."
            )
        used.add(key)
        out.append(np.asarray(value, dtype=np.float32))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "best_encoder_weights.npz")
    parser.add_argument("--also-head", action="store_true",
                        help="extrae también la cabeza (normalmente NO se quiere: "
                             "la del artefacto es la de inicialización aleatoria)")
    args = parser.parse_args()

    tf_env.assert_legacy_keras()

    if not args.tflite.exists():
        raise SystemExit(f"No existe {args.tflite}")

    print("=" * 70)
    print("  Extracción del encoder desde el TFLite del cuadernillo")
    print("=" * 70)
    print(f"Origen: {args.tflite} ({args.tflite.stat().st_size / 1024 / 1024:.2f} MB)")

    cfg = FedPerConfig()
    tf.keras.backend.clear_session()
    encoder = fp.encoder_from_config(cfg)
    head = fp.head_from_config(cfg, encoder.output_shape[-1])

    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "notebook_ckpt"
        print("\n[1/3] Volcando el checkpoint con la firma 'save'...")
        dump_checkpoint(args.tflite, prefix)

        print("[2/3] Leyendo el checkpoint...")
        reader = tf.train.load_checkpoint(str(prefix))
        n_tensors = len(reader.get_variable_to_shape_map())
        print(f"  {n_tensors} tensores en el checkpoint")

        print("[3/3] Emparejando con el encoder de referencia...")
        encoder_weights = resolve_weights(encoder, reader, "encoder")
        head_weights = resolve_weights(head, reader, "cabeza") if args.also_head else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, *encoder_weights)

    total = sum(int(np.prod(w.shape)) for w in encoder_weights)
    print(f"\nEncoder extraído: {len(encoder_weights)} tensores, {total:,} floats")
    print(f"  -> {args.out}")

    if head_weights is not None:
        head_path = args.out.with_name(args.out.stem + "_head.npz")
        np.savez(head_path, *head_weights)
        print(f"  -> {head_path} (cabeza; normalmente no hace falta)")

    # Un encoder que sale todo a cero significa que las variables no llegaron a
    # inicializarse; conviene detectarlo aquí y no tres pasos más adelante.
    if all(np.allclose(w, 0.0) for w in encoder_weights):
        raise SystemExit(
            "\nTODOS los pesos salieron a cero: el checkpoint se escribió antes "
            "de que las variables se inicializasen. Revisa que el .tflite sea "
            "el exportado tras entrenar y no uno recién convertido."
        )

    print("\nSiguiente paso:")
    print(f"  python export_tflite_model.py --encoder-weights {args.out} \\")
    print("      --scaler-stats scaler_stats.json \\")
    print("      --background-train bg_train_scaled.npy \\")
    print("      --background-calib bg_calib_scaled.npy \\")
    print("      --android-assets-dir ../autenticacionContinua/app/src/main/assets")


if __name__ == "__main__":
    main()
