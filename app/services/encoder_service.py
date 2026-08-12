"""
Servicio del encoder compartido (FedPer) — Fase 3.

Sustituye a `ModelService`, que servía la arquitectura DeepConvLSTM anterior.

Responsabilidad única: custodiar el vector plano del encoder que viaja por
FedAvg, y persistirlo en disco.

NO IMPORTA TENSORFLOW A PROPÓSITO. El servidor no entrena ni infiere: sólo
promedia `encoder_flat_size` floats por ronda. Cargar Keras aquí costaba ~1 GB
de RAM y 25 s de arranque para nada. La reconstrucción de los tensores con
forma se hace desde `weight_layout.json`, que ya describe el aplanado; sólo
hace falta si alguien quiere volver a meter los pesos en un modelo de Keras,
y para eso está `unflatten_to_tensors`.

La cabeza personal nunca llega aquí: se queda en cada dispositivo.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MANIFEST_NAME = "model_manifest.json"
LAYOUT_NAME = "weight_layout.json"
INITIAL_ENCODER_NAME = "initial_encoder.npz"
CHECKPOINTS_DIRNAME = "checkpoints"


class ArtifactsMissing(RuntimeError):
    """Los artefactos de la Fase 1 no están donde deberían."""


class EncoderService:
    """
    Mantiene el encoder global en memoria y lo guarda en disco.

    El contrato con el cliente es un único número, `encoder_flat_size`, que se
    lee del manifiesto en lugar de estar escrito a mano. Manifiesto y `.tflite`
    del APK salen de la misma ejecución de `export_tflite_model.py`, así que
    leerlo aquí hace imposible que servidor y cliente se desincronicen.
    """

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self._dir = Path(artifacts_dir or ARTIFACTS_DIR)
        self._manifest = self._load_manifest()
        self._layout = self._load_layout()

        self.encoder_flat_size: int = int(
            self._manifest["weights"]["encoder_flat_size"]
        )
        if self._layout["total_size"] != self.encoder_flat_size:
            raise ArtifactsMissing(
                f"{LAYOUT_NAME} describe {self._layout['total_size']} floats pero "
                f"{MANIFEST_NAME} declara {self.encoder_flat_size}. Los dos "
                f"ficheros tienen que salir de la misma exportación."
            )

        self._encoder: np.ndarray = self._load_initial_encoder()
        self._round_of_current: int = 0

        self._checkpoints = self._dir / CHECKPOINTS_DIRNAME
        self._checkpoints.mkdir(parents=True, exist_ok=True)

        logger.info(
            "EncoderService listo: %d floats, %d tensores, sensor_config=%s",
            self.encoder_flat_size,
            len(self._layout["tensors"]),
            self._manifest["signal"]["sensor_config"],
        )

    # ------------------------------------------------------------------
    # Carga de artefactos
    # ------------------------------------------------------------------

    def _require(self, name: str) -> Path:
        path = self._dir / name
        if not path.exists():
            raise ArtifactsMissing(
                f"Falta {path}. Ejecuta primero:\n"
                f"    cd backend && python export_tflite_model.py "
                f"--encoder-weights ... --scaler-stats ...\n"
                f"La Fase 3 sirve el artefacto de la Fase 1; no lo genera."
            )
        return path

    def _load_manifest(self) -> Dict[str, Any]:
        return json.loads(self._require(MANIFEST_NAME).read_text(encoding="utf-8"))

    def _load_layout(self) -> Dict[str, Any]:
        """
        `weight_layout.json` describe los DOS bloques (`encoder` y `head`).
        Aquí sólo interesa el encoder: la cabeza no sale nunca del dispositivo.
        """
        completo = json.loads(self._require(LAYOUT_NAME).read_text(encoding="utf-8"))
        if "encoder" not in completo:
            raise ArtifactsMissing(
                f"{LAYOUT_NAME} no tiene el bloque 'encoder'. Claves: "
                f"{sorted(completo)}"
            )
        return completo["encoder"]

    def _load_initial_encoder(self) -> np.ndarray:
        """
        Vector inicial que reciben los clientes en la ronda 1.

        Si existe un checkpoint de una federación anterior se retoma desde ahí;
        si no, se parte del encoder horneado en el artefacto, que ya trae el
        pre-entrenamiento de la simulación (evita repetir ~110 rondas).
        """
        latest = self._dir / CHECKPOINTS_DIRNAME / "encoder_latest.npz"
        if latest.exists():
            flat = self._flat_from_npz(latest)
            logger.info("Retomando el encoder desde %s", latest.name)
            return flat

        flat = self._flat_from_npz(self._require(INITIAL_ENCODER_NAME))
        logger.info("Encoder inicial horneado en el artefacto de la Fase 1")
        return flat

    def _flat_from_npz(self, path: Path) -> np.ndarray:
        """
        Lee un .npz y lo aplana en el orden de `weight_layout.json`.

        Un `.npz` guardado con `np.savez(f, *arrays)` usa claves `arr_0`,
        `arr_1`… cuyo orden alfabético NO es el numérico (`arr_10` va antes que
        `arr_2`). Se ordena por índice a propósito: equivocarse aquí produce un
        encoder con los tensores permutados, que no falla en ninguna
        comprobación de tamaño y degrada el modelo en silencio.
        """
        with np.load(path) as data:
            keys = list(data.files)
            if all(k.startswith("arr_") for k in keys):
                keys.sort(key=lambda k: int(k.split("_")[1]))
            arrays = [np.asarray(data[k], dtype=np.float32) for k in keys]

        expected = len(self._layout["tensors"])
        if len(arrays) != expected:
            raise ArtifactsMissing(
                f"{path.name} trae {len(arrays)} tensores, el layout describe "
                f"{expected}."
            )
        for arr, slot in zip(arrays, self._layout["tensors"]):
            if list(arr.shape) != list(slot["shape"]):
                raise ArtifactsMissing(
                    f"{path.name}: el tensor '{slot['name']}' tiene forma "
                    f"{list(arr.shape)}, el layout espera {slot['shape']}."
                )

        flat = np.concatenate([a.reshape(-1) for a in arrays]).astype(np.float32)
        if flat.size != self.encoder_flat_size:
            raise ArtifactsMissing(
                f"{path.name} aplana a {flat.size} floats, se esperaban "
                f"{self.encoder_flat_size}."
            )
        return flat

    # ------------------------------------------------------------------
    # Parámetros para Flower
    # ------------------------------------------------------------------

    def get_parameters(self) -> List[np.ndarray]:
        """
        Un ÚNICO tensor 1-D, que es lo que el cliente Android espera.

        `FlowerGrpcClient.decodeEncoder` rechaza cualquier cosa que no sean
        exactamente 1 tensor, así que devolver la lista de 15 arrays de Keras
        —como hacía ModelService— rompería la ronda en el cliente.
        """
        return [self._encoder.copy()]

    def set_encoder(self, flat: np.ndarray, server_round: int) -> None:
        flat = np.asarray(flat, dtype=np.float32).reshape(-1)
        if flat.size != self.encoder_flat_size:
            raise ValueError(
                f"El encoder agregado tiene {flat.size} floats, se esperaban "
                f"{self.encoder_flat_size}."
            )
        self._encoder = flat
        self._round_of_current = server_round

    @property
    def encoder(self) -> np.ndarray:
        return self._encoder

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def unflatten_to_tensors(self, flat: np.ndarray | None = None) -> List[np.ndarray]:
        """Reconstruye la lista de tensores con forma, en el orden de Keras."""
        flat = self._encoder if flat is None else np.asarray(flat, dtype=np.float32)
        out: List[np.ndarray] = []
        for slot in self._layout["tensors"]:
            chunk = flat[slot["offset"]: slot["offset"] + slot["size"]]
            out.append(chunk.reshape(slot["shape"]).astype(np.float32))
        return out

    def save_checkpoint(self, server_round: int, tag: str | None = None) -> Path:
        """
        Guarda el encoder como `.npz` con la misma estructura que
        `initial_encoder.npz`, para que `_flat_from_npz` pueda releerlo y para
        que sea consumible por `fedper_model.unflatten_weights`.
        """
        arrays = self.unflatten_to_tensors()
        name = tag or f"encoder_round_{server_round:03d}"
        path = self._checkpoints / f"{name}.npz"
        np.savez(path, *arrays)
        shutil.copyfile(path, self._checkpoints / "encoder_latest.npz")
        return path

    def promote_best(self, server_round: int, val_auc: float) -> Path:
        """
        Marca la ronda actual como la mejor vista hasta ahora.

        Se copia en lugar de moverse para que el histórico por ronda siga
        disponible al escribir la memoria.
        """
        src = self.save_checkpoint(server_round)
        best = self._checkpoints / "encoder_best.npz"
        shutil.copyfile(src, best)
        (self._checkpoints / "encoder_best.json").write_text(
            json.dumps(
                {
                    "round": server_round,
                    "val_auc": float(val_auc),
                    "encoder_flat_size": self.encoder_flat_size,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return best

    # ------------------------------------------------------------------
    # Metadatos para el cliente
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        """
        Contrato que verifica `ModelInfoFetcher.requireCompatibleWith`.

        Los tres campos que compara la app son `sensor_config`, `window_size` y
        `encoder_flat_size`, y los tres salen del manifiesto, no de constantes
        escritas aquí: si el APK y el servidor cargan artefactos distintos, la
        app aborta la ronda en vez de entrenar contra un modelo incompatible.
        """
        signal = self._manifest["signal"]
        weights = self._manifest["weights"]
        return {
            "architecture": self._manifest["architecture"],
            "schema_version": self._manifest["schema_version"],
            "sensor_config": signal["sensor_config"],
            "window_size": signal["window_size"],
            "n_features": signal["n_features"],
            "channel_order": signal["channel_order"],
            "target_hz": signal["target_hz"],
            "window_step": signal["window_step"],
            "encoder_flat_size": weights["encoder_flat_size"],
            "head_flat_size": weights["head_flat_size"],
            "federated_block": weights["federated_block"],
            "scaler_is_identity_placeholder": self._manifest["assets"][
                "scaler_is_identity_placeholder"
            ],
            "current_round": self._round_of_current,
        }
