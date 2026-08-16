#!/usr/bin/env python3
"""
Cliente simulado — criterio de aprobación de la FASE 3.

Emula lo que hace `FlowerGrpcClient.kt` sin necesitar un teléfono:

  · recibe UN tensor (el vector plano del encoder) y comprueba su tamaño;
  · lee `local_epochs` y `lr` de la config de la ronda;
  · devuelve UN tensor del mismo tamaño más `num_examples`;
  · reporta `train_loss` en fit y `val_auc` / `val_eer` en evaluate, con los
    mismos nombres de métrica que el cliente Android.

Sirve para tres cosas: que la agregación no falle por dimensiones, que el
early stopping por `val_auc` se dispare cuando debe, y —lo menos obvio— que el
protocolo gRPC de flwr 1.23 siga siendo el que habla el SDK de Android.

Uso:
    python mock_client.py --cid 0
    python mock_client.py --cid 1 --auc-schedule plateau

`--auc-schedule`:
    improving  el val_auc sube ronda a ronda (federación sana)
    plateau    sube y se estanca: debe disparar el early stopping
    noisy      ruido sin tendencia: comprueba que la EMA no fija best_round
                con una ronda afortunada
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import flwr as fl
from flwr.common import Config, NDArrays, Scalar

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  cliente: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mock")


class ClienteSimulado(fl.client.NumPyClient):
    def __init__(self, cid: int, flat_size: int, n_examples: int,
                 schedule: str, delay: float,
                 client_id: str | None = None) -> None:
        self.cid = cid
        self.flat_size = flat_size
        self.n_examples = n_examples
        self.schedule = schedule
        self.delay = delay
        # Identidad del "dispositivo". Dos procesos lanzados con el MISMO
        # --client-id simulan el fallo del 2026-08-15: un participante que
        # pulsa INICIAR FL dos veces y abre dos sesiones desde el mismo móvil.
        self.client_id = client_id
        self.rng = np.random.RandomState(1000 + cid)
        # Estado "local" persistente, como la cabeza personal del móvil.
        self.encoder = np.zeros(flat_size, dtype=np.float32)
        self.rondas_vistas = 0

    # -- utilidades ----------------------------------------------------

    def _comprobar(self, parameters: NDArrays) -> np.ndarray:
        if len(parameters) != 1:
            raise SystemExit(
                f"El servidor envió {len(parameters)} tensores; FedPer espera 1. "
                f"El cliente Android abortaría aquí."
            )
        flat = np.asarray(parameters[0], dtype=np.float32).reshape(-1)
        if flat.size != self.flat_size:
            raise SystemExit(
                f"El servidor envió {flat.size} floats, se esperaban "
                f"{self.flat_size}. Servidor y APK no comparten artefacto."
            )
        return flat

    def _auc_de_la_ronda(self, ronda: int) -> float:
        """AUC de validación sintético, según el calendario elegido."""
        if self.schedule == "improving":
            base = 0.60 + 0.03 * ronda
        elif self.schedule == "plateau":
            # Sube 4 rondas y se queda plano: con warmup_rounds=5 y
            # early_stop_patience=15, la parada debe llegar sobre la ronda 20.
            base = 0.60 + 0.04 * min(ronda, 4)
        else:  # noisy
            base = 0.70
        ruido = self.rng.normal(0, 0.01)
        return float(np.clip(base + ruido, 0.0, 1.0))

    # -- protocolo de Flower -------------------------------------------

    def get_parameters(self, config: Config) -> NDArrays:
        return [self.encoder]

    def fit(self, parameters: NDArrays, config: Config
            ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.rondas_vistas += 1
        ronda = int(config.get("server_round", self.rondas_vistas))
        recibido = self._comprobar(parameters)

        epochs = config.get("local_epochs")
        lr = config.get("lr")
        if epochs is None or lr is None:
            raise SystemExit(
                "El servidor no envió local_epochs/lr en la config de fit; el "
                "cliente Android caería a los valores del manifiesto y el "
                "decaimiento del learning rate no se aplicaría."
            )

        if self.delay:
            time.sleep(self.delay)

        # "Entrenar": desplazar un poco el encoder en una dirección propia de
        # este cliente. Así la media de FedAvg es comprobable: con dos clientes
        # simétricos, el agregado debe quedar entre ambos.
        deriva = 0.01 * (1 if self.cid % 2 == 0 else -1)
        self.encoder = (recibido + deriva).astype(np.float32)

        train_loss = float(1.5 * np.exp(-0.12 * ronda) + 0.25 + self.rng.normal(0, 0.01))
        log.info(
            "[%d] fit  ronda %-3d epochs=%s lr=%-8s n=%d  train_loss=%.4f",
            self.cid, ronda, epochs, lr, self.n_examples, train_loss,
        )
        metricas: Dict[str, Scalar] = {
            "train_loss": train_loss,
            "recon_loss": train_loss * 0.7,
            "cls_loss": train_loss * 0.3,
            "duration_ms": float(self.delay * 1000),
        }
        if self.client_id is not None:
            metricas["client_id"] = self.client_id
        return ([self.encoder], self.n_examples, metricas)

    def evaluate(self, parameters: NDArrays, config: Config
                 ) -> Tuple[float, int, Dict[str, Scalar]]:
        ronda = int(config.get("server_round", self.rondas_vistas))
        self._comprobar(parameters)
        auc = self._auc_de_la_ronda(ronda)
        eer = float(np.clip(1.0 - auc + self.rng.normal(0, 0.005), 0.0, 1.0))
        loss = float(0.9 * np.exp(-0.1 * ronda) + 0.3)
        log.info("[%d] eval ronda %-3d val_auc=%.4f val_eer=%.4f", self.cid, ronda, auc, eer)
        return (
            loss,
            max(1, self.n_examples // 4),
            {
                "val_auc": auc,
                "val_eer": eer,
                "val_far": eer,
                "val_frr": eer,
                "val_accuracy": auc,
                "calibrated_threshold": 0.5,
            },
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cid", type=int, default=0)
    p.add_argument("--server", default="127.0.0.1:8080")
    p.add_argument("--examples", type=int, default=250,
                   help="num_examples reportado (pondera FedAvg)")
    p.add_argument("--auc-schedule", default="improving",
                   choices=["improving", "plateau", "noisy"])
    p.add_argument("--delay", type=float, default=0.0,
                   help="Segundos de 'entrenamiento' simulado por ronda")
    p.add_argument("--manifest", type=Path, default=ARTIFACTS / "model_manifest.json")
    p.add_argument("--client-id", default=None,
                   help="Identidad del dispositivo (metric `client_id`). Dos "
                        "procesos con el mismo valor simulan un teléfono que "
                        "abrió dos sesiones; el servidor debe descartar una.")
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    flat_size = int(manifest["weights"]["encoder_flat_size"])
    log.info("[%d] conectando a %s  (encoder_flat_size=%d)",
             args.cid, args.server, flat_size)

    fl.client.start_client(
        server_address=args.server,
        client=ClienteSimulado(
            args.cid, flat_size, args.examples, args.auc_schedule, args.delay,
            args.client_id,
        ).to_client(),
    )


if __name__ == "__main__":
    main()
