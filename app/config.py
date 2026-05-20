"""
Configuración de la aplicación.

Principio de responsabilidad única: centraliza todos los valores
configurables del servidor en un único lugar.
"""

from dataclasses import dataclass, field
from typing import Literal, Tuple


# ---------------------------------------------------------------------------
# Tipos del dominio
# ---------------------------------------------------------------------------

SensorConfig = Literal["gyro", "gyro_acc", "gyro_acc_touch"]


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuración inmutable de la arquitectura del modelo.

    Agrupa los hiperparámetros que definen la topología de la red
    DeepConvLSTM. Al ser frozen, garantiza que no se modifique
    accidentalmente después de la inicialización.
    """

    # — Dominio de sensores —
    sensor_config: SensorConfig = "gyro_acc"
    window_size: int = 128

    # — Rama IMU —
    conv_filters: Tuple[int, ...] = (64, 64, 128, 128)
    lstm_units: Tuple[int, ...] = (128, 64)

    # — Rama táctil —
    touch_units: Tuple[int, ...] = (32, 16)

    # — Regularización —
    dropout_rate: float = 0.3

    # — Cabeza —
    head_units: int = 32

    # — Optimizador —
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class AppConfig:
    """
    Configuración general del servidor Flask.
    """

    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True

    model: ModelConfig = field(default_factory=ModelConfig)
