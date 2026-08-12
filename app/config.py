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
class FedPerConfig:
    """
    Configuración de la arquitectura FedPer (encoder compartido + cabeza
    personal) tal y como quedó validada en el cuadernillo `mejor.py`.

    Los valores son copia literal del diccionario CONFIG de `mejor.py`
    (celda 2). Cambiar cualquiera de ellos invalida la compatibilidad de
    pesos con la simulación, así que la clase es `frozen` a propósito.

    Sólo el ENCODER viaja por FedAvg. La cabeza (decoder + clasificador)
    nunca sale del dispositivo.
    """

    # — Señal —
    window_size: int = 128
    target_hz: int = 50
    window_step: int = 96
    n_features: int = 6
    sensor_config: str = "acc_gyro"

    # Orden de canales con el que se entrenó en `mejor.py`: el stack de
    # `_resample_session` pone acelerómetro primero y `SENSOR_SLICES` toma
    # slice(0, 6). Cualquier otro orden en el cliente Android produce
    # basura silenciosa.
    channel_order: Tuple[str, ...] = (
        "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
    )

    # — Encoder compartido —
    conv_filters: Tuple[int, ...] = (24, 48)
    kernel_size: int = 5
    encoder_lstm_units: int = 48

    # — Cabeza personal —
    lstm_units: Tuple[int, ...] = (48,)
    bottleneck_dim: int = 16
    dropout: float = 0.3

    # — Regularización y optimización —
    l2_reg: float = 1e-4
    learning_rate: float = 1e-3
    lr_decay_factor: float = 0.5
    lr_decay_at: float = 0.7

    # — Entrenamiento local —
    local_epochs: int = 5
    batch_size: int = 32
    cls_loss_weight: float = 2.0
    decision_threshold: float = 0.5

    # — Muestreo genuino / impostor —
    background_ratio: float = 1.0
    test_ratio: float = 0.20

    # Fracción del remanente (tras apartar test) que se reserva para
    # validación. NO existe en `mejor.py`: es la corrección al sesgo de
    # usar el test set como criterio de early stopping.
    val_ratio: float = 0.20

    # — Formas fijas de las firmas TFLite —
    # Se fijan a propósito: la API de SignatureRunner de TFLite/Java no
    # redimensiona de forma fiable entradas dinámicas, y los cambios de
    # forma en tiempo de ejecución son la causa habitual de crashes en
    # entrenamiento on-device.
    train_genuine_per_batch: int = 16
    train_background_per_batch: int = 16
    infer_batch: int = 32


@dataclass(frozen=True)
class FLConfig:
    """
    Configuración de la estrategia y el servidor de aprendizaje federado (Flower).

    Los valores de parada temprana y decaimiento son copia de CONFIG de
    `mejor.py` (celda 2), salvo donde se indica.
    """
    grpc_port: int = 8080

    # — Selección de clientes —
    # A 1.0 porque la prueba usa 3 dispositivos: con fraction_fit=0.3, Flower
    # seleccionaría uno solo por ronda y la federación dejaría de serlo.
    # `mejor.py` podía permitirse muestrear porque simulaba decenas de clientes.
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 1
    min_evaluate_clients: int = 1
    min_available_clients: int = 1

    # — Duración y parada temprana —
    # 50 rondas, no las 110 del cuadernillo: aquí el encoder no parte de cero,
    # ya viene pre-entrenado horneado en el artefacto de la Fase 1.
    num_rounds: int = 50
    warmup_rounds: int = 5
    early_stop_patience: int = 15
    eval_ema_alpha: float = 0.3

    # — Presupuesto por ronda —
    # Medido en un Redmi Note 11 Pro: 250 pasos a ~139 ms = ~35 s de cómputo
    # por cliente, más la evaluación y el transporte de 25 464 floats.
    #
    # 900 s (15 min) da ~25x de margen sobre ese dispositivo. Es deliberadamente
    # generoso porque no sabemos cuánto rinden los otros dos teléfonos y porque
    # el coste de pasarse es asimétrico: un timeout corto EXPULSA al cliente
    # lento de la ronda y su encoder no entra en la media, mientras que uno
    # largo sólo hace esperar. Con dispositivos reales, además, el sistema
    # puede recortar CPU en segundo plano y estirar mucho el tiempo por paso.
    #
    # Lo que NO se puede es dejarlo sin poner: Flower esperaría indefinidamente
    # y un teléfono que se queda sin batería bloquearía la federación entera.
    # Ajustable con --round-timeout cuando midas los otros equipos.
    round_timeout: float = 900.0

    # — Entrenamiento local que se envía en on_fit_config_fn —
    local_epochs: int = 5
    learning_rate: float = 1e-3
    lr_decay_factor: float = 0.5
    lr_decay_at: float = 0.7


@dataclass(frozen=True)
class AppConfig:
    """
    Configuración general del servidor Flask y Flower.
    """

    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True

    model: ModelConfig = field(default_factory=ModelConfig)
    fl: FLConfig = field(default_factory=FLConfig)
