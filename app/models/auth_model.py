"""
Arquitectura DeepConvLSTM para autenticación continua basada en
biometría conductual móvil.

Arquitectura base: Mekruksavanich & Jitpattanakul (2021)
Extensión multimodal: fusión tardía para gestos táctiles (SHEN 2024)
Compatible con TensorFlow Lite para despliegue on-device.

Configuraciones de sensores soportadas:
  - 'gyro'            : Solo giroscopio         → ventana (128, 3)
  - 'gyro_acc'        : Giroscopio + Acelerómetro → ventana (128, 6)
  - 'gyro_acc_touch'  : Giroscopio + Acelerómetro + Gestos táctiles

Principios aplicados:
  - SRP: esta clase solo se encarga de construir la arquitectura.
  - OCP: se puede extender con nuevas configuraciones sin modificar
         el código existente (añadiendo entradas a IMU_CHANNELS).
  - DIP: el servicio depende de la abstracción (interfaz de build),
         no de los detalles internos de Keras.

"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers, Input
from typing import Dict, Tuple

from app.config import ModelConfig, SensorConfig


# ---------------------------------------------------------------------------
# Constantes del dominio
# ---------------------------------------------------------------------------

IMU_CHANNELS: Dict[str, int] = {
    "gyro":           3,   # gx, gy, gz
    "gyro_acc":       6,   # gx, gy, gz, ax, ay, az
    "gyro_acc_touch": 6,   # mismos canales IMU; gestos van en rama separada
}

TOUCH_FEATURE_DIM: int = 12


class AuthModelBuilder:
    """
    Constructor del modelo DeepConvLSTM de autenticación continua.

    Responsabilidad única: construir y devolver un modelo Keras compilado
    a partir de una configuración dada. No gestiona estado, no persiste
    pesos, no sirve endpoints.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._validate_config()

    # -------------------------------------------------------------------
    # Validación
    # -------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Valida que la configuración de sensores sea soportada."""
        if self._config.sensor_config not in IMU_CHANNELS:
            raise ValueError(
                f"sensor_config debe ser uno de {list(IMU_CHANNELS.keys())}, "
                f"recibido: '{self._config.sensor_config}'"
            )

    # -------------------------------------------------------------------
    # Bloques internos
    # -------------------------------------------------------------------

    @staticmethod
    def _conv_block(
        x,
        filters: int,
        kernel_size: int = 5,
        dropout_rate: float = 0.2,
        name_prefix: str = "conv",
    ):
        """Bloque Conv1D → BatchNorm → ReLU → Dropout."""
        x = layers.Conv1D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"{name_prefix}_conv",
        )(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu")(x)
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)
        return x

    def _build_imu_branch(self, imu_input):
        """
        Rama DeepConvLSTM para datos IMU.

        4 bloques Conv1D → 2 capas LSTM → 1 Dense embedding.
        """
        cfg = self._config
        x = imu_input

        # Bloques convolucionales con kernels decrecientes
        kernel_sizes = (7, 5, 3, 3)
        for i, (filters, ks) in enumerate(zip(cfg.conv_filters, kernel_sizes)):
            x = self._conv_block(
                x, filters, kernel_size=ks,
                dropout_rate=cfg.dropout_rate,
                name_prefix=f"imu_c{i + 1}",
            )

        # LSTM 1: secuencia completa
        x = layers.LSTM(
            cfg.lstm_units[0],
            return_sequences=True,
            dropout=0.0,            # MODIFICADO
            recurrent_dropout=0.0,  # MODIFICADO
            unroll=True,            # Añadido para TFLite ODT nativo
            name="imu_lstm1",
        )(x)

        # LSTM 2: vector de contexto final
        x = layers.LSTM(
            cfg.lstm_units[1],
            return_sequences=False,
            dropout=0.0,            # MODIFICADO
            recurrent_dropout=0.0,  # MODIFICADO
            unroll=True,            # Añadido para TFLite ODT nativo
            name="imu_lstm2",
        )(x)

        # Proyección a espacio de embedding
        x = layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name="imu_embed",
        )(x)

        return x  # (batch, 64)

    def _build_touch_branch(self, touch_input):
        """
        Rama MLP para features de gestos táctiles.

        Los gestos son eventos discretos representados como vectores
        de estadísticos (12 features).
        """
        cfg = self._config
        x = touch_input

        for i, units in enumerate(cfg.touch_units):
            x = layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
                name=f"touch_fc{i + 1}",
            )(x)
            x = layers.Dropout(cfg.dropout_rate, name=f"touch_drop{i + 1}")(x)

        return x  # (batch, touch_units[-1])

    @staticmethod
    def _build_classification_head(
        fused,
        hidden_units: int = 32,
        dropout_rate: float = 0.3,
    ):
        """
        Cabeza de clasificación binaria.

        Salida: probabilidad ∈ [0, 1]
          ≥ umbral → usuario legítimo
          < umbral → posible impostor
        """
        x = layers.Dense(
            hidden_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name="head_fc",
        )(fused)
        x = layers.Dropout(dropout_rate, name="head_drop")(x)
        output = layers.Dense(1, activation="sigmoid", name="auth_output")(x)
        return output

    # -------------------------------------------------------------------
    # Método público: construir el modelo completo
    # -------------------------------------------------------------------

    def build(self) -> Model:
        """
        Construye y compila el modelo DeepConvLSTM según la configuración.

        Returns:
            Modelo Keras compilado con pesos aleatorios, listo para
            ser distribuido a los clientes federados.
        """
        cfg = self._config
        imu_ch = IMU_CHANNELS[cfg.sensor_config]

        # ── Entradas ──────────────────────────────────────────────
        imu_input = Input(
            shape=(cfg.window_size, imu_ch),
            name="imu_input",
        )
        inputs = [imu_input]

        # ── Rama IMU ──────────────────────────────────────────────
        imu_features = self._build_imu_branch(imu_input)

        # ── Rama táctil (opcional) ────────────────────────────────
        if cfg.sensor_config == "gyro_acc_touch":
            touch_input = Input(
                shape=(TOUCH_FEATURE_DIM,),
                name="touch_input",
            )
            inputs.append(touch_input)

            touch_features = self._build_touch_branch(touch_input)

            # Fusión tardía
            fused = layers.Concatenate(name="fusion")(
                [imu_features, touch_features]
            )
        else:
            fused = imu_features

        # ── Cabeza de clasificación ───────────────────────────────
        output = self._build_classification_head(
            fused,
            hidden_units=cfg.head_units,
            dropout_rate=cfg.dropout_rate,
        )

        # ── Ensamblar y compilar ──────────────────────────────────
        model = Model(
            inputs=inputs,
            outputs=output,
            name=f"AuthDeepConvLSTM_{cfg.sensor_config}",
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=cfg.learning_rate
            ),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc", curve="ROC"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        return model
