"""
Arquitectura FedPer para autenticación continua: encoder compartido +
cabeza personal.

Este módulo es la traducción literal de las funciones `build_shared_encoder`,
`build_personal_head`, `build_full_model` y `build_pretrain_*` del cuadernillo
`mejor.py` (celda 8). Cualquier divergencia rompe la compatibilidad de pesos
con la simulación validada (~19% EER), así que las capas conservan nombres,
orden y hiperparámetros exactos.

Reparto de responsabilidades:
  · SharedEncoder  -> viaja por FedAvg (server <-> dispositivo).
  · PersonalHead   -> decoder + clasificador; NUNCA sale del dispositivo.

`encoder_layout()` / `head_layout()` describen cómo se aplana cada bloque de
pesos a un único vector float32. Ese vector plano es el contrato de
serialización entre Python y Kotlin: reduce el acuerdo entre ambos lados a un
solo número (el tamaño total) en lugar de a 15 tensores de formas distintas,
y FedAvg promedia elemento a elemento sin enterarse de la diferencia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from app import tf_env  # noqa: F401  (debe importarse antes que tensorflow)

import numpy as np
import tensorflow as tf

from app.config import FedPerConfig


# ---------------------------------------------------------------------------
# Constructores de arquitectura (copia literal de mejor.py)
# ---------------------------------------------------------------------------


def build_shared_encoder(
    window_size: int,
    n_features: int,
    conv_filters: Sequence[int],
    kernel_size: int,
    l2_reg: float,
    lstm_units: int = 48,
) -> tf.keras.Model:
    """Encoder compartido. Es lo único que se agrega con FedAvg."""
    reg = tf.keras.regularizers.l2(l2_reg)
    inp = tf.keras.Input(shape=(window_size, n_features), name="imu_window")
    x = inp
    for i, n_filters in enumerate(conv_filters):
        x = tf.keras.layers.Conv1D(
            n_filters, kernel_size, padding="same",
            kernel_regularizer=reg, name=f"enc_conv{i + 1}_{n_filters}f",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"enc_bn{i + 1}")(x)
        x = tf.keras.layers.Activation("relu", name=f"enc_relu{i + 1}")(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, name=f"enc_pool{i + 1}")(x)
    feat = tf.keras.layers.LSTM(
        lstm_units, return_sequences=False,
        kernel_regularizer=reg, name=f"enc_lstm1_{lstm_units}u",
    )(x)
    return tf.keras.Model(inp, feat, name="SharedEncoder")


def build_personal_head(
    feat_dim: int,
    window_size: int,
    n_features: int,
    lstm_units: Sequence[int],
    bottleneck_dim: int,
    dropout_rate: float,
    l2_reg: float,
) -> tf.keras.Model:
    """
    Cabeza personal multi-tarea: reconstrucción (autoencoder) + clasificación
    binaria genuino/impostor. Se queda siempre en el dispositivo.
    """
    reg = tf.keras.regularizers.l2(l2_reg)
    feat_in = tf.keras.Input(shape=(feat_dim,), name="shared_features")

    latent = tf.keras.layers.Dense(
        bottleneck_dim, activation="relu", kernel_regularizer=reg, name="latent",
    )(feat_in)
    latent = tf.keras.layers.Dropout(dropout_rate, name="latent_dropout")(latent)
    d = tf.keras.layers.RepeatVector(window_size, name="repeat")(latent)
    for i, units in enumerate(reversed(list(lstm_units))):
        d = tf.keras.layers.LSTM(
            units, return_sequences=True,
            kernel_regularizer=reg, name=f"dec_lstm{i + 1}_{units}u",
        )(d)
    recon_out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_features), name="reconstruction",
    )(d)

    c = tf.keras.layers.Dense(
        16, activation="relu", kernel_regularizer=reg, name="cls_dense",
    )(feat_in)
    c = tf.keras.layers.Dropout(dropout_rate, name="cls_dropout")(c)
    cls_out = tf.keras.layers.Dense(1, activation="sigmoid", name="classification")(c)

    return tf.keras.Model(feat_in, [recon_out, cls_out], name="PersonalHead")


def build_full_model(
    encoder: tf.keras.Model,
    head: tf.keras.Model,
    window_size: int,
    n_features: int,
    lr: float,
    freeze_encoder: bool = False,
    cls_loss_weight: float = 1.0,
) -> tf.keras.Model:
    """
    Ensambla encoder + cabeza.

    `freeze_encoder=True` reproduce la Celda 15 del cuadernillo (artefacto de
    personalización post-FL). Para el cliente federado hay que dejarlo en
    False: `AuthClient.fit` de `mejor.py` entrena el modelo COMPLETO y sólo
    después devuelve los pesos del encoder.
    """
    inp = tf.keras.Input(shape=(window_size, n_features))
    # `encoder(inp)` SIN `training=`: el flag se hereda de quien llame al
    # modelo, que es lo que las firmas TFLite esperan (True en train_step,
    # False en infer/infer_batch).
    #
    # Antes ponía `training=not freeze_encoder`, que con freeze_encoder=False
    # HORNEABA training=True en el grafo. Keras no lo sobrescribe con el
    # `training=False` de la llamada externa, así que BatchNormalization
    # normalizaba con estadísticas DEL LOTE durante la inferencia y actualizaba
    # sus medias móviles al evaluar. Consecuencias medidas:
    #   · la puntuación de una ventana dependía de las demás de su lote;
    #   · `infer` (lote de 1) e `infer_batch` (lote de 32) discrepaban hasta
    #     0.12 en el score, y en TFLite la puntuación derivaba en llamadas
    #     sucesivas sobre la MISMA entrada;
    #   · on-device, con lotes de 1, normalizar por la estadística de una sola
    #     muestra destruye la señal.
    # `mejor.py:595` tiene el mismo defecto, y ni `AuthClient` (695) ni
    # `build_user_model` (1172) pasaban freeze_encoder, de modo que la tabla
    # final de EER se calculó con BatchNorm en modo entrenamiento.
    feat = encoder(inp)
    recon_out, cls_out = head(feat)
    model = tf.keras.Model(inp, [recon_out, cls_out], name="FullAuthModel")
    if freeze_encoder:
        # `trainable = False` es la forma correcta de congelar: además de
        # excluir los pesos del gradiente, pone BatchNormalization en modo
        # inferencia de manera permanente.
        encoder.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),
        loss=["mse", "binary_crossentropy"],
        loss_weights=[1.0, cls_loss_weight],
        weighted_metrics=[[], []],
    )
    return model


def build_pretrain_decoder(
    feat_dim: int, window_size: int, n_features: int,
    lstm_units: Sequence[int], l2_reg: float,
) -> tf.keras.Model:
    """Decoder usado sólo durante el pre-entrenamiento no supervisado."""
    reg = tf.keras.regularizers.l2(l2_reg)
    feat_in = tf.keras.Input(shape=(feat_dim,), name="shared_features")
    d = tf.keras.layers.RepeatVector(window_size, name="pt_repeat")(feat_in)
    for i, units in enumerate(reversed(list(lstm_units))):
        d = tf.keras.layers.LSTM(
            units, return_sequences=True, kernel_regularizer=reg,
            name=f"pt_dec_lstm{i + 1}_{units}u",
        )(d)
    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_features), name="pt_reconstruction",
    )(d)
    return tf.keras.Model(feat_in, out, name="PretrainDecoder")


# ---------------------------------------------------------------------------
# Fábricas basadas en FedPerConfig
# ---------------------------------------------------------------------------


def encoder_from_config(cfg: FedPerConfig) -> tf.keras.Model:
    return build_shared_encoder(
        cfg.window_size, cfg.n_features, cfg.conv_filters,
        cfg.kernel_size, cfg.l2_reg, cfg.encoder_lstm_units,
    )


def head_from_config(cfg: FedPerConfig, feat_dim: int) -> tf.keras.Model:
    return build_personal_head(
        feat_dim, cfg.window_size, cfg.n_features,
        cfg.lstm_units, cfg.bottleneck_dim, cfg.dropout, cfg.l2_reg,
    )


def full_model_from_config(
    cfg: FedPerConfig, freeze_encoder: bool = False,
) -> tf.keras.Model:
    """Devuelve el modelo completo listo para entrenar (encoder + cabeza)."""
    encoder = encoder_from_config(cfg)
    head = head_from_config(cfg, encoder.output_shape[-1])
    model = build_full_model(
        encoder, head, cfg.window_size, cfg.n_features, cfg.learning_rate,
        freeze_encoder=freeze_encoder, cls_loss_weight=cfg.cls_loss_weight,
    )
    return model


# ---------------------------------------------------------------------------
# Layout de aplanado (contrato de serialización Python <-> Kotlin)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSlot:
    """Posición de un tensor de pesos dentro del vector plano."""

    name: str
    shape: List[int]
    offset: int
    size: int

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "offset": self.offset,
            "size": self.size,
        }


@dataclass(frozen=True)
class WeightLayout:
    """Descripción completa del aplanado de un bloque de pesos."""

    slots: List[TensorSlot]
    total_size: int

    def to_dict(self) -> Dict:
        return {
            "total_size": self.total_size,
            "num_tensors": len(self.slots),
            "tensors": [s.to_dict() for s in self.slots],
        }


def layout_of(model: tf.keras.Model) -> WeightLayout:
    """
    Construye el layout siguiendo el orden EXACTO de `model.weights`.

    Ese orden es el mismo que devuelve `model.get_weights()`, que es lo que
    `AuthClient.get_parameters` de `mejor.py` envía a FedAvg. Incluye a
    propósito las estadísticas no entrenables de BatchNormalization
    (`moving_mean` / `moving_variance`): el cuadernillo también las agrega.
    """
    slots: List[TensorSlot] = []
    offset = 0
    for w in model.weights:
        shape = [int(d) for d in w.shape]
        size = int(np.prod(shape)) if shape else 1
        slots.append(TensorSlot(name=w.name, shape=shape, offset=offset, size=size))
        offset += size
    return WeightLayout(slots=slots, total_size=offset)


def flatten_weights(weights: Sequence[np.ndarray]) -> np.ndarray:
    """Aplana una lista de arrays (formato `get_weights()`) a un vector."""
    return np.concatenate(
        [np.asarray(w, dtype=np.float32).reshape(-1) for w in weights], axis=0,
    ).astype(np.float32)


def unflatten_weights(flat: np.ndarray, layout: WeightLayout) -> List[np.ndarray]:
    """Inversa de `flatten_weights`, usando el layout para reconstruir formas."""
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    if flat.size != layout.total_size:
        raise ValueError(
            f"Vector plano de tamaño {flat.size}, se esperaban "
            f"{layout.total_size} elementos."
        )
    out: List[np.ndarray] = []
    for slot in layout.slots:
        chunk = flat[slot.offset: slot.offset + slot.size]
        out.append(chunk.reshape(slot.shape).astype(np.float32))
    return out
