"""
Envoltorio `tf.Module` que expone el modelo FedPer como firmas TFLite
entrenables on-device.

Es la contraparte real de la Celda 15 de `mejor.py`, con dos diferencias
deliberadas y documentadas:

1. EL ENCODER NO SE CONGELA.
   La Celda 15 exporta con `freeze_encoder=True`, así que su firma `train`
   sólo actualiza la cabeza. Ese artefacto sirve para la personalización
   posterior a la federación, no para participar en ella. Lo que hay que
   reproducir en producción es `AuthClient.fit` (mejor.py:703-721), que
   entrena el modelo COMPLETO y sólo después devuelve `encoder.get_weights()`
   al servidor. Aquí `trainable_variables` incluye encoder y cabeza.

2. EL FLAG `training` NO SE HORNEA EN EL GRAFO.
   `build_full_model` fija `encoder(inp, training=not freeze_encoder)` en
   tiempo de construcción, de modo que en el cuadernillo el encoder usa
   estadísticas de lote en BatchNormalization incluso durante `predict()`.
   On-device eso sería catastrófico: la autenticación en tiempo real infiere
   con lotes de 1 y normalizar por la estadística de una sola muestra destruye
   la señal. Aquí el flag se propaga por firma: `training=True` al entrenar,
   `training=False` al inferir (BN usa sus medias móviles). Las medias móviles
   siguen viajando dentro del vector del encoder, igual que en el cuadernillo.

Contrato de pesos: un único vector plano float32 por bloque
(`encoder_flat`, `head_flat`), con offsets estáticos horneados en el grafo.
Ver `fedper_model.layout_of`.

Formas FIJAS en todas las firmas: la API de SignatureRunner de TFLite/Java
no redimensiona entradas dinámicas de forma fiable, y los cambios de forma en
tiempo de ejecución son la causa habitual de crashes de entrenamiento
on-device.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from app import tf_env  # noqa: F401  (debe importarse antes que tensorflow)

import tensorflow as tf

from app.config import FedPerConfig
from app.models import fedper_model as fp


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _optimizer_variables(optimizer) -> List[tf.Variable]:
    """
    Devuelve las variables de estado del optimizador.

    Se normaliza el acceso porque `variables` es propiedad en el optimizador
    base de Keras 2.11+ y método en el `optimizer_v2` antiguo.
    """
    attr = getattr(optimizer, "variables", None)
    if callable(attr):
        attr = attr()
    return list(attr or [])


def _l2_regularized_kernels(models: Sequence[tf.keras.Model]) -> List[tf.Variable]:
    """
    Recolecta los kernels que llevan `kernel_regularizer` en la arquitectura.

    La pérdida se calcula explícitamente en lugar de leer `model.losses` para
    que sea totalmente determinista bajo `tf.function`: `model.losses` se
    reconstruye en cada llamada y su comportamiento al trazar depende de la
    versión de Keras. Con `l2(x)` de Keras el término es `l2 * sum(k**2)`,
    que es exactamente lo que se reproduce aquí.
    """
    kernels: List[tf.Variable] = []
    for model in models:
        for layer in model.layers:
            if getattr(layer, "kernel_regularizer", None) is None:
                continue
            kernel = getattr(layer, "kernel", None)
            if kernel is not None:
                kernels.append(kernel)
    return kernels


# ---------------------------------------------------------------------------
# Módulo exportable
# ---------------------------------------------------------------------------


class TrainableAuthModel(tf.Module):
    """
    Modelo FedPer con firmas de entrenamiento, inferencia e intercambio de
    pesos, listo para convertirse a TFLite.

    Firmas expuestas:
      initialize()                          -> {status}
      train_step(x_genuine, x_background)   -> {loss, recon_loss, cls_loss}
      infer(x, threshold)                   -> {genuine_score, reconstruction_error, is_genuine}
      infer_batch(x, threshold)             -> idem, lote de `infer_batch`
      save_encoder()                        -> {encoder_flat}
      restore_encoder(encoder_flat)         -> {status}
      save_head()                           -> {head_flat}
      restore_head(head_flat)               -> {status}
      set_lr(lr)                            -> {lr}
      reset_optimizer()                     -> {status}
    """

    def __init__(
        self,
        cfg: FedPerConfig,
        encoder: tf.keras.Model | None = None,
        head: tf.keras.Model | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = encoder if encoder is not None else fp.encoder_from_config(cfg)
        feat_dim = self.encoder.output_shape[-1]
        self.head = head if head is not None else fp.head_from_config(cfg, feat_dim)

        # Se ensambla SIN compilar y SIN fijar `training`: el flag lo aporta
        # cada firma. Compilar añadiría un segundo optimizador al grafo.
        inp = tf.keras.Input(shape=(cfg.window_size, cfg.n_features), name="imu_window")
        recon_out, cls_out = self.head(self.encoder(inp))
        self.model = tf.keras.Model(inp, [recon_out, cls_out], name="FullAuthModel")

        # `clipnorm=1.0` reproduce el optimizador con el que se compila el
        # modelo en `build_full_model` de `mejor.py`.
        self.optimizer = tf.keras.optimizers.Adam(cfg.learning_rate, clipnorm=1.0)
        self.bce_fn = tf.keras.losses.BinaryCrossentropy()

        self._l2_kernels = _l2_regularized_kernels([self.encoder, self.head])
        self._l2_reg = float(cfg.l2_reg)
        self._cls_loss_weight = float(cfg.cls_loss_weight)

        self.encoder_layout = fp.layout_of(self.encoder)
        self.head_layout = fp.layout_of(self.head)

        # Los slots de Adam deben existir ANTES de trazar las firmas: crear
        # variables dentro de una `tf.function` rompe la conversión a TFLite.
        self.optimizer.build(self.model.trainable_variables)
        self._reset_targets = [
            v for v in _optimizer_variables(self.optimizer)
            if v is not self.optimizer.learning_rate
        ]

    # ------------------------------------------------------------------
    # Pérdida (réplica de AuthClient.fit)
    # ------------------------------------------------------------------

    def _l2_loss(self) -> tf.Tensor:
        if not self._l2_kernels:
            return tf.constant(0.0, dtype=tf.float32)
        return self._l2_reg * tf.add_n(
            [tf.reduce_sum(tf.square(k)) for k in self._l2_kernels]
        )

    def _multitask_loss(self, x, y, recon, cls_prob):
        """
        Réplica de `model.fit(x, [x, y], sample_weight=[y, ones])` con
        `loss=["mse","binary_crossentropy"]` y `loss_weights=[1.0, w]`.

        El peso de muestra `y` sobre la rama de reconstrucción hace que el
        autoencoder sólo aprenda a reconstruir ventanas GENUINAS; los
        impostores contribuyen únicamente al clasificador.
        """
        recon_w = tf.reshape(y, [-1])
        per_sample_mse = tf.reduce_mean(tf.square(x - recon), axis=[1, 2])
        recon_loss = tf.reduce_mean(recon_w * per_sample_mse)
        cls_loss = self.bce_fn(y, cls_prob)
        total = recon_loss + self._cls_loss_weight * cls_loss + self._l2_loss()
        return total, recon_loss, cls_loss

    # ------------------------------------------------------------------
    # Firmas
    # ------------------------------------------------------------------

    def build_signatures(self) -> Dict[str, object]:
        """
        Construye las `tf.function` con las formas fijas de `cfg` y devuelve
        sus funciones concretas, listas para `tf.saved_model.save`.

        Se crean aquí (y no como decoradores de método) porque las formas
        dependen de la configuración, que es un parámetro de instancia.
        """
        cfg = self.cfg
        ws, nf = cfg.window_size, cfg.n_features
        n_gen = cfg.train_genuine_per_batch
        n_bg = cfg.train_background_per_batch
        n_infer = cfg.infer_batch
        enc_size = self.encoder_layout.total_size
        head_size = self.head_layout.total_size

        # NOTA SOBRE `dummy` — no borrar.
        #
        # Cuatro firmas (initialize, reset_optimizer, save_encoder, save_head)
        # no necesitan entradas conceptualmente, pero el wrapper Java de TFLite
        # rechaza invocarlas:
        #
        #     Interpreter.runSignature(inputs, outputs, key)
        #       if (inputs == null || inputs.isEmpty())
        #         throw new IllegalArgumentException(
        #             "Input error: Inputs should not be null or empty.");
        #
        # El intérprete de Python no tiene esa restricción, así que
        # `verify_tflite_model.py` daba 8/8 mientras el dispositivo fallaba las
        # 7 pruebas en @Before. Un escalar ignorado es la forma más barata de
        # que la firma sea invocable desde Android. No afecta a los pesos ni a
        # encoder_flat_size.
        _dummy_spec = tf.TensorSpec([], tf.float32, name="dummy")

        @tf.function(input_signature=[_dummy_spec])
        def initialize(dummy):
            del dummy  # ver NOTA SOBRE `dummy`
            return {"status": tf.constant(1, dtype=tf.int32)}

        @tf.function(input_signature=[
            tf.TensorSpec([n_gen, ws, nf], tf.float32, name="x_genuine"),
            tf.TensorSpec([n_bg, ws, nf], tf.float32, name="x_background"),
        ])
        def train_step(x_genuine, x_background):
            x = tf.concat([x_genuine, x_background], axis=0)
            y = tf.concat([
                tf.ones([n_gen, 1], dtype=tf.float32),
                tf.zeros([n_bg, 1], dtype=tf.float32),
            ], axis=0)
            with tf.GradientTape() as tape:
                recon, cls_prob = self.model(x, training=True)
                loss, recon_loss, cls_loss = self._multitask_loss(x, y, recon, cls_prob)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return {"loss": loss, "recon_loss": recon_loss, "cls_loss": cls_loss}

        def _infer_body(x, threshold):
            recon, cls_prob = self.model(x, training=False)
            cls_prob = tf.reshape(cls_prob, [-1])
            mse = tf.reduce_mean(tf.square(x - recon), axis=[1, 2])
            return {
                "genuine_score": cls_prob,
                "reconstruction_error": mse,
                "is_genuine": tf.cast(cls_prob >= threshold, tf.int32),
            }

        @tf.function(input_signature=[
            tf.TensorSpec([1, ws, nf], tf.float32, name="x"),
            tf.TensorSpec([], tf.float32, name="threshold"),
        ])
        def infer(x, threshold):
            return _infer_body(x, threshold)

        @tf.function(input_signature=[
            tf.TensorSpec([n_infer, ws, nf], tf.float32, name="x"),
            tf.TensorSpec([], tf.float32, name="threshold"),
        ])
        def infer_batch(x, threshold):
            return _infer_body(x, threshold)

        @tf.function(input_signature=[_dummy_spec])
        def save_encoder(dummy):
            del dummy  # ver NOTA SOBRE `dummy`
            return {"encoder_flat": self._flatten(self.encoder.weights)}

        @tf.function(input_signature=[
            tf.TensorSpec([enc_size], tf.float32, name="encoder_flat"),
        ])
        def restore_encoder(encoder_flat):
            self._assign_flat(self.encoder.weights, self.encoder_layout, encoder_flat)
            return {"status": tf.constant(1, dtype=tf.int32)}

        @tf.function(input_signature=[_dummy_spec])
        def save_head(dummy):
            del dummy  # ver NOTA SOBRE `dummy`
            return {"head_flat": self._flatten(self.head.weights)}

        @tf.function(input_signature=[
            tf.TensorSpec([head_size], tf.float32, name="head_flat"),
        ])
        def restore_head(head_flat):
            self._assign_flat(self.head.weights, self.head_layout, head_flat)
            return {"status": tf.constant(1, dtype=tf.int32)}

        @tf.function(input_signature=[tf.TensorSpec([], tf.float32, name="lr")])
        def set_lr(lr):
            # `fit_config` de mejor.py decae el learning rate a partir de la
            # ronda 0.7 * num_rounds y lo envía en la config de la ronda;
            # `AuthClient.fit` lo aplica con este mismo `assign`.
            self.optimizer.learning_rate.assign(lr)
            return {"lr": self.optimizer.learning_rate}

        @tf.function(input_signature=[_dummy_spec])
        def reset_optimizer(dummy):
            del dummy  # ver NOTA SOBRE `dummy`
            # En la simulación, Ray reconstruye `AuthClient` (y con él, el
            # optimizador) en cada ronda, así que Adam arranca sin momentos
            # acumulados. On-device el intérprete es persistente, de modo que
            # hay que resetearlo explícitamente al inicio de cada ronda para
            # reproducir esa dinámica.
            for v in self._reset_targets:
                v.assign(tf.zeros_like(v))
            return {"status": tf.constant(1, dtype=tf.int32)}

        self._signature_fns = {
            "initialize": initialize,
            "train_step": train_step,
            "infer": infer,
            "infer_batch": infer_batch,
            "save_encoder": save_encoder,
            "restore_encoder": restore_encoder,
            "save_head": save_head,
            "restore_head": restore_head,
            "set_lr": set_lr,
            "reset_optimizer": reset_optimizer,
        }
        return {
            name: fn.get_concrete_function()
            for name, fn in self._signature_fns.items()
        }

    # ------------------------------------------------------------------
    # Aplanado / desaplanado dentro del grafo
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(variables: Sequence[tf.Variable]) -> tf.Tensor:
        return tf.concat([tf.reshape(v.read_value(), [-1]) for v in variables], axis=0)

    @staticmethod
    def _assign_flat(
        variables: Sequence[tf.Variable],
        layout: fp.WeightLayout,
        flat: tf.Tensor,
    ) -> None:
        for var, slot in zip(variables, layout.slots):
            chunk = tf.slice(flat, [slot.offset], [slot.size])
            var.assign(tf.reshape(chunk, slot.shape))

    # ------------------------------------------------------------------
    # Calentamiento
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """
        Ejecuta cada firma una vez en modo eager.

        Obligatorio antes de exportar: fuerza la creación de todas las
        variables de recurso (slots de Adam incluidos) fuera de cualquier
        `tf.function`. Sin esto, la conversión a TFLite falla al encontrar
        creación de variables dentro del grafo.
        """
        cfg = self.cfg
        ws, nf = cfg.window_size, cfg.n_features
        x_gen = tf.zeros([cfg.train_genuine_per_batch, ws, nf], dtype=tf.float32)
        x_bg = tf.zeros([cfg.train_background_per_batch, ws, nf], dtype=tf.float32)
        thr = tf.constant(cfg.decision_threshold, dtype=tf.float32)

        fns = getattr(self, "_signature_fns", None)
        if fns is None:
            raise RuntimeError("Llama a build_signatures() antes de warmup().")

        # Escalar ignorado que exigen las cuatro firmas sin entradas reales;
        # ver NOTA SOBRE `dummy` en build_signatures().
        dummy = tf.constant(0.0, dtype=tf.float32)

        fns["initialize"](dummy)
        fns["set_lr"](tf.constant(cfg.learning_rate, dtype=tf.float32))
        fns["train_step"](x_gen, x_bg)
        fns["reset_optimizer"](dummy)
        fns["infer"](tf.zeros([1, ws, nf], dtype=tf.float32), thr)
        fns["infer_batch"](tf.zeros([cfg.infer_batch, ws, nf], dtype=tf.float32), thr)
        enc = fns["save_encoder"](dummy)["encoder_flat"]
        fns["restore_encoder"](enc)
        hd = fns["save_head"](dummy)["head_flat"]
        fns["restore_head"](hd)
