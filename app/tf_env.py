"""
Variables de entorno de TensorFlow.

DEBE importarse ANTES que `tensorflow` en cualquier proceso que construya
o exporte el modelo FedPer. El cuadernillo `mejor.py` fija estas mismas
variables en su primera celda; reproducir el entrenamiento exige reproducir
también el backend de Keras que se usó.

`TF_USE_LEGACY_KERAS=1` es el más importante: obliga a que `tf.keras` apunte
a `tf_keras` (Keras 2). Keras 3 cambia el orden de `model.weights`, la API de
los optimizadores y la interacción con `tf.Module`, lo que rompería tanto la
compatibilidad de pesos con el cuadernillo como la conversión a TFLite.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def assert_legacy_keras() -> None:
    """Verifica que el backend activo sea Keras 2 (tf_keras)."""
    import tensorflow as tf

    version = getattr(tf.keras, "__version__", "")
    if version.startswith("3."):
        raise RuntimeError(
            "tf.keras está resolviendo a Keras 3 "
            f"(version={version}). El modelo FedPer de `mejor.py` se entrenó "
            "con Keras 2. Instala `tf-keras` y asegúrate de importar "
            "`app.tf_env` antes que `tensorflow`."
        )
