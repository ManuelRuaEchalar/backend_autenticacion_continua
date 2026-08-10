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

import importlib
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

UNKNOWN_VERSION = "desconocida"


def keras_version() -> str:
    """
    Versión del Keras al que resuelve `tf.keras`, o `UNKNOWN_VERSION`.

    No basta con leer `tf.keras.__version__`: `tf.keras` es un módulo de carga
    diferida y varias combinaciones de TF/Keras no reexportan `__version__` en
    ese espacio de nombres, lo que provoca un AttributeError. Se cae entonces
    al paquete real que hay detrás.
    """
    import tensorflow as tf

    version = getattr(tf.keras, "__version__", None)
    if version:
        return str(version)

    root = (getattr(tf.keras, "__name__", "") or "").split(".")[0]
    for name in (root, "tf_keras", "keras"):
        if not name:
            continue
        try:
            found = getattr(importlib.import_module(name), "__version__", None)
        except ImportError:
            continue
        if found:
            return str(found)

    return UNKNOWN_VERSION


def assert_legacy_keras() -> None:
    """Verifica que el backend activo sea Keras 2 (tf_keras)."""
    version = keras_version()

    if version.startswith("3."):
        raise RuntimeError(
            "tf.keras está resolviendo a Keras 3 "
            f"(version={version}). El modelo FedPer de `mejor.py` se entrenó "
            "con Keras 2. Instala `tf-keras` y asegúrate de importar "
            "`app.tf_env` antes que `tensorflow`."
        )

    if version == UNKNOWN_VERSION:
        # Esto antes pasaba en silencio: un `getattr` fallido devolvía "", que
        # no empieza por "3.", y la comprobación daba por bueno un Keras 3.
        print(
            "AVISO: no se pudo determinar la versión de Keras. Si `tf-keras` no "
            "está instalado y TensorFlow es >= 2.16, el artefacto saldrá con el "
            "orden de pesos de Keras 3 e incompatible con el cuadernillo."
        )
