"""
Servicio de gestión del modelo global para aprendizaje federado.

Responsabilidad única: mantener el modelo global en memoria y
serializar sus parámetros para distribución a los clientes.

Principios aplicados:
  - SRP: solo gestiona el ciclo de vida del modelo global
         (inicialización + serialización de pesos).
  - DIP: depende de la abstracción ModelConfig, no de Keras directamente.
  - ISP: expone solo los métodos que necesita la capa de rutas.
"""

import io
import numpy as np
from typing import Dict, Any, List

from tensorflow.keras import Model

from app.config import ModelConfig
from app.models.auth_model import AuthModelBuilder


class ModelService:
    """
    Servicio que encapsula el modelo global del servidor federado.

    Gestiona:
      1. Inicialización del modelo con pesos aleatorios.
      2. Serialización de parámetros para envío a clientes.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: Model = self._initialize_model()

    # -------------------------------------------------------------------
    # Inicialización
    # -------------------------------------------------------------------

    def _initialize_model(self) -> Model:
        """
        Construye el modelo con pesos aleatorios (inicialización de Keras).

        En aprendizaje federado, el servidor inicia con pesos aleatorios
        y los clientes los descargan como punto de partida antes de la
        primera ronda de entrenamiento local.
        """
        builder = AuthModelBuilder(self._config)
        model = builder.build()
        return model

    # -------------------------------------------------------------------
    # Acceso a parámetros
    # -------------------------------------------------------------------

    def get_parameters(self) -> List[np.ndarray]:
        """
        Obtiene los pesos actuales del modelo global.

        Returns:
            Lista de arrays NumPy, uno por capa con parámetros
            entrenables del modelo.
        """
        return self._model.get_weights()

    def get_parameters_serialized(self) -> bytes:
        """
        Serializa los parámetros del modelo en formato binario (NPZ)
        para transmisión eficiente a los clientes.

        El formato NPZ es nativo de NumPy, compacto y deserializable
        directamente en el cliente Android vía TensorFlow Lite.

        Returns:
            Bytes del archivo NPZ conteniendo todos los pesos.
        """
        weights = self.get_parameters()
        buffer = io.BytesIO()
        arrays_dict = {
            f"layer_{i}": w for i, w in enumerate(weights)
        }
        np.savez(buffer, **arrays_dict)
        buffer.seek(0)
        return buffer.read()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Devuelve metadatos del modelo global.

        Útil para que los clientes verifiquen compatibilidad antes
        de descargar los pesos.

        Returns:
            Diccionario con información del modelo.
        """
        return {
            "model_name": self._model.name,
            "sensor_config": self._config.sensor_config,
            "window_size": self._config.window_size,
            "total_parameters": int(self._model.count_params()),
            "input_shapes": [
                list(inp.shape) for inp in self._model.inputs
            ],
            "num_layers": len(self._model.layers),
        }
