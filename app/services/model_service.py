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
from app.services.s3_service import S3Service


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
        Construye el modelo intentando cargar primero un archivo preentrenado (.keras).
        Si no se encuentra, construye uno desde cero con pesos aleatorios.
        """
        import os
        from tensorflow.keras.models import load_model

        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "startModel.keras")
        
        s3_service = S3Service()
        # Intentar descargar de S3 si no existe localmente o para asegurar la última versión
        downloaded = s3_service.download_start_model(model_path, "startModel.keras")
        
        if os.path.exists(model_path):
            print(f"Cargando modelo inicial desde: {model_path}")
            model = load_model(model_path)
            # Si existe localmente pero no se pudo descargar del S3 (ej. el bucket estaba vacío), lo subimos
            if not downloaded:
                print("Subiendo startModel.keras local a S3 para inicializar el bucket...")
                s3_service.upload_checkpoint(model_path, "startModel.keras")
            return model
            
        print("Archivo startModel.keras no encontrado ni en S3 ni local, inicializando con pesos aleatorios...")
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

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Actualiza los pesos del modelo global en memoria.
        Usualmente llamado después de la agregación de una ronda federada.

        Args:
            weights: Lista de arrays NumPy con los nuevos parámetros.
        """
        self._model.set_weights(weights)

    def save_checkpoint(self, round_number: int) -> None:
        """
        Guarda el modelo global en formato .keras y lo sube a S3.
        """
        import os
        
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(models_dir, exist_ok=True)
        checkpoint_filename = f"model_round_{round_number}.keras"
        local_path = os.path.join(models_dir, checkpoint_filename)
        
        print(f"Guardando checkpoint local en: {local_path}")
        self._model.save(local_path)
        
        s3_service = S3Service()
        # Subir con el nombre de la ronda específica
        s3_service.upload_checkpoint(local_path, checkpoint_filename)
        # Subir como latestModel.keras para tener una referencia fija al último
        s3_service.upload_checkpoint(local_path, "latestModel.keras")

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
