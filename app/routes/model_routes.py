"""
Rutas REST para distribución de parámetros del modelo global.

Endpoints:
  GET /api/model/parameters  → descarga los pesos del modelo (binario NPZ)
  GET /api/model/info        → metadatos del modelo (JSON)

Principios aplicados:
  - SRP: solo define endpoints y delega la lógica al servicio.
  - DIP: depende de ModelService (abstracción), no de Keras/TF.
"""

from flask import Blueprint, Response, jsonify, current_app

from app.services.model_service import ModelService


model_bp = Blueprint("model", __name__, url_prefix="/api/model")


def _get_service() -> ModelService:
    """
    Obtiene la instancia de ModelService del contexto de la aplicación.

    Evita acoplamiento directo entre las rutas y la instancia global
    del servicio; se inyecta en tiempo de ejecución vía app context.
    """
    return current_app.config["MODEL_SERVICE"]


@model_bp.route("/parameters", methods=["GET"])
def get_parameters():
    """
    Endpoint para que los clientes federados descarguen los parámetros
    (pesos) actuales del modelo global.

    Response:
        200: archivo binario NPZ con los pesos del modelo.
        Content-Type: application/octet-stream
    """
    service = _get_service()
    data = service.get_parameters_serialized()

    return Response(
        data,
        status=200,
        mimetype="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=global_model_weights.npz",
        },
    )


@model_bp.route("/info", methods=["GET"])
def get_model_info():
    """
    Endpoint que devuelve metadatos del modelo global.

    Permite a los clientes verificar compatibilidad (configuración
    de sensores, tamaño de ventana, etc.) antes de descargar los pesos.

    Response:
        200: JSON con información del modelo.
    """
    service = _get_service()
    info = service.get_model_info()
    return jsonify(info), 200
