"""
Rutas REST del modelo global.

Endpoints:
  GET /api/model/info    → contrato que verifica la app antes de entrenar
  GET /api/model/status  → estado de la federación en curso

Principios aplicados:
  - SRP: solo define endpoints y delega la lógica al servicio.
  - DIP: depende de EncoderService (abstracción), no de Keras/TF.
"""

from flask import Blueprint, current_app, jsonify

from app.services.encoder_service import EncoderService


model_bp = Blueprint("model", __name__, url_prefix="/api/model")


def _get_service() -> EncoderService:
    return current_app.config["ENCODER_SERVICE"]


@model_bp.route("/info", methods=["GET"])
def get_model_info():
    """
    Metadatos del modelo global.

    `ModelInfoFetcher.requireCompatibleWith` del cliente compara
    `sensor_config`, `window_size` y `encoder_flat_size` con su manifiesto
    empaquetado y ABORTA la ronda si difieren. Todos estos valores salen de
    `artifacts/model_manifest.json`, el mismo fichero que va en el APK, así que
    coinciden por construcción mientras ambos vengan de la misma exportación.

    Antes de la Fase 3 este endpoint devolvía `sensor_config="gyro_acc"` y no
    incluía `encoder_flat_size`, que es exactamente por lo que la app abortaba.
    """
    info = _get_service().get_model_info()
    # El modo de ablación se añade aquí y no en EncoderService: es un ajuste
    # del experimento, no una propiedad del modelo, y mezclarlo con el
    # manifiesto invitaría a que acabase comparándose en
    # `requireCompatibleWith` como si fuera parte del contrato.
    info["ablation"] = current_app.config.get("ABLATION", "full")
    return jsonify(info), 200


@model_bp.route("/status", methods=["GET"])
def get_status():
    """Progreso de la federación, para mirarlo desde el móvil o con curl."""
    federation = current_app.config["FEDERATION_SERVICE"]
    service = _get_service()
    return jsonify({
        "encoder_flat_size": service.encoder_flat_size,
        "current_round": service.get_model_info()["current_round"],
        "federation": federation.get_status(),
        "history": federation.get_history(),
    }), 200
