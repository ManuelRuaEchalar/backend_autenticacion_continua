"""
Factory de la aplicación Flask.

Patrón Application Factory: permite crear múltiples instancias
con diferentes configuraciones (útil para testing).

Principios aplicados:
  - SRP: solo ensambla componentes (no contiene lógica de negocio).
  - DIP: los componentes se inyectan vía configuración.
"""

from flask import Flask
import threading

_fl_lock = threading.Lock()

from app.config import AppConfig
from app.services.encoder_service import EncoderService
from app.services.federation_service import FederationService
from app.routes.model_routes import model_bp


def create_app(config: AppConfig | None = None) -> Flask:
    """
    Crea y configura la aplicación Flask.

    Args:
        config: configuración de la aplicación. Si es None,
                se usa la configuración por defecto.

    Returns:
        Instancia de Flask configurada y lista para ejecutar.
    """
    if config is None:
        config = AppConfig()

    app = Flask(__name__)

    # ── Inicializar servicios ──────────────────────────────────────
    # EncoderService sustituye a ModelService (Fase 3): sirve el vector plano
    # del encoder FedPer en vez del DeepConvLSTM completo, y no carga TF.
    encoder_service = EncoderService()
    federation_service = FederationService()

    app.config["ENCODER_SERVICE"] = encoder_service
    app.config["FEDERATION_SERVICE"] = federation_service
    # Modo de ablación, que la app consulta antes de construir su partición.
    # Vive aquí y no en EncoderService porque es un ajuste del EXPERIMENTO, no
    # una propiedad del modelo.
    app.config["ABLATION"] = config.fl.ablation

    # ── Rutas base ────────────────────────────────────────────────
    @app.route("/health", methods=["GET"])
    def health_check():
        return {"status": "ok", "message": "Federated Learning Server is running"}, 200

    # ── Registrar blueprints ──────────────────────────────────────
    app.register_blueprint(model_bp)

    # ── Iniciar Servidor FL (Flower) ──────────────────────────────
    import os
    from app.fl_server import start_fl_server

    # Usamos una variable de entorno para evitar que se lance múltiples veces
    # si el servidor (ej. Flask dev server) recarga la aplicación.
    with _fl_lock:
        if os.environ.get("FLOWER_STARTED") != "1":
            os.environ["FLOWER_STARTED"] = "1"
            fl_thread = threading.Thread(
                target=start_fl_server,
                args=(encoder_service, federation_service, config.fl),
                daemon=True,
            )
            fl_thread.start()

    return app
