"""
Factory de la aplicación Flask.

Patrón Application Factory: permite crear múltiples instancias
con diferentes configuraciones (útil para testing).

Principios aplicados:
  - SRP: solo ensambla componentes (no contiene lógica de negocio).
  - DIP: los componentes se inyectan vía configuración.
"""

from flask import Flask

from app.config import AppConfig
from app.services.model_service import ModelService
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
    model_service = ModelService(config.model)
    federation_service = FederationService()
    
    app.config["MODEL_SERVICE"] = model_service
    app.config["FEDERATION_SERVICE"] = federation_service

    # ── Rutas base ────────────────────────────────────────────────
    @app.route("/health", methods=["GET"])
    def health_check():
        return {"status": "ok", "message": "Federated Learning Server is running"}, 200

    # ── Registrar blueprints ──────────────────────────────────────
    app.register_blueprint(model_bp)

    # ── Iniciar Servidor FL (Flower) ──────────────────────────────
    import os
    import threading
    from app.fl_server import start_fl_server

    # Usamos una variable de entorno para evitar que se lance múltiples veces
    # si el servidor (ej. Flask dev server) recarga la aplicación.
    if os.environ.get("FLOWER_STARTED") != "1":
        os.environ["FLOWER_STARTED"] = "1"
        fl_thread = threading.Thread(
            target=start_fl_server,
            args=(model_service, federation_service, config.fl),
            daemon=True,
        )
        fl_thread.start()

    return app
