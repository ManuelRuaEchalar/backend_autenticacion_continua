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

    # ── Inicializar servicio del modelo ────────────────────────────
    model_service = ModelService(config.model)
    app.config["MODEL_SERVICE"] = model_service

    # ── Registrar blueprints ──────────────────────────────────────
    app.register_blueprint(model_bp)

    return app
