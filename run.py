"""
Punto de entrada del servidor de autenticación federada.

Uso:
    python run.py
    python run.py --sensor-config gyro_acc_touch --port 8080
"""

import argparse

from app.config import AppConfig, ModelConfig
from app.factory import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Servidor FL de autenticación continua"
    )
    parser.add_argument(
        "--sensor-config",
        choices=["gyro", "gyro_acc", "gyro_acc_touch"],
        default="gyro_acc",
        help="Configuración de sensores del modelo (default: gyro_acc)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host del servidor (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Puerto del servidor (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Modo debug (default: True)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_config = ModelConfig(sensor_config=args.sensor_config)
    app_config = AppConfig(
        host=args.host,
        port=args.port,
        debug=args.debug,
        model=model_config,
    )

    app = create_app(app_config)

    print(f"\n{'=' * 60}")
    print(f"  Servidor FL — Autenticación Continua")
    print(f"  Configuración de sensores: {args.sensor_config}")
    print(f"  Escuchando en: http://{args.host}:{args.port}")
    print(f"{'=' * 60}")
    print(f"  Endpoints:")
    print(f"    GET /api/model/parameters  → Descargar pesos (NPZ)")
    print(f"    GET /api/model/info        → Metadatos del modelo")
    print(f"{'=' * 60}\n")

    app.run(host=app_config.host, port=app_config.port, debug=app_config.debug)


if __name__ == "__main__":
    main()
