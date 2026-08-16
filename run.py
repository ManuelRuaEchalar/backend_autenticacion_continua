"""
Punto de entrada del servidor de autenticación federada (FedPer, Fase 3).

Uso:
    python run.py
    python run.py --port 5000 --grpc-port 8080

Todo corre en local: no hace falta AWS. El registro de métricas en PostgreSQL
y los checkpoints en S3 son opcionales y están desactivados salvo que se
definan DATABASE_URL y AWS_S3_BUCKET_NAME. Los checkpoints del encoder se
guardan siempre en `artifacts/checkpoints/`.
"""

import argparse
import logging
import os
import sys
from dataclasses import replace

from dotenv import load_dotenv

load_dotenv()

from app.config import AppConfig, FLConfig
from app.database import init_db
from app.factory import create_app
from app.services.encoder_service import ArtifactsMissing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Servidor FL de autenticación continua (FedPer)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000,
                        help="Puerto de la API REST (default: 5000)")
    parser.add_argument("--grpc-port", type=int, default=8080,
                        help="Puerto gRPC de Flower (default: 8080)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Número de rondas federadas (default: 50)")
    parser.add_argument("--min-clients", type=int, default=None,
                        help="Clientes mínimos para empezar una ronda")
    parser.add_argument("--round-timeout", type=float, default=None,
                        help="Segundos máximos por ronda (default: 300)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Rondas sin mejora de val_auc antes de parar (default: 15)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Rondas iniciales que no cuentan para la paciencia")
    parser.add_argument("--ablation", default="full", choices=["full", "baseline", "matched_off", "peer"],
                        help="'full': filtro de actividad + impostores "
                             "emparejados por energia. 'baseline': sin filtro y "
                             "con impostores al azar (comportamiento previo a "
                             "la v1.6). Lo lee la app de /api/model/info, asi "
                             "que ambas corridas usan el MISMO APK.")
    parser.add_argument("--debug", action="store_true",
                        help="Modo debug de Flask")
    return parser.parse_args()


def puerto_ocupado(puerto: int) -> bool:
    """
    ¿Hay ya algo escuchando en ese puerto?

    Se comprueba ANTES de cargar nada: descubrirlo después, con el gRPC
    fallando en un hilo secundario, deja el proceso a medias —REST arriba,
    Flower abajo— y el motivo real se pierde entre los reintentos.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", puerto)) == 0


def main() -> None:
    args = parse_args()

    for puerto, nombre in ((args.port, "REST"), (args.grpc_port, "gRPC de Flower")):
        if puerto_ocupado(puerto):
            print(
                f"\n  El puerto {puerto} ({nombre}) ya está ocupado.\n"
                f"\n  Suele ser un servidor anterior que sigue vivo. Para verlo"
                f" y cerrarlo:\n"
                f"\n      Get-NetTCPConnection -State Listen -LocalPort {puerto} |"
                f"\n        ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force }}\n",
                file=sys.stderr,
            )
            raise SystemExit(1)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    init_db()  # no-op si no hay DATABASE_URL alcanzable

    fl_config = FLConfig(grpc_port=args.grpc_port)
    if args.rounds is not None:
        fl_config = replace(fl_config, num_rounds=args.rounds)
    if args.round_timeout is not None:
        fl_config = replace(fl_config, round_timeout=args.round_timeout)
    if args.patience is not None:
        fl_config = replace(fl_config, early_stop_patience=args.patience)
    if args.warmup is not None:
        fl_config = replace(fl_config, warmup_rounds=args.warmup)
    if args.min_clients is not None:
        fl_config = replace(
            fl_config,
            min_fit_clients=args.min_clients,
            min_evaluate_clients=args.min_clients,
            min_available_clients=args.min_clients,
        )
    fl_config = replace(fl_config, ablation=args.ablation)

    app_config = AppConfig(
        host=args.host, port=args.port, debug=args.debug, fl=fl_config,
    )

    # Flower lo arranca `create_app`, UNA sola vez. Antes lo lanzaban run.py y
    # factory.py por separado y chocaban en el 8080; por eso hacía falta
    # exportar FLOWER_STARTED=1 a mano. Se limpia aquí para que una variable
    # heredada de la shell no impida que arranque el gRPC.
    os.environ["FLOWER_STARTED"] = "0"

    try:
        app = create_app(app_config)
    except ArtifactsMissing as e:
        print(f"\nNo se puede arrancar:\n\n{e}\n", file=sys.stderr)
        raise SystemExit(1)

    info = app.config["ENCODER_SERVICE"].get_model_info()
    linea = "=" * 62
    print(f"\n{linea}")
    print("  Servidor FL — Autenticación Continua (FedPer)")
    print(linea)
    print(f"  Arquitectura      : {info['architecture']}")
    print(f"  sensor_config     : {info['sensor_config']}  "
          f"(ventana {info['window_size']}x{info['n_features']})")
    print(f"  encoder_flat_size : {info['encoder_flat_size']}  <- debe coincidir con el APK")
    print(f"  head_flat_size    : {info['head_flat_size']}  (nunca sale del móvil)")
    print(linea)
    print(f"  REST  : http://{args.host}:{args.port}")
    print(f"  gRPC  : {args.host}:{args.grpc_port}")
    print(f"  Rondas: {fl_config.num_rounds}   timeout/ronda: {fl_config.round_timeout:.0f}s"
          f"   clientes mínimos: {fl_config.min_available_clients}")
    # En grande y en su propia línea: es lo que distingue una corrida de otra
    # y lo que se leerá al comparar los dos resultados.
    modo = {
        "full": "filtro de actividad + impostores emparejados",
        "baseline": "SIN filtro, impostores al azar (linea base)",
        "matched_off": "filtro SI, impostores AL AZAR (aisla el emparejado)",
        "peer": "impostores de OTRO USUARIO REAL (pendiente G)",
    }[fl_config.ablation]
    print(f"  ABLACIÓN: {fl_config.ablation.upper()}  -> {modo}")
    print(linea)
    print("  GET /health")
    print("  GET /api/model/info    metadatos y contrato")
    print("  GET /api/model/status  progreso de la federación")
    print(f"{linea}\n")

    if info["scaler_is_identity_placeholder"]:
        print("  AVISO: el artefacto usa un normalizador identidad. La app se "
              "negará a entrenar. Reexporta con --scaler-stats.\n")

    app.run(
        host=app_config.host,
        port=app_config.port,
        debug=app_config.debug,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
