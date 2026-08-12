"""
Arranque del servidor gRPC de Flower — Fase 3 (FedPer).

El servidor sirve y agrega un único vector plano: el encoder compartido. No
carga TensorFlow ni construye ningún modelo de Keras; ver `EncoderService`.
"""

import logging
import socket
import time

import flwr as fl
from flwr.common import ndarrays_to_parameters

from app.config import FLConfig
from app.services.encoder_service import EncoderService
from app.services.federation_service import FederationService
from app.strategy import EarlyStopReached, FedPerStrategy

logger = logging.getLogger(__name__)


def build_strategy(
    encoder_service: EncoderService,
    federation_service: FederationService,
    fl_config: FLConfig,
) -> fl.server.strategy.Strategy:
    """
    Construye la estrategia con el encoder actual como parámetros iniciales.

    Los pesos iniciales NO son aleatorios: salen de `initial_encoder.npz`, que
    trae el pre-entrenamiento de la simulación horneado por la Fase 1. Si hay
    un `encoder_latest.npz` de una federación anterior, se retoma desde ahí.
    """
    return FedPerStrategy(
        encoder_service=encoder_service,
        federation_service=federation_service,
        fl_config=fl_config,
        fraction_fit=fl_config.fraction_fit,
        fraction_evaluate=fl_config.fraction_evaluate,
        min_fit_clients=fl_config.min_fit_clients,
        min_evaluate_clients=fl_config.min_evaluate_clients,
        min_available_clients=fl_config.min_available_clients,
        initial_parameters=ndarrays_to_parameters(encoder_service.get_parameters()),
    )


def wait_for_port_free(port: int, timeout: float = 60.0) -> bool:
    """
    Espera a que el puerto se pueda volver a abrir antes de reiniciar.

    El `time.sleep(5)` fijo no era una espera suficiente: con la sesión
    terminando por excepción, el servidor gRPC anterior nunca se cerraba y el
    socket no se liberaba nunca, así que el reinicio moría con WSA 10048 y
    dejaba el gRPC caído aunque la API REST siguiera respondiendo. Ahora la
    causa está corregida en `strategy.py`, pero sondear el puerto de verdad
    cuesta nada y cubre el cierre lento de conexiones en TIME_WAIT.

    No se usa SO_REUSEADDR a propósito: en Windows permite BIND sobre un
    socket que otro proceso tiene abierto, que es justo el fallo silencioso
    que este sondeo pretende detectar.
    """
    fin = time.monotonic() + timeout
    while time.monotonic() < fin:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                time.sleep(1.0)
    return False


def start_fl_server(
    encoder_service: EncoderService,
    federation_service: FederationService,
    fl_config: FLConfig,
):
    """
    Arranca Flower y lo reinicia al terminar, para aceptar nuevas sesiones.

    El reinicio importa en las pruebas reales: los dispositivos no están
    disponibles a la vez ni de forma continua, así que la federación se hace
    a ratos y el servidor tiene que seguir escuchando entre tandas.
    """
    import signal
    import threading

    # WORKAROUND: Flower registra handlers de salida elegante y `signal.signal`
    # sólo funciona en el hilo principal. Aquí corremos en un hilo secundario
    # porque Flask ocupa el principal.
    _original_signal = signal.signal
    if threading.current_thread() != threading.main_thread():
        signal.signal = lambda *args, **kwargs: None

    try:
        while True:
            try:
                strategy = build_strategy(
                    encoder_service, federation_service, fl_config
                )
                logger.info(
                    "Servidor Flower gRPC listo en el puerto %d. Esperando clientes "
                    "para %d rondas (encoder de %d floats, timeout %.0fs/ronda)...",
                    fl_config.grpc_port,
                    fl_config.num_rounds,
                    encoder_service.encoder_flat_size,
                    fl_config.round_timeout,
                )
                fl.server.start_server(
                    server_address=f"0.0.0.0:{fl_config.grpc_port}",
                    config=fl.server.ServerConfig(
                        num_rounds=fl_config.num_rounds,
                        # Sin timeout, un cliente que se va con la pantalla
                        # apagada bloquea la ronda indefinidamente.
                        round_timeout=fl_config.round_timeout,
                    ),
                    strategy=strategy,
                )
                if getattr(strategy, "stopped", False):
                    logger.info(
                        "Sesión FL cerrada por parada temprana en la ronda %s "
                        "(mejor val_auc(ema)=%.4f en la ronda %s). Los clientes "
                        "han recibido RECONNECT_INS; su medición final sobre el "
                        "test ciego sale en el logcat del dispositivo.",
                        strategy.stopped_at_round, strategy.best_auc,
                        strategy.best_round,
                    )
                else:
                    logger.info("Sesión FL completada (%d rondas).",
                                fl_config.num_rounds)

            except EarlyStopReached as e:
                # No es un error: es el criterio de parada haciendo su trabajo.
                logger.info("%s", e)
                logger.info(
                    "Encoder de la mejor ronda en artifacts/checkpoints/"
                    "encoder_best.npz. Reiniciando para una nueva sesión en 5 s..."
                )

            except RuntimeError as e:
                # Reintentar un puerto ocupado no sirve de nada: lo tiene otro
                # proceso y no lo va a soltar. Antes el bucle escupía un
                # traceback cada 5 s indefinidamente y enterraba el motivo real.
                if "Failed to bind" in str(e):
                    logger.error(
                        "\n"
                        "  El puerto %d ya está ocupado por otro proceso.\n"
                        "\n"
                        "  Casi siempre es un servidor anterior que sigue vivo.\n"
                        "  Para localizarlo y cerrarlo, en PowerShell:\n"
                        "\n"
                        "      Get-NetTCPConnection -State Listen -LocalPort %d |\n"
                        "        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }\n"
                        "\n"
                        "  El servidor gRPC NO se ha iniciado. La API REST sí sigue\n"
                        "  respondiendo, pero ningún teléfono podrá entrenar.\n",
                        fl_config.grpc_port, fl_config.grpc_port,
                    )
                    return
                logger.exception("Error en la sesión FL: %s. Reiniciando en 5 s...", e)

            except Exception as e:
                logger.exception("Error en la sesión FL: %s. Reiniciando en 5 s...", e)

            if not wait_for_port_free(fl_config.grpc_port, timeout=60.0):
                logger.error(
                    "El puerto %d sigue ocupado 60 s después de cerrar la "
                    "sesión. No se reinicia el servidor gRPC.",
                    fl_config.grpc_port,
                )
                return
    finally:
        signal.signal = _original_signal
