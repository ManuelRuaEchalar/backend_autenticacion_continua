import logging
import flwr as fl
from flwr.common import ndarrays_to_parameters


from app.config import FLConfig
from app.services.model_service import ModelService
from app.services.federation_service import FederationService
from app.strategy import FedAvgCustom

logger = logging.getLogger(__name__)


def build_strategy(
    model_service: ModelService,
    federation_service: FederationService,
    fl_config: FLConfig,
) -> fl.server.strategy.Strategy:
    """
    Construye la estrategia personalizada de Flower.
    Obtiene los pesos iniciales desde Keras (model_service) para que
    el servidor envíe la primera versión del modelo a los clientes.
    """
    # 1. Obtener pesos del modelo global (recién inicializado con pesos aleatorios)
    initial_weights = model_service.get_parameters()
    initial_parameters = ndarrays_to_parameters(initial_weights)

    # 2. Configurar la estrategia
    strategy = FedAvgCustom(
        model_service=model_service,
        federation_service=federation_service,
        fraction_fit=fl_config.fraction_fit,
        fraction_evaluate=fl_config.fraction_evaluate,
        min_fit_clients=fl_config.min_fit_clients,
        min_evaluate_clients=fl_config.min_evaluate_clients,
        min_available_clients=fl_config.min_available_clients,
        initial_parameters=initial_parameters,
    )
    return strategy


def start_fl_server(
    model_service: ModelService,
    federation_service: FederationService,
    fl_config: FLConfig,
):
    """
    Inicia el servidor gRPC de Flower.
    Esta llamada es bloqueante, se recomienda ejecutar en un hilo separado.
    """
    logger.info(f"Iniciando servidor Flower gRPC en puerto {fl_config.grpc_port}...")
    strategy = build_strategy(model_service, federation_service, fl_config)

    # WORKAROUND: Prevenir "ValueError: signal only works in main thread"
    # parcheando el módulo signal porque Flower intenta registrar handlers 
    # de salida elegante estando dentro de un thread secundario.
    import signal
    import threading
    
    _original_signal = signal.signal
    if threading.current_thread() != threading.main_thread():
        signal.signal = lambda *args, **kwargs: None

    try:
        fl.server.start_server(
            server_address=f"0.0.0.0:{fl_config.grpc_port}",
            config=fl.server.ServerConfig(num_rounds=fl_config.num_rounds),
            strategy=strategy,
        )
    finally:
        # Restaurar si es necesario
        signal.signal = _original_signal

    logger.info("Servidor Flower detenido.")
