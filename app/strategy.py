from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from app.services.model_service import ModelService
from app.services.federation_service import FederationService


class FedAvgCustom(fl.server.strategy.FedAvg):
    """
    Estrategia personalizada de FedAvg.

    Extiende FedAvg estándar para inyectar los pesos agregados
    en nuestro ModelService global y registrar métricas en FederationService.
    """

    def __init__(
        self,
        model_service: ModelService,
        federation_service: FederationService,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._model_service = model_service
        self._federation_service = federation_service

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Agrega los pesos de los clientes y los guarda en el modelo global.
        """
        # Llamar a FedAvg estándar para hacer la agregación
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convertir a numpy ndarrays
            ndarrays: NDArrays = parameters_to_ndarrays(aggregated_parameters)

            # Actualizar el modelo en memoria
            self._model_service.set_weights(ndarrays)

            # Extraer métricas adicionales si es necesario (y agregarlas)
            total_samples = sum([fit_res.num_examples for _, fit_res in results])
            n_clients = len(results)

            self._federation_service.record_round(
                round_number=server_round,
                n_clients=n_clients,
                total_samples=total_samples,
                aggregated_metrics=aggregated_metrics,
            )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Agrega las evaluaciones de los clientes.
        Calcula EER ponderado si los clientes lo reportan en res.metrics.
        """
        # Llamar a FedAvg estándar para la agregación del loss (loss es el primer float)
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        if not results:
            return loss_aggregated, metrics_aggregated

        # Calcular EER ponderado si existe la clave "eer"
        total_samples = sum([eval_res.num_examples for _, eval_res in results])
        eer_values = [
            (eval_res.num_examples, eval_res.metrics.get("eer", None))
            for _, eval_res in results
        ]

        # Solo ponderamos si al menos uno reportó "eer" de forma válida
        valid_eers = [(n, eer) for n, eer in eer_values if eer is not None]

        if valid_eers and total_samples > 0:
            weighted_eer = sum(n * eer for n, eer in valid_eers) / sum(n for n, _ in valid_eers)
            self._federation_service.record_evaluation(
                round_number=server_round,
                weighted_eer=weighted_eer,
                n_clients=len(results),
            )
            metrics_aggregated["eer"] = weighted_eer

        return loss_aggregated, metrics_aggregated
