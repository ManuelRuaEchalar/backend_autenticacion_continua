"""
Estrategia FedAvg para FedPer — Fase 3.

Sustituye a la versión anterior, que agregaba los 23 tensores del
DeepConvLSTM. Ahora circula UN ÚNICO tensor: el vector plano del encoder
compartido (`encoder_flat_size` floats). La cabeza personal nunca sale del
dispositivo, que es justamente lo que distingue FedPer de FedAvg a secas.

Réplica de `FedAvgCapture` de `mejor.py:768`, con dos diferencias
deliberadas:

  · El early stopping mira `val_auc`, calculado por los clientes sobre su
    partición de VALIDACIÓN. En el cuadernillo miraba el AUC sobre `x_test`,
    que además era el conjunto con el que se reportaba el resultado final:
    el criterio de parada veía el test. Aquí el test permanece ciego durante
    toda la federación.
  · Los checkpoints se guardan en disco local. No hay S3.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from app.config import FLConfig
from app.services.encoder_service import EncoderService
from app.services.federation_service import FederationService

logger = logging.getLogger(__name__)


class EarlyStopReached(Exception):
    """La federación alcanzó el criterio de parada."""


def _hora() -> str:
    return datetime.now().strftime("%H:%M:%S")


#: Nombre del metric con el que el cliente se identifica en cada `fit`.
#: Debe coincidir con FlowerGrpcClient.handleFit (app Android).
METRIC_CLIENT_ID = "client_id"


def _identidad(res: FitRes) -> Optional[str]:
    """
    Identidad estable del DISPOSITIVO que envió este resultado.

    Flower identifica clientes por CONEXIÓN (`proxy.cid`), no por aparato, así
    que dos sesiones abiertas desde el mismo teléfono le parecen dos clientes
    legítimos. El `client_id` es un UUID que la app persiste en preferencias
    (ClientIdentity.kt), y es lo único que permite distinguir "tres usuarios"
    de "dos usuarios y uno contado dos veces".

    Devuelve `None` si el cliente no lo manda: un APK anterior a la v1.4 no
    conoce este metric.
    """
    valor = res.metrics.get(METRIC_CLIENT_ID)
    return valor if isinstance(valor, str) and valor else None


def _media_ponderada(pares: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """
    Media de cada métrica ponderada por número de muestras.

    Se ignoran los clientes con `num_examples <= 0`: un dispositivo que no
    llegó al mínimo de ventanas reporta 0 y no debe arrastrar la media hacia
    ningún lado.
    """
    if not pares:
        return {}
    salida: Dict[str, Scalar] = {}
    claves = {k for _, m in pares for k, v in m.items() if isinstance(v, (int, float))}
    for clave in claves:
        num = 0.0
        den = 0
        for n, metricas in pares:
            if n > 0 and clave in metricas and isinstance(metricas[clave], (int, float)):
                num += n * float(metricas[clave])
                den += n
        if den:
            salida[clave] = num / den
    return salida


class FedPerStrategy(fl.server.strategy.FedAvg):
    """FedAvg sobre el vector del encoder, con parada temprana por `val_auc`."""

    def __init__(
        self,
        encoder_service: EncoderService,
        federation_service: FederationService,
        fl_config: FLConfig,
        *args,
        **kwargs,
    ) -> None:
        kwargs.setdefault("on_fit_config_fn", self._fit_config)
        kwargs.setdefault("on_evaluate_config_fn", self._evaluate_config)
        kwargs.setdefault("fit_metrics_aggregation_fn", _media_ponderada)
        kwargs.setdefault("evaluate_metrics_aggregation_fn", _media_ponderada)
        super().__init__(*args, **kwargs)

        self._encoders = encoder_service
        self._federation = federation_service
        self._cfg = fl_config

        self.best_auc: float = -1.0
        self.best_round: Optional[int] = None
        self.auc_ema: Optional[float] = None
        self.rounds_no_improve: int = 0
        # La parada temprana NO se señaliza con una excepción, aunque sea lo
        # que pide el cuerpo. `flwr.server.run_fl` sólo llama a
        # `disconnect_all_clients()` —y con ello sólo envía RECONNECT_INS— si
        # `Server.fit()` RETORNA. Una excepción que sale de `aggregate_evaluate`
        # se salta esa línea, y entonces:
        #   · el cliente se queda colgado en `responseFlow.collect` para
        #     siempre, y nunca ejecuta `evaluateHeldOutTest()`, que es la
        #     medición sobre el conjunto ciego. El número de la tesis no se
        #     llegaba a calcular NUNCA por el camino normal de terminación.
        #   · el servidor gRPC no se cierra, el socket del 8080 queda tomado y
        #     el reinicio automático fallaba siempre con WSA 10048.
        # Con una bandera, las rondas restantes se saltan en microsegundos
        # (`fit_round` cancela si `configure_fit` devuelve lista vacía) y
        # Flower cierra la sesión por su camino de siempre.
        self.stopped: bool = False
        self.stopped_at_round: Optional[int] = None
        self.history: Dict[str, list] = {
            "round": [], "train_loss": [], "val_auc": [], "val_eer": [], "n_clients": [],
        }

    # ------------------------------------------------------------------
    # Configuración que viaja a los clientes
    # ------------------------------------------------------------------

    def _fit_config(self, server_round: int) -> Dict[str, Scalar]:
        """
        Mismo calendario de learning rate que `fit_config` de mejor.py:843.

        `FlowerGrpcClient.handleFit` lee `local_epochs` como sint64 y `lr` como
        double; los tipos importan porque el cliente cae a los valores del
        manifiesto si la clave no tiene el tipo esperado.
        """
        ronda_de_decaimiento = self._cfg.num_rounds * self._cfg.lr_decay_at
        lr = (
            self._cfg.learning_rate
            if server_round <= ronda_de_decaimiento
            else self._cfg.learning_rate * self._cfg.lr_decay_factor
        )
        return {
            "local_epochs": int(self._cfg.local_epochs),
            "lr": float(lr),
            "server_round": int(server_round),
        }

    def _evaluate_config(self, server_round: int) -> Dict[str, Scalar]:
        return {"server_round": int(server_round)}

    # ------------------------------------------------------------------
    # Ronda de entrenamiento
    # ------------------------------------------------------------------

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        if self.stopped:
            # Lista vacía -> `fit_round` cancela la ronda y sigue. Las rondas
            # que quedan hasta `num_rounds` se consumen al instante y sin
            # tocar a los clientes.
            return []
        logger.info("[%s] === RONDA %d — esperando clientes ===", _hora(), server_round)
        instrucciones = super().configure_fit(server_round, parameters, client_manager)
        if instrucciones:
            cfg = instrucciones[0][1].config
            logger.info(
                "[%s] %d cliente(s) seleccionados. Enviando encoder de %d floats "
                "(local_epochs=%s, lr=%s)",
                _hora(), len(instrucciones), self._encoders.encoder_flat_size,
                cfg.get("local_epochs"), cfg.get("lr"),
            )
        else:
            logger.warning("[%s] Ningún cliente seleccionado en la ronda %d",
                           _hora(), server_round)
        return instrucciones

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            logger.error("[%s] %d cliente(s) fallaron durante el fit de la ronda %d",
                         _hora(), len(failures), server_round)
            for f in failures:
                if isinstance(f, BaseException):
                    logger.error("    %s: %s", type(f).__name__, f)

        if not results:
            logger.warning("[%s] Ronda %d sin resultados de fit", _hora(), server_round)
            return None, {}

        esperado = self._encoders.encoder_flat_size
        utiles: List[Tuple[ClientProxy, FitRes]] = []
        # client_id -> cid de la conexión que lo reclamó primero.
        vistos: Dict[str, str] = {}
        sin_identidad = 0
        for proxy, res in results:
            # — Identidad del dispositivo, ANTES que nada —
            # Un duplicado es una violación del protocolo, no un peso malo: se
            # descarta pase lo que pase con sus tensores.
            identidad = _identidad(res)
            if identidad is None:
                sin_identidad += 1
            elif identidad in vistos:
                logger.error(
                    "[%s] DISPOSITIVO DUPLICADO en la ronda %d: la conexión %s "
                    "reporta el mismo client_id (%s) que %s. Se descarta la "
                    "segunda. Alguien ha pulsado INICIAR FL dos veces en el "
                    "mismo teléfono; sin este corte pesaría DOBLE en FedAvg.",
                    _hora(), server_round, proxy.cid, identidad[:8],
                    vistos[identidad],
                )
                continue
            else:
                vistos[identidad] = proxy.cid

            arrays = parameters_to_ndarrays(res.parameters)
            if len(arrays) != 1:
                logger.error(
                    "[%s] Cliente %s envió %d tensores; FedPer espera exactamente 1 "
                    "(el vector plano del encoder). Se descarta.",
                    _hora(), proxy.cid, len(arrays),
                )
                continue
            if arrays[0].size != esperado:
                logger.error(
                    "[%s] Cliente %s envió %d floats, se esperaban %d. ¿APK y "
                    "servidor con artefactos distintos? Se descarta.",
                    _hora(), proxy.cid, arrays[0].size, esperado,
                )
                continue
            if not np.all(np.isfinite(arrays[0])):
                # Un solo NaN contamina el encoder global de todos los usuarios,
                # y a partir de ahí la federación no se recupera.
                logger.error(
                    "[%s] Cliente %s envió valores no finitos. Se descarta.",
                    _hora(), proxy.cid,
                )
                continue
            if res.num_examples <= 0:
                logger.warning(
                    "[%s] Cliente %s reportó 0 ejemplos. Se descarta (pesaría 0 "
                    "en FedAvg y sólo añade ruido a las métricas).",
                    _hora(), proxy.cid,
                )
                continue
            # Un encoder que vuelve BIT A BIT idéntico al que se envió no es
            # una ronda que aprendió poco: es una ronda que no entrenó. Pasó
            # durante 18 rondas seguidas —el cliente recibía `local_epochs=0`
            # por un desajuste de campos en el proto— sin un solo error, y
            # FedAvg se dedicó a promediar copias del mismo vector mientras el
            # early stopping elegía "la mejor" de 18 réplicas idénticas.
            # Aquí se corta en la ronda 1.
            delta = float(np.abs(arrays[0] - self._encoders.encoder).max())
            if delta == 0.0:
                logger.error(
                    "[%s] Cliente %s devolvió el encoder SIN CAMBIOS "
                    "(max|dw|=0). Esa ronda no entrenó: revisa `local_epochs` "
                    "y `lr` en el logcat del dispositivo. Se descarta.",
                    _hora(), proxy.cid,
                )
                continue
            logger.debug(
                "[%s] Cliente %s: max|dw| del encoder = %.3e",
                _hora(), proxy.cid, delta,
            )
            utiles.append((proxy, res))

        if sin_identidad:
            # No se descarta a nadie por esto: rompería la compatibilidad con
            # un APK viejo. Pero tiene que constar, porque mientras haya un
            # cliente sin identidad la comprobación de duplicados NO cubre la
            # ronda, y el silencio es justo lo que hizo caro este fallo.
            logger.warning(
                "[%s] Ronda %d: %d cliente(s) no enviaron client_id (APK "
                "anterior a la v1.4). No se puede garantizar que sean "
                "dispositivos distintos.",
                _hora(), server_round, sin_identidad,
            )

        if not utiles:
            logger.error("[%s] Ronda %d: ningún cliente aportó pesos válidos",
                         _hora(), server_round)
            return None, {}

        # Censo de la ronda. Con participantes remotos y sin telemetría en el
        # móvil, este log es lo único que dice QUIÉN entrenó de verdad.
        if vistos:
            logger.info(
                "[%s] Ronda %d: %d dispositivo(s) distintos [%s]",
                _hora(), server_round, len(vistos),
                ", ".join(sorted(i[:8] for i in vistos)),
            )

        if len(utiles) < self.min_fit_clients:
            # Se agrega igualmente —tirar la ronda entera perdería el trabajo
            # de los clientes sanos— pero el número resultante ya no es el de
            # la federación que se pidió, y eso no puede pasar inadvertido.
            logger.error(
                "[%s] Ronda %d agregada con %d cliente(s), por debajo de los "
                "%d exigidos. La media NO representa la federación completa.",
                _hora(), server_round, len(utiles), self.min_fit_clients,
            )

        parametros, metricas = super().aggregate_fit(server_round, utiles, [])

        if parametros is not None:
            agregado = parameters_to_ndarrays(parametros)[0]
            self._encoders.set_encoder(agregado, server_round)
            ruta = self._encoders.save_checkpoint(server_round)

            total = sum(r.num_examples for _, r in utiles)
            train_loss = metricas.get("train_loss")
            self.history["round"].append(server_round)
            self.history["train_loss"].append(train_loss)
            self.history["n_clients"].append(len(utiles))

            logger.info(
                "[%s] [FIT]  Ronda %d: train_loss=%s  clientes=%d  ventanas=%d  -> %s",
                _hora(), server_round,
                f"{train_loss:.4f}" if isinstance(train_loss, float) else "n/d",
                len(utiles), total, ruta.name,
            )

            self._federation.record_round(
                round_number=server_round,
                n_clients=len(utiles),
                total_samples=total,
                aggregated_metrics=dict(metricas),
            )

        return parametros, metricas

    # ------------------------------------------------------------------
    # Ronda de evaluación y parada temprana
    # ------------------------------------------------------------------

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Contrapartida de `configure_fit` para la parada temprana."""
        if self.stopped:
            return []
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if failures:
            logger.error("[%s] %d cliente(s) fallaron al evaluar la ronda %d",
                         _hora(), len(failures), server_round)

        utiles = [(p, r) for p, r in results if r.num_examples > 0]
        if not utiles:
            logger.warning("[%s] Ronda %d sin evaluaciones utilizables",
                           _hora(), server_round)
            return None, {}

        loss, metricas = super().aggregate_evaluate(server_round, utiles, [])

        # `val_` a propósito: deja constancia en los logs de que la parada
        # temprana NUNCA mira el conjunto de test, que sigue ciego.
        auc = metricas.get("val_auc")
        eer = metricas.get("val_eer")
        self.history["val_auc"].append(auc)
        self.history["val_eer"].append(eer)

        if isinstance(eer, float):
            self._federation.record_evaluation(
                round_number=server_round, weighted_eer=eer, n_clients=len(utiles),
            )

        if not isinstance(auc, float):
            logger.warning(
                "[%s] Ronda %d: los clientes no reportaron val_auc; no se puede "
                "decidir la parada temprana.", _hora(), server_round,
            )
            return loss, metricas

        # EMA para que una ronda afortunada no fije el best_round. Mismo
        # alpha que el cuadernillo.
        self.auc_ema = (
            auc if self.auc_ema is None
            else self._cfg.eval_ema_alpha * auc
            + (1 - self._cfg.eval_ema_alpha) * self.auc_ema
        )

        if self.auc_ema > self.best_auc + 1e-4:
            self.best_auc = self.auc_ema
            self.best_round = server_round
            self.rounds_no_improve = 0
            self._encoders.promote_best(server_round, self.auc_ema)
        else:
            self.rounds_no_improve += 1

        en_warmup = server_round <= self._cfg.warmup_rounds
        logger.info(
            "[%s] [EVAL] Ronda %d: loss=%s val_auc=%.4f ema=%.4f "
            "best=%.4f (ronda %s) sin_mejora=%d/%d%s  clientes=%d",
            _hora(), server_round,
            f"{loss:.4f}" if isinstance(loss, float) else "n/d",
            auc, self.auc_ema, self.best_auc, self.best_round,
            self.rounds_no_improve, self._cfg.early_stop_patience,
            "  (warmup)" if en_warmup else "", len(utiles),
        )

        if not en_warmup and self.rounds_no_improve >= self._cfg.early_stop_patience:
            self.stopped = True
            self.stopped_at_round = server_round
            logger.info(
                "[%s] EARLY STOPPING en la ronda %d. Mejor val_auc(ema)=%.4f en la "
                "ronda %s. El encoder de esa ronda está en checkpoints/encoder_best.npz",
                _hora(), server_round, self.best_auc, self.best_round,
            )
            logger.info(
                "[%s] Saltando las rondas restantes para que Flower cierre la "
                "sesión por su camino normal: los clientes recibirán "
                "RECONNECT_INS y ejecutarán su medición final sobre el test "
                "ciego.", _hora(),
            )

        return loss, metricas
