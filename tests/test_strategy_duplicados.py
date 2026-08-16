"""
Pendiente J: el servidor tiene que distinguir DISPOSITIVOS, no conexiones.

El 2026-08-15, en campo, un participante pulsó INICIAR FL dos veces. La app no
tenía guarda de reentrada, así que abrió una segunda sesión Flower completa en
el mismo proceso. El servidor la contó como un tercer cliente y registró
`clientes=3, 0 failures` durante dos rondas cuando en realidad eran DOS
teléfonos, uno con peso DOBLE en la media de FedAvg.

Sólo se descubrió mirando el `netstat` del servidor. Estos tests fijan el
comportamiento para que no dependa de que alguien mire.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
from flwr.common import Code, FitRes, Scalar, Status, ndarrays_to_parameters

from app.config import FLConfig
from app.strategy import FedPerStrategy

FLAT_SIZE = 8


class _EncoderServiceFake:
    def __init__(self) -> None:
        self.encoder_flat_size = FLAT_SIZE
        self._encoder = np.zeros(FLAT_SIZE, dtype=np.float32)
        self.checkpoints: List[int] = []

    @property
    def encoder(self) -> np.ndarray:
        return self._encoder

    def set_encoder(self, flat: np.ndarray, server_round: int) -> None:
        self._encoder = np.asarray(flat, dtype=np.float32)

    def save_checkpoint(self, server_round: int, tag: Optional[str] = None) -> Path:
        self.checkpoints.append(server_round)
        return Path(f"encoder_round_{server_round:03d}.npz")

    def get_parameters(self) -> List[np.ndarray]:
        return [self._encoder]


class _FederationServiceFake:
    def __init__(self) -> None:
        self.rondas: List[Dict] = []

    def record_round(self, round_number, n_clients, total_samples,
                     aggregated_metrics) -> None:
        self.rondas.append(
            {"round": round_number, "n_clients": n_clients,
             "total_samples": total_samples}
        )


class _ProxyFake:
    """Sustituye a ClientProxy: `aggregate_fit` sólo le mira el `cid`."""

    def __init__(self, cid: str) -> None:
        self.cid = cid


def _fit_res(valor: float, client_id: Optional[str], n: int = 100) -> FitRes:
    """Un FitRes válido, con el encoder distinto del global (max|dw| != 0)."""
    metrics: Dict[str, Scalar] = {"train_loss": 0.5}
    if client_id is not None:
        metrics["client_id"] = client_id
    return FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(
            [np.full(FLAT_SIZE, valor, dtype=np.float32)]
        ),
        num_examples=n,
        metrics=metrics,
    )


@pytest.fixture
def estrategia():
    encoders = _EncoderServiceFake()
    federacion = _FederationServiceFake()
    strategy = FedPerStrategy(
        encoder_service=encoders,
        federation_service=federacion,
        fl_config=FLConfig(),
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )
    return strategy, encoders, federacion


def _agregar(strategy, resultados: List[Tuple[_ProxyFake, FitRes]]):
    return strategy.aggregate_fit(server_round=1, results=resultados, failures=[])


def test_tres_dispositivos_distintos_se_agregan(estrategia):
    strategy, _, federacion = estrategia
    resultados = [
        (_ProxyFake("conn-a"), _fit_res(1.0, "movil-A")),
        (_ProxyFake("conn-b"), _fit_res(2.0, "movil-B")),
        (_ProxyFake("conn-c"), _fit_res(3.0, "movil-C")),
    ]
    parametros, _ = _agregar(strategy, resultados)

    assert parametros is not None
    assert federacion.rondas[0]["n_clients"] == 3
    assert federacion.rondas[0]["total_samples"] == 300


def test_dos_conexiones_del_mismo_movil_cuentan_una_vez(estrategia):
    """El caso real del 2026-08-15."""
    strategy, _, federacion = estrategia
    resultados = [
        (_ProxyFake("conn-a"), _fit_res(1.0, "movil-A")),
        (_ProxyFake("conn-b"), _fit_res(2.0, "movil-B")),
        # Mismo teléfono que conn-b, otra conexión: es el duplicado.
        (_ProxyFake("conn-b2"), _fit_res(9.0, "movil-B")),
    ]
    parametros, _ = _agregar(strategy, resultados)

    assert parametros is not None
    assert federacion.rondas[0]["n_clients"] == 2, "el duplicado no se descartó"
    assert federacion.rondas[0]["total_samples"] == 200


def test_el_duplicado_no_arrastra_la_media(estrategia):
    """
    Lo que hace caro este fallo no es el conteo, es el peso en FedAvg.

    Sin el corte, el encoder de `movil-B` entraría dos veces y la media se
    desplazaría hacia él. Con dos clientes de 100 ejemplos y valores 1.0 y 2.0
    la media es 1.5; si el duplicado contase, sería (1+2+2)/3 = 1.667.
    """
    strategy, encoders, _ = estrategia
    resultados = [
        (_ProxyFake("conn-a"), _fit_res(1.0, "movil-A")),
        (_ProxyFake("conn-b"), _fit_res(2.0, "movil-B")),
        (_ProxyFake("conn-b2"), _fit_res(2.0, "movil-B")),
    ]
    _agregar(strategy, resultados)

    assert encoders.encoder[0] == pytest.approx(1.5)


def test_se_conserva_la_primera_conexion_no_la_ultima(estrategia):
    strategy, encoders, _ = estrategia
    resultados = [
        (_ProxyFake("conn-a"), _fit_res(1.0, "movil-A")),
        (_ProxyFake("conn-b"), _fit_res(2.0, "movil-B")),
        (_ProxyFake("conn-b2"), _fit_res(50.0, "movil-B")),
    ]
    _agregar(strategy, resultados)

    # Si se hubiera quedado con la última, la media sería (1+50)/2 = 25.5.
    assert encoders.encoder[0] == pytest.approx(1.5)


def test_apk_viejo_sin_client_id_no_se_descarta(estrategia):
    """
    Compatibilidad: un APK anterior a la v1.4 no manda `client_id`.

    Descartarlo dejaría fuera a un participante por no haber actualizado, que
    es peor que no poder comprobar duplicados. Se agrega y se avisa por log.
    """
    strategy, _, federacion = estrategia
    resultados = [
        (_ProxyFake("conn-a"), _fit_res(1.0, None)),
        (_ProxyFake("conn-b"), _fit_res(2.0, None)),
        (_ProxyFake("conn-c"), _fit_res(3.0, None)),
    ]
    parametros, _ = _agregar(strategy, resultados)

    assert parametros is not None
    assert federacion.rondas[0]["n_clients"] == 3


def test_avisa_cuando_falta_client_id(estrategia, caplog):
    strategy, _, _ = estrategia
    resultados = [
        (_ProxyFake("conn-a"), _fit_res(1.0, "movil-A")),
        (_ProxyFake("conn-b"), _fit_res(2.0, None)),
        (_ProxyFake("conn-c"), _fit_res(3.0, "movil-C")),
    ]
    with caplog.at_level("WARNING"):
        _agregar(strategy, resultados)

    assert "no enviaron client_id" in caplog.text


def test_avisa_al_agregar_por_debajo_del_minimo(estrategia, caplog):
    """
    Descartar el duplicado deja la ronda en 2 de 3 exigidos. Se agrega igual
    —tirarla perdería el trabajo de los clientes sanos— pero tiene que constar
    que la media ya no representa la federación pedida.
    """
    strategy, _, _ = estrategia
    resultados = [
        (_ProxyFake("conn-a"), _fit_res(1.0, "movil-A")),
        (_ProxyFake("conn-b"), _fit_res(2.0, "movil-B")),
        (_ProxyFake("conn-b2"), _fit_res(2.0, "movil-B")),
    ]
    with caplog.at_level("ERROR"):
        _agregar(strategy, resultados)

    assert "por debajo de los 3 exigidos" in caplog.text
    assert "DISPOSITIVO DUPLICADO" in caplog.text


def test_el_censo_de_dispositivos_queda_en_el_log(estrategia, caplog):
    """
    Con participantes remotos y sin telemetría en el móvil, este log es lo
    único que dice quién entrenó de verdad en cada ronda.
    """
    strategy, _, _ = estrategia
    resultados = [
        (_ProxyFake("conn-a"), _fit_res(1.0, "aaaaaaaa-1111")),
        (_ProxyFake("conn-b"), _fit_res(2.0, "bbbbbbbb-2222")),
        (_ProxyFake("conn-c"), _fit_res(3.0, "cccccccc-3333")),
    ]
    with caplog.at_level("INFO"):
        _agregar(strategy, resultados)

    assert "3 dispositivo(s) distintos" in caplog.text
    assert "aaaaaaaa" in caplog.text
