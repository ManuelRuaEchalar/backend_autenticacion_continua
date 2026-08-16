"""
Contrato de /api/model/info.

Este test llevaba rojo desde la refactorizacion de Fase 3: esperaba un campo
`model_name` y `sensor_config` en ["gyro","gyro_acc","gyro_acc_touch"], que era
el contrato del DeepConvLSTM anterior. El endpoint pasó a servir el vector
plano del encoder FedPer y devuelve "acc_gyro". Era el test el que estaba
obsoleto, no la ruta — y precisamente por eso hay que arreglarlo: un test rojo
permanente deja de avisar de nada.
"""


def test_get_model_info(client):
    """El contrato que `ModelInfoFetcher.requireCompatibleWith` verifica."""
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.get_json()

    # Los tres campos que la app compara con su manifiesto empaquetado. Si
    # alguno falta, el cliente aborta la ronda antes de abrir el gRPC.
    for campo in ("sensor_config", "window_size", "encoder_flat_size"):
        assert campo in data, f"falta '{campo}', la app abortaria la ronda"

    assert data["sensor_config"] == "acc_gyro"
    assert isinstance(data["window_size"], int) and data["window_size"] > 0
    assert isinstance(data["encoder_flat_size"], int)
    assert data["encoder_flat_size"] > 0

    # El orden de canales importa tanto como el número: acelerometro primero.
    # Intercambiarlos por parejas no hace fallar nada, sólo devuelve basura.
    assert data["channel_order"] == [
        "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"
    ]
    assert data["n_features"] == len(data["channel_order"])


def test_info_publica_el_modo_de_ablacion(client):
    """
    La app lee `ablation` de aquí ANTES de construir su partición, porque el
    filtro de actividad actúa durante el ventaneo. Si el campo desapareciera,
    el cliente caería a "full" por defecto y las dos condiciones del
    experimento dejarían de distinguirse en silencio.
    """
    data = client.get("/api/model/info").get_json()
    assert "ablation" in data
    assert data["ablation"] in ("full", "baseline")


def test_health(client):
    assert client.get("/health").status_code == 200
