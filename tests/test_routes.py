import pytest
import io
import numpy as np

def test_get_model_info(client):
    """Test model info endpoint returns valid JSON metadata"""
    response = client.get('/api/model/info')
    assert response.status_code == 200
    
    data = response.get_json()
    assert "model_name" in data
    assert "sensor_config" in data
    assert "total_parameters" in data
    assert "input_shapes" in data
    
    assert data["sensor_config"] in ["gyro", "gyro_acc", "gyro_acc_touch"]
    assert isinstance(data["total_parameters"], int)
    assert data["total_parameters"] > 0


