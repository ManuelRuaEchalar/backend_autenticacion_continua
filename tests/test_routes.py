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

def test_get_model_parameters(client):
    """Test model parameters endpoint returns valid NPZ file"""
    response = client.get('/api/model/parameters')
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'application/octet-stream'
    assert 'attachment; filename=' in response.headers['Content-Disposition']
    
    # Load the NPZ data from the response to verify it's a valid numpy archive
    with io.BytesIO(response.data) as f:
        npz_file = np.load(f)
        # Should contain at least one array of weights
        assert len(npz_file.files) > 0
        # The first array should have weights
        first_key = npz_file.files[0]
        assert isinstance(npz_file[first_key], np.ndarray)
