import pytest
from app.config import ModelConfig
from app.models.auth_model import AuthModelBuilder

def test_model_builder_gyro_acc():
    """Test model compilation with gyro_acc configuration"""
    config = ModelConfig(sensor_config="gyro_acc")
    builder = AuthModelBuilder(config)
    model = builder.build()
    
    assert model.name == "AuthDeepConvLSTM_gyro_acc"
    # gyro_acc expects window_size (128) and 6 channels
    assert model.input_shape == (None, 128, 6)
    # Output should be a single probability
    assert model.output_shape == (None, 1)

def test_model_builder_gyro_acc_touch():
    """Test model compilation with multimodal configuration"""
    config = ModelConfig(sensor_config="gyro_acc_touch")
    builder = AuthModelBuilder(config)
    model = builder.build()
    
    assert model.name == "AuthDeepConvLSTM_gyro_acc_touch"
    # Expects IMU input and Touch input
    assert len(model.inputs) == 2
    assert model.inputs[0].shape == (None, 128, 6) # IMU
    assert model.inputs[1].shape == (None, 12)     # Touch
    
def test_invalid_sensor_config():
    """Test builder raises error with invalid configuration"""
    config = ModelConfig(sensor_config="invalid_sensor")
    with pytest.raises(ValueError, match="sensor_config debe ser uno de"):
        AuthModelBuilder(config)
