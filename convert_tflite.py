import os
import tensorflow as tf
from app.models.auth_model import AuthModelBuilder
from app.config import ModelConfig

class TFLiteODTModel(tf.Module):
    def __init__(self, keras_model):
        super().__init__()
        self.model = keras_model
        # Rastrear explícitamente todas las variables (incluyendo seed_generators de Dropout)
        self.all_variables = keras_model.variables

    @tf.function(input_signature=[
        tf.TensorSpec([None, 128, 6], tf.float32),
        tf.TensorSpec([None, 1], tf.float32)
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            prediction = self.model(x, training=True)
            loss = tf.keras.losses.BinaryCrossentropy()(y, prediction)
        gradients = tape.gradient(loss, self.model.trainable_weights)
        tf.group([w.assign_sub(0.001 * g) for w, g in zip(self.model.trainable_weights, gradients)])
        return {"loss": loss}

    @tf.function(input_signature=[tf.TensorSpec([None, 128, 6], tf.float32)])
    def infer(self, x):
        return {"output": self.model(x, training=False)}

    @tf.function
    def restore(self, **kwargs):
        for i, w in enumerate(self.model.weights):
            w.assign(kwargs[f"var_{i:04d}"])
        return {}

    @tf.function(input_signature=[])
    def save(self):
        return {f"var_{i:04d}": w for i, w in enumerate(self.model.weights)}

    @tf.function(input_signature=[tf.TensorSpec([1], tf.float32)])
    def initialize(self, dummy):
        _ = self.model(tf.zeros([1, 128, 6]), training=False)
        return {"status": tf.constant([[1]])}


def convert_model():
    print("Iniciando conversión de modelo TFLite con LSTM congelado...")
    
    model_path = os.path.join(os.path.dirname(__file__), "app", "models", "startModel.keras")
    if os.path.exists(model_path):
        print(f"Cargando modelo existente: {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Modelo existente no encontrado, construyendo uno nuevo...")
        config = ModelConfig()
        builder = AuthModelBuilder(config)
        model = builder.build()
    
    print("Congelando extractor de características (Conv1D y LSTM)...")
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Dense):
            layer.trainable = False
            
    print(f"Total de tensores de pesos (state): {len(model.weights)}")
    print(f"Tensores entrenables (actualizados on-device): {len(model.trainable_weights)}")
    
    module = TFLiteODTModel(model)
    
    # Warm-up
    print("Realizando warm-up...")
    dummy_x = tf.zeros([1, 128, 6], dtype=tf.float32)
    dummy_y = tf.zeros([1, 1], dtype=tf.float32)
    module.train(dummy_x, dummy_y)
    module.infer(dummy_x)
    
    tensor_specs = {}
    for i, w in enumerate(model.weights):
        tensor_specs[f"var_{i:04d}"] = tf.TensorSpec(w.shape, w.dtype)

    signatures = {
        'train': module.train.get_concrete_function(),
        'infer': module.infer.get_concrete_function(),
        'save': module.save.get_concrete_function(),
        'restore': module.restore.get_concrete_function(**tensor_specs),
        'initialize': module.initialize.get_concrete_function(),
    }

    saved_model_dir = "saved_model_temp"
    tf.saved_model.save(module, saved_model_dir, signatures=signatures)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.experimental_enable_resource_variables = True
    
    print("Convirtiendo modelo a formato TensorFlow Lite...")
    tflite_model = converter.convert()
    
    out_path = os.path.join(os.path.dirname(__file__), "startModel.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Modelo guardado exitosamente en: {out_path}")

    model.save(model_path)
    print(f"Modelo Keras sincronizado en: {model_path}")

if __name__ == "__main__":
    convert_model()
