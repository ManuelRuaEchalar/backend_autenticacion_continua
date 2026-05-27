import tensorflow as tf
from app.models.auth_model import AuthModelBuilder
from app.config import ModelConfig

print("Building model...")
config = ModelConfig()
builder = AuthModelBuilder(config)
model = builder.build()

print(f"Model built with {len(model.weights)} weights, {len(model.trainable_weights)} trainable.")

# Check the shapes
for w in model.weights:
    print(w.name, w.shape)
