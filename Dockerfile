FROM python:3.12-slim

# Evitar que Python escriba archivos .pyc y forzar stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Deshabilitar CUDA y suprimir logs de C++ de TensorFlow ya que el servidor FL solo usa CPU
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_CPP_MIN_LOG_LEVEL="2"

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar paquetes si es necesario
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para aprovechar la caché de capas de Docker
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copiar el resto del código
COPY . .

# Exponer puerto 5000
EXPOSE 5000

# Ejecutar con Gunicorn para producción
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app.factory:create_app()"]
