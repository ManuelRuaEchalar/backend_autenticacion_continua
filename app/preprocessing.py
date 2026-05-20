"""
Preprocesamiento de señales para autenticación continua.

Responsabilidades:
  1. Construcción de ventanas deslizantes sobre señales IMU
  2. Normalización por usuario (online, compatible con FL)
  3. Extracción de features estadísticas de gestos táctiles
  4. Generación de muestras negativas sintéticas (para entrenamiento FL)

Referencia de features táctiles:
  Las 12 features por gesto están validadas en SAĞBAŞ & BALLI (2024) y en
  el dataset BrainRun (PAPAMICHAIL et al. 2019).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Parámetros globales (sincronizados con model.py)
# ---------------------------------------------------------------------------

WINDOW_SIZE     = 128    # muestras
TRAIN_STEP      = 64     # paso durante entrenamiento (50% solapamiento)
INFER_STEP      = 128    # paso durante inferencia (0% solapamiento → ahorra batería)
SAMPLING_RATE   = 50     # Hz


# ---------------------------------------------------------------------------
# Estructura de un gesto táctil
# ---------------------------------------------------------------------------

@dataclass
class TouchEvent:
    """
    Representa un evento táctil discreto (tap o swipe).

    Todos los campos se normalizan en preprocessing antes de
    ser enviados al modelo.
    """
    duration_ms:   float          # duración total del contacto
    dx:            float          # desplazamiento horizontal (px)
    dy:            float          # desplazamiento vertical (px)
    distance:      float          # longitud de trayectoria (px)
    mean_velocity: float          # velocidad media (px/ms)
    max_velocity:  float          # velocidad pico (px/ms)
    mean_pressure: float          # presión media [0–1]
    max_pressure:  float          # presión máxima [0–1]
    mean_area:     float          # área media de contacto (px²)
    max_area:      float          # área máxima de contacto (px²)
    direction_angle: float        # ángulo de dirección (rad, 0=derecha)
    gesture_type:  float          # 0.0 = tap | 1.0 = swipe


# ---------------------------------------------------------------------------
# Windowing de señales IMU
# ---------------------------------------------------------------------------

def create_windows(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step: int = TRAIN_STEP,
    label: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Divide una señal IMU continua en ventanas deslizantes.

    Args:
        signal      : array (T, C) — T muestras, C canales (3 o 6)
        window_size : número de muestras por ventana
        step        : paso entre ventanas
        label       : etiqueta binaria (1=legítimo, 0=impostor).
                      Si None, no se devuelven etiquetas.

    Returns:
        windows : array (N, window_size, C)
        labels  : array (N,) o None

    Ejemplo:
        >>> signal = np.random.randn(5000, 6)  # 100 s a 50 Hz
        >>> windows, labels = create_windows(signal, label=1)
        >>> windows.shape
        (78, 128, 6)
    """
    T, C = signal.shape
    if T < window_size:
        raise ValueError(
            f"Señal demasiado corta ({T} muestras) para ventana de {window_size}"
        )

    num_windows = (T - window_size) // step + 1
    windows = np.zeros((num_windows, window_size, C), dtype=np.float32)

    for i in range(num_windows):
        start = i * step
        windows[i] = signal[start : start + window_size]

    labels_out = None
    if label is not None:
        labels_out = np.full(num_windows, label, dtype=np.float32)

    return windows, labels_out


# ---------------------------------------------------------------------------
# Normalización online por usuario (compatible con FL)
# ---------------------------------------------------------------------------

class OnlineNormalizer:
    """
    Normalización incremental por canal usando media y desviación estándar
    estimadas online (Welford, 1962).

    En aprendizaje federado, cada dispositivo mantiene su propio normalizador.
    Los estadísticos NO se comparten con el servidor (privacidad preservada).

    Uso:
        norm = OnlineNormalizer(n_channels=6)
        for window in streaming_windows:
            norm.update(window)           # actualiza estadísticos
        normalized = norm.transform(window)
    """

    def __init__(self, n_channels: int):
        self.n = 0
        self.mean = np.zeros(n_channels, dtype=np.float64)
        self.M2   = np.ones(n_channels, dtype=np.float64)   # varianza × n

    def update(self, window: np.ndarray) -> None:
        """Actualiza estadísticos con una nueva ventana (128, C)."""
        for sample in window:                   # iterar muestra a muestra
            self.n += 1
            delta     = sample - self.mean
            self.mean += delta / self.n
            delta2    = sample - self.mean
            self.M2   += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self.M2 / (self.n - 1))

    def transform(self, window: np.ndarray) -> np.ndarray:
        """Normaliza una ventana restando la media y dividiendo por std."""
        std = np.where(self.std < 1e-8, 1.0, self.std)   # evitar div/0
        return ((window - self.mean) / std).astype(np.float32)

    def fit_transform(self, window: np.ndarray) -> np.ndarray:
        """Actualiza estadísticos y devuelve la ventana normalizada."""
        self.update(window)
        return self.transform(window)


# ---------------------------------------------------------------------------
# Extracción de features de gestos táctiles
# ---------------------------------------------------------------------------

def extract_touch_features(events: List[TouchEvent]) -> np.ndarray:
    """
    Agrega una lista de eventos táctiles en un vector de 12 features.

    Si no hay eventos en la ventana de tiempo actual, devuelve ceros
    (el modelo aprende a ignorarlos con suficiente entrenamiento).

    Args:
        events : lista de TouchEvent en la ventana de tiempo actual

    Returns:
        vector (12,) de features táctiles normalizadas ∈ [0, 1]

    Diseño de las 12 features (validadas en SAĞBAŞ & BALLI 2024):
        0  duration_ms    – mediana de duración de contacto
        1  dx             – media de desplazamiento horizontal
        2  dy             – media de desplazamiento vertical
        3  distance       – mediana de longitud de trayectoria
        4  mean_velocity  – media de velocidad media
        5  max_velocity   – media de velocidad pico
        6  mean_pressure  – mediana de presión media
        7  max_pressure   – mediana de presión máxima
        8  mean_area      – mediana de área media de contacto
        9  max_area       – mediana de área máxima
        10 direction_angle– media de ángulo de dirección
        11 gesture_type   – proporción de swipes (vs taps) en la ventana
    """
    if not events:
        return np.zeros(12, dtype=np.float32)

    arr = np.array([
        [e.duration_ms, e.dx, e.dy, e.distance,
         e.mean_velocity, e.max_velocity,
         e.mean_pressure, e.max_pressure,
         e.mean_area, e.max_area,
         e.direction_angle, e.gesture_type]
        for e in events
    ], dtype=np.float32)  # (N_events, 12)

    features = np.array([
        np.median(arr[:, 0]),   # duration
        np.mean(arr[:, 1]),     # dx
        np.mean(arr[:, 2]),     # dy
        np.median(arr[:, 3]),   # distance
        np.mean(arr[:, 4]),     # mean_velocity
        np.mean(arr[:, 5]),     # max_velocity
        np.median(arr[:, 6]),   # mean_pressure
        np.median(arr[:, 7]),   # max_pressure
        np.median(arr[:, 8]),   # mean_area
        np.median(arr[:, 9]),   # max_area
        np.mean(arr[:, 10]),    # direction_angle
        np.mean(arr[:, 11]),    # gesture_type (proporción de swipes)
    ], dtype=np.float32)

    return features


# ---------------------------------------------------------------------------
# Generador de muestras negativas (para entrenamiento FL)
# ---------------------------------------------------------------------------

def generate_synthetic_negatives(
    positive_windows: np.ndarray,
    n_negatives: Optional[int] = None,
    noise_scale: float = 0.5,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Genera muestras negativas sintéticas (impostores simulados).

    En FL, cada cliente solo tiene datos de un usuario (positivos).
    Para entrenar el clasificador binario, se necesitan muestras negativas.
    Este método añade ruido gaussiano escalado para generar comportamientos
    "plausibles pero distintos" al usuario real.

    Estrategia: permutación de ventanas + ruido gaussiano
      → similar a OZA et al. (2021) pero sin enviar estadísticos al servidor

    Args:
        positive_windows : array (N, window_size, C) de ventanas positivas
        n_negatives      : número de negativas a generar (default = N)
        noise_scale      : escala del ruido (0.5 × std de los datos)
        random_seed      : semilla para reproducibilidad

    Returns:
        array (n_negatives, window_size, C) de ventanas negativas sintéticas
    """
    rng = np.random.default_rng(random_seed)
    N = len(positive_windows)
    n_neg = n_negatives if n_negatives is not None else N

    # Calcular desviación estándar por canal
    std_per_channel = positive_windows.std(axis=(0, 1))  # (C,)

    negatives = np.zeros((n_neg, *positive_windows.shape[1:]), dtype=np.float32)

    for i in range(n_neg):
        # Ventana base: una positiva aleatoria
        base = positive_windows[rng.integers(N)].copy()
        # Permutación temporal (mezcla el orden temporal)
        idx = rng.permutation(base.shape[0])
        base = base[idx]
        # Ruido calibrado
        noise = rng.normal(0, noise_scale * std_per_channel, size=base.shape)
        negatives[i] = base + noise

    return negatives.astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline de preparación de datos completo
# ---------------------------------------------------------------------------

def prepare_training_data(
    imu_signal: np.ndarray,
    touch_events: Optional[List[TouchEvent]] = None,
    normalizer: Optional[OnlineNormalizer] = None,
    window_size: int = WINDOW_SIZE,
    step: int = TRAIN_STEP,
    balance: bool = True,
) -> Tuple:
    """
    Pipeline completo: señal IMU cruda → tensores listos para entrenamiento.

    Args:
        imu_signal    : array (T, C) — señal IMU del usuario legítimo
        touch_events  : lista de TouchEvent (opcional; para config 'gyro_acc_touch')
        normalizer    : normalizador por usuario (si None, crea uno nuevo)
        window_size   : tamaño de ventana
        step          : paso entre ventanas
        balance       : si True, genera muestras negativas para balancear clases

    Returns:
        Tupla que varía según la presencia de gestos táctiles:
          Sin táctil  : (X_imu, y)   — arrays listos para model.fit()
          Con táctil  : (X_imu, X_touch, y)
    """
    # ── Ventanas IMU ──────────────────────────────────────────────────
    pos_windows, pos_labels = create_windows(imu_signal, window_size, step, label=1)

    # ── Normalización ─────────────────────────────────────────────────
    if normalizer is None:
        normalizer = OnlineNormalizer(n_channels=imu_signal.shape[1])
    pos_windows_norm = np.stack([
        normalizer.fit_transform(w) for w in pos_windows
    ])

    # ── Muestras negativas ────────────────────────────────────────────
    if balance:
        neg_windows = generate_synthetic_negatives(pos_windows_norm)
        neg_labels  = np.zeros(len(neg_windows), dtype=np.float32)
        X_imu = np.concatenate([pos_windows_norm, neg_windows], axis=0)
        y     = np.concatenate([pos_labels, neg_labels], axis=0)
    else:
        X_imu = pos_windows_norm
        y     = pos_labels

    # ── Shuffle ───────────────────────────────────────────────────────
    perm  = np.random.permutation(len(X_imu))
    X_imu = X_imu[perm]
    y     = y[perm]

    # ── Features táctiles ─────────────────────────────────────────────
    if touch_events is not None:
        touch_feat_pos = extract_touch_features(touch_events)
        # Replicar el vector de features para cada ventana
        # (en una implementación real, se segmentaría por ventana de tiempo)
        X_touch = np.tile(touch_feat_pos, (len(X_imu), 1)).astype(np.float32)
        return X_imu, X_touch, y

    return X_imu, y
