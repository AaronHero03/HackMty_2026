"""Procesamiento Acústico y extracción de características (Fase 2 - Versión Híbrida).

Combina el filtrado temporal IIR pasabanda con el cálculo espectral LFCC 
estricto (descarte de C0, filtro de tramas por energía y precálculo de matriz).
"""
import numpy as np
import scipy.signal as signal
from scipy.fft import dct, stft

SR = 8000
N_FILTROS = 20
N_LFCC = 12
VENTANA = 256          # 32 ms a 8 kHz
SALTO = 80             # 10 ms a 8 kHz
UMBRAL_TRAMAS_DB = 30  # Eliminación de tramas 30 dB por debajo del p95


def _crear_banco_filtros_lineal(sr=SR, n_fft=VENTANA, n_filtros=N_FILTROS, fmin=300, fmax=3400):
    """Genera la matriz de filtros triangulares lineales."""
    frecuencias = np.fft.rfftfreq(n_fft, d=1 / sr)
    bordes = np.linspace(fmin, fmax, n_filtros + 2)
    banco = np.zeros((n_filtros, len(frecuencias)))
    for i in range(n_filtros):
        izq, centro, der = bordes[i], bordes[i + 1], bordes[i + 2]
        subida = (frecuencias - izq) / (centro - izq)
        bajada = (der - frecuencias) / (der - centro)
        banco[i] = np.clip(np.minimum(subida, bajada), 0, None)
    return banco


# Banco precalculado global para alto rendimiento
BANCO_FILTROS = _crear_banco_filtros_lineal()


def aplicar_filtro_pasabanda(y, sr=SR, fmin=300, fmax=3400, orden=4):
    """Filtro IIR Butterworth usando Secciones de Segundo Orden (SOS)."""
    if len(y) == 0:
        return y
    sos = signal.butter(N=orden, Wn=[fmin, fmax], btype='bandpass', fs=sr, output='sos')
    return signal.sosfilt(sos, y)


def extraer_features_acusticas(y_voz, sr=SR, n_lfcc=N_LFCC):
    """Extrae 24 métricas acústicas (12 medianas y 12 std) de los coeficientes LFCC (1 a 12)."""
    if sr != SR:
        raise ValueError(f"Se esperaba audio a {SR} Hz y llegó a {sr} Hz")

    nombres = [f"lfcc_{est}_{i}" for i in range(1, n_lfcc + 1) for est in ("mediana", "std")]

    # Retorno limpio para segmentos sin voz suficiente
    if len(y_voz) < 2 * VENTANA:
        return {nombre: np.nan for nombre in nombres}

    # 1. Filtro temporal IIR Butterworth (300 - 3400 Hz)
    y_filtrado = aplicar_filtro_pasabanda(np.asarray(y_voz, dtype=np.float32), sr=sr)

    # 2. STFT y banco de filtros espectrales (32 ms / 10 ms)
    _, _, Z = stft(y_filtrado, fs=sr, nperseg=VENTANA, noverlap=VENTANA - SALTO, boundary=None, padded=False)
    energia_espectral = BANCO_FILTROS @ (np.abs(Z) ** 2)

    # 3. Discriminación de tramas de silencio/eco
    energia_tramas_db = 10 * np.log10(energia_espectral.sum(axis=0) + 1e-12)
    umbral_corte = np.percentile(energia_tramas_db, 95) - UMBRAL_TRAMAS_DB
    tramas_validas = energia_tramas_db >= umbral_corte

    if not np.any(tramas_validas):
        return {nombre: np.nan for nombre in nombres}

    # 4. DCT y aislamiento de coeficientes 1 a 12 (descarta C0)
    coeficientes = dct(np.log(energia_espectral + 1e-10), type=2, axis=0, norm="ortho")
    coeficientes_validos = coeficientes[1 : n_lfcc + 1, tramas_validas]

    # 5. Cálculo de agregaciones estadísticas
    features = {}
    for i in range(n_lfcc):
        features[f"lfcc_mediana_{i+1}"] = float(np.median(coeficientes_validos[i]))
        features[f"lfcc_std_{i+1}"] = float(np.std(coeficientes_validos[i]))

    return features