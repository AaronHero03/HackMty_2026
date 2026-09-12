"""Procesamiento Acústico y extracción de características (Fase 2).

Contiene los algoritmos matemáticos puros para el filtrado pasabanda
y la extracción de Coeficientes Cepstrales de Frecuencia Lineal (LFCC).
Estas funciones son importadas tanto por los scripts de prueba como por la API.
"""
import numpy as np
import scipy.signal as signal
from scipy.fftpack import dct

def aplicar_filtro_pasabanda(y, sr=8000, fmin=300, fmax=3400, orden=4):
    """Filtro IIR Butterworth usando Secciones de Segundo Orden (SOS)."""
    if len(y) == 0:
        return y
    sos = signal.butter(N=orden, Wn=[fmin, fmax], btype='bandpass', fs=sr, output='sos')
    return signal.sosfilt(sos, y)

def banco_filtros_lineal(n_filtros=20, n_fft=512, sr=8000, fmin=300, fmax=3400):
    """Banco de filtros triangulares con espaciado lineal (300-3400 Hz)."""
    num_bins = n_fft // 2 + 1
    freqs_fft = np.linspace(0, sr / 2, num_bins)
    freqs_filtros = np.linspace(fmin, fmax, n_filtros + 2)

    banco = np.zeros((n_filtros, num_bins))
    for i in range(n_filtros):
        f_izq, f_cen, f_der = freqs_filtros[i:i+3]
        up = (freqs_fft - f_izq) / (f_cen - f_izq)
        down = (f_der - freqs_fft) / (f_der - f_cen)
        banco[i] = np.maximum(0, np.minimum(up, down))
    return banco

def extraer_features_acusticas(y_voz, sr=8000, n_lfcc=12, n_filtros=20, n_fft=512, hop_length=160):
    """Aplica el filtro pasabanda, extrae LFCCs y devuelve estadísticos."""
    features = {}

    # 1. Filtro de telefonía estricto
    y_filtrado = aplicar_filtro_pasabanda(y_voz, sr=sr, fmin=300, fmax=3400)

    if len(y_filtrado) < n_fft:
        for i in range(1, n_lfcc + 1):
            features[f"lfcc_mediana_{i}"] = 0.0
            features[f"lfcc_std_{i}"] = 0.0
        return features

    # 2. Espectro de Potencia
    ventana = np.hamming(n_fft)
    pasos = range(0, len(y_filtrado) - n_fft, hop_length)
    espectro_potencia = np.array([np.abs(np.fft.rfft(y_filtrado[p : p + n_fft] * ventana, n=n_fft))**2 for p in pasos]).T

    # 3. Banco Lineal y Transformada Discreta del Coseno
    banco = banco_filtros_lineal(n_filtros=n_filtros, n_fft=n_fft, sr=sr, fmin=300, fmax=3400)
    energias = np.dot(banco, espectro_potencia)
    energias = np.where(energias == 0, np.finfo(float).eps, energias)
    lfccs = dct(np.log(energias), type=2, axis=0, norm='ortho')[:n_lfcc]

    # 4. Estadísticos clave
    for i in range(n_lfcc):
        features[f"lfcc_mediana_{i+1}"] = float(np.median(lfccs[i]))
        features[f"lfcc_std_{i+1}"] = float(np.std(lfccs[i]))

    return features