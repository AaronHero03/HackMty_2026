"""Fase 2 — Métricas acústicas optimizadas para audio telefónico (8 kHz).

Extrae 27 características representativas:
- 24 LFCCs (12 medianas y 12 desviaciones estándar de coeficientes 1 a 12).
- CPP (Cepstral Peak Prominence): armonicidad sin depender de estimación de F0.
- Inclinación Espectral: balance de energía graves (300-1000 Hz) vs agudos (2000-3400 Hz).
- Vocoder Periodicity Peak: huella de framing/bloque del motor sintético en la envolvente.

Contrato:
    metricas_acusticas = extraer_metricas_acusticas(voz_recortada, sr)
"""

from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.signal import butter, hilbert, sosfilt, stft, welch
import scipy.signal as signal

SR = 8000
N_FILTROS = 20          # filtros triangulares del mismo ancho entre 300 y 3400 Hz
N_LFCC = 12             # se guardan los coeficientes 1 a 12 (el 0 es casi el volumen)
VENTANA = 256           # 32 ms a 8 kHz
SALTO = 80              # 10 ms
UMBRAL_TRAMAS_DB = 30   # se quitan las tramas 30 dB por debajo de las más fuertes (silencio digital y eco)


def _banco_filtros():
    frecuencias = np.fft.rfftfreq(VENTANA, d=1 / SR)
    bordes = np.linspace(300, 3400, N_FILTROS + 2)
    banco = np.zeros((N_FILTROS, len(frecuencias)))
    for i in range(N_FILTROS):
        izq, centro, der = bordes[i], bordes[i + 1], bordes[i + 2]
        subida = (frecuencias - izq) / (centro - izq)
        bajada = (der - frecuencias) / (der - centro)
        banco[i] = np.clip(np.minimum(subida, bajada), 0, None)
    return banco


BANCO = _banco_filtros()


def aplicar_filtro_pasabanda(y, sr=SR, fmin=300, fmax=3400, orden=4):
    """Filtro IIR Butterworth usando Secciones de Segundo Orden (SOS)."""
    if len(y) == 0:
        return y
    sos = signal.butter(N=orden, Wn=[fmin, fmax], btype='bandpass', fs=sr, output='sos')
    return signal.sosfilt(sos, y)

def _lfcc(y):
    """LFCC por trama. Devuelve (coeficientes: matriz (20, tramas), energía de cada trama en dB)."""
    _, _, Z = stft(y, fs=SR, nperseg=VENTANA, noverlap=VENTANA - SALTO, boundary=None, padded=False)
    energia = BANCO @ (np.abs(Z) ** 2)
    coeficientes = dct(np.log(energia + 1e-10), type=2, axis=0, norm="ortho")
    return coeficientes, 10 * np.log10(energia.sum(axis=0) + 1e-12)

def _tramos_activos(y):
    """Reconstruye el audio temporal conservando solo las tramas útiles de voz."""
    n_tramas = 1 + (len(y) - VENTANA) // SALTO
    if n_tramas < 1:
        return np.asarray(y, dtype=np.float32)

    energia = np.array([
        10 * np.log10(np.sum(y[i * SALTO : i * SALTO + VENTANA].astype(np.float64) ** 2)+ 1e-12)
            for i in range(n_tramas)
    ])
    umbral = np.percentile(energia, 95) - UMBRAL_TRAMAS_DB
    activos = energia >= umbral
    trozos = [y[i * SALTO : i * SALTO + SALTO] for i in range(n_tramas) if activos[i]]
    return (np.concatenate(trozos).astype(np.float32) if trozos else np.array([], dtype=np.float32))

# Features nuevos 

def cpp(y):
    """Cepstral Peak Prominence: mide periodicidad limpia sin requerir pitch-tracking."""
    if len(y) < int(0.04 * SR):
        return np.nan
    frame_len, hop = int(0.04 * SR), int(0.01 * SR)
    q_lo, q_hi = int(SR / 400), int(SR / 75)

    valores = []
    for start in range(0, len(y) - frame_len, hop):
        frame = y[start : start + frame_len] * np.hanning(frame_len)
        espectro_log = np.log(np.abs(np.fft.rfft(frame)) + 1e-10)
        cepstrum = np.fft.irfft(espectro_log)
        q_hi_c = min(q_hi, len(cepstrum) // 2)
        if q_hi_c <= q_lo:
            continue
        region = cepstrum[q_lo:q_hi_c]
        x = np.arange(len(cepstrum) // 2)
        coef = np.polyfit(x, cepstrum[: len(cepstrum) // 2], 1)
        base = np.polyval(coef, q_lo + np.argmax(region))
        valores.append(np.max(region) - base)
    return float(np.mean(valores)) if valores else np.nan


def inclinacion_espectral(y, banda_baja=(300, 1000), banda_alta=(2000, 3400)):
    """Mide la diferencia de potencia espectral entre graves y agudos."""
    if len(y) < 256:
        return np.nan
    freqs, psd = welch(y.astype(np.float64), fs=SR, nperseg=min(512, len(y)))
    D = 10 * np.log10(psd + 1e-20)
    m_baja = (freqs >= banda_baja[0]) & (freqs <= banda_baja[1])
    m_alta = (freqs >= banda_alta[0]) & (freqs <= banda_alta[1])
    if not m_baja.any() or not m_alta.any():
        return np.nan
    return float(D[m_baja].mean() - D[m_alta].mean())


def vocoder_periodicity_peak(y):
    """Prominencia de modulación en la envolvente de Hilbert (artefactos de framing TTS)."""
    if len(y) < int(0.1 * SR):
        return np.nan
    envolvente = np.abs(hilbert(y.astype(np.float64)))
    freqs, psd = welch(envolvente, fs=SR, nperseg=min(1024, len(envolvente)))
    banda = (freqs >= 40) & (freqs <= 200)
    if not banda.any() or psd.sum() == 0:
        return np.nan
    return float(psd[banda].max() / (np.median(psd) + 1e-10))


# Funcion principal (Contrato)
def extraer_metricas_acusticas(voz_recortada, sr=SR):
    """Extrae las 27 características acústicas del segmento de voz."""
    if sr != SR:
        raise ValueError(f"Se esperaba audio a {SR} Hz y llegó a {sr} Hz")

    nombres_lfcc = [
        f"lfcc_{estadistico}_{i}"
        for i in range(1, N_LFCC + 1)
        for estadistico in ("mediana", "std")
    ]
    nombres_extra = ["cpp", "inclinacion_espectral", "vocoder_periodicity_peak"]

    todas_las_llaves = nombres_lfcc + nombres_extra

    if len(voz_recortada) < 2 * VENTANA:
        return {nombre: np.nan for nombre in todas_las_llaves}

    voz_recortada = np.asarray(voz_recortada, dtype=np.float32)

    # 1. Extracción de LFCC (descartando C0)
    coef, energia_db = _lfcc(voz_recortada)
    umbral_corte = np.percentile(energia_db, 95) - UMBRAL_TRAMAS_DB
    tramas_validas = energia_db >= umbral_corte

    metricas = {}
    if np.any(tramas_validas):
        coef = coef[:, tramas_validas]
        for i in range(1, N_LFCC + 1):
            metricas[f"lfcc_mediana_{i}"] = float(np.median(coef[i]))
            metricas[f"lfcc_std_{i}"] = float(np.std(coef[i]))
    else:
        for i in range(1, N_LFCC + 1):
            metricas[f"lfcc_mediana_{i}"] = np.nan
            metricas[f"lfcc_std_{i}"] = np.nan

    # 2. Extracción de métricas de dominio temporal sobre voz activa
    activo = _tramos_activos(voz_recortada)
    if len(activo) < int(0.05 * SR):
        metricas.update({nombre: np.nan for nombre in nombres_extra})
        return metricas

    metricas["cpp"] = cpp(activo)
    metricas["inclinacion_espectral"] = inclinacion_espectral(activo)
    metricas["vocoder_periodicity_peak"] = vocoder_periodicity_peak(activo)

    return metricas