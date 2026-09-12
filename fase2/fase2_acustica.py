"""Fase 2 — Métricas acústicas de la voz de quien llama (LFCC). Versión de Fer.

Hay otra versión de la Fase 2 en src/dsp.py (Andrés); el equipo elige cuál usa main.py.

Contrato con main.py:
    metricas_acusticas = f2.extraer_metricas_acusticas(voz_recortada, sr)

- voz_recortada: canal 0 ya normalizado y recortado a los turnos de voz
  (Fase 1: procesar_audio_base -> cargar_turnos -> recortar_voz_activa).
- Devuelve 24 métricas: mediana y desviación de los LFCC 1 a 12.

"""
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.signal import stft
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

def lfcc(y):
    """LFCC por trama. Devuelve (coeficientes: matriz (20, tramas), energía de cada trama en dB)."""
    _, _, Z = stft(y, fs=SR, nperseg=VENTANA, noverlap=VENTANA - SALTO, boundary=None, padded=False)
    energia = BANCO @ (np.abs(Z) ** 2)
    coeficientes = dct(np.log(energia + 1e-10), type=2, axis=0, norm="ortho")
    return coeficientes, 10 * np.log10(energia.sum(axis=0) + 1e-12)


def extraer_metricas_acusticas(voz_recortada, sr):
    """Mediana y desviación de los LFCC 1 a 12 de la voz de quien llama."""
    if sr != SR:
        raise ValueError(f"Se esperaba audio a {SR} Hz y llegó a {sr} Hz")
    nombres = [f"lfcc_{estadistico}_{i}" for i in range(1, N_LFCC + 1) for estadistico in ("mediana", "std")]
    if len(voz_recortada) < 2 * VENTANA:
        return {nombre: np.nan for nombre in nombres}   # sin voz suficiente: vacío, no ceros falsos

    coef, energia_db = lfcc(np.asarray(voz_recortada, dtype=np.float32))
    coef = coef[:, energia_db >= np.percentile(energia_db, 95) - UMBRAL_TRAMAS_DB]

    metricas = {}
    for i in range(1, N_LFCC + 1):
        metricas[f"lfcc_mediana_{i}"] = float(np.median(coef[i]))
        metricas[f"lfcc_std_{i}"] = float(np.std(coef[i]))
    return metricas
