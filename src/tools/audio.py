"""Carga y normalización de llamadas.

Las mismas funciones se usan al entrenar y en la API, para que el modelo vea siempre el audio
preparado de la misma forma.
"""
import base64
import binascii
import io
import os
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 8000
TRAMA = 160               # 20 ms
OBJETIVO_DBFS = -20.0     # nivel al que se lleva la parte fuerte de cada canal
GANANCIA_MAX_DB = 30.0    # tope para no amplificar un canal casi vacío


def cargar_llamada(fuente):
    """Devuelve (canal0, canal1) como float32 a 8 kHz, sin mezclar canales.

    fuente: ruta al WAV, bytes del WAV o texto base64 (con o sin prefijo "data:...;base64,").
    Canal 0 = quien llama, canal 1 = agente. Si llega mono, el canal 1 queda en ceros.
    """
    if isinstance(fuente, (bytes, bytearray)):
        archivo = io.BytesIO(fuente)
    elif isinstance(fuente, os.PathLike) or _es_ruta(fuente):
        archivo = fuente
    else:
        texto = fuente.split("base64,", 1)[-1]
        try:
            archivo = io.BytesIO(base64.b64decode(texto))
        except (binascii.Error, ValueError) as e:
            raise ValueError("El audio no es base64 válido") from e

    y, sr = sf.read(archivo, dtype="float32", always_2d=True)   # (muestras, canales)
    if sr != SR:
        g = gcd(SR, sr)
        y = resample_poly(y, SR // g, sr // g, axis=0).astype(np.float32)
    llama = y[:, 0]
    agente = y[:, 1] if y.shape[1] > 1 else np.zeros_like(llama)
    return llama, agente


def _es_ruta(texto):
    if not isinstance(texto, str) or len(texto) > 4096:
        return False
    try:
        return os.path.isfile(texto)
    except (OSError, ValueError):
        return False


def normalizar(x):
    """Ganancia constante para un canal.

    Lleva el nivel de su parte fuerte (percentil 95 del volumen en tramas de 20 ms, que casi
    siempre es voz) a OBJETIVO_DBFS. No usa turns/, así que funciona igual en la API.
    Una ganancia constante no cambia latencias, jitter ni la forma del espectro.
    """
    k = len(x) // TRAMA
    if k == 0:
        return x
    rms = np.sqrt(np.mean(x[:k * TRAMA].reshape(k, TRAMA) ** 2, axis=1))
    referencia = np.percentile(rms, 95)
    if referencia <= 0:
        return x
    ganancia_db = min(OBJETIVO_DBFS - 20 * np.log10(referencia), GANANCIA_MAX_DB)
    return np.clip(x * 10 ** (ganancia_db / 20), -1.0, 1.0).astype(np.float32)
