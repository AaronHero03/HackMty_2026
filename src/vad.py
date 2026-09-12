"""Detector de voz por canal (Silero VAD).

Dos pasos separados a propósito:
1. probabilidades_voz(): corre la red una vez y da la probabilidad de voz cada 32 ms.
2. tramos_voz(): convierte esas probabilidades en tramos con un umbral y un silencio mínimo.

Así se pueden probar muchos ajustes sin volver a correr la red. tramos_voz() reproduce la
lógica de silero_vad.get_speech_timestamps (sin límite de duración máxima por tramo).
"""
import numpy as np
import torch
from silero_vad import load_silero_vad

SR = 8000
VENTANA = 256            # muestras por probabilidad a 8 kHz (32 ms)

# Ajustes por defecto, elegidos con fase1/evaluar_vad.py comparando contra turns/ solo en train
# (282 llamadas). Entre 0.7 y 0.8 de umbral y 200-250 ms el error casi no cambia (4.28-4.31 %).
UMBRAL = 0.7
MIN_SILENCIO_MS = 200
MIN_VOZ_MS = 250
RELLENO_MS = 30

_modelo = None


def _cargar_modelo():
    global _modelo
    if _modelo is None:
        _modelo = load_silero_vad()
    return _modelo


def probabilidades_voz(x):
    """x: UN canal (numpy float32 a 8 kHz). Devuelve una probabilidad de voz por ventana de 32 ms."""
    modelo = _cargar_modelo()
    modelo.reset_states()
    with torch.no_grad():
        p = modelo.audio_forward(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), sr=SR)
    return p.squeeze(0).numpy()


def tramos_voz(probs, n_muestras, umbral=UMBRAL, min_silencio_ms=MIN_SILENCIO_MS,
               min_voz_ms=MIN_VOZ_MS, relleno_ms=RELLENO_MS):
    """Tramos de voz [(inicio_s, fin_s), ...] a partir de las probabilidades de un canal."""
    umbral_bajo = max(umbral - 0.15, 0.01)     # histéresis: para cortar, la probabilidad debe bajar más
    min_voz = SR * min_voz_ms / 1000
    min_silencio = SR * min_silencio_ms / 1000
    relleno = SR * relleno_ms / 1000

    tramos, activo, inicio, fin_tentativo = [], False, 0, 0
    for i, p in enumerate(probs):
        muestra = VENTANA * i
        if p >= umbral and fin_tentativo:
            fin_tentativo = 0
        if p >= umbral and not activo:
            activo, inicio = True, muestra
            continue
        if p < umbral_bajo and activo:
            if not fin_tentativo:
                fin_tentativo = muestra
            if muestra - fin_tentativo < min_silencio:
                continue
            if fin_tentativo - inicio > min_voz:
                tramos.append([inicio, fin_tentativo])
            activo, fin_tentativo = False, 0
    if activo and n_muestras - inicio > min_voz:
        tramos.append([inicio, n_muestras])

    # Relleno alrededor de cada tramo sin encimar tramos vecinos (igual que Silero)
    for i, t in enumerate(tramos):
        if i == 0:
            t[0] = int(max(0, t[0] - relleno))
        if i != len(tramos) - 1:
            hueco = tramos[i + 1][0] - t[1]
            if hueco < 2 * relleno:
                t[1] += int(hueco // 2)
                tramos[i + 1][0] = int(max(0, tramos[i + 1][0] - hueco // 2))
            else:
                t[1] = int(min(n_muestras, t[1] + relleno))
                tramos[i + 1][0] = int(max(0, tramos[i + 1][0] - relleno))
        else:
            t[1] = int(min(n_muestras, t[1] + relleno))
    return [(a / SR, b / SR) for a, b in tramos]


def detectar_voz(x, **ajustes):
    """Atajo: tramos de voz de un canal ya normalizado."""
    return tramos_voz(probabilidades_voz(x), len(x), **ajustes)
