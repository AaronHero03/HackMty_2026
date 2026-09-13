import pandas as pd
import numpy as np
import soundfile as sf
import json
from math import gcd
from pathlib import Path
from scipy.signal import resample_poly

SR_OBJETIVO = 8000

def leer_manifiesto(ruta_base):
    """
    Carga el manifest.csv para obtener el mapa de las llamadas.
    Devuelve un DataFrame de Pandas.
    """
    ruta_manifest = Path(ruta_base) / "manifest.csv"
    return pd.read_csv(ruta_manifest)

def normalizar_volumen(y, target_db=-20):
    """
    Ajusta el volumen del audio a un nivel RMS constante.
    Esto elimina la variable de confusión de la ganancia descubierta en Fase 0.
    """
    rms_actual = np.sqrt(np.mean(y**2))
    if rms_actual == 0:
        return y
    target_rms = 10 ** (target_db / 20.0)
    return y * (target_rms / rms_actual)

def procesar_audio_base(ruta_wav):
    """
    Lee el .wav (mono o estéreo), aísla el Canal 0 (el llamador), lo normaliza y resamplea a 8 kHz.
    Retorna el arreglo numpy del audio y el sample rate (siempre SR_OBJETIVO).
    """
    crudo, sr = sf.read(ruta_wav)

    # Manejar audio mono (1D) o estéreo (2D)
    if crudo.ndim == 1:
        y_c0 = crudo.astype(np.float32)
    else:
        y_c0 = crudo[:, 0].astype(np.float32)

    # Resamplear si el archivo no está a 8 kHz
    if sr != SR_OBJETIVO:
        g = gcd(SR_OBJETIVO, sr)
        y_c0 = resample_poly(y_c0, SR_OBJETIVO // g, sr // g).astype(np.float32)

    y_c0_norm = normalizar_volumen(y_c0)
    return y_c0_norm, SR_OBJETIVO

def cargar_turnos(ruta_json):
    """
    Lee el archivo de anotaciones y separa los turnos por participante.
    Retorna dos listas de diccionarios: (turnos_llamador, turnos_agente).
    """
    with open(ruta_json, 'r') as f:
        datos = json.load(f)
        
    turnos = datos.get("turns", [])
    
    c0_turnos = [t for t in turnos if t["channel"] == 0] # Llamador
    c1_turnos = [t for t in turnos if t["channel"] == 1] # Agente
    
    return c0_turnos, c1_turnos

def recortar_voz_activa(y, turnos, sr):
    """
    Usa las marcas de tiempo para recortar los silencios y devolver 
    un único arreglo continuo solo con la voz del participante.
    """
    if not turnos:
        return np.array([])
        
    tramos = []
    for t in turnos:
        ini = int(t["start"] * sr)
        fin = int(t["end"] * sr)
        tramos.append(y[ini:fin])
        
    return np.concatenate(tramos)
