import pandas as pd
import numpy as np
import soundfile as sf
import json
from pathlib import Path

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
    Lee el .wav estéreo, aísla el Canal 0 (el llamador) y lo normaliza.
    Retorna el arreglo numpy del audio y el sample rate.
    """
    crudo, sr = sf.read(ruta_wav)
    
    # Extraer Canal 0 y convertir a float32
    y_c0 = crudo[:, 0].astype(np.float32)
    
    # Normalización de defensa contra la trampa de volumen
    y_c0_norm = normalizar_volumen(y_c0)
    
    return y_c0_norm, sr

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
