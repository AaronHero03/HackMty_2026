# Fase 1: Pipeline de Datos y Estructura Base

# Responsable ideal: Estudiante de Computer Science 1 (Perfil de datos/backend).

#     Lectura del Manifiesto: Crear un script que cargue manifest.csv para mapear los identificadores de llamada (anon_id) con su clase y split (train o val).
#     Separación de Canales: Escribir una función que lea los .wav estéreo y extraiga el Canal 0 (el llamador) y el Canal 1 (el agente).
#     Segmentación de Entrenamiento: Implementar una función que lea los archivos turns/*.json y recorte el audio exactamente en los segmentos donde hay voz, guardando estas coordenadas para la extracción de características.

import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import json
from pathlib import Path
from scipy.signal import butter, lfilter
import warnings

# Suprimir advertencias de librosa por audios cortos
warnings.filterwarnings("ignore")

def bandpass_filter(y, sr, lowcut=300, highcut=3400):
    """Aplica un filtro pasa banda para mantener solo frecuencias entre lowcut y highcut."""
    
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y)

def normalizar_volumen(y, target_db=-20):
    """Ajusta el volumen del audio a un nivel RMS constante."""
    rms_actual = np.sqrt(np.mean(y**2))
    if rms_actual == 0:
        return y
    target_rms = 10 ** (target_db / 20.0)
    return y * (target_rms / rms_actual)

def extraer_features(ruta_wav, ruta_json):
    features = {}
    sr = 8000
    
    # --- CAPA CONVERSACIONAL (Desde turns/) ---
    with open(ruta_json, 'r') as f:
        turnos = json.load(f)["turns"]
        
    c0 = [t for t in turnos if t["channel"] == 0] # Llamador
    c1 = [t for t in turnos if t["channel"] == 1] # Agente
    
    latencias = []
    # Calcular latencia de respuesta (Llamador responde al Agente)
    for agente_t in c1:
        # Respuestas del llamador que inician DESPUÉS de que el agente termine
        resp = [llam_t["start"] - agente_t["end"] for llam_t in c0 if llam_t["start"] >= agente_t["end"]]
        if resp:
            latencias.append(resp[0])
            
    # Métricas de tiempo robustas
    features["latencia_mediana"] = np.median(latencias) if latencias else 0.0
    features["latencia_std"] = np.std(latencias) if latencias else 0.0
    
    # --- CAPA ACÚSTICA (Desde audio/ Canal 0) ---
    crudo, sr_file = sf.read(ruta_wav)
    y_c0 = crudo[:, 0].astype(np.float32)
    
    # 1. Normalizar volumen para evitar la trampa de dB detectada en Fase 0
    y_c0 = normalizar_volumen(y_c0)

    # 1. Aplicar filtro pasa banda
    y_c0 = bandpass_filter(y_c0, sr)
    
    # 2. Recortar solo los segmentos donde el llamador habla
    tramos_voz = []
    for t in c0:
        ini = int(t["start"] * sr)
        fin = int(t["end"] * sr)
        tramos_voz.extend(y_c0[ini:fin])
        
    y_voz = np.array(tramos_voz)
    
    if len(y_voz) < 512: # Si no hay suficiente voz, devolver ceros
        for i in range(1, 13):
            features[f"mfcc_mediana_{i}"] = 0.0
            features[f"mfcc_std_{i}"] = 0.0
        return features

    # 3. Extraer coeficientes limitados a 300-3400 Hz (Evita trampa de graves/agudos)
    # Usamos MFCC aquí por velocidad, pero restringiendo fmin y fmax para ser fieles al canal telefónico
    mfccs = librosa.feature.mfcc(
        y=y_voz, 
        sr=sr, 
        n_mfcc=12, 
        fmin=300, 
        fmax=3400
    )
    
    # Calcular estadística (mediana y desviación) de cada coeficiente
    for i in range(12):
        features[f"mfcc_mediana_{i+1}"] = np.median(mfccs[i])
        features[f"mfcc_std_{i+1}"] = np.std(mfccs[i])
        
    return features

def procesar_dataset(ruta_base):
    ruta_base = Path(ruta_base)
    manifest = pd.read_csv(ruta_base / "manifest.csv")
    
    datos_procesados = []
    
    print("Iniciando extracción de características...")
    for idx, fila in manifest.iterrows():
        id_llamada = fila["anon_id"]
        ruta_wav = ruta_base / "audio" / f"{id_llamada}.wav"
        ruta_json = ruta_base / "turns" / f"{id_llamada}.json"
        
        try:
            feats = extraer_features(ruta_wav, ruta_json)
            # Agregar metadatos
            feats["anon_id"] = id_llamada
            feats["label"] = fila["label"]
            feats["split"] = fila["split"]
            
            datos_procesados.append(feats)
        except Exception as e:
            print(f"Error procesando {id_llamada}: {e}")
            
        if (idx + 1) % 50 == 0:
            print(f"  Procesadas {idx + 1}/{len(manifest)} llamadas...")
            
    df_features = pd.DataFrame(datos_procesados)
    ruta_salida = ruta_base / "features.csv"
    df_features.to_csv(ruta_salida, index=False)
    print(f"\n¡Listo! Matriz guardada en {ruta_salida}")

if __name__ == "__main__":
    # Ajusta la ruta a tu carpeta de datos
    RUTA_DATOS = "/home/aaronhero/Workspace/HackMty2026/Altur"
    procesar_dataset(RUTA_DATOS)