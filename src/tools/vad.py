import librosa
import numpy as np
import glob
import json
from pathlib import Path

def generar_turnos_vad(ruta_wav, umbral_db=40, duracion_minima=0.2):
    """
    Lee un audio estéreo a 8kHz y genera el diccionario de turnos detectando la voz activa.
    """
    # Cargar estéreo (mono=False devuelve matriz [canales, muestras])
    y, sr = librosa.load(ruta_wav, sr=8000, mono=False)
    turnos = []
    
    # Asegurar que tiene 2 canales
    if y.ndim == 1:
        y = np.vstack((y, np.zeros_like(y)))

    for canal_idx in [0, 1]:
        # librosa.effects.split devuelve los índices de inicio y fin de tramos con sonido
        tramos = librosa.effects.split(y[canal_idx], top_db=umbral_db)
        
        for tramo in tramos:
            inicio_seg = tramo[0] / sr
            fin_seg = tramo[1] / sr
            
            # Filtramos ruidos microscópicos (ej. respiraciones menores a 0.2s)
            if (fin_seg - inicio_seg) >= duracion_minima:
                turnos.append({
                    "start": inicio_seg,
                    "end": fin_seg,
                    "channel": canal_idx
                })
                
    turnos = [{
        "start": round(float(t["start"]), 3),
        "end": round(float(t["end"]), 3),
        "channel": int(t["channel"]),
    } for t in turnos ]
                
    # Ordenar cronológicamente para que la Fase 3 pueda calcular la latencia correctamente
    turnos = sorted(turnos, key=lambda x: x["start"])
    return {"turns": turnos}

def fusionar_turnos(turnos, max_pausa_s=0.5):
    """
    Une turnos consecutivos del mismo canal si la pausa entre ellos es pequeña.
    """
    if not turnos:
        return []
        
    turnos_limpios = [turnos[0].copy()]
    
    for turno_actual in turnos[1:]:
        turno_anterior = turnos_limpios[-1]
        
        # Si es la misma persona y la pausa es menor al límite
        if (turno_actual["channel"] == turno_anterior["channel"] and 
           (turno_actual["start"] - turno_anterior["end"]) <= max_pausa_s):
            
            # Fusionamos: El fin del turno anterior ahora es el fin del actual
            turno_anterior["end"] = turno_actual["end"]
        else:
            # Si es otra persona o una pausa muy larga, lo agregamos como turno nuevo
            turnos_limpios.append(turno_actual.copy())
            
    return turnos_limpios


def generar_turnos(directorio_muestras):
    archivos_wav = glob.glob(f"{directorio_muestras}/*.wav")
    
    if not archivos_wav:
        print(f"No se encontraron archivos .wav en {directorio_muestras}.")
        return

    print(f"\nProcesando {len(archivos_wav)} muestras con VAD...\n" + "-"*40)

    for ruta_audio in archivos_wav:
        # .stem saca solo el nombre sin la extensión (ej. "call_01bf49059daf")
        nombre_base = Path(ruta_audio).stem
        
        # --- PIPELINE DE PROCESAMIENTO ---
        datos_turnos = generar_turnos_vad(ruta_audio, umbral_db=40)
        turnos_limpios = fusionar_turnos(datos_turnos["turns"], max_pausa_s=1.2)
                
        ruta_turns = Path(directorio_muestras).parent / "turns"
        ruta_turns.mkdir(parents=True, exist_ok=True)
        ruta_json_salida = ruta_turns / f"{nombre_base}.json"

        with open(ruta_json_salida, "w", encoding="utf-8") as archivo:
            # Envolvemos la lista en un diccionario con la llave "turns"
            json.dump({"turns": turnos_limpios}, archivo, indent=4, ensure_ascii=False)