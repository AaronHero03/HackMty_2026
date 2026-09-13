import json
import os
from pathlib import Path
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

def procesar_con_diarizacion(ruta_audio, ruta_json_salida, ruta_audio_salida, hf_token):
    # Cargar el modelo pre-entrenado de diarización
    pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token
    )
    
    # Ejecutar la inteligencia sobre la nota de voz
    print(f"Analizando hablantes en {Path(ruta_audio).name}...")
    diarization = pipeline(str(ruta_audio))
    anotacion = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization
    
    y_original, sr = sf.read(str(ruta_audio))
    y_mono = y_original[:, 0] if len(y_original.shape) > 1 else y_original
    
    pista_c0 = np.zeros_like(y_mono)
    pista_c1 = np.zeros_like(y_mono)
    
    hablantes_mapeo = {}
    turnos_finales = []
    
    # Recorrer cada intervención detectada por la IA
    for turn, _, speaker in anotacion.itertracks(yield_label=True):
        if speaker not in hablantes_mapeo:
            # Asigna el primer hablante al Canal 0 y el segundo al Canal 1
            hablantes_mapeo[speaker] = len(hablantes_mapeo) % 2
            
        canal = hablantes_mapeo[speaker]
        start, end = turn.start, turn.end
        
        turnos_finales.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "channel": canal
        })
        
        # Aislar físicamente el audio en su pista correspondiente
        ini_idx = int(start * sr)
        fin_idx = int(end * sr)
        
        if canal == 0:
            pista_c0[ini_idx:fin_idx] = y_mono[ini_idx:fin_idx]
        else:
            pista_c1[ini_idx:fin_idx] = y_mono[ini_idx:fin_idx]
            
    # Guardar el JSON adaptado al contrato de Altur
    with open(ruta_json_salida, "w", encoding="utf-8") as f:
        json.dump({"turns": turnos_finales}, f, indent=4, ensure_ascii=False)
        
    # Guardar el WAV estéreo real para que tu Fase 1 lo lea perfecto
    y_stereo = np.column_stack((pista_c0, pista_c1))
    sf.write(str(ruta_audio_salida), y_stereo, sr)
    print(f"✅ Procesado con éxito: {len(turnos_finales)} turnos detectados y canales separados.")


if __name__ == "__main__":
    TOKEN_HF = os.getenv("HF_TOKEN")
    if not TOKEN_HF:
        raise EnvironmentError("No se encontró 'HF_TOKEN' en el archivo .env")
    
    _raiz = Path(__file__).resolve().parents[2]
    dir_entrada = _raiz / "tests" / "dataset"
    dir_salida = _raiz / "tests" / "dataset" / "stereo"
    
    # Crear la carpeta de salida si no existe
    dir_salida.mkdir(parents=True, exist_ok=True)
    
    # Buscar todos los archivos .wav en la carpeta de entrada
    # (puedes cambiar o agregar extensiones como .mp3 usando or)
    archivos_audio = list(dir_entrada.glob("*.wav"))
    
    if not archivos_audio:
        print(f"⚠️ No se encontraron archivos .wav en {dir_entrada}")
    
    for archivo_audio in archivos_audio:
        # Definir nombres únicos de salida basados en el archivo original
        nombre_base = archivo_audio.stem
        ruta_json = dir_salida / f"{nombre_base}_turns.json"
        ruta_stereo = dir_salida / f"{nombre_base}_stereo.wav"
        
        print(f"\n--- Procesando archivo: {archivo_audio.name} ---")
        try:
            procesar_con_diarizacion(
                ruta_audio=str(archivo_audio),
                ruta_json_salida=str(ruta_json),
                ruta_audio_salida=str(ruta_stereo),
                hf_token=TOKEN_HF
            )
        except Exception as e:
            print(f"❌ Error procesando {archivo_audio.name}: {e}")