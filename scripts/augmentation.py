import librosa
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path

def guardar_audio_seguro(ruta, audio, sr):
    """Guarda el audio usando scipy para evitar los crashes de soundfile"""
    # Si es estéreo (canales, muestras), lo invertimos para scipy (muestras, canales)
    if audio.ndim > 1 and audio.shape[0] < audio.shape[1]:
        audio = audio.T
    
    # Asegurar memoria contigua en float32
    audio_limpio = np.ascontiguousarray(audio, dtype=np.float32)
    
    # Scipy toma (ruta, frecuencia, datos) y sí soporta PathLib nativo
    wavfile.write(ruta, sr, audio_limpio)

def multiplicar_dataset(directorio_entrada, directorio_salida):
    directorio_entrada = Path(directorio_entrada)
    directorio_salida = Path(directorio_salida)
    directorio_salida.mkdir(parents=True, exist_ok=True)
    
    archivos = list(directorio_entrada.glob("*.wav"))
    print(f"Iniciando Data Augmentation para {len(archivos)} audios...")
    
    for i, ruta in enumerate(archivos):
        nombre = ruta.stem
        
        # Cargar audio
        y, sr = librosa.load(ruta, sr=None, mono=False)
        
        # 1. Guardar original
        guardar_audio_seguro(directorio_salida / f"{nombre}_orig.wav", y, sr)
        
        # 2. Crear clon Agudo (+2 semitonos)
        y_agudo = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        guardar_audio_seguro(directorio_salida / f"{nombre}_agudo.wav", y_agudo, sr)
        
        # 3. Crear clon Grave (-2 semitonos)
        y_grave = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
        guardar_audio_seguro(directorio_salida / f"{nombre}_grave.wav", y_grave, sr)
        
        if (i + 1) % 10 == 0:
            print(f"Procesados {i + 1}/{len(archivos)} audios...")
            
    print("✅ Augmentation completado. ¡A entrenar!")

if __name__ == "__main__":
    source_folder = "/home/aaronhero/Workspace/HackMty2026/Tigres_Del_Sur/data/audio/original"
    result_folder = "/home/aaronhero/Workspace/HackMty2026/Tigres_Del_Sur/data/audio/aumentado"
    
    multiplicar_dataset(source_folder, result_folder)