from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile

# Determina la raíz del proyecto dinámicamente (si está en 'scripts/')
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ_PROYECTO / "Altur_Data"


def guardar_audio_seguro(ruta: Path, audio: np.ndarray, sr: int):
    if audio.ndim > 1 and audio.shape[0] < audio.shape[1]:
        audio = audio.T
    audio_limpio = np.ascontiguousarray(audio, dtype=np.float32)
    wavfile.write(ruta, sr, audio_limpio)


def agregar_ruido_gaussiano(audio: np.ndarray, snr_db: float = 28.0) -> np.ndarray:
    potencia_senal = np.mean(audio**2)
    if potencia_senal == 0:
        return audio
    potencia_ruido = potencia_senal / (10 ** (snr_db / 10))
    ruido = np.random.normal(0, np.sqrt(potencia_ruido), audio.shape)
    return (audio + ruido).astype(np.float32)


def cambiar_ganancia(audio: np.ndarray, factor: float) -> np.ndarray:
    return (audio * factor).astype(np.float32)


def aumentar_dataset_segun_manifest(ruta_base: Path):
    manifest_path = ruta_base / "manifest.csv"
    dir_audio = ruta_base / "audio"
    dir_salida = ruta_base / "audio_augmented"

    if not manifest_path.exists():
        print(f"❌ Error: No se encontró manifest.csv en {ruta_base}")
        return

    dir_salida.mkdir(parents=True, exist_ok=True)
    df_manifest = pd.read_csv(manifest_path)
    df_train = df_manifest[df_manifest["split"] == "train"]

    print(f"🚀 Generando aumentación en: {dir_salida.resolve()}")

    cont = 0
    for idx, fila in df_train.iterrows():
        anon_id = fila["anon_id"]
        ruta_wav = dir_audio / f"{anon_id}.wav"

        if not ruta_wav.exists():
            continue

        y, sr = librosa.load(ruta_wav, sr=None, mono=False)

        guardar_audio_seguro(dir_salida / f"{anon_id}_orig.wav", y, sr)

        y_agudo = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        guardar_audio_seguro(dir_salida / f"{anon_id}_agudo.wav", y_agudo, sr)

        y_grave = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
        guardar_audio_seguro(dir_salida / f"{anon_id}_grave.wav", y_grave, sr)

        y_ruido = cambiar_ganancia(y, 0.85)
        y_ruido = agregar_ruido_gaussiano(y_ruido, snr_db=28.0)
        guardar_audio_seguro(dir_salida / f"{anon_id}_ruido.wav", y_ruido, sr)

        cont += 4

    print(f"\n✅ Total de audios aumentados generados: {cont}")


if __name__ == "__main__":
    aumentar_dataset_segun_manifest(RUTA_DATOS)