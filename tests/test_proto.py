import json
from pathlib import Path
import warnings
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")

# Determina la raíz del proyecto dinámicamente (si el script está en 'tests/')
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ_PROYECTO / "Altur_Data"


def normalizar_volumen(y, target_db=-20):
    rms_actual = np.sqrt(np.mean(y**2))
    if rms_actual == 0:
        return y
    target_rms = 10 ** (target_db / 20.0)
    return y * (target_rms / rms_actual)


def extraer_features(ruta_wav, ruta_json):
    features = {}
    sr = 8000

    with open(ruta_json, "r") as f:
        turnos = json.load(f)["turns"]

    c0 = [t for t in turnos if t["channel"] == 0]
    c1 = [t for t in turnos if t["channel"] == 1]

    latencias = []
    for agente_t in c1:
        resp = [
            llam_t["start"] - agente_t["end"]
            for llam_t in c0
            if llam_t["start"] >= agente_t["end"]
        ]
        if resp:
            latencias.append(resp[0])

    features["latencia_mediana"] = np.median(latencias) if latencias else 0.0
    features["latencia_std"] = np.std(latencias) if latencias else 0.0

    crudo, _ = sf.read(ruta_wav)
    y_c0 = (
        crudo[:, 0].astype(np.float32)
        if crudo.ndim > 1
        else crudo.astype(np.float32)
    )
    y_c0 = normalizar_volumen(y_c0)

    tramos_voz = []
    for t in c0:
        ini = int(t["start"] * sr)
        fin = int(t["end"] * sr)
        tramos_voz.extend(y_c0[ini:fin])

    y_voz = np.array(tramos_voz)

    if len(y_voz) < 512:
        for i in range(1, 13):
            features[f"mfcc_mediana_{i}"] = 0.0
            features[f"mfcc_std_{i}"] = 0.0
        return features

    mfccs = librosa.feature.mfcc(
        y=y_voz, sr=sr, n_mfcc=12, fmin=300, fmax=3400
    )

    for i in range(12):
        features[f"mfcc_mediana_{i+1}"] = np.median(mfccs[i])
        features[f"mfcc_std_{i+1}"] = np.std(mfccs[i])

    return features


def procesar_dataset(ruta_base: Path):
    manifest_path = ruta_base / "manifest.csv"
    if not manifest_path.exists():
        print(f"❌ Error: No se encontró manifest.csv en {ruta_base}")
        return

    manifest = pd.read_csv(manifest_path)
    dir_audio_orig = ruta_base / "audio"
    dir_audio_aug = ruta_base / "audio_augmented"
    dir_turns = ruta_base / "turns"

    sufijos_train = ["_orig", "_agudo", "_grave", "_ruido"]
    datos_procesados = []

    print(f"🚀 Procesando datos desde: {ruta_base.resolve()}")

    for idx, fila in manifest.iterrows():
        id_llamada = fila["anon_id"]
        split = fila["split"]
        label = fila["label"]
        ruta_json = dir_turns / f"{id_llamada}.json"

        if not ruta_json.exists():
            continue

        if split == "train":
            for sufijo in sufijos_train:
                id_variante = f"{id_llamada}{sufijo}"
                ruta_wav = dir_audio_aug / f"{id_variante}.wav"

                if not ruta_wav.exists():
                    continue

                try:
                    feats = extraer_features(ruta_wav, ruta_json)
                    feats["anon_id"] = id_variante
                    feats["label"] = label
                    feats["split"] = "train"
                    datos_procesados.append(feats)
                except Exception as e:
                    print(f"⚠️ Error en {id_variante}: {e}")
        else:
            ruta_wav = dir_audio_orig / f"{id_llamada}.wav"
            if not ruta_wav.exists():
                continue

            try:
                feats = extraer_features(ruta_wav, ruta_json)
                feats["anon_id"] = id_llamada
                feats["label"] = label
                feats["split"] = split
                datos_procesados.append(feats)
            except Exception as e:
                print(f"⚠️ Error en {id_llamada}: {e}")

    df_features = pd.DataFrame(datos_procesados)
    ruta_salida = ruta_base / "features.csv"
    df_features.to_csv(ruta_salida, index=False)
    print(f"\n✅ Matriz guardada en: {ruta_salida.resolve()}")


if __name__ == "__main__":
    procesar_dataset(RUTA_DATOS)