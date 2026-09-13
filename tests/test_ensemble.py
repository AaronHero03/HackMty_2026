from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from src.fase1.audio_base import procesar_audio_base, recortar_voz_activa
from src.fase2.acustica import aplicar_filtro_pasabanda, extraer_metricas_acusticas
from src.fase3.conversacional import extraer_metricas_tiempo
from src.tools.vad import fusionar_turnos, generar_turnos_vad

# Rutas principales
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ_PROYECTO / "models"
DIR_MUESTRAS_DEF = RAIZ_PROYECTO / "Altur_Data" / "audio_augmented"
RUTA_MANIFEST_DEF = RAIZ_PROYECTO / "Altur_Data" / "manifest.csv"


# DEFINICIÓN DE MODELOS REPRESENTATIVOS (Ajusta los nombres de archivo)
MODELOS_REPRESENTATIVOS = {
    "RF": "modelo_rf_altur.joblib",
    "CatBoost": "modelo_catboost_altur.cbm",
    "XGBoostv4": "modelo_xgboost_altur_v4.json"
}


def cargar_manifest_ground_truth(ruta_manifest: Path) -> dict:
    if not ruta_manifest.exists():
        print(f"⚠️ Manifiesto no encontrado en: {ruta_manifest.resolve()}")
        return {}

    df = pd.read_csv(ruta_manifest)
    ground_truth = {}
    for _, row in df.iterrows():
        anon_id = str(row["anon_id"]).strip()
        label = str(row["label"]).strip().lower()
        ground_truth[anon_id] = "synthetic" if "synth" in label else "human"
    return ground_truth


def cargar_modelos_clave(diccionario_modelos: dict) -> dict:
    cargados = {}
    for alias, nombre_archivo in diccionario_modelos.items():
        ruta = DIR_MODELOS / nombre_archivo
        if not ruta.exists():
            print(f"⚠️ Modelo clave omitido (no existe): {ruta.resolve()}")
            continue

        ext = ruta.suffix.lower()
        try:
            if ext in [".pkl", ".joblib"]:
                cargados[alias] = joblib.load(ruta)
            elif ext == ".json" and xgb is not None:
                m = xgb.XGBClassifier()
                m.load_model(str(ruta))
                cargados[alias] = m
            elif ext == ".cbm" and CatBoostClassifier is not None:
                m = CatBoostClassifier()
                m.load_model(str(ruta))
                cargados[alias] = m
        except Exception as e:
            print(f"⚠️ Error cargando {alias} desde {nombre_archivo}: {e}")

    return cargados


def extraer_probabilidad(modelo, df_features: pd.DataFrame) -> float:
    df_temp = df_features.copy()
    if hasattr(modelo, "feature_names_in_"):
        cols = modelo.feature_names_in_
        for col in cols:
            if col not in df_temp.columns:
                df_temp[col] = float("nan")
        df_temp = df_temp[cols]

    if hasattr(modelo, "predict_proba"):
        probs = modelo.predict_proba(df_temp)[0]
        return float(probs[1]) if len(probs) > 1 else float(probs[0])
    else:
        pred = modelo.predict(df_temp)[0]
        return 1.0 if pred == 1 else 0.0


def evaluar_ensambles_representativos(
    directorio_muestras: Path = DIR_MUESTRAS_DEF,
    ruta_manifest: Path = RUTA_MANIFEST_DEF,
    umbral_ia: float = 0.40,
):
    modelos = cargar_modelos_clave(MODELOS_REPRESENTATIVOS)
    labels_reales = cargar_manifest_ground_truth(Path(ruta_manifest))

    if not modelos:
        print("❌ No se pudo cargar ninguno de los modelos representativos.")
        print(f"Asegúrate de que existen en la carpeta: {DIR_MODELOS.resolve()}")
        return

    print(f"🌟 Ensamble configurado con ({len(modelos)} modelos): {', '.join(modelos.keys())}")
    archivos_wav = list(Path(directorio_muestras).glob("*.wav"))

    if not archivos_wav:
        print(f"⚠️ No hay archivos .wav en {Path(directorio_muestras).resolve()}")
        return

    print(f"\n🚀 Evaluando Hard & Soft Ensembles sobre {len(archivos_wav)} muestras...\n" + "=" * 75)

    y_true = []
    y_pred_soft = []
    y_pred_hard = []

    for ruta_audio in archivos_wav:
        anon_id = (
            ruta_audio.stem.replace("_orig", "")
            .replace("_agudo", "")
            .replace("_grave", "")
            .replace("_ruido", "")
        )

        try:
            datos_turnos = generar_turnos_vad(str(ruta_audio))
            turnos_limpios = fusionar_turnos(datos_turnos["turns"], max_pausa_s=0.5)

            turnos_llamador = [t for t in turnos_limpios if t["channel"] == 0]
            turnos_agente = [t for t in turnos_limpios if t["channel"] == 1]

            y_c0_norm, sr = procesar_audio_base(str(ruta_audio))
            voz_recortada = recortar_voz_activa(y_c0_norm, turnos_llamador, sr)

            voz_filtrada = aplicar_filtro_pasabanda(voz_recortada, sr)
            metricas_ac = extraer_metricas_acusticas(voz_filtrada, sr)
            metricas_tiempo = extraer_metricas_tiempo(
                tramos_agente=turnos_agente, tramos_llamante=turnos_llamador
            )

            df_inferencia = pd.DataFrame([{**metricas_ac, **metricas_tiempo}])

            probs = {}
            for alias, mod in modelos.items():
                probs[alias] = extraer_probabilidad(mod, df_inferencia)

            # Soft Voting (Promedio continuo)
            prob_promedio = float(np.mean(list(probs.values())))
            veredicto_soft = "synthetic" if prob_promedio >= umbral_ia else "human"

            # Hard Voting (Conteo por mayoría binaria)
            votos_ia = sum(1 for p in probs.values() if p >= umbral_ia)
            veredicto_hard = "synthetic" if votos_ia > (len(probs) / 2) else "human"

            label_real = labels_reales.get(anon_id, None)

            print(f"🎙️ Audio: {ruta_audio.name}")
            for alias, p in probs.items():
                print(f"   ├── {alias:<15}: Prob IA = {p:.4f}")

            print(f"   📊 SOFT ENSEMBLE : {veredicto_soft.upper()} (Prob Promedio: {prob_promedio:.4f})")
            print(f"   🗳️ HARD ENSEMBLE : {veredicto_hard.upper()} ({votos_ia}/{len(probs)} votos IA)")

            if label_real:
                y_true.append(label_real)
                y_pred_soft.append(veredicto_soft)
                y_pred_hard.append(veredicto_hard)
                print(f"   🎯 REAL (GT)     : {label_real.upper()}\n" + "-" * 75)
            else:
                print("   ⚠️ REAL (GT)     : NO REGISTRADO EN MANIFEST\n" + "-" * 75)

        except Exception as e:
            print(f"⚠️ Error procesando {ruta_audio.name}: {e}\n")

    if y_true:
        print("\n" + "=" * 75)
        print("📊 REPORTE FINAL - SOFT VOTING ENSEMBLE (VS MANIFEST)")
        print("=" * 75)
        print(classification_report(y_true, y_pred_soft, target_names=["human", "synthetic"]))
        print("Matriz de Confusión (Soft):")
        print(confusion_matrix(y_true, y_pred_soft, labels=["human", "synthetic"]))

        print("\n" + "=" * 75)
        print("📊 REPORTE FINAL - HARD VOTING ENSEMBLE (VS MANIFEST)")
        print("=" * 75)
        print(classification_report(y_true, y_pred_hard, target_names=["human", "synthetic"]))
        print("Matriz de Confusión (Hard):")
        print(confusion_matrix(y_true, y_pred_hard, labels=["human", "synthetic"]))


if __name__ == "__main__":
    evaluar_ensambles_representativos()