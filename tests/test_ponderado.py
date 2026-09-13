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


# Modelos y pesos asignados basados en su desempeño individual previo
MODELOS_Y_PESOS = {
    "CatBoost": {"archivo": "modelo_catboost_altur.cbm", "peso": 0.70},
    "RandomForest": {"archivo": "modelo_rf_altur.joblib", "peso": 0.15},
    "XGBoost": {"archivo": "modelo_xgboost_altur_v4.json", "peso": 0.15},
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


def cargar_modelos_ponderados(diccionario_config: dict) -> tuple:
    cargados = {}
    pesos_activos = {}
    
    for alias, config in diccionario_config.items():
        ruta = DIR_MODELOS / config["archivo"]
        if not ruta.exists():
            print(f"⚠️ Modelo omitido (no existe): {ruta.resolve()}")
            continue

        ext = ruta.suffix.lower()
        try:
            if ext in [".pkl", ".joblib"]:
                cargados[alias] = joblib.load(ruta)
                pesos_activos[alias] = config["peso"]
            elif ext == ".json" and xgb is not None:
                m = xgb.XGBClassifier()
                m.load_model(str(ruta))
                cargados[alias] = m
                pesos_activos[alias] = config["peso"]
            elif ext == ".cbm" and CatBoostClassifier is not None:
                m = CatBoostClassifier()
                m.load_model(str(ruta))
                cargados[alias] = m
                pesos_activos[alias] = config["peso"]
        except Exception as e:
            print(f"⚠️ Error cargando {alias} desde {config['archivo']}: {e}")

    # Re-normalizar pesos si algún modelo no se pudo cargar
    if pesos_activos:
        suma_pesos = sum(pesos_activos.values())
        pesos_activos = {k: v / suma_pesos for k, v in pesos_activos.items()}

    return cargados, pesos_activos


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


def evaluar_ensamble_ponderado(
    directorio_muestras: Path = DIR_MUESTRAS_DEF,
    ruta_manifest: Path = RUTA_MANIFEST_DEF,
    umbral_ia: float = 0.40,
):
    modelos, pesos = cargar_modelos_ponderados(MODELOS_Y_PESOS)
    labels_reales = cargar_manifest_ground_truth(Path(ruta_manifest))

    if not modelos:
        print("❌ No se pudo cargar ninguno de los modelos configurados.")
        return

    print(f"🌟 Ensamble Ponderado Configurado:")
    for alias, p in pesos.items():
        print(f"   ├── {alias:<15}: Peso = {p:.2f}")

    archivos_wav = list(Path(directorio_muestras).glob("*.wav"))
    if not archivos_wav:
        print(f"⚠️ No hay archivos .wav en {Path(directorio_muestras).resolve()}")
        return

    print(f"\n🚀 Evaluando Soft Voting Ponderado sobre {len(archivos_wav)} muestras...\n" + "=" * 75)

    y_true = []
    y_pred_weighted = []

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

            # Soft Voting Ponderado (Suma ponderada de probabilidades)
            prob_ponderada = sum(probs[alias] * pesos[alias] for alias in modelos.keys())
            veredicto_weighted = "synthetic" if prob_ponderada >= umbral_ia else "human"

            label_real = labels_reales.get(anon_id, None)

            print(f"🎙️ Audio: {ruta_audio.name}")
            for alias, p in probs.items():
                print(f"   ├── {alias:<15}: Prob IA = {p:.4f} (Peso: {pesos[alias]:.2f})")

            print(f"   ⚖️ SOFT PONDERADO: {veredicto_weighted.upper()} (Prob Final: {prob_ponderada:.4f})")

            if label_real:
                y_true.append(label_real)
                y_pred_weighted.append(veredicto_weighted)
                print(f"   🎯 REAL (GT)     : {label_real.upper()}\n" + "-" * 75)
            else:
                print("   ⚠️ REAL (GT)     : NO REGISTRADO EN MANIFEST\n" + "-" * 75)

        except Exception as e:
            print(f"⚠️ Error procesando {ruta_audio.name}: {e}\n")

    if y_true:
        print("\n" + "=" * 75)
        print("📊 REPORTE FINAL - SOFT VOTING PONDERADO (VS MANIFEST)")
        print("=" * 75)
        print(classification_report(y_true, y_pred_weighted, target_names=["human", "synthetic"]))
        print("Matriz de Confusión (Ponderado):")
        print(confusion_matrix(y_true, y_pred_weighted, labels=["human", "synthetic"]))


if __name__ == "__main__":
    evaluar_ensamble_ponderado()