from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

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
DIR_MUESTRAS_DEF = RAIZ_PROYECTO / "Altur_Data" / "audio"
RUTA_MANIFEST_DEF = RAIZ_PROYECTO / "Altur_Data" / "manifest.csv"
DIR_GRAFICAS = RAIZ_PROYECTO / "reports" / "graficas"
DIR_GRAFICAS.mkdir(parents=True, exist_ok=True)


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


def cargar_todos_los_modelos() -> tuple:
    cargados = {}
    pesos_activos = {}
    
    for alias, config in MODELOS_Y_PESOS.items():
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
            print(f"⚠️ Error cargando {alias}: {e}")

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


def graficar_comparativa_roc(y_true_bin, probs_catboost, probs_ensemble, nombre_archivo="comparativa_curvas_roc.png"):
    auc_cb = roc_auc_score(y_true_bin, probs_catboost)
    auc_ens = roc_auc_score(y_true_bin, probs_ensemble)

    fpr_cb, tpr_cb, _ = roc_curve(y_true_bin, probs_catboost)
    fpr_ens, tpr_ens, _ = roc_curve(y_true_bin, probs_ensemble)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_cb, tpr_cb, color="forestgreen", lw=2, label=f"CatBoost Individual (AUC = {auc_cb:.4f})")
    plt.plot(fpr_ens, tpr_ens, color="darkorange", lw=2, label=f"Soft Ensemble Ponderado (AUC = {auc_ens:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Azar (0.50)")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=11)
    plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=11)
    plt.title("Comparativa Curvas ROC: CatBoost vs Ensamble Ponderado", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    ruta_guardado = DIR_GRAFICAS / nombre_archivo
    plt.savefig(ruta_guardado, dpi=300)
    plt.close()
    print(f"📈 Gráfica comparativa guardada: {ruta_guardado.resolve()}")


def evaluar_comparativa_completa(
    directorio_muestras: Path = DIR_MUESTRAS_DEF,
    ruta_manifest: Path = RUTA_MANIFEST_DEF,
    umbral_ia: float = 0.40,
):
    modelos, pesos = cargar_todos_los_modelos()
    labels_reales = cargar_manifest_ground_truth(Path(ruta_manifest))

    if "CatBoost" not in modelos:
        print("❌ El modelo CatBoost es obligatorio para esta comparativa.")
        return

    archivos_wav = list(Path(directorio_muestras).glob("*.wav"))
    if not archivos_wav:
        print(f"⚠️ No hay archivos .wav en {Path(directorio_muestras).resolve()}")
        return

    print(f"\n🚀 Extrayendo características de {len(archivos_wav)} audios para evaluación paralela...\n" + "=" * 75)

    y_true = []
    probs_catboost_list = []
    probs_ensemble_list = []

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

            # Probabilidades individuales y del ensamble
            probs_dict = {}
            for alias, mod in modelos.items():
                probs_dict[alias] = extraer_probabilidad(mod, df_inferencia)

            p_cb = probs_dict["CatBoost"]
            p_ens = sum(probs_dict[alias] * pesos[alias] for alias in modelos.keys())

            label_real = labels_reales.get(anon_id, None)
            if label_real:
                y_true.append(label_real)
                probs_catboost_list.append(p_cb)
                probs_ensemble_list.append(p_ens)

        except Exception as e:
            continue

    if y_true:
        y_true_bin = np.array([1 if label == "synthetic" else 0 for label in y_true])
        p_cb_arr = np.array(probs_catboost_list)
        p_ens_arr = np.array(probs_ensemble_list)

        # Predicciones binarias con el umbral
        pred_cb = (p_cb_arr >= umbral_ia).astype(int)
        pred_ens = (p_ens_arr >= umbral_ia).astype(int)

        print("\n" + "=" * 75)
        print("📊 REPORTE 1: CATBOOST INDIVIDUAL")
        print("=" * 75)
        print(classification_report(y_true_bin, pred_cb, target_names=["human", "synthetic"]))
        auc_cb = roc_auc_score(y_true_bin, p_cb_arr)
        print(f"📈 ROC-AUC CatBoost : {auc_cb:.4f}")

        print("\n" + "=" * 75)
        print("📊 REPORTE 2: SOFT VOTING PONDERADO")
        print("=" * 75)
        print(classification_report(y_true_bin, pred_ens, target_names=["human", "synthetic"]))
        auc_ens = roc_auc_score(y_true_bin, p_ens_arr)
        print(f"📈 ROC-AUC Ensamble : {auc_ens:.4f}")

        # Análisis de errores y distancias al umbral
        print("\n" + "=" * 75)
        print("🔍 ANÁLISIS DE MÁRGENES DE ERROR (Cercanía al umbral)")
        print("=" * 75)
        
        err_cb = y_true_bin != pred_cb
        if np.sum(err_cb) > 0:
            dist_cb = np.abs(p_cb_arr[err_cb] - umbral_ia)
            print(f"CatBoost Errores ({np.sum(err_cb)}): Distancia promedio = {np.mean(dist_cb):.4f} | Leves (<0.15) = {np.sum(dist_cb < 0.15)}")

        err_ens = y_true_bin != pred_ens
        if np.sum(err_ens) > 0:
            dist_ens = np.abs(p_ens_arr[err_ens] - umbral_ia)
            print(f"Ensamble Errores ({np.sum(err_ens)}): Distancia promedio = {np.mean(dist_ens):.4f} | Leves (<0.15) = {np.sum(dist_ens < 0.15)}")

        # Generar Gráficas
        print("\n🎨 Generando gráfica comparativa de curvas ROC...")
        graficar_comparativa_roc(y_true_bin, p_cb_arr, p_ens_arr)
        print("✅ ¡Gráfica de curvas ROC guardada en 'reports/graficas/'!")


if __name__ == "__main__":
    evaluar_comparativa_completa()