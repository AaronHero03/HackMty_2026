import warnings
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# Carga condicional de modelos
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# Importar pipeline del proyecto
from src.fase1.audio_base import procesar_audio_base, recortar_voz_activa
from src.fase2.acustica import aplicar_filtro_pasabanda, extraer_metricas_acusticas
from src.fase3.conversacional import extraer_metricas_tiempo
from src.tools.vad import fusionar_turnos, generar_turnos_vad

# Definición de rutas base del proyecto
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ_PROYECTO / "models"
DIR_MUESTRAS_DEF = RAIZ_PROYECTO / "Altur_Data" / "audio_augmented"
RUTA_MANIFEST_DEF = RAIZ_PROYECTO / "Altur_Data" / "manifest.csv"


def cargar_manifest_ground_truth(ruta_manifest: Path) -> dict:
    """Carga el manifest.csv y devuelve un mapa anon_id -> label normalizado."""
    if not ruta_manifest.exists():
        print(f"⚠️ Manifiesto no encontrado en: {ruta_manifest.resolve()}")
        return {}

    df = pd.read_csv(ruta_manifest)
    ground_truth = {}
import xgboost as xgb
import pandas as pd
import glob
import os
import sys
from pathlib import Path
from pymongo import MongoClient
from datetime import datetime, timezone

# Agrega la carpeta raíz (HackMty_2026) al PATH de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa tus módulos de procesamiento (ajusta las rutas según tu estructura)
from src.fase1.main import procesar_audio_base, recortar_voz_activa
from src.fase2.fase2_acustica import aplicar_filtro_pasabanda, extraer_metricas_acusticas
from src.fase3.fase3_conversacional import extraer_metricas_tiempo
from src.tools.vad import generar_turnos_vad, fusionar_turnos # Asegúrate de tener la función de fusión aquí

MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://adminAltur:adminaltur67@clusteraltur.6yjrmxp.mongodb.net/?appName=ClusterAltur")
cliente_mongo = MongoClient(MONGO_URI) 
db = cliente_mongo["proyectoAltur"]
coleccion = db["historial_predicciones"]

def probar_audios_locales(directorio_muestras, ruta_modelo="modelo_xgboost_altur_v4.json"):
    # 1. Cargar el modelo XGBoost
    print(f"Cargando modelo desde {ruta_modelo}...")
    modelo = xgb.XGBClassifier()
    modelo.load_model(ruta_modelo)
    
    # 2. Buscar todos los archivos WAV en la carpeta de muestras
    archivos_wav = glob.glob(f"{directorio_muestras}/*.wav")
    
    for _, row in df.iterrows():
        anon_id = str(row["anon_id"]).strip()
        label = str(row["label"]).strip().lower()
        # Normalizar etiqueta
        label_norm = "synthetic" if "synth" in label else "human"
        ground_truth[anon_id] = label_norm

    return ground_truth


def cargar_modelo_universal(ruta_modelo: Path):
    """Carga modelos guardados en formatos .pkl, .joblib, .json o .cbm."""
    ext = ruta_modelo.suffix.lower()

    if ext in [".pkl", ".joblib"]:
        return joblib.load(ruta_modelo)
    elif ext == ".json":
        if xgb is None:
            raise ImportError("xgboost no está instalado en el entorno virtual.")
        modelo = xgb.XGBClassifier()
        modelo.load_model(str(ruta_modelo))
        return modelo
    elif ext == ".cbm":
        if CatBoostClassifier is None:
            raise ImportError("catboost no está instalado en el entorno virtual.")
        modelo = CatBoostClassifier()
        modelo.load_model(str(ruta_modelo))
        return modelo
    else:
        raise ValueError(f"Formato de modelo no soportado: {ext}")


def obtener_probabilidad_sintetico(modelo, df_features: pd.DataFrame) -> float:
    """Extrae la probabilidad predictiva para la clase sintética (1)."""
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


def probar_audios_locales(
    directorio_muestras: Path = DIR_MUESTRAS_DEF,
    nombre_modelo: str = "modelo_xgboost_altur_v4.json",
    ruta_modelo: Path = None,
    ruta_manifest: Path = RUTA_MANIFEST_DEF,
    umbral_ia: float = 0.40,
):
    """Ejecuta la extracción e inferencia evaluando cada audio contra manifest.csv."""
    # Resolver la ruta del modelo
    if ruta_modelo is not None:
        ruta_final = Path(ruta_modelo)
    else:
        ruta_final = Path(nombre_modelo)
        if not ruta_final.is_absolute() and not ruta_final.exists():
            ruta_final = DIR_MODELOS / nombre_modelo

    directorio_muestras = Path(directorio_muestras)

    if not ruta_final.exists():
        print(f"❌ No existe el archivo de modelo en: {ruta_final.resolve()}")
        return

    print(f"🤖 Cargando modelo: {ruta_final.name}")
    modelo = cargar_modelo_universal(ruta_final)

    labels_reales = cargar_manifest_ground_truth(Path(ruta_manifest))
    archivos_wav = list(directorio_muestras.glob("*.wav"))

    if not archivos_wav:
        print(f"⚠️ No se encontraron archivos .wav en {directorio_muestras.resolve()}")
        return

    print(f"\n🚀 Inferencia iniciada sobre {len(archivos_wav)} muestras...\n" + "-" * 70)

    y_true = []
    y_pred = []

    for ruta_audio in archivos_wav:
        # Extraer el id limpiando posibles sufijos de aumentación
        anon_id = (
            ruta_audio.stem.replace("_orig", "")
            .replace("_agudo", "")
            .replace("_grave", "")
            .replace("_ruido", "")
        )

        nombre_archivo = Path(ruta_audio).name
        nombre_sin_extension = nombre_archivo.replace('.wav', '')
        try:
            # 1. Pipeline de extracción
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

            # 2. Inferencia
            df_inferencia = pd.DataFrame([{**metricas_ac, **metricas_tiempo}])
            prob_ia = obtener_probabilidad_sintetico(modelo, df_inferencia)

            veredicto_pred = "synthetic" if prob_ia >= umbral_ia else "human"
            
            # 3. Verificación con Manifest
            label_real = labels_reales.get(anon_id, None)

            print(f"🎙️ {ruta_audio.name}")
            print(f"   Predicción : {veredicto_pred.upper()} (Prob IA: {prob_ia:.4f})")

            if label_real:
                y_true.append(label_real)
                y_pred.append(veredicto_pred)
                es_correcto = veredicto_pred == label_real
                status = "✅ CORRECTO" if es_correcto else "❌ ERROR"
                print(f"   Real (GT)  : {label_real.upper()} | {status}\n")
            else:
                print("   Real (GT)  : ⚠️ NO REGISTRADO EN MANIFEST\n")

                veredicto = "👤 Human"
            
            confianza = prob_ia * 100 if prediccion == 1 else (1 - prob_ia) * 100

            documento = { #objeto de mongodb
                "id": nombre_sin_extension,
                "resultado": prediccion,
                "confianza": prob_ia, 
                "fecha_analisis": datetime.now(timezone.utc)
            }
            
            try: #publicacion a la db
                coleccion.insert_one(documento)
                estado_db = "💾 Guardado en Atlas"
                print(f"✅ Guardado en DB: {nombre_archivo}")
            except Exception as e:
                estado_db = f"⚠️ Error DB: {e}"
            
            print(f" Vector generado: {fila}")
            print(f"🎙️ {nombre_archivo}")
            print(f"   Veredicto: {veredicto}")
            print(f"   Confianza: {confianza:.2f}%")
            print(f"   -> Latencia detectada: {fila.get('latencia_mediana', 0):.2f}s | Vocoder Peak: {fila.get('vocoder_periodicity_peak', 0):.2f}\n")
            
        except Exception as e:
            print(f"⚠️ Error procesando {ruta_audio.name}: {e}\n")

# 4. Reporte de Métricas Finales
    if y_true:
        print("=" * 70)
        print(f"📊 REPORTE DE EVALUACIÓN GENERAL | Modelo: {ruta_final.name} (VS MANIFEST)")
        print("=" * 70)
        print(classification_report(y_true, y_pred, target_names=["human", "synthetic"]))
        print("Matriz de Confusión:")
        print(confusion_matrix(y_true, y_pred, labels=["human", "synthetic"]))


if __name__ == "__main__":
    probar_audios_locales(
        directorio_muestras=RAIZ_PROYECTO / "Altur_Data" / "audio_augmented",
        nombre_modelo="modelo_catboost_altur.cbm",
        ruta_manifest=RUTA_MANIFEST_DEF,
    )