import sys
import os
import base64
import tempfile
from pathlib import Path
import pandas as pd
import xgboost as xgb
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# Evitar el ModuleNotFoundError de 'src'
RAIZ = Path(__file__).resolve().parent
sys.path.append(str(RAIZ))

from src.fase1.main import procesar_audio_base, recortar_voz_activa
from src.fase2.fase2_acustica import aplicar_filtro_pasabanda, extraer_metricas_acusticas
from src.fase3.fase3_conversacional import extraer_metricas_tiempo
from src.tools.vad import generar_turnos_vad, fusionar_turnos

# Modelo junto a este archivo, para no depender de la carpeta desde donde se arranca el servidor
RUTA_MODELO = RAIZ / "modelo_xgboost_altur_v4.json"

# Probabilidad de IA a partir de la cual se responde sintético
UMBRAL = float(os.environ.get("UMBRAL", "0.40"))

# Páginas que pueden llamar a la API desde el navegador, separadas por comas
ORIGENES_CORS = os.environ.get("CORS_ORIGINS", "https://fernandox89.github.io").split(",")

# Configuración MongoDB Atlas: la cadena de conexión solo viene de la variable MONGO_URI, nunca del código.
# Sin MONGO_URI la API responde igual, solo que no guarda historial.
coleccion = None
MONGO_URI = os.environ.get("MONGO_URI")
if MONGO_URI:
    try:
        # Tiempos cortos: si Atlas no responde, el guardado falla rápido
        cliente_mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000, socketTimeoutMS=3000)
        coleccion = cliente_mongo["proyectoAltur"]["historial_predicciones"]
    except Exception as e:
        print(f"⚠️ MongoDB desactivado: MONGO_URI no es válida ({e})")

# Diccionario para mantener el modelo en memoria RAM
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Cargando modelo XGBoost...")
    modelo = xgb.XGBClassifier()
    modelo.load_model(str(RUTA_MODELO))
    ml_models["xgboost"] = modelo
    print(f"✅ Modelo cargado y listo. Umbral: {UMBRAL}. MongoDB: {'sí' if coleccion is not None else 'no'}.")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGENES_CORS, allow_methods=["POST"], allow_headers=["Content-Type"])

# Esquemas de datos para el juez
class DetectRequest(BaseModel):
    call_id: str
    audio_base64: str
    sample_rate: int = 8000
    channels: int = 2

class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float

def guardar_en_mongo(documento):
    # Corre después de responder: si falla, solo queda en el log y la respuesta al juez no cambia
    try:
        coleccion.insert_one(documento)
    except Exception as e:
        print(f"⚠️ No se guardó {documento['id']} en MongoDB: {e}")

@app.get("/health")
def health():
    return {"ok": "xgboost" in ml_models}

@app.post("/detect", response_model=DetectResponse)
def detect_call(payload: DetectRequest, tareas: BackgroundTasks):
    modelo = ml_models.get("xgboost")
    ruta_audio = None

    try:
        # --- 1. DECODIFICAR Y GUARDAR TEMPORALMENTE ---
        audio_bytes = base64.b64decode(payload.audio_base64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            ruta_audio = tmp_file.name

        # --- 2. PIPELINE EXACTO DE TU TEST.PY ---
        datos_turnos = generar_turnos_vad(ruta_audio)
        turnos_limpios = fusionar_turnos(datos_turnos["turns"], max_pausa_s=0.5)

        turnos_llamador = [t for t in turnos_limpios if t["channel"] == 0]
        turnos_agente = [t for t in turnos_limpios if t["channel"] == 1]

        y_c0_norm, sr = procesar_audio_base(ruta_audio)
        voz_recortada = recortar_voz_activa(y_c0_norm, turnos_llamador, sr)

        voz_filtrada = aplicar_filtro_pasabanda(voz_recortada, sr)
        metricas_ac = extraer_metricas_acusticas(voz_filtrada, sr)
        metricas_tiempo = extraer_metricas_tiempo(
            tramos_agente=turnos_agente,
            tramos_llamante=turnos_llamador
        )

        # --- 3. INFERENCIA ---
        fila = {**metricas_ac, **metricas_tiempo}
        df_inferencia = pd.DataFrame([fila])

        columnas_esperadas = modelo.feature_names_in_
        for col in columnas_esperadas:
            if col not in df_inferencia.columns:
                df_inferencia[col] = float('nan')

        df_final = df_inferencia[columnas_esperadas]

        prob_ia = float(modelo.predict_proba(df_final)[0][1])

        # Tu misma regla de umbral
        is_synthetic = prob_ia >= UMBRAL

        # El juez lee confidence como la seguridad en la respuesta dada:
        # probabilidad de IA = confidence si is_synthetic es true, y 1 - confidence si es false
        confidence = prob_ia if is_synthetic else 1.0 - prob_ia

    except Exception as e:
        print(f"⚠️ Error procesando {payload.call_id}: {e}")
        # Si algo explota (ej. audio mudo), regresamos humano sin seguridad (0.5) para no tumbar el script del juez
        return DetectResponse(is_synthetic=False, confidence=0.5)

    finally:
        if ruta_audio and os.path.exists(ruta_audio):
            os.remove(ruta_audio)

    # --- 4. GUARDADO EN MONGODB (después de responder al juez) ---
    if coleccion is not None:
        tareas.add_task(guardar_en_mongo, {
            "id": payload.call_id,
            "resultado": 1 if is_synthetic else 0,
            "confianza": prob_ia,
            "latencia_mediana": fila.get('latencia_mediana', 0),
            "vocoder_peak": fila.get('vocoder_periodicity_peak', 0),
            "fecha_analisis": datetime.now(timezone.utc)
        })

    # --- 5. RESPUESTA AL JUEZ ---
    return DetectResponse(is_synthetic=is_synthetic, confidence=confidence)
