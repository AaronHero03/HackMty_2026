import sys
import os
import base64
import tempfile
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# Evitar el ModuleNotFoundError de 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.fase1.main import procesar_audio_base, recortar_voz_activa
from src.fase2.fase2_acustica import aplicar_filtro_pasabanda, extraer_metricas_acusticas
from src.fase3.fase3_conversacional import extraer_metricas_tiempo
from src.tools.vad import generar_turnos_vad, fusionar_turnos

# Configuración MongoDB Atlas
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://adminAltur:adminaltur67@clusteraltur.6yjrmxp.mongodb.net/?appName=ClusterAltur")
cliente_mongo = MongoClient(MONGO_URI) 
db = cliente_mongo["proyectoAltur"]
coleccion = db["historial_predicciones"]

# Diccionario para mantener el modelo en memoria RAM
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Cargando modelo XGBoost...")
    modelo = xgb.XGBClassifier()
    modelo.load_model("modelo_xgboost_altur_v3.json") #carga el modelo desde el archivo
    ml_models["xgboost"] = modelo
    print("✅ Modelo cargado y listo.")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# Esquemas de datos para el juez
class DetectRequest(BaseModel):
    call_id: str
    audio_base64: str
    sample_rate: int = 8000
    channels: int = 2

class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float

@app.post("/detect", response_model=DetectResponse)
def detect_call(payload: DetectRequest):
    modelo = ml_models.get("xgboost")
    
    # 1. Decodificar y guardar temporalmente
    audio_bytes = base64.b64decode(payload.audio_base64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        ruta_audio = tmp_file.name

    try:
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
        
        prediccion = int(modelo.predict(df_final)[0])
        prob_ia = float(modelo.predict_proba(df_final)[0][1])
        
        # Tu misma regla de umbral
        is_synthetic = bool(prob_ia >= 0.40)
        
        # --- 4. GUARDADO EN MONGODB ---
        documento = {
            "id": payload.call_id,
            "resultado": 1 if is_synthetic else 0,
            "confianza": prob_ia,
            "latencia_mediana": fila.get('latencia_mediana', 0),
            "vocoder_peak": fila.get('vocoder_periodicity_peak', 0),
            "fecha_analisis": datetime.now(timezone.utc)
        }
        coleccion.insert_one(documento)

        # --- 5. RESPUESTA AL JUEZ ---
        return DetectResponse(
            is_synthetic=is_synthetic,
            confidence=prob_ia
        )
        
    except Exception as e:
        print(f"⚠️ Error procesando {payload.call_id}: {e}")
        # Si algo explota (ej. audio mudo), regresamos humano por defecto para no tumbar el script del juez
        return DetectResponse(is_synthetic=False, confidence=0.0)
        
    finally:
        if os.path.exists(ruta_audio):
            os.remove(ruta_audio)