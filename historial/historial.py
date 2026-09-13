"""Historial del equipo: lee de MongoDB lo que guarda la API (/detect) para mostrarlo en la página.

Es un servicio aparte de la API del juez: si este falla, /detect sigue funcionando.
Solo lee; no guarda ni borra nada.
"""
import math
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import DESCENDING, MongoClient

# Páginas que pueden leer el historial desde el navegador, separadas por comas
ORIGENES_CORS = os.environ.get("CORS_ORIGINS", "https://fernandox89.github.io,http://localhost:8000").split(",")
LIMITE_MAXIMO = 1000

# Misma base y colección donde guarda la API. La cadena de conexión solo viene de MONGO_URI.
coleccion = None
MONGO_URI = os.environ.get("MONGO_URI")
if MONGO_URI:
    try:
        cliente_mongo = MongoClient(MONGO_URI, tz_aware=True, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000, socketTimeoutMS=10000)
        coleccion = cliente_mongo["proyectoAltur"]["historial_predicciones"]
    except Exception as e:
        print(f"⚠️ MongoDB desactivado: MONGO_URI no es válida ({e})")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=ORIGENES_CORS, allow_methods=["GET"])


def numero(valor):
    # JSON no acepta NaN: la latencia queda en NaN cuando no hubo respuestas que medir
    if isinstance(valor, (int, float)) and not (isinstance(valor, float) and math.isnan(valor)):
        return float(valor)
    return None


def a_json(doc):
    prob_ia = numero(doc.get("confianza"))          # la API guarda la probabilidad de IA en "confianza"
    resultado = doc.get("resultado")
    is_synthetic = bool(resultado) if resultado is not None else None
    fecha = doc.get("fecha_analisis")
    if isinstance(fecha, datetime) and fecha.tzinfo is None:
        fecha = fecha.replace(tzinfo=timezone.utc)   # se guarda en UTC; sin zona, el navegador la leería como hora local
    return {
        "call_id": doc.get("id"),
        "is_synthetic": is_synthetic,
        # Igual que la respuesta de /detect: seguridad en la respuesta dada
        "confidence": None if prob_ia is None or is_synthetic is None else (prob_ia if is_synthetic else 1 - prob_ia),
        "latencia_mediana_s": numero(doc.get("latencia_mediana")),
        "fecha": fecha.isoformat() if isinstance(fecha, datetime) else fecha,
    }


@app.get("/health")
def health():
    return {"ok": True, "mongodb": coleccion is not None}


@app.get("/historial")
def historial(limite: int = 200):
    if coleccion is None:
        raise HTTPException(503, "Falta MONGO_URI")
    limite = max(1, min(limite, LIMITE_MAXIMO))
    try:
        docs = coleccion.find({}, {"_id": 0}).sort("fecha_analisis", DESCENDING).limit(limite)
        return {"total": coleccion.count_documents({}), "llamadas": [a_json(d) for d in docs]}
    except Exception as e:
        print(f"⚠️ No se pudo leer MongoDB: {e}")
        raise HTTPException(503, "No se pudo leer MongoDB")
