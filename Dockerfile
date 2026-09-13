# API de Yasu (POST /detect) para Railway.
# Solo copia lo que usa la API: nada de audios ni datos de Altur.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY api.py modelo_xgboost_altur_v4.json ./
COPY src/__init__.py src/
COPY src/fase1/main.py src/fase1/
COPY src/fase2/fase2_acustica.py src/fase2/
COPY src/fase3/fase3_conversacional.py src/fase3/
COPY src/tools/vad.py src/tools/

# Railway indica en PORT el puerto donde debe escuchar el servidor
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
