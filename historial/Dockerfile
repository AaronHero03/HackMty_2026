# Servicio del historial (GET /historial) para la página. Va aparte de la API del juez.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY historial.py .

# Railway indica en PORT el puerto donde debe escuchar el servidor
CMD uvicorn historial:app --host 0.0.0.0 --port ${PORT:-8000}
