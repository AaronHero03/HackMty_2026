import json
from pathlib import Path

import numpy as np


def extraer_metricas_tiempo(audio_id, ruta_base):
    """Devuelve latencia, varianza, conteo de interrupciones y % de participación del agente."""
    ruta_json = Path(ruta_base) / "turns" / f"{audio_id}.json"

    with ruta_json.open("r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    turnos = datos.get("turns", [])
    latencias = []
    interrupciones = 0
    tiempo_agente = 0.0
    tiempo_total_llamada = 0.0

    if turnos:
        # El final del último turno nos da la duración total aproximada de la llamada
        tiempo_total_llamada = turnos[-1]["end"]

    # Calcular tiempo total hablado por el agente (Canal 1)
    for turno in turnos:
        if turno["channel"] == 1:
            tiempo_agente += (turno["end"] - turno["start"])

    # Calcular latencias e interrupciones
    for turno_anterior, turno_actual in zip(turnos, turnos[1:]):
        if turno_anterior["channel"] == 1 and turno_actual["channel"] == 0:
            latencia = turno_actual["start"] - turno_anterior["end"]
            latencias.append(latencia)
            
            # Si empieza a hablar antes de que el agente termine, es una interrupción
            if latencia < 0:
                interrupciones += 1

    pct_agente = (tiempo_agente / tiempo_total_llamada * 100) if tiempo_total_llamada > 0 else 0.0

    if not latencias:
        return {
            "latenciaMedia": np.nan,
            "varianza": np.nan,
            "interrupciones": interrupciones,
            "pct_habla_agente": pct_agente
        }

    return {
        "latenciaMedia": float(np.mean(latencias)),
        "varianza": float(np.var(latencias)),
        "interrupciones": interrupciones,
        "pct_habla_agente": pct_agente
    }