import json
from pathlib import Path

import numpy as np


def extraer_metricas_tiempo(audio_id, ruta_base):
    """Devuelve la media y la varianza de las latencias de una llamada."""
    ruta_json = Path(ruta_base) / "turns" / f"{audio_id}.json"

    with ruta_json.open("r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    turnos = datos.get("turns", [])
    latencias = []

    # Canal 1 es la empresa y canal 0 es la persona que queremos detectar.
    for turno_anterior, turno_actual in zip(turnos, turnos[1:]):
        if turno_anterior["channel"] == 1 and turno_actual["channel"] == 0:
            latencias.append(turno_actual["start"] - turno_anterior["end"])

    if not latencias:
        return {
            "latenciaMedia": None,
            "variaza": None,
        }

    return {
        "latenciaMedia": float(np.mean(latencias)),
        "variaza": float(np.var(latencias)),
    }
