import json
from pathlib import Path

import numpy as np


UMBRAL_RESPUESTA_CORTA_S = 0.4

def _normalizar_tramos(tramos) -> list[tuple[float, float]]:
    """Convierte listas de tuplas o diccionarios a formato estándar [(inicio, fin), ...]"""
    if not tramos:
        return []
    if isinstance(tramos[0], dict):
        return [(float(t["start"]), float(t["end"])) for t in tramos]
    return [(float(a), float(b)) for a, b in tramos]

def calcular_latencias(tramos_agente, tramos_llamante) -> list[float]:
    """Calcula el tiempo entre el fin de un turno del agente y el inicio del llamante."""
    agente = _normalizar_tramos(tramos_agente)
    llamante = _normalizar_tramos(tramos_llamante)
    
    # Ordenar eventos cronológicamente (canal 1: agente, canal 0: llamante)
    eventos = sorted([(a, b, 1) for a, b in agente] + [(a, b, 0) for a, b in llamante])
    
    return [
        sig[0] - act[1]
        for act, sig in zip(eventos, eventos[1:])
        if act[2] == 1 and sig[2] == 0
    ]


def extraer_metricas_tiempo(*, tramos_agente, tramos_llamante) -> dict:
    """Extrae mediana, varianza, coeficiente de variación y % de respuestas cortas."""
    lat = np.array(calcular_latencias(tramos_agente, tramos_llamante))

    if len(lat) == 0:
        return {
            "latencia_mediana": np.nan,
            "latencia_varianza": np.nan,
            "latencia_cv": np.nan,
            "pct_respuestas_cortas": np.nan,
        }

    mediana = float(np.median(lat))
    std = float(np.std(lat))

    return {
        "latencia_mediana": mediana,
        "latencia_varianza": float(np.var(lat)),
        "latencia_cv": float(std / mediana) if mediana > 1e-3 else np.nan,
        "pct_respuestas_cortas": float(100 * np.mean(lat < UMBRAL_RESPUESTA_CORTA_S)),
    }
