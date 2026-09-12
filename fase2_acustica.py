"""Fase 2 — Métricas acústicas de la voz de quien llama (LFCC). Versión de Fer.

Hay otra versión de la Fase 2 en src/dsp.py (Andrés); el equipo elige cuál usa main.py.

Contrato con main.py:
    metricas_acusticas = f2.extraer_metricas_acusticas(voz_recortada, sr)

- voz_recortada: canal 0 ya normalizado y recortado a los turnos de voz
  (Fase 1: procesar_audio_base -> cargar_turnos -> recortar_voz_activa).
- Devuelve 24 métricas: mediana y desviación de los LFCC 1 a 12.

Uso por sí sola (encadenada con la Fase 1 de Aarón, fase1/main.py):
    python fase2_acustica.py --datos ../hackmty26
    En Colab: !python /content/HackMty_2026/fase2_acustica.py --datos /content/hackmty26
Genera <datos>/features_lfcc.csv con las primeras 348 llamadas (las últimas 5 son la prueba de la Fase 3).

Solo usa numpy y scipy (y pandas en el uso por sí sola, igual que la Fase 1).
"""
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.signal import stft

SR = 8000
N_FILTROS = 20          # filtros triangulares del mismo ancho entre 300 y 3400 Hz
N_LFCC = 12             # se guardan los coeficientes 1 a 12 (el 0 es casi el volumen)
VENTANA = 256           # 32 ms a 8 kHz
SALTO = 80              # 10 ms
UMBRAL_TRAMAS_DB = 30   # se quitan las tramas 30 dB por debajo de las más fuertes (silencio digital y eco)


def _banco_filtros():
    frecuencias = np.fft.rfftfreq(VENTANA, d=1 / SR)
    bordes = np.linspace(300, 3400, N_FILTROS + 2)
    banco = np.zeros((N_FILTROS, len(frecuencias)))
    for i in range(N_FILTROS):
        izq, centro, der = bordes[i], bordes[i + 1], bordes[i + 2]
        subida = (frecuencias - izq) / (centro - izq)
        bajada = (der - frecuencias) / (der - centro)
        banco[i] = np.clip(np.minimum(subida, bajada), 0, None)
    return banco


BANCO = _banco_filtros()


def lfcc(y):
    """LFCC por trama. Devuelve (coeficientes: matriz (20, tramas), energía de cada trama en dB)."""
    _, _, Z = stft(y, fs=SR, nperseg=VENTANA, noverlap=VENTANA - SALTO, boundary=None, padded=False)
    energia = BANCO @ (np.abs(Z) ** 2)
    coeficientes = dct(np.log(energia + 1e-10), type=2, axis=0, norm="ortho")
    return coeficientes, 10 * np.log10(energia.sum(axis=0) + 1e-12)


def extraer_metricas_acusticas(voz_recortada, sr):
    """Mediana y desviación de los LFCC 1 a 12 de la voz de quien llama."""
    if sr != SR:
        raise ValueError(f"Se esperaba audio a {SR} Hz y llegó a {sr} Hz")
    nombres = [f"lfcc_{estadistico}_{i}" for i in range(1, N_LFCC + 1) for estadistico in ("mediana", "std")]
    if len(voz_recortada) < 2 * VENTANA:
        return {nombre: np.nan for nombre in nombres}   # sin voz suficiente: vacío, no ceros falsos

    coef, energia_db = lfcc(np.asarray(voz_recortada, dtype=np.float32))
    coef = coef[:, energia_db >= np.percentile(energia_db, 95) - UMBRAL_TRAMAS_DB]

    metricas = {}
    for i in range(1, N_LFCC + 1):
        metricas[f"lfcc_mediana_{i}"] = float(np.median(coef[i]))
        metricas[f"lfcc_std_{i}"] = float(np.std(coef[i]))
    return metricas


def procesar_dataset(ruta_datos, n_llamadas=348, ruta_salida=None):
    """Fase 1 (fase1/main.py de Aarón) -> Fase 2 para las primeras n_llamadas del manifest."""
    import pandas as pd
    from fase1 import main as f1

    ruta_datos = Path(ruta_datos)
    ruta_salida = Path(ruta_salida) if ruta_salida else ruta_datos / "features_lfcc.csv"
    manifest = f1.leer_manifiesto(ruta_datos).head(n_llamadas)

    filas, errores = [], []
    for idx, fila in manifest.iterrows():
        try:
            y_c0_norm, sr = f1.procesar_audio_base(ruta_datos / "audio" / f"{fila['anon_id']}.wav")    # Fase 1
            turnos_llamador, _ = f1.cargar_turnos(ruta_datos / "turns" / f"{fila['anon_id']}.json")    # Fase 1
            voz_recortada = f1.recortar_voz_activa(y_c0_norm, turnos_llamador, sr)                     # Fase 1

            fila_final = {"anon_id": fila["anon_id"], "label": fila["label"], "split": fila["split"]}
            fila_final.update(extraer_metricas_acusticas(voz_recortada, sr))                           # Fase 2
            filas.append(fila_final)
        except Exception as e:
            errores.append((fila["anon_id"], repr(e)))
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(manifest)} llamadas")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(ruta_salida, index=False)
    print(f"Guardado: {ruta_salida} {tabla.shape} | llamadas con error: {len(errores)} {errores[:3]}")
    return tabla


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fase 2 (versión de Fer) encadenada con la Fase 1: LFCC por llamada.")
    p.add_argument("--datos", required=True, help="carpeta con manifest.csv, audio/ y turns/")
    p.add_argument("--llamadas", type=int, default=348, help="primeras N llamadas del manifest (por defecto 348)")
    p.add_argument("--salida", default=None, help="CSV de salida (por defecto <datos>/features_lfcc.csv)")
    args = p.parse_args()
    procesar_dataset(args.datos, args.llamadas, args.salida)
