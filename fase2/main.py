"""Fase 2 — Procesamiento Acústico (Biomédica).

Este script ejecuta la capa de extracción acústica (Fase 2) sobre todo el dataset.
Importa la lógica de audio base de src.audio y los algoritmos matemáticos de src.dsp.

Uso, desde la raíz del repo:
    python fase2/main.py --datos ../hackmty26
"""
import argparse
import json
import csv
import numpy as np
from pathlib import Path

# Importaciones limpias desde la carpeta src/ del proyecto
from src.audio import cargar_llamada, normalizar, SR  #[cite: 3]
from src.dsp import extraer_features_acusticas

def extraer_voz_activa(y, turnos_c0, sr=SR):
    """Recorta el audio basándose en los turnos del JSON para obtener voz continua."""
    tramos = []
    for t in turnos_c0:
        ini = int(t["start"] * sr)
        fin = int(t["end"] * sr)
        tramos.append(y[ini:fin])
    return np.concatenate(tramos) if tramos else np.array([], dtype=np.float32)

def procesar_fase2(datos, salida):
    with open(datos / "manifest.csv", encoding="utf-8") as fh:
        manifiesto = list(csv.DictReader(fh))
        
    resultados = []
    print(f"Iniciando extracción Fase 2 (LFCCs + Pasabanda) en {len(manifiesto)} llamadas...")

    for i, fila in enumerate(manifiesto):
        anon_id = fila["anon_id"]
        ruta_wav = datos / "audio" / f"{anon_id}.wav"
        ruta_json = datos / "turns" / f"{anon_id}.json"

        try:
            # 1. Pipeline base (Fase 1 importada)
            llama, _ = cargar_llamada(ruta_wav)  #[cite: 3]
            llama_norm = normalizar(llama)       #[cite: 3]

            # 2. Extraer turnos
            with open(ruta_json, "r", encoding="utf-8") as f:
                turnos = json.load(f).get("turns", [])
            c0_turnos = [t for t in turnos if t["channel"] == 0]

            # 3. Recortar voz y aplicar Fase 2 (importada desde src.dsp)
            y_voz = extraer_voz_activa(llama_norm, c0_turnos)
            feats = extraer_features_acusticas(y_voz, sr=SR)

            # 4. Metadatos
            feats["anon_id"] = anon_id
            feats["label"] = 1 if fila["label"] == "synthetic" else 0
            feats["split"] = fila["split"]

            resultados.append(feats)

        except Exception as e:
            print(f"Error en {anon_id}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(manifiesto)} procesadas...")

    # Guardar CSV de resultados
    salida.mkdir(parents=True, exist_ok=True)
    ruta_csv = salida / "features_acusticas.csv"
    
    with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(resultados[0].keys()))
        writer.writeheader()
        writer.writerows(resultados)
        
    print(f"\n¡Extracción Fase 2 lista! Archivo guardado en: {ruta_csv}")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", required=True, type=Path, help="carpeta del dataset")
    args = p.parse_args()
    
    salida = Path(__file__).resolve().parent / "resultados"
    procesar_fase2(args.datos, salida)

if __name__ == "__main__":
    main()