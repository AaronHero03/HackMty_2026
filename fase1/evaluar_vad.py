"""Fase 1 — Detector de voz contra turns/.

1. Carga cada llamada, normaliza cada canal y corre Silero una vez (probabilidades en caché).
   También sin normalizar, para medir si normalizar cambia el error entre clases.
2. Prueba combinaciones de umbral y silencio mínimo; elige la de menor error promedio en train.
3. Con esa combinación reporta el error por canal y por clase, en train y en val.
4. Revisa que la latencia de quien llama, medida con NUESTRO detector, siga separando clases.

Uso, desde la raíz del repo:
    python fase1/evaluar_vad.py --datos ../hackmty26

Salidas:
    <datos>/fase1/probs/<id>.npz       probabilidades por canal (no se suben)
    <datos>/fase1/desacuerdos.csv      tramos largos donde detector y turns/ no coinciden, para escuchar
    <datos>/fase1/ejemplos/*.png       3 llamadas con los tramos de turns/ y del detector
    fase1/resultados/rejilla.csv       error promedio en train por combinación
    fase1/resultados/resumen.csv       error por canal, clase y split con la combinación elegida

Recuerda: turns/ también lo generó una máquina. Parecerse a turns/ no es la meta; sirve para
encontrar desacuerdos grandes y escucharlos.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio import SR, cargar_llamada, normalizar  # noqa: E402

RESOLUCION = 0.01                        # 10 ms para comparar tramos
UMBRALES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_SILENCIOS_MS = (100, 150, 200, 250, 300, 500, 800)
DESACUERDO_MIN_S = 0.5
VENTANA = 256                            # muestras por probabilidad de Silero a 8 kHz


# ---------- 1. Probabilidades ----------

def calcular_probs(tarea):
    ruta_wav, destino = tarea
    if destino.exists():
        return
    import torch
    torch.set_num_threads(1)
    from src.vad import probabilidades_voz
    llama, agente = cargar_llamada(ruta_wav)
    np.savez_compressed(
        destino, n=len(llama),
        p0=probabilidades_voz(normalizar(llama)), p1=probabilidades_voz(normalizar(agente)),
        p0_crudo=probabilidades_voz(llama), p1_crudo=probabilidades_voz(agente))


# ---------- 2 y 3. Comparación con turns/ ----------

def rejilla(tramos, duracion):
    m = np.zeros(int(round(duracion / RESOLUCION)), bool)
    for a, b in tramos:
        m[int(a / RESOLUCION):int(b / RESOLUCION)] = True
    return m


def errores(pred, real):
    return {"error": 100 * np.mean(pred != real),
            "perdida": 100 * np.mean(real & ~pred),      # turns/ dice voz y el detector no
            "falsa": 100 * np.mean(pred & ~real)}         # el detector dice voz y turns/ no


def latencia_mediana(t0, t1):
    """Mediana de lo que tarda quien llama (canal 0) en hablar después de que termina el agente."""
    ev = sorted([(a, b, 0) for a, b in t0] + [(a, b, 1) for a, b in t1])
    lat = [y[0] - x[1] for x, y in zip(ev, ev[1:]) if x[2] == 1 and y[2] == 0]
    return float(np.median(lat)) if lat else np.nan


def evaluar_llamada(tarea):
    from src.vad import tramos_voz
    ruta_probs, ruta_turns, combinaciones, detalle = tarea
    z = np.load(ruta_probs)
    n = int(z["n"])
    dur = n / SR
    with open(ruta_turns, encoding="utf-8") as fh:
        turnos = json.load(fh)["turns"]
    real_t = {c: [(t["start"], t["end"]) for t in turnos if t["channel"] == c] for c in (0, 1)}
    real = {c: rejilla(real_t[c], dur) for c in (0, 1)}

    filas = []
    for version, u, s in combinaciones:
        suf = "" if version == "normalizado" else "_crudo"
        fila = {"version": version, "umbral": u, "min_silencio_ms": s}
        pred_t = {c: tramos_voz(z[f"p{c}{suf}"], n, umbral=u, min_silencio_ms=s) for c in (0, 1)}
        for c in (0, 1):
            for k, v in errores(rejilla(pred_t[c], dur), real[c]).items():
                fila[f"c{c}_{k}"] = v
        if detalle:
            fila["lat_detector"] = latencia_mediana(pred_t[0], pred_t[1])
            fila["lat_turns"] = latencia_mediana(real_t[0], real_t[1])
            fila["desacuerdos"] = desacuerdos(pred_t, real, {c: z[f"p{c}{suf}"] for c in (0, 1)})
            fila["tramos"] = pred_t
        filas.append(fila)
    return filas


def desacuerdos(pred_t, real, probs):
    """Tramos largos donde detector y turns/ no coinciden, con la probabilidad media de Silero
    y qué parte del tramo cae mientras habla el agente según turns/ (pista de eco)."""
    salida = []
    for c in (0, 1):
        pred = rejilla(pred_t[c], len(real[c]) * RESOLUCION)[:len(real[c])]
        distinto = np.concatenate([[False], pred != real[c], [False]])
        cambios = np.flatnonzero(distinto[1:] != distinto[:-1])
        for a, b in zip(cambios[::2], cambios[1::2]):
            if (b - a) * RESOLUCION >= DESACUERDO_MIN_S:
                tipo = "detector: voz / turns: silencio" if pred[a] else "turns: voz / detector: silencio"
                v0 = int(a * RESOLUCION * SR / VENTANA)
                v1 = max(int(b * RESOLUCION * SR / VENTANA), v0 + 1)
                salida.append({"canal": c, "inicio_s": round(a * RESOLUCION, 2),
                               "fin_s": round(b * RESOLUCION, 2), "tipo": tipo,
                               "prob_media": round(float(probs[c][v0:v1].mean()), 3),
                               "encima_agente_pct": round(100 * float(real[1][a:b].mean()), 1)})
    return salida


def auc(v, es_ia):
    ok = ~np.isnan(v)
    p, q = v[ok & es_ia], v[ok & ~es_ia]
    return mannwhitneyu(p, q).statistic / (len(p) * len(q))


# ---------- Salidas ----------

def ejemplos(datos, manifiesto, resultados, destino, tramos_por_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destino.mkdir(parents=True, exist_ok=True)
    val = [i for i, f in enumerate(manifiesto) if f["split"] == "val"]
    humana = next(i for i in val if manifiesto[i]["label"] == "human")
    ia = next(i for i in val if manifiesto[i]["label"] == "synthetic")
    peor = max(range(len(resultados)), key=lambda i: resultados[i]["c0_error"] + resultados[i]["c1_error"])
    for nombre, i in (("humana", humana), ("ia", ia), ("mayor_error", peor)):
        f = manifiesto[i]
        llama, agente = cargar_llamada(datos / "audio" / f"{f['anon_id']}.wav")
        with open(datos / "turns" / f"{f['anon_id']}.json", encoding="utf-8") as fh:
            turnos = json.load(fh)["turns"]
        fig, ejes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
        for c, x in ((0, normalizar(llama)), (1, normalizar(agente))):
            ax = ejes[c]
            t = np.arange(len(x))[::40] / SR
            ax.plot(t, np.abs(x[::40]), color="0.6", lw=0.4)
            for t0 in turnos:
                if t0["channel"] == c:
                    ax.axvspan(t0["start"], t0["end"], ymin=0.55, ymax=0.95, color="tab:green", alpha=0.35)
            for a, b in tramos_por_id[f["anon_id"]][c]:
                ax.axvspan(a, b, ymin=0.05, ymax=0.45, color="tab:purple", alpha=0.35)
            ax.set_ylabel(f"canal {c}")
            ax.set_yticks([])
        ejes[0].set_title(f"{f['anon_id']} ({f['label']}, {f['split']}) · verde arriba = turns/ · morado abajo = detector")
        ejes[1].set_xlabel("segundos")
        fig.tight_layout()
        fig.savefig(destino / f"{nombre}.png", dpi=90)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", required=True, type=Path)
    p.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()
    datos, cache = args.datos, args.datos / "fase1"
    (cache / "probs").mkdir(parents=True, exist_ok=True)
    salida = Path(__file__).resolve().parent / "resultados"
    salida.mkdir(parents=True, exist_ok=True)
    with open(datos / "manifest.csv", encoding="utf-8") as fh:
        manifiesto = list(csv.DictReader(fh))
    ctx = mp.get_context("spawn")      # igual en Windows, Linux y Mac

    print(f"1. Probabilidades de voz con Silero ({args.procesos} procesos)...")
    t = time.time()
    tareas = [(datos / "audio" / f"{f['anon_id']}.wav", cache / "probs" / f"{f['anon_id']}.npz") for f in manifiesto]
    with ctx.Pool(args.procesos) as pool:
        pool.map(calcular_probs, tareas, chunksize=4)
    print(f"   listo en {time.time() - t:.0f} s")

    es_ia = np.array([f["label"] == "synthetic" for f in manifiesto])
    split = np.array([f["split"] for f in manifiesto])
    rutas = [(cache / "probs" / f"{f['anon_id']}.npz", datos / "turns" / f"{f['anon_id']}.json") for f in manifiesto]

    print("\n2. Rejilla de ajustes (solo train, audio normalizado)")
    combis = [("normalizado", u, s) for u in UMBRALES for s in MIN_SILENCIOS_MS]
    entrenamiento = [i for i in range(len(manifiesto)) if split[i] == "train"]
    with ctx.Pool(args.procesos) as pool:
        por_llamada = pool.map(evaluar_llamada, [(*rutas[i], combis, False) for i in entrenamiento], chunksize=4)
    rejilla_filas = []
    for k, (version, u, s) in enumerate(combis):
        e0 = np.mean([r[k]["c0_error"] for r in por_llamada])
        e1 = np.mean([r[k]["c1_error"] for r in por_llamada])
        rejilla_filas.append({"umbral": u, "min_silencio_ms": s, "c0_error": round(e0, 2),
                              "c1_error": round(e1, 2), "promedio": round((e0 + e1) / 2, 2)})
    rejilla_filas.sort(key=lambda r: r["promedio"])
    print(f"   {'umbral':>6} {'silencio mín':>12} {'error c0':>9} {'error c1':>9} {'promedio':>9}")
    for r in rejilla_filas[:5]:
        print(f"   {r['umbral']:>6} {r['min_silencio_ms']:>9} ms {r['c0_error']:>8.2f}% {r['c1_error']:>8.2f}% {r['promedio']:>8.2f}%")
    mejor = rejilla_filas[0]
    u, s = mejor["umbral"], mejor["min_silencio_ms"]
    print(f"   Elegido: umbral {u}, silencio mínimo {s} ms")
    with open(salida / "rejilla.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rejilla_filas[0]))
        w.writeheader()
        w.writerows(rejilla_filas)

    print("\n3. Error con el ajuste elegido, por canal y clase (% del tiempo de la llamada)")
    with ctx.Pool(args.procesos) as pool:
        final = pool.map(evaluar_llamada, [(*r, [("normalizado", u, s), ("crudo", u, s)], True) for r in rutas], chunksize=4)
    resumen = []
    print(f"   {'versión':<12} {'split':<5} {'canal':<6} {'clase':<7} {'error':>6} {'pérdida':>8} {'falsa':>6}")
    for vi, version in enumerate(("normalizado", "crudo")):
        for sp in ("train", "val"):
            for c in (0, 1):
                for clase, m in (("humana", ~es_ia), ("IA", es_ia)):
                    idx = np.flatnonzero((split == sp) & m)
                    fila = {"version": version, "split": sp, "canal": c, "clase": clase}
                    for k in ("error", "perdida", "falsa"):
                        fila[k] = round(float(np.mean([final[i][vi][f"c{c}_{k}"] for i in idx])), 2)
                    resumen.append(fila)
                    print(f"   {version:<12} {sp:<5} {c:<6} {clase:<7} {fila['error']:>5.2f}% {fila['perdida']:>7.2f}% {fila['falsa']:>5.2f}%")
    with open(salida / "resumen.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumen[0]))
        w.writeheader()
        w.writerows(resumen)

    print("\n4. Latencia de quien llama: nuestro detector contra turns/")
    lat_d = np.array([r[0]["lat_detector"] for r in final])
    lat_t = np.array([r[0]["lat_turns"] for r in final])
    ok = ~np.isnan(lat_d) & ~np.isnan(lat_t)
    print(f"   Correlación (Spearman) entre ambas: {spearmanr(lat_d[ok], lat_t[ok])[0]:.2f}")
    for nombre, v in (("detector", lat_d), ("turns/", lat_t)):
        print(f"   {nombre:<9} mediana humana {np.nanmedian(v[~es_ia]):.2f} s, IA {np.nanmedian(v[es_ia]):.2f} s | "
              f"AUC train {auc(v[split == 'train'], es_ia[split == 'train']):.2f}, val {auc(v[split == 'val'], es_ia[split == 'val']):.2f}")

    filas_desacuerdo = [{"anon_id": manifiesto[i]["anon_id"], "clase": manifiesto[i]["label"],
                         "split": manifiesto[i]["split"], **d}
                        for i, r in enumerate(final) for d in r[0]["desacuerdos"]]
    filas_desacuerdo.sort(key=lambda d: d["fin_s"] - d["inicio_s"], reverse=True)

    print(f"\n5. Canal 0: ¿qué son los desacuerdos? (tramos de {DESACUERDO_MIN_S} s o más)")
    for clase, n_clase in (("human", int((~es_ia).sum())), ("synthetic", int(es_ia.sum()))):
        for tipo in ("turns: voz / detector: silencio", "detector: voz / turns: silencio"):
            g = [d for d in filas_desacuerdo if d["canal"] == 0 and d["clase"] == clase and d["tipo"] == tipo]
            if g:
                print(f"   {clase:<9} {tipo:<33} {len(g) / n_clase:4.1f} por llamada | "
                      f"prob. Silero mediana {np.median([d['prob_media'] for d in g]):.2f} | "
                      f"encima del agente {np.median([d['encima_agente_pct'] for d in g]):.0f}%")

    with open(cache / "desacuerdos.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["anon_id", "clase", "split", "canal", "inicio_s", "fin_s", "tipo",
                                           "prob_media", "encima_agente_pct"])
        w.writeheader()
        w.writerows(filas_desacuerdo)
    ejemplos(datos, manifiesto, [r[0] for r in final], cache / "ejemplos",
             {manifiesto[i]["anon_id"]: final[i][0]["tramos"] for i in range(len(manifiesto))})
    print(f"\nGuardado: {salida}/rejilla.csv, {salida}/resumen.csv")
    print(f"Para revisar (no se suben): {cache}/desacuerdos.csv ({len(filas_desacuerdo)} tramos) y {cache}/ejemplos/")


if __name__ == "__main__":
    main()
