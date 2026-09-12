"""Fase 0 — Trampas en los datos.

Busca diferencias entre llamadas humanas y de IA que NO tengan que ver con ser IA
(volumen, ruido de fondo, espectro, comportamiento del agente). Si un rasgo así separa
las clases, el modelo podría aprender ese atajo y fallar con las llamadas de Altur.

Solo usa numpy, scipy y matplotlib (vienen en Google Colab).

Uso, desde la raíz del repo:
    python fase0/trampas.py --datos ../hackmty26
    En Colab:  !python fase0/trampas.py --datos /content/hackmty26

Salidas:
    <datos>/fase0/por_llamada.csv y psd.npz   mediciones por llamada (NO se suben: llevan etiquetas)
    fase0/resultados/rasgos.csv               comparación humano vs IA por rasgo (promedios)
    fase0/resultados/espectro.png             espectro promedio por clase

Nota: los tramos de voz salen de turns/ del dataset. Aquí sirven para explorar; el modelo
final no debe depender de ellos (ver Paso 4 de IMPLEMENTACION1.1.md).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
from scipy.stats import mannwhitneyu

SR = 8000
MARGEN_S = 0.25   # segundos alrededor de cada turno que no cuentan como silencio
NPERSEG = 512     # ventana de Welch: 64 ms, resolución de 15.6 Hz
TRAMA = 160       # 20 ms, para medir el ruido de fondo
TRAMOS_PSD = ("c0_voz", "c0_silencio", "c1_voz", "c1_silencio")
FRECS = np.fft.rfftfreq(NPERSEG, d=1 / SR)
PLANO_DB = 3.0    # inclinación del silencio por debajo de esto = silencio digital, sin ruido de fondo

# Rasgos agrupados por parte de la Fase 0: (columna, descripción corta)
PARTES = {
    "Parte 1 — Volumen y ruido (canal 0, quien llama)": [
        ("c0_voz_db", "volumen de la voz (dBFS)"),
        ("c0_silencio_db", "ruido de fondo en silencios (dBFS)"),
        ("c0_snr_db", "voz menos ruido (dB)"),
        ("c0_ceros_pct", "% de muestras exactamente 0 en silencios"),
        ("c0_saturado_pct", "% de muestras saturadas en la voz"),
        ("c0_dc", "desplazamiento DC (muestras)"),
        ("c0_eco_db", "sube el canal 0 cuando solo habla el agente (dB)"),
        ("c0_silencio_inclinacion_db", "inclinación del ruido en silencios (dB; ~0 = digital)"),
    ],
    "Parte 2 — Espectro (canal 0, quien llama)": [
        ("c0_voz_graves_pct", "% energía de la voz bajo 300 Hz"),
        ("c0_voz_agudos_pct", "% energía de la voz sobre 3400 Hz"),
        ("c0_voz_centroide_hz", "centro de gravedad del espectro de la voz (Hz)"),
        ("c0_silencio_graves_pct", "% energía del silencio bajo 300 Hz"),
        ("c0_silencio_agudos_pct", "% energía del silencio sobre 3400 Hz"),
    ],
    "Parte 3a — Fuga: cómo SUENA el agente (canal 1)": [
        ("c1_voz_db", "volumen de la voz del agente (dBFS)"),
        ("c1_silencio_db", "ruido de fondo del agente (dBFS)"),
        ("c1_ceros_pct", "% de muestras exactamente 0 en silencios"),
        ("c1_silencio_inclinacion_db", "inclinación del ruido en silencios (dB)"),
        ("c1_voz_graves_pct", "% energía bajo 300 Hz"),
        ("c1_voz_agudos_pct", "% energía sobre 3400 Hz"),
        ("c1_voz_centroide_hz", "centro de gravedad del espectro (Hz)"),
    ],
    "Parte 3b — Fuga: TIEMPOS del agente (canal 1)": [
        ("agente_pct", "% de la llamada en que habla el agente"),
        ("agente_turnos_min", "turnos del agente por minuto"),
        ("agente_turno_mediana_s", "duración típica de un turno del agente (s)"),
        ("agente_latencia_mediana_s", "tarda el agente en contestar (s)"),
        ("agente_interrupciones_min", "veces por minuto que el agente empieza a hablar encima"),
    ],
    "Referencia — señales que esperamos que SÍ separen": [
        ("llamante_latencia_mediana_s", "tarda quien llama en contestar (s)"),
        ("silencio_pct", "% de la llamada en que nadie habla"),
        ("duracion_s", "duración de la llamada (s)"),
    ],
}


def mascara(turnos, canal, n, margen_s=0.0):
    """True donde el canal tiene voz según turns/, ampliado margen_s por lado."""
    m = np.zeros(n, bool)
    for t in turnos:
        if t["channel"] == canal:
            ini = max(0, int((t["start"] - margen_s) * SR))
            fin = min(n, int((t["end"] + margen_s) * SR))
            m[ini:fin] = True
    return m


def rms_db(x):
    if len(x) < SR // 2:
        return np.nan
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-7)


def piso_db(x):
    """Ruido de fondo robusto: mediana del volumen de tramas de 20 ms (necesita ≥ 1 s)."""
    k = len(x) // TRAMA
    if k < SR // TRAMA:
        return np.nan
    tramas = x[:k * TRAMA].reshape(k, TRAMA)
    return 20 * np.log10(np.median(np.sqrt(np.mean(tramas ** 2, axis=1))) + 1e-7)


def espectro(x):
    if len(x) < 4 * NPERSEG:
        return None
    return welch(x, fs=SR, nperseg=NPERSEG)[1]


def rasgos_espectro(f, P, prefijo):
    if P is None:
        return {f"{prefijo}_graves_pct": np.nan, f"{prefijo}_agudos_pct": np.nan,
                f"{prefijo}_centroide_hz": np.nan}
    total = P.sum() + 1e-30
    return {
        f"{prefijo}_graves_pct": 100 * P[f < 300].sum() / total,
        f"{prefijo}_agudos_pct": 100 * P[f > 3400].sum() / total,
        f"{prefijo}_centroide_hz": (f * P).sum() / total,
    }


def inclinacion_db(P):
    """dB entre 200-1000 Hz menos dB entre 3000-3600 Hz. El ruido real de una línea o un cuarto
    tiene más graves que agudos (valor positivo); el silencio digital es plano (cerca de 0)."""
    if P is None:
        return np.nan
    D = 10 * np.log10(P + 1e-20)
    return D[(FRECS >= 200) & (FRECS <= 1000)].mean() - D[(FRECS >= 3000) & (FRECS <= 3600)].mean()


def rasgos_tiempo(turnos, duracion):
    t = sorted(turnos, key=lambda x: x["start"])
    c0 = [x for x in t if x["channel"] == 0]
    c1 = [x for x in t if x["channel"] == 1]
    minutos = duracion / 60
    lat_agente = [b["start"] - a["end"] for a, b in zip(t, t[1:])
                  if a["channel"] == 0 and b["channel"] == 1]
    lat_llamante = [b["start"] - a["end"] for a, b in zip(t, t[1:])
                    if a["channel"] == 1 and b["channel"] == 0]
    encima = sum(1 for b in c1 if any(a["start"] < b["start"] < a["end"] for a in c0))
    return {
        "agente_pct": 100 * sum(x["end"] - x["start"] for x in c1) / duracion,
        "agente_turnos_min": len(c1) / minutos,
        "agente_turno_mediana_s": np.median([x["end"] - x["start"] for x in c1]) if c1 else np.nan,
        "agente_latencia_mediana_s": np.median(lat_agente) if lat_agente else np.nan,
        "agente_interrupciones_min": encima / minutos,
        "llamante_latencia_mediana_s": np.median(lat_llamante) if lat_llamante else np.nan,
    }


def medir_llamada(datos, fila):
    sr, crudo = wavfile.read(datos / "audio" / f"{fila['anon_id']}.wav")
    if sr != SR or crudo.ndim != 2 or crudo.shape[1] != 2:
        raise ValueError(f"{fila['anon_id']}: se esperaba estéreo a {SR} Hz, llegó {sr} Hz {crudo.shape}")
    y = crudo.astype(np.float32) / 32768.0
    n = len(y)
    with open(datos / "turns" / f"{fila['anon_id']}.json", encoding="utf-8") as fh:
        turnos = json.load(fh)["turns"]

    voz = {0: mascara(turnos, 0, n), 1: mascara(turnos, 1, n)}
    ocupado0 = mascara(turnos, 0, n, MARGEN_S)
    ocupado1 = mascara(turnos, 1, n, MARGEN_S)
    silencio = ~ocupado0 & ~ocupado1          # nadie habla
    solo_agente = voz[1] & ~ocupado0          # habla el agente y quien llama calla

    out = {"anon_id": fila["anon_id"], "label": fila["label"], "split": fila["split"],
           "duracion_s": n / SR, "silencio_pct": 100 * silencio.mean()}
    psd = {}
    for c in (0, 1):
        x = y[:, c]
        out[f"c{c}_voz_db"] = rms_db(x[voz[c]])
        out[f"c{c}_silencio_db"] = piso_db(x[silencio])
        out[f"c{c}_snr_db"] = out[f"c{c}_voz_db"] - out[f"c{c}_silencio_db"]
        out[f"c{c}_ceros_pct"] = 100 * np.mean(crudo[silencio, c] == 0) if silencio.sum() >= SR else np.nan
        out[f"c{c}_saturado_pct"] = 100 * np.mean(np.abs(crudo[voz[c], c].astype(np.int32)) >= 32000)
        out[f"c{c}_dc"] = float(np.mean(crudo[:, c]))
        out[f"c{c}_valores_distintos"] = len(np.unique(crudo[:, c]))   # μ-law deja ~256 como máximo
        out[f"c{c}_pico"] = int(np.abs(crudo[:, c].astype(np.int32)).max())
        for tramo, m in (("voz", voz[c]), ("silencio", silencio)):
            P = espectro(x[m])
            psd[f"c{c}_{tramo}"] = P
            out.update(rasgos_espectro(FRECS, P, f"c{c}_{tramo}"))
            if tramo == "silencio":
                out[f"c{c}_silencio_inclinacion_db"] = inclinacion_db(P)
    out["c0_eco_db"] = piso_db(y[solo_agente, 0]) - out["c0_silencio_db"]
    out.update(rasgos_tiempo(turnos, n / SR))
    return out, psd


def medir_todo(datos, cache):
    with open(datos / "manifest.csv", encoding="utf-8") as fh:
        manifiesto = list(csv.DictReader(fh))
    filas, psd = [], {k: np.full((len(manifiesto), len(FRECS)), np.nan) for k in TRAMOS_PSD}
    for i, fila in enumerate(manifiesto):
        out, P = medir_llamada(datos, fila)
        filas.append(out)
        for k in TRAMOS_PSD:
            if P[k] is not None:
                psd[k][i] = P[k]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(manifiesto)} llamadas medidas")
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / "por_llamada.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    np.savez_compressed(cache / "psd.npz", frecuencias=FRECS, **psd)


def auc(valores, es_ia):
    """Probabilidad de que una llamada de IA tenga un valor mayor que una humana (0.5 = azar)."""
    pos, neg = valores[es_ia & ~np.isnan(valores)], valores[~es_ia & ~np.isnan(valores)]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    return mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))


def veredicto(a_train, a_val):
    """FUERTE / moderada solo si separa en train Y en val, en la misma dirección."""
    if np.isnan(a_train) or np.isnan(a_val):
        return "sin datos"
    if (a_train - 0.5) * (a_val - 0.5) <= 0:
        return "no separa"
    fuerza = 0.5 + min(abs(a_train - 0.5), abs(a_val - 0.5))
    return "FUERTE" if fuerza >= 0.70 else "moderada" if fuerza >= 0.60 else "no separa"


def resumir(cache, salida):
    with open(cache / "por_llamada.csv", encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    es_ia = np.array([f["label"] == "synthetic" for f in filas])
    split = np.array([f["split"] for f in filas])
    resumen = []
    for parte, rasgos in PARTES.items():
        print(f"\n{parte}")
        print(f"  {'rasgo':<58} {'humano':>8} {'IA':>8} {'AUC train':>10} {'AUC val':>8}  veredicto")
        for col, desc in rasgos:
            v = np.array([float(f[col]) for f in filas])
            med_h, med_ia = np.nanmedian(v[~es_ia]), np.nanmedian(v[es_ia])
            a_tr = auc(v[split == "train"], es_ia[split == "train"])
            a_va = auc(v[split == "val"], es_ia[split == "val"])
            ver = veredicto(a_tr, a_va)
            print(f"  {desc:<58} {med_h:>8.2f} {med_ia:>8.2f} {a_tr:>10.2f} {a_va:>8.2f}  {ver}")
            resumen.append({"parte": parte, "rasgo": col, "descripcion": desc,
                            "mediana_humano": round(med_h, 3), "mediana_ia": round(med_ia, 3),
                            "auc_train": round(a_tr, 3), "auc_val": round(a_va, 3), "veredicto": ver})
    col = lambda k: np.array([float(f[k]) for f in filas])

    print("\nCódec telefónico (μ-law deja como máximo ~256 valores distintos y un pico de 32124)")
    for clase, m in (("humanas", ~es_ia), ("IA", es_ia)):
        for c in (0, 1):
            vd, pico = col(f"c{c}_valores_distintos")[m], col(f"c{c}_pico")[m]
            print(f"  {clase:>7} canal {c}: valores distintos mediana {np.median(vd):.0f} "
                  f"(máx {vd.max():.0f}), pico máximo {pico.max():.0f}")

    print(f"\nSilencio digital en el canal 0 (inclinación < {PLANO_DB:.0f} dB)")
    incl = col("c0_silencio_inclinacion_db")
    plano = incl < PLANO_DB
    for s in ("train", "val"):
        m = (split == s) & ~np.isnan(incl)
        print(f"  {s}: {plano[m & ~es_ia].sum()}/{(m & ~es_ia).sum()} humanas y "
              f"{plano[m & es_ia].sum()}/{(m & es_ia).sum()} de IA")
    m = (split == "val") & ~np.isnan(incl)
    print(f"  Regla 'silencio digital = IA' acierta {(plano[m] == es_ia[m]).sum()}/{m.sum()} en val")

    print("\n¿El agente habla menos solo porque la IA tarda más en contestar?")
    agente, lat = col("agente_pct"), col("llamante_latencia_mediana_s")
    for lo, hi in ((0.5, 1.5), (1.5, 2.5)):
        m = (lat >= lo) & (lat < hi)
        print(f"  Quien llama tarda {lo}-{hi} s: {(m & ~es_ia).sum()} humanas, {(m & es_ia).sum()} de IA | "
              f"el agente habla {np.median(agente[m & ~es_ia]):.1f}% vs {np.median(agente[m & es_ia]):.1f}% | "
              f"AUC {auc(agente[m], es_ia[m]):.2f}")

    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / "rasgos.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumen[0]))
        w.writeheader()
        w.writerows(resumen)
    graficar(np.load(cache / "psd.npz"), es_ia, salida / "espectro.png")
    print(f"\nAUC: 0.5 = azar; >0.5 la IA tiene valores más altos; <0.5 más bajos.")
    print(f"Guardado: {salida / 'rasgos.csv'} y {salida / 'espectro.png'}")


def graficar(psd, es_ia, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = psd["frecuencias"]
    fig, ejes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    titulos = {"c0": "Canal 0 (quien llama)", "c1": "Canal 1 (agente)"}
    for i, canal in enumerate(("c0", "c1")):
        for j, tramo in enumerate(("voz", "silencio")):
            ax = ejes[i, j]
            P = psd[f"{canal}_{tramo}"]
            for clase, m, color in (("Humano", ~es_ia, "tab:blue"), ("IA", es_ia, "tab:orange")):
                D = 10 * np.log10(P[m] + 1e-20)
                D = D[~np.isnan(D).any(axis=1)]
                q1, med, q3 = np.percentile(D, [25, 50, 75], axis=0)
                ax.plot(f, med, color=color, label=f"{clase} (n={len(D)})")
                ax.fill_between(f, q1, q3, color=color, alpha=0.2)
            for corte in (300, 3400):
                ax.axvline(corte, color="gray", ls="--", lw=0.8)
            ax.set_title(f"{titulos[canal]} — {tramo}")
            ax.set_ylabel("Potencia (dB)")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right")
    for ax in ejes[1]:
        ax.set_xlabel("Frecuencia (Hz)   ·   líneas punteadas: 300 y 3400 Hz (banda telefónica)")
    fig.suptitle("Espectro por clase: línea = mediana, sombra = 50% central de las llamadas")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", required=True, type=Path, help="carpeta del dataset (manifest.csv, ../../Altur/audio/, ../../Altur/turns/)")
    p.add_argument("--rehacer", action="store_true", help="volver a medir aunque exista el caché")
    args = p.parse_args()
    cache = args.datos / "fase0"
    if args.rehacer or not (cache / "por_llamada.csv").exists():
        print("Midiendo las llamadas...")
        medir_todo(args.datos, cache)
    resumir(cache, Path(__file__).resolve().parent / "resultados")


if __name__ == "__main__":
    main()
