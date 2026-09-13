from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def cargar_datos():
    root = Path(__file__).resolve().parent.parent
    ruta_csv = root / "Altur_Data" / "features_final_vad.csv"
    print(f"📊 Cargando datos desde: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    val_df = df[df["split"] == "val"].copy()
    train_df = df[df["split"] == "train"].copy()

    X_train = train_df.drop(
        columns=["anon_id", "label", "split"], errors="ignore"
    )
    y_train = train_df["label"].map({"human": 0, "synthetic": 1})

    X_val = val_df.drop(columns=["anon_id", "label", "split"], errors="ignore")
    y_val = val_df["label"].map({"human": 0, "synthetic": 1})

    X_tr, X_es, y_tr, y_es = train_test_split(
        X_train, y_train, random_state=42, test_size=0.15, stratify=y_train
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_es": X_es,
        "y_es": y_es,
        "val_df": val_df,
    }


def calcular_intervalo_confianza(y_true, y_pred):
    from scipy.stats import beta

    aciertos = (np.array(y_true) == np.array(y_pred)).sum()
    n = len(y_true)
    acc = (aciertos / n) * 100
    alpha = 0.05
    lower = beta.ppf(alpha / 2, aciertos, n - aciertos + 1) * 100
    upper = beta.ppf(1 - alpha / 2, aciertos + 1, n - aciertos) * 100
    return acc, lower, upper, n


def obtener_dir_modelo_docs(nombre_modelo):
    root = Path(__file__).resolve().parent.parent
    tag = nombre_modelo.lower().replace(" ", "_")
    dir_modelo = root / "docs" / tag
    dir_modelo.mkdir(parents=True, exist_ok=True)
    return dir_modelo


def graficar_matriz_confusion(y_true, y_pred, nombre_modelo, dir_out):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Humano (0)", "IA (1)"],
        yticklabels=["Humano (0)", "IA (1)"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(
        f"Matriz de Confusión - {nombre_modelo}", fontsize=12, pad=10
    )
    ax.set_xlabel("Predicción del Modelo")
    ax.set_ylabel("Clase Real")
    plt.tight_layout()

    ruta_salida = dir_out / "matriz_confusion.png"
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"🖼️ Matriz de confusión guardada en: {ruta_salida}")


def graficar_metricas_clase(y_true, y_pred, nombre_modelo, dir_out):
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Humano (0)", "IA (1)"],
        output_dict=True,
    )
    df_metrics = pd.DataFrame(report).iloc[:-1, :2].T

    fig, ax = plt.subplots(figsize=(7, 5))
    df_metrics.plot(
        kind="bar",
        ax=ax,
        colormap="viridis",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title(
        f"Métricas por Clase - {nombre_modelo}", fontsize=12, pad=10
    )
    ax.set_ylabel("Puntuación (Score)")
    ax.set_ylim(0, 1.15)
    ax.set_xticklabels(["Humano (0)", "IA (1)"], rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
            )

    plt.tight_layout()
    ruta_salida = dir_out / "metricas_clase.png"
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"📊 Gráfica de métricas guardada en: {ruta_salida}")


def evaluar_y_reportar(nombre_modelo, y_true, y_pred, val_df):
    acc, lo, hi, n = calcular_intervalo_confianza(y_true, y_pred)
    dir_out = obtener_dir_modelo_docs(nombre_modelo)

    print(f"\n--- RESULTADOS {nombre_modelo.upper()} EN VALIDACIÓN ---")
    print(f"Precisión: {acc:.2f}% (IC 95%: {lo:.1f}%–{hi:.1f}%, n={n})\n")
    print(
        classification_report(
            y_true, y_pred, target_names=["Humano (0)", "IA (1)"]
        )
    )

    graficar_matriz_confusion(y_true, y_pred, nombre_modelo, dir_out)
    graficar_metricas_clase(y_true, y_pred, nombre_modelo, dir_out)

    return dir_out