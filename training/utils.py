from pathlib import Path
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def obtener_ruta_features() -> Path:
    """Busca la matriz de características preferida (VAD o estándar)."""
    raiz = Path(__file__).resolve().parent.parent
    ruta_vad = raiz / "Altur_Data" / "features_final_vad.csv"
    ruta_std = raiz / "Altur_Data" / "features_final.csv"

    if ruta_vad.exists():
        return ruta_vad
    elif ruta_std.exists():
        return ruta_std
    else:
        raise FileNotFoundError(
            "No se encontró features_final_vad.csv ni features_final.csv en Altur_Data/"
        )


def cargar_datos(ruta_csv=None):
    """Carga y prepara los splits de datos sin contaminar el conjunto de validación."""
    if ruta_csv is None:
        ruta_csv = obtener_ruta_features()

    print(f"📊 Cargando datos desde: {ruta_csv}")
    df = pd.read_csv(ruta_csv)

    df["target"] = df["label"].apply(lambda x: 1 if x == "synthetic" else 0)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    columnas_ignorar = ["anon_id", "label", "split", "target"]
    X_train = train_df.drop(columns=columnas_ignorar)
    y_train = train_df["target"]
    X_val = val_df.drop(columns=columnas_ignorar)
    y_val = val_df["target"]

    # Split de Early Stopping
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_es": X_es,
        "y_es": y_es,
        "X_val": X_val,
        "y_val": y_val,
        "val_df": val_df,
    }


def calcular_intervalo_confianza(y_true, y_pred):
    """Calcula el Intervalo de Confianza del 95% mediante distribución Beta (Clopper-Pearson)."""
    n = len(y_true)
    aciertos = int((y_pred == y_true).sum())
    lo = beta.ppf(0.025, aciertos, n - aciertos + 1) * 100
    hi = beta.ppf(0.975, aciertos + 1, n - aciertos) * 100
    acc = accuracy_score(y_true, y_pred) * 100
    return acc, lo, hi, n


def evaluar_y_reportar(nombre_modelo, y_val, y_pred, val_df):
    """Imprime métricas finales y detecta fallos de clasificación."""
    acc, lo, hi, n = calcular_intervalo_confianza(y_val, y_pred)
    print(f"\n--- RESULTADOS {nombre_modelo.upper()} EN VALIDACIÓN ---")
    print(f"Precisión: {acc:.2f}% (IC 95%: {lo:.1f}%–{hi:.1f}%, n={n})\n")
    print(
        classification_report(
            y_val, y_pred, target_names=["Humano (0)", "IA (1)"]
        )
    )

    val_df_copy = val_df.copy()
    val_df_copy["prediccion"] = y_pred
    errores = val_df_copy[val_df_copy["target"] != val_df_copy["prediccion"]]

    if not errores.empty:
        print(f"Llamadas que engañaron a {nombre_modelo}:")
        print(errores[["anon_id", "label", "prediccion"]])
    else:
        print(
            f"{nombre_modelo} clasificó correctamente el 100% de las muestras."
        )

    return acc