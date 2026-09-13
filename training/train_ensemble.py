from pathlib import Path
from catboost import CatBoostClassifier
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import accuracy_score

from training.utils import cargar_datos, evaluar_y_reportar


def obtener_probabilidades_svm(modelo_svm, X):
    """Obtiene las probabilidades para la clase 1 (IA).

    Si el SVM no fue entrenado con probability=True, aplica la transformación
    Sigmoide (expit) sobre decision_function.
    """
    try:
        return modelo_svm.predict_proba(X)[:, 1]
    except (AttributeError, NotImplementedError):
        if hasattr(modelo_svm, "decision_function"):
            scores = modelo_svm.decision_function(X)
            return expit(scores)
        raise RuntimeError("El modelo SVM no soporta estimación de certezas.")


def calcular_permutation_importance_ensemble(
    modelos_dict, X_val, y_val, feature_names, n_repeats=5
):
    print("🔍 Calculando Permutation Importance para el Ensamble...")
    y_val_arr = np.array(y_val)

    def predict_probs_ensemble(X_data):
        p_cb = modelos_dict["catboost"].predict_proba(X_data)[:, 1]
        p_et = modelos_dict["extratrees"].predict_proba(X_data)[:, 1]
        p_svm = obtener_probabilidades_svm(modelos_dict["svm"], X_data)
        return (p_cb + p_et + p_svm) / 3.0

    p_base = predict_probs_ensemble(X_val)
    preds_base = (p_base >= 0.5).astype(int)
    acc_base = accuracy_score(y_val_arr, preds_base)

    importances = []
    for col in X_val.columns:
        drop_accs = []
        for _ in range(n_repeats):
            X_permuted = X_val.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)

            p_perm = predict_probs_ensemble(X_permuted)
            preds_perm = (p_perm >= 0.5).astype(int)
            acc_perm = accuracy_score(y_val_arr, preds_perm)

            drop_accs.append(acc_base - acc_perm)

        importances.append(np.mean(drop_accs))

    return pd.Series(importances, index=feature_names)


def entrenar_evaluar_ensamble():
    datos = cargar_datos()

    root = Path(__file__).resolve().parent.parent
    dir_models = root / "models"

    ruta_cb = dir_models / "modelo_catboost_altur.cbm"
    ruta_et = dir_models / "modelo_et_altur.joblib"
    ruta_svm = dir_models / "modelo_svm_altur.joblib"

    # Validar existencia de artefactos
    modelos_faltantes = []
    if not ruta_cb.exists():
        modelos_faltantes.append("CatBoost (.cbm)")
    if not ruta_et.exists():
        modelos_faltantes.append("ExtraTrees (.joblib)")
    if not ruta_svm.exists():
        modelos_faltantes.append("SVM (.joblib)")

    if modelos_faltantes:
        print(
            f"❌ Error: Faltan los siguientes modelos en 'models/': {', '.join(modelos_faltantes)}"
        )
        print(
            "Entrena los modelos individuales antes de ejecutar el ensamble."
        )
        return

    print("📦 Cargando modelos para el Soft Voting Ensemble...")

    cb = CatBoostClassifier()
    cb.load_model(ruta_cb)

    et = joblib.load(ruta_et)
    svm = joblib.load(ruta_svm)

    X_val = datos["X_val"]
    y_val = datos["y_val"]

    print("🗳️ Calculando combinación por Soft Voting (Promedio de certezas)...")

    prob_cb = cb.predict_proba(X_val)[:, 1]
    prob_et = et.predict_proba(X_val)[:, 1]
    prob_svm = obtener_probabilidades_svm(svm, X_val)

    # Promedio de probabilidades
    prob_ensemble = (prob_cb + prob_et + prob_svm) / 3.0
    y_pred_ensemble = (prob_ensemble >= 0.5).astype(int)

    # Generación de reportes
    dir_out = evaluar_y_reportar(
        "Ensemble", y_val, y_pred_ensemble, datos["val_df"]
    )

    # Permutation Importance
    modelos_dict = {"catboost": cb, "extratrees": et, "svm": svm}
    importancias = calcular_permutation_importance_ensemble(
        modelos_dict, X_val, y_val, datos["X_train"].columns
    )
    top15 = importancias.nlargest(15).sort_values(ascending=True)

    ruta_grafica = dir_out / "importancia.png"
    plt.figure(figsize=(10, 8))
    top15.plot(kind="barh", color="#2b5c8f")
    plt.title(
        "Soft Voting Ensemble - Top 15 Métricas por Permutation Importance"
    )
    plt.xlabel("Caída Promedio en Accuracy al Permutar Variable")
    plt.tight_layout()
    plt.savefig(ruta_grafica, dpi=150)
    plt.close()
    print(f"📉 Gráfica de importancia del Ensamble guardada en: {ruta_grafica}")


if __name__ == "__main__":
    entrenar_evaluar_ensamble()