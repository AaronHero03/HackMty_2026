from pathlib import Path
from catboost import CatBoostClassifier
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from training.utils import cargar_datos, evaluar_y_reportar


def calcular_permutation_importance_hard(
    modelos_dict, X_val, y_val, feature_names, n_repeats=5
):
    print("🔍 Calculando Permutation Importance para el Hard Voting Ensemble...")
    y_val_arr = np.array(y_val)

    def predict_hard(X_data):
        pred_cb = modelos_dict["catboost"].predict(X_data).flatten()
        pred_et = modelos_dict["extratrees"].predict(X_data).flatten()
        pred_svm = modelos_dict["svm"].predict(X_data).flatten()
        return ((pred_cb + pred_et + pred_svm) >= 2).astype(int)

    preds_base = predict_hard(X_val)
    acc_base = accuracy_score(y_val_arr, preds_base)

    importances = []
    for col in X_val.columns:
        drop_accs = []
        for _ in range(n_repeats):
            X_permuted = X_val.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)

            preds_perm = predict_hard(X_permuted)
            acc_perm = accuracy_score(y_val_arr, preds_perm)

            drop_accs.append(acc_base - acc_perm)

        importances.append(np.mean(drop_accs))

    return pd.Series(importances, index=feature_names)


def entrenar_evaluar_hard_ensemble():
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
        return

    print("📦 Cargando modelos para el Hard Voting Ensemble...")

    cb = CatBoostClassifier()
    cb.load_model(ruta_cb)
    et = joblib.load(ruta_et)
    svm = joblib.load(ruta_svm)

    X_val = datos["X_val"]
    y_val = datos["y_val"]

    print("🗳️ Calculando combinación por Hard Voting (Mayoría simple)...")

    # Predicciones discretas (0 o 1)
    pred_cb = cb.predict(X_val).flatten()
    pred_et = et.predict(X_val).flatten()
    pred_svm = svm.predict(X_val).flatten()

    # Suma de votos: si 2 o más modelos votan 1 (IA), la etiqueta final es 1
    votos = pred_cb + pred_et + pred_svm
    y_pred_hard = (votos >= 2).astype(int)

    # Generación de reportes en docs/hard_ensemble/
    dir_out = evaluar_y_reportar(
        "Hard Ensemble", y_val, y_pred_hard, datos["val_df"]
    )

    # Permutation Importance
    modelos_dict = {"catboost": cb, "extratrees": et, "svm": svm}
    importancias = calcular_permutation_importance_hard(
        modelos_dict, X_val, y_val, datos["X_train"].columns
    )
    top15 = importancias.nlargest(15).sort_values(ascending=True)

    ruta_grafica = dir_out / "importancia.png"
    plt.figure(figsize=(10, 8))
    top15.plot(kind="barh", color="#7b1fa2")
    plt.title(
        "Hard Voting Ensemble - Top 15 Métricas por Permutation Importance"
    )
    plt.xlabel("Caída Promedio en Accuracy al Permutar Variable")
    plt.tight_layout()
    plt.savefig(ruta_grafica, dpi=150)
    plt.close()
    print(f"📉 Gráfica de importancia del Ensamble guardada en: {ruta_grafica}")


if __name__ == "__main__":
    entrenar_evaluar_hard_ensemble()