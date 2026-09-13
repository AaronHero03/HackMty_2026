from pathlib import Path
from catboost import CatBoostClassifier
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from training.train_cnn import CNN1D
from training.utils import cargar_datos
import xgboost as xgb


def evaluar_modelo(y_true, y_pred, nombre_modelo):
    acc = accuracy_score(y_true, y_pred)
    prec_h = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_h = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_h = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    prec_ia = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_ia = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_ia = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return {
        "Modelo": nombre_modelo,
        "Accuracy": acc,
        "Macro F1": macro_f1,
        "F1 Humano": f1_h,
        "F1 IA": f1_ia,
        "Precision Humano": prec_h,
        "Recall Humano": rec_h,
        "Precision IA": prec_ia,
        "Recall IA": rec_ia,
    }


def main():
    datos = cargar_datos()
    root = Path(__file__).resolve().parent.parent
    dir_models = root / "models"
    dir_docs = root / "docs"
    dir_docs.mkdir(parents=True, exist_ok=True)

    X_val = datos["X_val"]
    y_val = datos["y_val"]

    resultados = []

    # 1. Modelos scikit-learn (.joblib)
    mapa_sklearn = {
        "SVM": dir_models / "modelo_svm_altur.joblib",
        "ExtraTrees": dir_models / "modelo_et_altur.joblib",
        "RandomForest": dir_models / "modelo_rf_altur.joblib",
        "LDA": dir_models / "modelo_lda_altur.joblib",
        "MLP": dir_models / "modelo_mlp_altur.joblib",
    }

    for nombre, ruta in mapa_sklearn.items():
        if ruta.exists():
            modelo = joblib.load(ruta)
            y_pred = modelo.predict(X_val)
            resultados.append(evaluar_modelo(y_val, y_pred, nombre))

    # 2. CatBoost (.cbm)
    ruta_cb = dir_models / "modelo_catboost_altur.cbm"
    if ruta_cb.exists():
        cb = CatBoostClassifier()
        cb.load_model(ruta_cb)
        y_pred = cb.predict(X_val)
        resultados.append(evaluar_modelo(y_val, y_pred, "CatBoost"))

    # 3. XGBoost (.json)
    ruta_xgb = dir_models / "modelo_xgboost_altur.json"
    if ruta_xgb.exists():
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(ruta_xgb)
        y_pred = xgb_model.predict(X_val)
        resultados.append(evaluar_modelo(y_val, y_pred, "XGBoost"))

    # 4. PyTorch CNN 1D (.pt)
    ruta_cnn = dir_models / "best_cnn_weights.pt"
    ruta_prep_cnn = dir_models / "preprocesadores_cnn_altur.joblib"

    if ruta_cnn.exists() and ruta_prep_cnn.exists():
        preprocs = joblib.load(ruta_prep_cnn)
        X_val_clean = preprocs["imputer"].transform(X_val)
        X_val_scaled = preprocs["scaler"].transform(X_val_clean)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn = CNN1D(num_features=X_val.shape[1]).to(device)
        cnn.load_state_dict(torch.load(ruta_cnn, map_location=device))
        cnn.eval()

        X_val_tensor = (
            torch.tensor(X_val_scaled, dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        with torch.no_grad():
            probs = cnn(X_val_tensor).cpu().numpy().flatten()
        y_pred_cnn = (probs >= 0.5).astype(int)

        resultados.append(evaluar_modelo(y_val, y_pred_cnn, "CNN 1D (PyTorch)"))

    if not resultados:
        print("⚠️ No se encontraron modelos guardados en la carpeta 'models/'")
        return

    df_res = (
        pd.DataFrame(resultados)
        .sort_values(by="Accuracy", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 65)
    print("📊 COMPARATIVA GLOBAL DE MODELOS EN VALIDACIÓN")
    print("=" * 65)
    print(
        df_res[
            ["Modelo", "Accuracy", "Macro F1", "F1 Humano", "F1 IA"]
        ].to_string(index=False, float_format=lambda x: f"{x * 100:.2f}%")
    )
    print("=" * 65 + "\n")

    x = np.arange(len(df_res))
    width = 0.35

    # --- GRÁFICA 1: Accuracy vs Macro F1 ---
    plt.figure(figsize=(11, 6))
    plt.bar(
        x - width / 2,
        df_res["Accuracy"] * 100,
        width,
        label="Accuracy (%)",
        color="#1f77b4",
    )
    plt.bar(
        x + width / 2,
        df_res["Macro F1"] * 100,
        width,
        label="Macro F1 (%)",
        color="#ff7f0e",
    )
    plt.ylabel("Porcentaje (%)")
    plt.title("Rendimiento General de Modelos (Accuracy vs Macro F1)")
    plt.xticks(x, df_res["Modelo"], rotation=15)
    plt.ylim(70, 103)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    ruta_comp = dir_docs / "comparativa_modelos.png"
    plt.savefig(ruta_comp, dpi=150)
    plt.close()

    # --- GRÁFICA 2: F1-Score por Clase (Humano vs IA) ---
    plt.figure(figsize=(11, 6))
    plt.bar(
        x - width / 2,
        df_res["F1 Humano"] * 100,
        width,
        label="F1 Humano (Clase 0)",
        color="#2ca02c",
    )
    plt.bar(
        x + width / 2,
        df_res["F1 IA"] * 100,
        width,
        label="F1 IA (Clase 1)",
        color="#d62728",
    )
    plt.ylabel("F1-Score (%)")
    plt.title("Comparativa de F1-Score por Clase (Humano vs IA)")
    plt.xticks(x, df_res["Modelo"], rotation=15)
    plt.ylim(70, 103)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    ruta_f1_clase = dir_docs / "f1_por_clase.png"
    plt.savefig(ruta_f1_clase, dpi=150)
    plt.close()

    print(f"🖼️ Gráfica general guardada en: {ruta_comp}")
    print(f"🖼️ Gráfica F1 por clase guardada en: {ruta_f1_clase}")


if __name__ == "__main__":
    main()