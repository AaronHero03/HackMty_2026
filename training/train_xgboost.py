from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from training.utils import cargar_datos, evaluar_y_reportar


def entrenar_xgboost():
    datos = cargar_datos()

    root = Path(__file__).resolve().parent.parent
    dir_models = root / "models"
    dir_models.mkdir(parents=True, exist_ok=True)

    modelo = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=15,
    )

    print("🚀 Entrenando XGBoost con Early Stopping...")
    modelo.fit(
        datos["X_tr"],
        datos["y_tr"],
        eval_set=[(datos["X_es"], datos["y_es"])],
        verbose=False,
    )

    y_pred = modelo.predict(datos["X_val"])
    dir_out = evaluar_y_reportar(
        "XGBoost", datos["y_val"], y_pred, datos["val_df"]
    )

    # Gráfica de Importancia
    importancias = pd.Series(
        modelo.feature_importances_, index=datos["X_train"].columns
    )
    top15 = importancias.nlargest(15).sort_values(ascending=True)

    ruta_grafica = dir_out / "importancia.png"
    plt.figure(figsize=(10, 8))
    top15.plot(kind="barh", color="#1b5e20")
    plt.title("XGBoost - Top 15 Métricas por Importancia")
    plt.xlabel("Feature Importance Score")
    plt.tight_layout()
    plt.savefig(ruta_grafica, dpi=150)
    plt.close()
    print(f"📉 Gráfica de importancia guardada en: {ruta_grafica}")

    # Guardar modelo
    ruta_modelo = dir_models / "modelo_xgboost_altur.json"
    modelo.save_model(ruta_modelo)
    print(f"💾 Modelo guardado en: {ruta_modelo}")


if __name__ == "__main__":
    entrenar_xgboost()