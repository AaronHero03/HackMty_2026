from pathlib import Path
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pandas as pd
from training.utils import cargar_datos, evaluar_y_reportar


def entrenar_catboost():
    datos = cargar_datos()

    root = Path(__file__).resolve().parent.parent
    dir_models = root / "models"
    dir_models.mkdir(parents=True, exist_ok=True)

    modelo = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=5,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=10,
    )

    print("🚀 Entrenando CatBoost con Early Stopping...")
    modelo.fit(
        datos["X_tr"],
        datos["y_tr"],
        eval_set=(datos["X_es"], datos["y_es"]),
        verbose=False,
    )

    y_pred = modelo.predict(datos["X_val"])
    dir_out = evaluar_y_reportar(
        "CatBoost", datos["y_val"], y_pred, datos["val_df"]
    )

    # Gráfica de Importancia
    importancias = pd.Series(
        modelo.get_feature_importance(), index=datos["X_train"].columns
    )
    top15 = importancias.nlargest(15).sort_values(ascending=True)

    ruta_grafica = dir_out / "importancia.png"
    plt.figure(figsize=(10, 8))
    top15.plot(kind="barh", color="#e65100")
    plt.title("CatBoost - Top 15 Métricas por Importancia")
    plt.xlabel("Feature Importance Score")
    plt.tight_layout()
    plt.savefig(ruta_grafica, dpi=150)
    plt.close()
    print(f"📉 Gráfica de importancia guardada en: {ruta_grafica}")

    # Guardar Modelo
    ruta_modelo = dir_models / "modelo_catboost_altur.cbm"
    modelo.save_model(str(ruta_modelo))
    print(f"💾 Modelo guardado en: {ruta_modelo}")


if __name__ == "__main__":
    entrenar_catboost()