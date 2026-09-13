from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from training.utils import cargar_datos, evaluar_y_reportar


def entrenar_rf():
    datos = cargar_datos()

    root = Path(__file__).resolve().parent.parent
    dir_models = root / "models"
    dir_models.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("🚀 Entrenando Random Forest...")
    pipeline.fit(datos["X_train"], datos["y_train"])

    y_pred = pipeline.predict(datos["X_val"])
    dir_out = evaluar_y_reportar(
        "RandomForest", datos["y_val"], y_pred, datos["val_df"]
    )

    # Gráfica de Importancia
    modelo_rf = pipeline.named_steps["rf"]
    importancias = pd.Series(
        modelo_rf.feature_importances_, index=datos["X_train"].columns
    )
    top15 = importancias.nlargest(15).sort_values(ascending=True)

    ruta_grafica = dir_out / "importancia.png"
    plt.figure(figsize=(10, 8))
    top15.plot(kind="barh", color="#2e7d32")
    plt.title("Random Forest - Top 15 Métricas por Importancia")
    plt.xlabel("Feature Importance Score")
    plt.tight_layout()
    plt.savefig(ruta_grafica, dpi=150)
    plt.close()
    print(f"📉 Gráfica de importancia guardada en: {ruta_grafica}")

    # Guardar Pipeline
    ruta_modelo = dir_models / "modelo_rf_altur.joblib"
    joblib.dump(pipeline, ruta_modelo)
    print(f"💾 Modelo guardado en: {ruta_modelo}")


if __name__ == "__main__":
    entrenar_rf()