from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from training.utils import cargar_datos, evaluar_y_reportar


def entrenar_lda():
    datos = cargar_datos()

    root = Path(__file__).resolve().parent.parent
    dir_models = root / "models"
    dir_models.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )

    print("🚀 Entrenando Linear Discriminant Analysis (LDA)...")
    pipeline.fit(datos["X_train"], datos["y_train"])

    y_pred = pipeline.predict(datos["X_val"])
    dir_out = evaluar_y_reportar("LDA", datos["y_val"], y_pred, datos["val_df"])

    # Magnitud absoluta de coeficientes
    modelo_lda = pipeline.named_steps["lda"]
    coeficientes = np.abs(modelo_lda.coef_[0])
    importancias = pd.Series(coeficientes, index=datos["X_train"].columns)
    top15 = importancias.nlargest(15).sort_values(ascending=True)

    ruta_grafica = dir_out / "importancia.png"
    plt.figure(figsize=(10, 8))
    top15.plot(kind="barh", color="#6a1b9a")
    plt.title("LDA - Top 15 Coeficientes Discriminantes")
    plt.xlabel("Valor Absoluto del Coeficiente")
    plt.tight_layout()
    plt.savefig(ruta_grafica, dpi=150)
    plt.close()
    print(f"📉 Gráfica de importancia guardada en: {ruta_grafica}")

    # Guardar Pipeline
    ruta_modelo = dir_models / "modelo_lda_altur.joblib"
    joblib.dump(pipeline, ruta_modelo)
    print(f"💾 Modelo guardado en: {ruta_modelo}")


if __name__ == "__main__":
    entrenar_lda()