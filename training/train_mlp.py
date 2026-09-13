from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from training.utils import cargar_datos, evaluar_y_reportar


def entrenar_mlp():
    datos = cargar_datos()

    root = Path(__file__).resolve().parent.parent
    dir_models = root / "models"
    dir_models.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=400,
                    alpha=0.01,
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        ]
    )

    print("🚀 Entrenando Multi-Layer Perceptron (MLP)...")
    pipeline.fit(datos["X_train"], datos["y_train"])

    y_pred = pipeline.predict(datos["X_val"])
    dir_out = evaluar_y_reportar("MLP", datos["y_val"], y_pred, datos["val_df"])

    # Permutation Importance
    print("🔍 Calculando Permutation Importance para MLP...")
    perm_importance = permutation_importance(
        pipeline,
        datos["X_val"],
        datos["y_val"],
        n_repeats=10,
        random_state=42,
    )

    importancias = pd.Series(
        perm_importance.importances_mean, index=datos["X_train"].columns
    )
    top15 = importancias.nlargest(15).sort_values(ascending=True)

    ruta_grafica = dir_out / "importancia.png"
    plt.figure(figsize=(10, 8))
    top15.plot(kind="barh", color="#0288d1")
    plt.title("MLP - Top 15 Métricas por Permutation Importance")
    plt.xlabel("Caída Promedio en Accuracy al Permutar Variable")
    plt.tight_layout()
    plt.savefig(ruta_grafica, dpi=150)
    plt.close()
    print(f"📉 Gráfica de importancia guardada en: {ruta_grafica}")

    # Guardar Pipeline
    ruta_modelo = dir_models / "modelo_mlp_altur.joblib"
    joblib.dump(pipeline, ruta_modelo)
    print(f"💾 Modelo guardado en: {ruta_modelo}")


if __name__ == "__main__":
    entrenar_mlp()