from pathlib import Path
import matplotlib.pyplot as plt
from training.utils import cargar_datos, evaluar_y_reportar
import xgboost as xgb


def entrenar_xgboost():
    datos = cargar_datos()

    modelo = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        colsample_bytree=0.5,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=10,
    )

    print("Entrenando XGBoost con Early Stopping...")
    modelo.fit(
        datos["X_tr"],
        datos["y_tr"],
        eval_set=[(datos["X_es"], datos["y_es"])],
        verbose=False,
    )

    y_pred = modelo.predict(datos["X_val"])
    evaluar_y_reportar("XGBoost", datos["y_val"], y_pred, datos["val_df"])

    # Gráfica de Importancia (Gain)
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(
        modelo, max_num_features=15, importance_type="gain", title="XGBoost - Top 15 Métricas por Gain"
    )
    plt.tight_layout()
    plt.savefig("importancia_xgboost_gain.png", dpi=150)
    print("Gráfica guardada como 'importancia_xgboost_gain.png'")

    # Guardar Artefacto
    ruta_guardado = Path("modelo_xgboost_altur.json")
    modelo.save_model(ruta_guardado)
    print(f"Modelo guardado en {ruta_guardado}")


if __name__ == "__main__":
    entrenar_xgboost()