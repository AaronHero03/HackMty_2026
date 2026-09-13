import joblib
from sklearn.ensemble import RandomForestClassifier
from training.utils import cargar_datos, evaluar_y_reportar


def entrenar_random_forest():
    datos = cargar_datos()

    modelo = RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
    )

    print("Entrenando Random Forest...")
    modelo.fit(datos["X_train"], datos["y_train"])

    y_pred = modelo.predict(datos["X_val"])
    evaluar_y_reportar("Random Forest", datos["y_val"], y_pred, datos["val_df"])

    joblib.dump(modelo, "modelo_rf_altur.joblib")
    print("Modelo guardado en modelo_rf_altur.joblib")


if __name__ == "__main__":
    entrenar_random_forest()