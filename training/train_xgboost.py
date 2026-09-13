import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

import pandas as pd
def entrenar_modelo(ruta_csv):
    print(f"Cargando datos desde: {ruta_csv}...")
    df = pd.read_csv(ruta_csv)

    # 1. Preparar las etiquetas (XGBoost necesita 0 y 1)
    # Suponiendo que las etiquetas son 'human' y 'synthetic' (ajusta si son diferentes)
    df['target'] = df['label'].apply(lambda x: 1 if x == 'synthetic' else 0)

    # 2. Separar según la columna 'split' original del dataset
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val'] # o 'test' dependiendo de tu dataset

    # Quitar columnas que no son características numéricas
    columnas_a_ignorar = ['anon_id', 'label', 'split', 'target']
    X_train = train_df.drop(columns=columnas_a_ignorar)
    y_train = train_df['target']
    X_val = val_df.drop(columns=columnas_a_ignorar)
    y_val = val_df['target']

    print(f"Entrenando con {len(X_train)} ejemplos. Validando con {len(X_val)} ejemplos.")

    # 3. Inicializar y entrenar el modelo XGBoost
    # Estos parámetros base suelen funcionar muy bien "out of the box"
    # 3. Inicializar el modelo XGBoost
    modelo = xgb.XGBClassifier(
      n_estimators=200,
      learning_rate=0.05,
      max_depth=5,
      colsample_bytree=0.5,  # <-- Agrega esta línea
      random_state=42,
      eval_metric='logloss',
      early_stopping_rounds=10
    )


    print("Entrenando modelo XGBoost...")
    modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 4. Hacer predicciones en validación
    y_pred = modelo.predict(X_val)

    # Guardamos las predicciones en una copia del dataframe de validación
    val_df_copy = val_df.copy()
    val_df_copy['prediccion'] = y_pred

    # Filtramos las filas donde la predicción no coincide con la realidad
    errores = val_df_copy[val_df_copy['target'] != val_df_copy['prediccion']]

    print("\n🕵️‍♂️ LLAMADAS QUE ENGAÑARON AL MODELO:")
    # Muestra el ID de la llamada, su etiqueta real y qué adivinó el modelo (0=Humano, 1=IA)
    print(errores[['anon_id', 'label', 'prediccion']])

    # 5. Evaluar los resultados
    print("\n--- RESULTADOS EN VALIDACIÓN ---")
    print(f"Precisión (Accuracy): {accuracy_score(y_val, y_pred) * 100:.2f}%\n")
    print("Reporte detallado:")
    print(classification_report(y_val, y_pred, target_names=['Humano (0)', 'IA (1)']))

    # 6. Graficar Importancia de las Características (Feature Importance)
    print("Generando gráfica de importancia de características...")
    plt.figure(figsize=(10, 8))
    # Selecciona las 15 métricas más importantes para no saturar la gráfica
    xgb.plot_importance(modelo, max_num_features=15, importance_type='weight')
    plt.title("Top 15 Métricas que delatan a la IA")
    plt.tight_layout()
    plt.savefig("importancia_metricas.png", dpi=150)
    print("Gráfica guardada como 'importancia_metricas.png'. ¡Úsala en la presentación!")

    ruta_modelo = "modelo_xgboost_altur.json"
    modelo.save_model(ruta_modelo)
    print(f"\n¡Modelo guardado exitosamente en: {ruta_modelo}!")

    return modelo

if __name__ == "__main__":
    # Asegúrate de tener instalados: pip install pandas xgboost scikit-learn matplotlib
    modelo_entrenado = entrenar_modelo("features_final.csv")