import xgboost as xgb
import pandas as pd
import glob
from pathlib import Path

# Importa tus módulos de procesamiento (ajusta las rutas según tu estructura)
from src.fase1.audio_base import procesar_audio_base, recortar_voz_activa
from src.fase2.acustica import aplicar_filtro_pasabanda, extraer_metricas_acusticas
from src.fase3.conversacional import extraer_metricas_tiempo
from src.tools.vad import generar_turnos_vad, fusionar_turnos # Asegúrate de tener la función de fusión aquí

def probar_audios_locales(directorio_muestras, ruta_modelo="modelo_xgboost_altur_v4.json"):
    # 1. Cargar el modelo XGBoost
    print(f"Cargando modelo desde {ruta_modelo}...")
    modelo = xgb.XGBClassifier()
    modelo.load_model(ruta_modelo)
    
    # 2. Buscar todos los archivos WAV en la carpeta de muestras
    archivos_wav = glob.glob(f"{directorio_muestras}/*.wav")
    
    if not archivos_wav:
        print("No se encontraron archivos .wav en el directorio.")
        return

    print(f"\nProcesando {len(archivos_wav)} muestras...\n" + "-"*40)

    for ruta_audio in archivos_wav:
        nombre_archivo = Path(ruta_audio).name
        try:
            # --- PIPELINE DE PROCESAMIENTO ---
            
            # A) Detección de Actividad de Voz (VAD) y limpieza de micropausas
            datos_turnos = generar_turnos_vad(ruta_audio)
            turnos_limpios = fusionar_turnos(datos_turnos["turns"], max_pausa_s=0.5)
            
            turnos_llamador = [t for t in turnos_limpios if t["channel"] == 0]
            turnos_agente = [t for t in turnos_limpios if t["channel"] == 1]
            
            # B) Fase 1: Recortar y Normalizar Canal 0
            y_c0_norm, sr = procesar_audio_base(ruta_audio)
            voz_recortada = recortar_voz_activa(y_c0_norm, turnos_llamador, sr)
            
            # C) Fase 2: Matemáticas Acústicas
            voz_filtrada = aplicar_filtro_pasabanda(voz_recortada, sr)
            metricas_ac = extraer_metricas_acusticas(voz_filtrada, sr)
            
            # D) Fase 3: Matemáticas de Tiempos
            metricas_tiempo = extraer_metricas_tiempo(
                tramos_agente=turnos_agente, 
                tramos_llamante=turnos_llamador
            )
            
            # --- INFERENCIA ---
            
            # Empaquetar todo en una fila de DataFrame
            fila = {**metricas_ac, **metricas_tiempo}
            df_inferencia = pd.DataFrame([fila])
            
            # Asegurar que el DataFrame tiene el mismo orden de columnas que el entrenamiento
            # XGBoost usualmente maneja esto por los nombres de las columnas, pero es buena práctica
            columnas_esperadas = modelo.feature_names_in_
            
            # Rellenar con NaN si falta alguna columna (por protección)
            for col in columnas_esperadas:
                if col not in df_inferencia.columns:
                    df_inferencia[col] = float('nan')
            
            df_final = df_inferencia[columnas_esperadas]
            
            # Ejecutar modelo
            prediccion = int(modelo.predict(df_final)[0])
            prob_ia = float(modelo.predict_proba(df_final)[0][1])
            
            # Mostrar resultados
            #veredicto = "🤖 SINTÉTICO (IA)" if prediccion == 1 else "👤 HUMANO"
            
            if prob_ia >= 0.40:  # Umbral más defensivo
                veredicto = "🤖 Synthetic"
            else:
                veredicto = "👤 Human"
            
            confianza = prob_ia * 100 if prediccion == 1 else (1 - prob_ia) * 100
            
            
            print(f" Vector generado: {fila}")
            print(f"🎙️ {nombre_archivo}")
            print(f"   Veredicto: {veredicto}")
            print(f"   Confianza: {confianza:.2f}%")
            print(f"   -> Latencia detectada: {fila.get('latencia_mediana', 0):.2f}s | Vocoder Peak: {fila.get('vocoder_periodicity_peak', 0):.2f}\n")
            
        except Exception as e:
            print(f"⚠️ Error procesando {nombre_archivo}: {e}\n")

if __name__ == "__main__":
    probar_audios_locales("tests")