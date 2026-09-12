import pandas as pd
from pathlib import Path

# Importamos los contratos de cada miembro del equipo
import fase1.main as f1
import fase2.fase2_acustica as f2
import fase3.fase3_conversacional as f3

def main():
    # Ajusta esta ruta si es necesario
    ruta_base = Path("~/Workspace/HackMty2026/Tigres_Del_Sur/Altur_Data").expanduser()
    
    # 1. Fase 1 arranca leyendo el mapa
    manifest = f1.leer_manifiesto(ruta_base)
    datos_procesados = []
    
    print("Iniciando orquestación del dataset...")
    
    for idx, fila in manifest.iterrows():
        id_llamada = fila["anon_id"]
        ruta_wav = ruta_base / "audio" / f"{id_llamada}.wav"
        ruta_json = ruta_base / "turns" / f"{id_llamada}.json"
        
        try:
            # --- Fase 1: Extracción base ---
            y_c0_norm, sr = f1.procesar_audio_base(ruta_wav)
            turnos_llamador, turnos_agente = f1.cargar_turnos(ruta_json)
            voz_recortada = f1.recortar_voz_activa(y_c0_norm, turnos_llamador, sr)
            
            # --- Fase 2: DSP (Biomédica) ---
            voz_filtrada = f2.aplicar_filtro_pasabanda(voz_recortada)
            metricas_acusticas = f2.extraer_metricas_acusticas(voz_filtrada, sr)
            
            # --- Fase 3: Lógica conversacional ---
            metricas_tiempo = f3.extraer_metricas_tiempo(id_llamada, ruta_base)
            
            # --- Fusión ---
            fila_final = {
                "anon_id": id_llamada,
                "label": fila["label"],
                "split": fila["split"]
            }
            fila_final.update(metricas_acusticas)
            fila_final.update(metricas_tiempo)
            
            datos_procesados.append(fila_final)
            
        except Exception as e:
            print(f"Error procesando {id_llamada}: {e}")
            
        if (idx + 1) % 50 == 0:
            print(f"  Procesadas {idx + 1}/{len(manifest)} llamadas...")
            
    # Exportación final
    df_final = pd.DataFrame(datos_procesados)
    ruta_salida = ruta_base / "features_final.csv"
    df_final.to_csv(ruta_salida, index=False)
    print(f"\n¡Matriz ensamblada y lista en {ruta_salida}!")

if __name__ == "__main__":
    main()