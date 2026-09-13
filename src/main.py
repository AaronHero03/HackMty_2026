import json
from pathlib import Path
import pandas as pd

# Importamos los contratos de cada miembro del equipo
import src.fase1.audio_base as f1
import src.fase2.acustica as f2
import src.fase3.conversacional as f3
from src.tools.vad import fusionar_turnos, generar_turnos_vad


def main():
    root = Path(__file__).resolve().parent.parent
    ruta_base = root / "Altur_Data"

    # Carpeta dedicada para guardar los nuevos JSONs de VAD (sin tocar 'turns/')
    carpeta_turns_vad = ruta_base / "turns_vad"
    carpeta_turns_vad.mkdir(parents=True, exist_ok=True)

    # 1. Fase 1 arranca leyendo el mapa
    manifest = f1.leer_manifiesto(ruta_base)
    datos_procesados = []

    print("Iniciando orquestación del dataset (Modo VAD Dinámico + Caché)...")

    for idx, fila in manifest.iterrows():
        id_llamada = fila["anon_id"]
        ruta_wav = ruta_base / "audio" / f"{id_llamada}.wav"
        ruta_json_vad = carpeta_turns_vad / f"{id_llamada}.json"

        try:
            # --- Fase 1: Extracción base ---
            y_c0_norm, sr = f1.procesar_audio_base(ruta_wav)

            # --- VAD Dinámico con lectura de Caché ---
            if ruta_json_vad.exists():
                # Si ya fue calculado anteriormente, se carga directo desde el JSON guardado
                with open(ruta_json_vad, "r", encoding="utf-8") as f:
                    datos_turnos = json.load(f)
            else:
                # Si no existe, se calcula y se guarda en la carpeta nueva 'turns_vad/'
                datos_turnos = generar_turnos_vad(str(ruta_wav))
                with open(ruta_json_vad, "w", encoding="utf-8") as f:
                    json.dump(datos_turnos, f, indent=2, ensure_ascii=False)

            # Filtrado y separación de turnos por canal
            turnos_limpios = fusionar_turnos(datos_turnos["turns"], max_pausa_s=0.5)
            turnos_llamador = [t for t in turnos_limpios if t["channel"] == 0]
            turnos_agente = [t for t in turnos_limpios if t["channel"] == 1]

            voz_recortada = f1.recortar_voz_activa(y_c0_norm, turnos_llamador, sr)

            # --- Fase 2: DSP (Biomédica) ---
            voz_filtrada = f2.aplicar_filtro_pasabanda(voz_recortada)
            metricas_acusticas = f2.extraer_metricas_acusticas(voz_filtrada, sr)

            # --- Fase 3: Lógica conversacional ---
            metricas_tiempo = f3.extraer_metricas_tiempo(tramos_agente=turnos_agente, tramos_llamante=turnos_llamador)

            # --- Fusión ---
            fila_final = {
                "anon_id": id_llamada,
                "label": fila["label"],
                "split": fila["split"],
            }
            fila_final.update(metricas_acusticas)
            fila_final.update(metricas_tiempo)

            datos_procesados.append(fila_final)

        except Exception as e:
            print(f"Error procesando {id_llamada}: {e}")

        if (idx + 1) % 50 == 0:
            print(f"  Procesadas {idx + 1}/{len(manifest)} llamadas...")

    # Exportación final manteniendo separado el CSV original del nuevo
    df_final = pd.DataFrame(datos_procesados)
    ruta_salida = ruta_base / "features_final_vad.csv"
    df_final.to_csv(ruta_salida, index=False)
    print(f"\n¡Matriz ensamblada con VAD dinámico lista en {ruta_salida}!")


if __name__ == "__main__":
    main()