from math import gcd
import io
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

# Carga las variables definidas en el archivo .env
load_dotenv()

# Lee la API Key desde las variables de entorno
API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Define la ruta absoluta de la carpeta en el mismo directorio que este script
DIRECTORIO_SCRIPT = Path(__file__).resolve().parent
CARPETA_SALIDA = DIRECTORIO_SCRIPT / "ElevenLabs"

# Lista con los IDs de las conversaciones que deseas descargar y procesar
CONVERSATION_IDS = [
    "conv_2501m2cfw3fnetvbrrcm96rn0xew",
    "conv_1201m2cfgqtffgrvc51et7vm04yg",
    "conv_1601m2cf81hxfab8t1cqjkvafr53",
    "conv_7901m2ceq1yafy29g45kda1wa6yt",
    "conv_5101m2cee157fhx8t56vvt0ceehb",
    "conv_7601m2cdxy3wf77txa7avdfxd62m",
    "conv_3101m2cd3dqgfjsa2745gf50assg",
]


def descargar_y_procesar_audio(conv_id: str, api_key: str, destino: Path):
    url = f"https://api.elevenlabs.io/v1/convai/conversations/{conv_id}/audio"
    headers = {"xi-api-key": api_key}
    print(f"Procesando {conv_id}...")

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # 1. Leer buffer de audio desde memoria
        data, sr_orig = sf.read(io.BytesIO(response.content))

        # 2. Convertir a estéreo (2 canales)
        if data.ndim == 1:
            data = np.column_stack((data, data))
        elif data.shape[1] > 2:
            data = data[:, :2]

        # 3. Cambiar tasa de muestreo a 8000 Hz si difiere
        if sr_orig != 8000:
            g = gcd(8000, sr_orig)
            data = resample_poly(data, 8000 // g, sr_orig // g, axis=0)

        # 4. Guardar archivo final
        ruta_archivo = destino / f"{conv_id}_8khz_stereo.wav"
        sf.write(ruta_archivo, data, 8000)
        print(f"  └─ Guardado exitoso: {ruta_archivo}")

    except requests.exceptions.HTTPError as http_err:
        print(f"  └─ Error de API HTTP en {conv_id}: {http_err}")
    except Exception as e:
        print(f"  └─ Error inesperado en {conv_id}: {e}")


def main():
    if not API_KEY:
        print("Error: No se encontró 'ELEVENLABS_API_KEY' en el archivo .env")
        return

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    for conv_id in CONVERSATION_IDS:
        descargar_y_procesar_audio(conv_id, API_KEY, CARPETA_SALIDA)
    print("\n¡Proceso finalizado!")


if __name__ == "__main__":
    main()