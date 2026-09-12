import soundfile as sf
import librosa
import numpy as np

def simular_llamada_telefonica(ruta_cliente, ruta_agente, archivo_salida, latencia_segundos=0.8):
    """
    Une dos audios independientes en una pista estéreo de 8kHz simulando turnos.
    Canal 1 (Der) = Agente empieza a hablar.
    Canal 0 (Izq) = Cliente responde con 'latencia_segundos' de pausa.
    """
    print(f"Ensamblando {archivo_salida}...")
    
    # 1. Leer y forzar a 8000 Hz Mono
    cliente, _ = librosa.load(ruta_cliente, sr=8000, mono=True)
    agente, _ = librosa.load(ruta_agente, sr=8000, mono=True)
    
    # 2. Calcular duraciones y pausas
    muestras_agente = len(agente)
    muestras_latencia = int(latencia_segundos * 8000)
    muestras_cliente = len(cliente)
    
    longitud_total = muestras_agente + muestras_latencia + muestras_cliente
    
    # 3. Crear lienzo estéreo vacío
    estereo = np.zeros((longitud_total, 2), dtype=np.float32)
    
    # 4. Insertar audios en sus respectivos canales y tiempos
    estereo[0:muestras_agente, 1] = agente  # Canal 1
    
    inicio_cliente = muestras_agente + muestras_latencia
    estereo[inicio_cliente:inicio_cliente + muestras_cliente, 0] = cliente # Canal 0
    
    # 5. Exportar a WAV de 16-bits (formato estándar de telefonía)
    sf.write(archivo_salida, estereo, 8000, subtype='PCM_16')
    print("¡Llamada de prueba lista!")

# Uso:
simular_llamada_telefonica("tests/ElevenLabs2.mp3", "tests/ElevenLabs1.mp3", "tests/demo_hackathon.wav")