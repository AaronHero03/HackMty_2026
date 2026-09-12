import os
import json

def detectar_canales(ruta_carpeta):
    # Recorrer todos los archivos en la carpeta especificada
    for nombre_archivo in os.listdir(ruta_carpeta):
        if nombre_archivo.endswith(".json"):
            ruta_archivo = os.path.join(ruta_carpeta, nombre_archivo)
            
            with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                try:
                    datos = json.load(archivo)
                    
                    # Extraer los canales únicos de la lista "turns"
                    canales = set(turno.get("channel") for turno in datos.get("turns", []))
                    cantidad_canales = len(canales)
                    
                    # Imprimir el resultado según la cantidad de canales encontrados
                    if cantidad_canales == 1:
                        print(f"{nombre_archivo}: 1 canal detectado {list(canales)}")
                    elif cantidad_canales >= 2:
                        print(f"{nombre_archivo}: {cantidad_canales} canales detectados {list(canales)}")
                    else:
                        print(f"{nombre_archivo}: No se encontraron canales (archivo vacío o formato incorrecto)")
                        
                except json.JSONDecodeError:
                    print(f"Error al leer {nombre_archivo}. Asegúrate de que el formato sea válido.")

# Reemplaza esto con la ruta donde tienes guardados tus archivos JSON
ruta_de_tus_jsons = "/home/aaronhero/Workspace/HackMty2026/Altur/turns" 
detectar_canales(ruta_de_tus_jsons)