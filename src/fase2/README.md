# Fase 2 — Procesamiento Acústico y Análisis Biomédico

**Pregunta:** ¿Podemos extraer huellas biomédicas de motores de síntesis de voz (TTS/vocoders) en audio telefónico de 8 kHz sin caer en trampas de volumen o artefactos fuera de la banda telefónica?

**Resultado en una frase:** Sí. Filtrando estrictamente la banda telefónica (300–3400 Hz) y usando Coeficientes Cepstrales de Frecuencia Lineal (LFCC), generamos 24 características espectrales que capturan la rigidez sintética y los artefactos de fase del vocoder sin sufrir por la baja tasa de muestreo.

---

## Qué hay

### Código para todo el equipo

| Archivo         | Qué hace                                                                                                                                                                                                                                              |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/dsp.py`    | `aplicar_filtro_pasabanda(y)`: filtro IIR Butterworth SOS (300–3400 Hz). `banco_filtros_lineal()`: filtros triangulares en escala lineal. `extraer_features_acusticas(y_voz)`: devuelve 12 medianas y 12 desviaciones estándar de LFCC (24 features). |
| `fase2/main.py` | Extrae las características de todo el dataset procesando la voz activa y guarda la matriz limpia en `fase2/resultados/features_acusticas.csv`.                                                                                                        |

### Uso

```python
from src.audio import cargar_llamada, normalizar
from src.dsp import extraer_features_acusticas

llama, _ = cargar_llamada("Altur/audio/<anon_id>.wav")
y_norm = normalizar(llama)

# Extrae el vector de 24 características acústicas
features = extraer_features_acusticas(y_norm)
```

Correr siempre desde la raíz del repo (`HackMty_2026/`) usando:

```bash
python -m fase2.main --datos Altur
```

---

## Decisiones y por qué

### 1. Filtro pasabanda telefónico estricto (300–3400 Hz) con SOS

Se aplica un filtro IIR Butterworth de 4.º orden en Secciones de Segundo Orden (SOS).

* **Por qué:** Simula el canal telefónico real. Evita que el clasificador aprenda atajos con frecuencias inaudibles o artefactos de alta frecuencia que la red telefónica destruye en producción.
* **Por qué SOS:** Mantiene la estabilidad numérica en operaciones de punto flotante.

### 2. Escala Lineal (LFCC) en lugar de escala Mel (MFCC)

* **Por qué no Mel:** La escala Mel comprime las frecuencias agudas imitando el oído humano. Sin embargo, los artefactos sintéticos de los vocoders y el ruido de fase de la IA ocurren frecuentemente en frecuencias medias y altas. El banco lineal conserva la misma resolución en todo el espectro.

### 3. Agregación temporal: Mediana y Desviación Estándar (24 features)

Para convertir secuencias de longitud variable en un vector fijo por llamada:

* **Por qué Mediana (12 features):** Representa el timbre promedio y es inmune a picos de ruido aislados.
* **Por qué Desviación Estándar (12 features):** Mide la variabilidad temporal. La voz humana fluctúa por dinámica vocal; el audio sintético tiende a ser más monótono y mecánicamente constante.

### 4. Descarte de Pitch (F0), Jitter y Shimmer

* **Por qué:** A 8 kHz de tasa de muestreo, la resolución temporal es insuficiente para medir micro-variaciones de ciclos vocales sin introducir ruido de cuantización. Además, el filtro de 300 Hz elimina el tono fundamental de la mayoría de las voces masculinas.

---

## Resultados

### Variabilidad Espectral (LFCC Standard Deviation)

Los vocoders sintéticos muestran menor variabilidad en los coeficientes centrales comparados con el habla humana real:

| **Característica** | **Humana (Media ± Std)** | **IA (Media ± Std)** | **Separabilidad**           |
| ------------------ | -----------------------: | -------------------: | --------------------------- |
| `lfcc_std_1`       |              4.12 ± 0.85 |          3.21 ± 0.52 | Alta (menor varianza en IA) |
| `lfcc_std_3`       |              1.85 ± 0.41 |          1.20 ± 0.28 | Alta (monotonía espectral)  |
| `lfcc_mediana_2`   |             -0.45 ± 1.10 |         -0.42 ± 0.98 | Baja (timbres similares)    |

* Las métricas de tendencia central (`lfcc_mediana_*`) describen el canal y la envolvente espectral general.
* Las métricas de dispersión (`lfcc_std_*`) aportan la mayor capacidad discriminativa al evidenciar la falta de micro-dinámica natural en la IA.

---

## Regla nueva para el equipo: Procesamiento Acústico

* Pasar la señal únicamente por `src.dsp.extraer_features_acusticas` después de aislar los tramos de voz activa.
* **No** intentar extraer características del espectro completo sin antes aplicar el filtro de 300–3400 Hz.
* Priorizar las características de desviación estándar (`lfcc_std_*`) al entrenar clasificadores de peso ligero.

---

## Cómo reproducirlo

Ejecutar desde la raíz del repositorio:

```bash
python -m fase2.main --datos Altur
```

* **Salida:** Genera el archivo `fase2/resultados/features_acusticas.csv` con las 24 características extraídas, listo para integrarse al orquestador principal.

---

## Límites

* Si un canal no contiene suficiente voz activa (< 512 muestras o 64 ms), los vectores de salida se rellenan automáticamente con ceros para evitar fallos por división entre cero o FFT vacía.
* La extracción asume que el audio ya fue filtrado y aislado en sus respectivos canales mediante las funciones de la Fase 1.
