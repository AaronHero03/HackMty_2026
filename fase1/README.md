# Fase 1 — Leer llamadas y encontrar la voz

**Pregunta:** ¿podemos leer cualquier llamada, separar los canales y encontrar dónde habla cada
uno **sin usar `turns/`**? En la API no tendremos `turns/`, así que el modelo debe aprender con
los mismos tramos que después verá en producción.

**Resultado en una frase:** sí. El detector se equivoca en 2–9 % del tiempo contra `turns/`, y la
latencia de quien llama medida con **nuestro** detector separa humanos de IAs **mejor** que con
`turns/` (AUC 0.88 en val). De paso encontramos **eco del agente** en el canal de las llamadas
humanas.

---

## Qué hay (código para todo el equipo)

| Archivo | Qué hace |
|---|---|
| `src/audio.py` | `cargar_llamada(fuente)`: lee una ruta, bytes o texto base64 y devuelve `(canal0, canal1)` a 8 kHz, sin mezclar canales. `normalizar(x)`: ganancia constante por canal |
| `src/vad.py` | `detectar_voz(x)`: tramos de voz `[(inicio_s, fin_s), ...]` de un canal. Por dentro: `probabilidades_voz(x)` + `tramos_voz(probs, n)` |
| `fase1/evaluar_vad.py` | Compara el detector contra `turns/` y elige los ajustes |
| `requirements.txt` | Dependencias del proyecto |

```python
from src.audio import cargar_llamada, normalizar
from src.vad import detectar_voz

llama, agente = cargar_llamada("../hackmty26/audio/<anon_id>.wav")   # o el base64 que manda Altur
tramos_llama = detectar_voz(normalizar(llama))      # [(inicio_s, fin_s), ...]
tramos_agente = detectar_voz(normalizar(agente))
```

Correr siempre desde la raíz del repo, para que `import src` funcione.

## Instalación

```bash
pip install -r requirements.txt
```

- **Colab:** `torch` ya viene instalado; basta esa línea.
- **Linux sin GPU:** antes, `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`.
  El `torch` normal de PyPI en Linux trae las librerías de GPU y pesa varios GB; esta versión es
  igual pero solo CPU.
- **Windows y Mac:** PyPI ya entrega la versión CPU; no hay que hacer nada extra.

---

## Decisiones y por qué

**1. Normalizar sin `turns/`, antes del detector.**
Se toma el volumen de la parte fuerte del canal (percentil 95 de tramas de 20 ms, que casi
siempre es voz) y se lleva a −20 dBFS con **una sola ganancia**, con tope de +30 dB para no
amplificar un canal vacío.
- *Por qué no sobre `turns/`* (como decía la guía): la API no los tiene.
- *Por qué antes del detector:* la IA suena 7 dB más fuerte (trampa 1 de la Fase 0); así el
  detector no la encuentra más fácil solo por volumen.
- *Por qué una ganancia constante:* no cambia latencias ni la forma del espectro.

**2. Silero en dos pasos.**
`probabilidades_voz()` corre la red una vez (probabilidad cada 32 ms) y `tramos_voz()` convierte
probabilidades en tramos. Así se prueban muchos ajustes sin volver a correr la red.
`tramos_voz()` reproduce la lógica de `silero_vad.get_speech_timestamps`: se verificó que da
**exactamente** los mismos tramos (2 llamadas, 2 canales, 2 ajustes).

**3. Ajustes elegidos: umbral 0.7 y silencio mínimo 200 ms.**
- **Umbral:** qué tan segura debe estar la red para marcar voz.
- **Silencio mínimo:** una pausa más corta no corta el tramo.
- Se probaron 49 combinaciones **solo con train** (282 llamadas). El error casi no cambia entre
  0.6–0.8 y 200–250 ms (4.28–4.34 %), así que el ajuste es estable.
- Duración mínima de voz (250 ms) y relleno (30 ms) quedan en los valores de Silero.

**4. Por qué Silero** (y no un detector por energía o WebRTC): tolera ruido y funciona a 8 kHz
(ver Paso 4 de `IMPLEMENTACION1.1.md`). Esta fase agregó otra razón: **ignora el eco del agente**
que un detector por energía contaría como voz.

---

## Resultados

### Error contra `turns/` (% del tiempo de la llamada)

| | Humanas | IA |
|---|---|---|
| Canal 0 (quien llama), train | 9.2 % | 4.4 % |
| Canal 0, val | 6.1 % | 3.7 % |
| Canal 1 (agente), train | 2.5 % | 2.1 % |
| Canal 1, val | 2.5 % | 2.3 % |

- Criterio de la guía (error menor a 10 %): ✅ en todos los casos.
- En el canal 0 humano el error es el doble; casi todo es **eco del agente** (siguiente punto).
- `turns/` también lo generó una máquina: parecerse a `turns/` no es la meta.

### Qué son los desacuerdos del canal 0

| Clase | Tipo | Por llamada | Prob. de Silero | Mientras habla el agente |
|---|---|---|---|---|
| Humana | `turns/` dice voz, el detector no | 6.7 | 0.01 | 65 % |
| IA | `turns/` dice voz, el detector no | 1.5 | 0.14 | 0 % |
| IA | el detector dice voz, `turns/` no | 1.0 | 0.73 | 6 % |

(Tramos de 0.5 s o más.)

- **Humanas:** son sonidos que Silero no considera voz, casi siempre **mientras habla el agente**
  y unos 20 dB más bajos que la voz normal (medido aparte). Todo apunta a **eco del agente** que
  se cuela en el canal de quien llama, algo normal en llamadas telefónicas reales.
  `turns/` los contaba como voz; nuestro detector, bien, no.
- **IA con ruido de fondo:** en la tercera parte de las IAs que traen ruido, el detector marca
  parte del fondo como voz. Pesa poco en el promedio, pero conviene escuchar algunos casos.

### ¿Normalizar cambia el error?

Casi nada: Silero tolera bien el volumen. Solo baja un poco las falsas alarmas en IA
(val: 2.8 % → 1.9 %). Se queda, porque los rasgos de voz la necesitan de todos modos.

### La prueba importante: la señal principal sigue funcionando

| Latencia de quien llama medida con… | Humana | IA | AUC train | AUC val |
|---|---|---|---|---|
| `turns/` | 0.90 s | 2.16 s | 0.86 | 0.85 |
| **Nuestro detector** | 1.28 s | 2.31 s | **0.88** | **0.88** |

Ambas medidas se parecen (correlación 0.83). Los humanos "tardan más" con nuestro detector porque
`turns/` contaba el eco como si la persona ya estuviera hablando.

---

## Regla nueva para el equipo: eco del agente

Las llamadas humanas traen eco del agente en el canal 0; las de IA, inyectadas digitalmente, no.
Es una trampa del mismo tipo que las de la Fase 0.
- **No** medir energía del canal 0 mientras habla el agente.
- Latencias y encimadas: **solo** con los tramos de `detectar_voz()`.
- Las encimadas calculadas con `turns/` en llamadas humanas están **infladas** por el eco.

## Cómo reproducirlo

```bash
python fase1/evaluar_vad.py --datos ../hackmty26
```

- Primera vez: ~10 min (Silero sobre las 353 llamadas, con y sin normalizar, en paralelo).
  Después: ~25 s, porque las probabilidades quedan guardadas.
- Al repo: `fase1/resultados/rejilla.csv` (error por combinación) y `resumen.csv` (error por
  canal, clase y split).
- Fuera del repo, junto al dataset: probabilidades, `desacuerdos.csv` (tramos para escuchar) y 3
  gráficas de ejemplo. No se suben: son datos por llamada.

## Límites

- La explicación del eco sale de indicadores (volumen bajo, probabilidad de Silero de 0.01,
  coincidencia con el agente). ⚠️ Falta confirmarla **escuchando** algunos tramos de
  `desacuerdos.csv`.
- El error se mide sobre el tiempo total de la llamada; para latencias importan más los bordes de
  cada tramo.
- No se probó con clips cortos (si Altur manda solo un pedazo de llamada).
