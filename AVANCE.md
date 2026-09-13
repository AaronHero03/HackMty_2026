# Avance del proyecto — Yasu

Tablero del equipo. Se lee en un minuto y dice **qué está hecho, quién lo hace y qué salió**.

**Reglas:**
1. Cada quien edita **solo su fila** de la tabla (así no chocamos al subir cambios).
2. Se actualiza cada vez que subes algo.
3. El resultado va en **una frase**, con un número si lo hay.

Estados: ⬜ sin empezar · 🟡 en curso · ✅ listo · 🔴 bloqueado

---

## Marcador

**¿Cuántas de las 71 llamadas de `val` acertamos?**

| Versión | Aciertos en `val` | Nota |
|---|---|---|
| Regla simple: "si tarda más de 1.6 s en contestar, es IA" | **63 / 71** | Marca a vencer. Medida con los `turns/` del dataset; con nuestro detector de voz puede bajar un poco |

- Los ajustes se deciden **solo con `train`**. `val` se usa únicamente para anotar el marcador; si ajustamos mirando `val`, nos engañamos.
- Cada mejora se anuncia así: "antes 63/71, ahora X/71".

---

## Fases

Las fases siguen `Implementacion_Simple.md`. Los detalles técnicos y el porqué de cada decisión están en `IMPLEMENTACION1.1.md`.

| Fase | Pregunta que responde | Quién | Estado | Resultado en una frase |
|---|---|---|---|---|
| 0. Trampas en los datos | ¿Los datos son confiables, o hay algo que delata la clase sin tener que ver con ser IA? | Fernando | ✅ | 2 trampas: la IA suena 7 dB más fuerte y 2 de cada 3 IAs tienen silencio digital (sin ruido de fondo). 2 dudas para Altur. La latencia de quien llama se ve legítima. Detalle y reglas por fase: `fase0/README.md` |
| 1. Leer llamadas | ¿Separamos los canales y encontramos dónde hay voz? | Fernando | ✅ | `src/audio.py` y `src/vad.py` listos para todos. Error contra `turns/` de 2–9 %. La latencia de quien llama medida con nuestro detector separa mejor que con `turns/` (AUC 0.88 en val). Encontramos eco del agente en llamadas humanas. Detalle: `fase1/README.md` |
| 2. Voz | ¿Cómo suena quien llama? (LFCC) | Carlos / Andrés (por confirmar) | ⬜ | |
| 3. Conversación | ¿Cómo responde quien llama? (latencias, encimadas) | Fernando | ⬜ | |
| 4. Modelo | ¿Cuántas llamadas acierta y qué tan confiable es su confianza? | Por definir | ⬜ | |
| 5. API | ¿Altur puede consultarnos y recibe `{"is_synthetic", "confidence"}`? | Aarón | ⬜ | |

---

## Metas que se pueden mostrar

1. **Fase 0:** 4 gráficas (voz y silencio, humanos contra IAs) y la conclusión "datos limpios" o "hay una trampa en X".
2. **Fase 1:** una función que lee cualquier llamada y marca dónde hay voz. Va primero porque todas las demás fases la usan.
3. **Fase 5 temprana (en paralelo desde ya):** la API responde siempre `0.5`. Así sabemos pronto que Altur puede consultarnos.
4. **Fases 2 y 3:** una tabla con una fila por llamada.
5. **Fase 4:** "acertamos X/71", con la confianza calibrada.
6. **Fase 5 final:** la API contesta con el modelo.

---

## Pendientes del equipo

- [ ] Confirmar que `Implementacion_Simple.md` es el mapa del equipo e `IMPLEMENTACION1.1.md` la consulta técnica.
- [ ] Revisar 4 ajustes propuestos a `Implementacion_Simple.md`:
  1. Usar el **mismo detector de voz** al entrenar y en la API (los `turns/` solo para comprobar).
  2. Medir el **tiempo típico en contestar**, no solo cuánto varía.
  3. Corregir la razón para dejar jitter y shimmer para después.
  4. Agregar la **Fase 0** de trampas.
- [ ] Definir quién hace la Fase 1 y la Fase 4.
- [ ] Decidir qué herramientas de patrocinadores usamos (propuesta abajo). Solo si encajan con el reto; ninguna va antes del núcleo, salvo Vultr.
- [ ] Leer las **reglas por fase** de `fase0/README.md` antes de programar las Fases 1 a 5 (normalizar volumen, nada de rasgos del silencio, latencia mediana).
- [ ] Fases 2, 3 y 5: leer y preparar el audio con `src/audio.py` y encontrar la voz con `src/vad.py`, **no** con `turns/` (instalación y ejemplo en `fase1/README.md`). Ojo con el eco del agente en llamadas humanas.
- [ ] Preguntar a Altur (salen de la Fase 0):
  1. ¿Las llamadas de IA se inyectaron como audio digital sin ruido de fondo? ¿Las de evaluación también?
  2. ¿Las llamadas humanas y las de IA usaron la misma versión y el mismo guion del agente?

### Herramientas de patrocinadores (propuesta)

| Herramienta | Propuesta | Para qué | Cuándo |
|---|---|---|---|
| **Vultr** | ✅ Usar | Servidor donde vive `POST /detect` (ya es el servidor del Paso 13). Sin GPU: los modelos no la necesitan | Desde el inicio (Fase 5) |
| **ElevenLabs** | 🟡 Usar como prueba | Grabar unas llamadas donde quien llama es un agente de ElevenLabs y ver si lo detectamos ("voz nunca vista"). **No para entrenar**: el modelo aprendería "audio hecho por nosotros = IA". Solo voces del equipo, con consentimiento, pasadas a 8 kHz | Después de la Fase 4 |
| **MongoDB Atlas** | 🟡 Si sobra tiempo | Guardar la huella de las voces **sintéticas** detectadas y avisar si una llamada nueva usa la misma voz de IA. Nunca voces de personas | Al final |
| **Gemini** | ⏸️ Esperar a Altur | Capa semántica: escuchar qué responde quien llama cuando el agente pregunta por algo que no existe. Requiere enviar el audio a Google (pregunta para mentores) y no puede ser necesario para que `/detect` responda | Solo si Altur lo permite |
| Snowflake, Solana, Tiger Data | ❌ No usar | No mejoran la detección: los datos caben en un CSV y el audio se procesa en memoria | — |

---

## Cómo reportar un avance

Un mensaje de 4 líneas en el chat del equipo, y la misma frase en tu fila de la tabla:

```
Fase: 3 — Conversación
Hice: latencia y encimadas de las 353 llamadas
Encontré: la IA tarda 2.2 s en contestar; un humano, 0.9 s
Sigue: pasarle la tabla a quien hace el modelo
```

---

## Bitácora

| Fecha | Quién | Qué pasó |
|---|---|---|
| 2026-09-12 | Fernando | Primera lectura de los `turns/`: quien llama tarda en contestar 0.9 s si es humano y 2.2 s si es IA (medianas). Con esa sola regla se aciertan 63 de 71 llamadas de `val` |
| 2026-09-12 | Fernando | Audio descargado del Release oficial y verificado. No se sube al repo (`audio/` está en `.gitignore`) |
| 2026-09-12 | Fernando | Fase 0 lista (`fase0/trampas.py`, ~25 s). Trampas: volumen (AUC 0.93 en val) y silencio digital ("silencio plano = IA" acierta 56/71 en val). Sospechosos: más graves en la voz de IA y el agente habla menos con IAs. Limpio: el sonido del agente. Legítima: la latencia de quien llama (AUC 0.85 en val) |
| 2026-09-12 | Fernando | Fase 1 lista: carga de llamadas, normalización y detector Silero (umbral 0.7, silencio mínimo 200 ms, elegidos con train). Error contra `turns/` de 2–9 %. Con nuestro detector la latencia de quien llama separa mejor (AUC 0.88 en val). Hallazgo: eco del agente en el canal 0 de las llamadas humanas, que `turns/` contaba como voz |
