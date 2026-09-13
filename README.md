# HackMTY 2026

HackMTY is a hackathon organized by Tecnologico de Monterrey feat Major League Hacking,
we have 32 hours for develop a project related to our challenge. This year are 4 global
enterprises related with a technology area.

## Team
- Carlos Gloria - Computer Science
- Fernando Garcia - Robotics 
- Aarón Hernández - Computer Science
- Andres Guzman - Biomedical Engineering

## Tools 

MLH offer us a lot of different technologies for develop our project, included:
- A gemini API
- Tiger data Toolkit API
- ElevenLabs
- Solana 
- MongoDB Atlas $50 free credits
- Snowflake. 120 day free trial
- Vultr 

Also, there are prizes for use this technologies but we don't have to use everithing, 
just the necessary.

## Sponsors
- Infosys
- CapitalOne
- Banorte
- Altur

## Altur Challenge

This mission is about voice recognizing and security on the banking area. They give us a
customized dataset. It is an intensive approach related with AI and signal processing. 


### Altur
Is an entreprises that makes voice agents for the finance sector. They cover all the
production process, as interaction with the users and client. They have their self 
communication system hostend by themselves.

More that 150M of calls per month but just 3M finish with a conversation.
They opear in Mexic, Brasil, Chile and Colombia

The challenge is about how a bank knows with who is talking.

They provide us a dataset of 300 calls. Half human-human and the other half agent-human

We have to expose a /detect endpoint that return:
{
  "is_synthetic": true
  "confidence": 0.87
}

## API (`POST /detect`)

`api.py` expone el endpoint que llama el juez de Altur, con el mismo proceso de `tests/test_1.py` y el modelo `modelo_xgboost_altur_v4.json`.

**Contrato:** recibe `{"call_id", "audio_base64", "sample_rate": 8000, "channels": 2}` y responde `{"is_synthetic": bool, "confidence": 0-1}`. `confidence` es la seguridad en la respuesta dada: el juez calcula la probabilidad de IA como `confidence` si `is_synthetic` es true, y como `1 - confidence` si es false.

**Variables de entorno** (en Railway, en Variables):

| Variable | Para qué | Si no está |
|---|---|---|
| `MONGO_URI` | Guardar cada resultado en MongoDB Atlas, después de responder | No guarda historial; la API responde igual |
| `UMBRAL` | Probabilidad de IA a partir de la cual responde sintético | 0.40 |
| `CORS_ORIGINS` | Páginas que pueden llamar a la API desde el navegador, separadas por comas | `https://fernandox89.github.io` |

**Correrla en tu computadora** (Windows, Ubuntu o Arch), con lo de `requirements.txt` ya instalado:

```bash
pip install fastapi uvicorn
uvicorn api:app --port 8000
```

**Desplegar en Railway**, desde la raíz del repo, con la CLI de Railway, la sesión iniciada y el servicio elegido una vez con `railway link`:

```bash
railway up
```

El `Dockerfile` instala `requirements-api.txt` (solo lo que usa el servidor, sin torch) y copia únicamente la API, el modelo y los módulos que usa. `.railwayignore` evita subir audios y datos de Altur.

## Historial del equipo (`GET /historial`)

`historial/` es un **servicio aparte** de la API del juez. Lee de MongoDB lo que guarda `/detect` para que la página muestre las llamadas de todo el equipo. Solo lee: no guarda ni borra nada. Si falla, `/detect` sigue funcionando.

- `GET /historial?limite=200` devuelve `{"total", "llamadas": [{"call_id", "is_synthetic", "confidence", "latencia_mediana_s", "fecha"}]}`, de la más reciente a la más vieja. `confidence` significa lo mismo que en `/detect`.
- Variables de entorno: `MONGO_URI` (la misma de la API) y `CORS_ORIGINS` (por defecto GitHub Pages y `http://localhost:8000`).

Se despliega en su propio servicio de Railway, desde la raíz del repo:

```bash
railway up historial --path-as-root --service voiceguard-historial
```
