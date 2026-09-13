# Yasu Analytics — Voice Deepfake Detection for Call Centers

**HackMTY 2026 · Altur Challenge Track — "Defend the Bank Against Voice Deepfakes"**

Yasu Analytics (internal codename **VoiceGuard**) decides, for a recorded phone call, whether the incoming caller is a real human or a synthetic voice. It's built around Altur's actual data format: stereo, telephony-grade 8kHz audio, channel 0 = caller, channel 1 = the AI agent, real conversations in Spanish over regular phone lines.

---

## The problem

A few seconds of public audio is enough to clone a voice convincingly. Contact centers — Altur's core business — are now a two-way attack surface: fraudsters impersonate customers to take over accounts, and impersonate banks to social-engineer customers. Most contact centers, human-staffed or AI-powered, have no reliable way to tell who — or what — is actually on the line.

## What it does

Given a call, the pipeline:

1. Isolates and normalizes the caller's channel (channel 0), resamples to 8kHz.
2. Builds its own voice-activity turns for both channels (rather than trusting the dataset's `turns/` blindly — see Phase 1).
3. Extracts **27 acoustic features** from the caller's active voice, and **4 conversational-timing features** from the agent↔caller turn-taking.
4. Runs those 31 features through a **weighted ensemble of three gradient/tree models** (CatBoost, Random Forest, XGBoost).
5. Returns `{"is_synthetic": bool, "confidence": float}` over `POST /detect`, matching Altur's spec.

---

## Pipeline, as it's actually implemented

```
Fase 0  src/fase0/trampas.py        Data integrity audit (before trusting any feature)
Fase 1  src/fase1/audio_base.py     Channel isolation, volume normalization, resample to 8kHz
        src/tools/vad.py            Custom voice-activity detection (generar_turnos_vad, fusionar_turnos)
Fase 2  src/fase2/acustica.py       27 acoustic features from the caller's voice
Fase 3  src/fase3/conversacional.py 4 conversational-timing features
Fase 4  models/                     CatBoost + Random Forest + XGBoost, weighted soft-voting
Fase 5  API                         POST /detect
```

### Fase 0 — `trampas.py`: auditing the data before trusting it

Before building a single production feature, this script measures **every call** in the manifest along four groups of traits and checks whether each one separates human vs. synthetic calls for reasons that have nothing to do with being AI:

- **Parte 1 — Volume & noise (channel 0):** voice level, background-noise floor, caller SNR, % of exactly-zero samples during silence, % of clipped samples, DC offset, echo level (`c0_eco_db`, see below), and the **spectral tilt of the silence itself** (`c0_silencio_inclinacion_db`).
- **Parte 2 — Spectrum (channel 0):** % energy below 300Hz / above 3400Hz, spectral centroid, for both voice and silence.
- **Parte 3a/3b — Agent leakage (channel 1):** whether the agent's own voice/spectrum or *timing behavior* (turns/minute, median turn length, agent latency, interruption rate) leaks class information it shouldn't.
- **Reference — expected legitimate signal:** caller response latency, overall silence %, call duration.

Two mechanisms make this an actual audit rather than a fishing expedition:

- **Digital silence check.** `inclinacion_db` compares energy in 200–1000Hz vs. 3000–3600Hz during silence. Real background/line noise skews toward low frequencies (positive value); a completely synthetic/injected silence is spectrally flat (near 0). Calls with tilt `< 3 dB` are flagged as "digital silence."
- **Two-split confirmation (`veredicto`).** A feature is only labeled `FUERTE` ("strong") or `moderada` if its AUC separates the classes **in the same direction on both `train` and `val`** — a feature that only separates on one split is marked "no separa" and discarded as noise/overfitting, not used as a real signal.
- **Codec sanity check.** Because the audio is telephony-grade, μ-law encoding caps the number of distinct sample values (~256) and the peak amplitude (~32124); this is checked directly as a basic authenticity/format sanity check on the dataset.
- **Echo detection.** `c0_eco_db` compares the caller channel's noise floor during segments where *only the agent is speaking* against the caller channel's normal silence floor — a rise indicates the agent's audio is bleeding into the caller's channel (acoustic echo), which would otherwise contaminate caller-only features.

Output: `fase0/resultados/rasgos.csv` (per-feature comparison with AUC on train and val) and `espectro.png` (median + interquartile spectrum per class, per channel, for voice and silence).

### Fase 1 — `audio_base.py` + `tools/vad.py`: building our own ears

- `procesar_audio_base`: reads the WAV (mono or stereo), isolates channel 0, resamples to 8kHz if needed, and **normalizes RMS volume to a fixed −20 dB target** — this directly neutralizes the ~7dB volume gap Fase 0 found between classes, so downstream features can't shortcut on loudness.
- `cargar_turnos` / `recortar_voz_activa`: load the manifest's turn timestamps and concatenate only the active-voice samples for a given speaker.
- `tools/vad.py` (`generar_turnos_vad`, `fusionar_turnos`): generates the team's **own** voice-activity turns instead of relying solely on the dataset's `turns/` labels, then merges turns separated by short pauses (`max_pausa_s=0.5`). This is what's actually used at both training and inference time in the evaluation scripts below — the dataset's `turns/` is treated as a reference/QA signal, not a dependency.

### Fase 2 — `acustica.py`: 27 acoustic features

Runs on the caller's voice after a 300–3400Hz Butterworth bandpass filter (`aplicar_filtro_pasabanda`, matching the telephony band) and an **active-frame filter** that drops any 32ms frame more than 30dB quieter than the loudest 5th percentile of the call — this discards residual digital-silence and echo-bleed frames before features are computed, addressing the exact contamination risks flagged in Fase 0.

| Feature group | Count | What it captures |
|---|---|---|
| `lfcc_mediana_1..12`, `lfcc_std_1..12` | 24 | Linear-Frequency Cepstral Coefficients (median + std) from a 20-triangular-filter bank spanning 300–3400Hz, DCT of log filterbank energies — LFCC over MFCC because a linear filter bank is more appropriate for the narrow, already-bandlimited telephony spectrum than mel spacing. |
| `cpp` | 1 | Cepstral Peak Prominence — measures how "clean"/periodic the voicing is, without needing pitch tracking (robust at 8kHz). |
| `inclinacion_espectral` | 1 | Spectral tilt: energy in 300–1000Hz minus energy in 2000–3400Hz. |
| `vocoder_periodicity_peak` | 1 | Modulation-spectrum prominence (40–200Hz) of the Hilbert envelope — designed to catch the frame/block periodicity artifacts a TTS/vocoder can leave in the amplitude envelope, which a real larynx doesn't produce. |

### Fase 3 — `conversacional.py`: 4 conversational-timing features

Computed purely from turn timestamps (agent = channel 1, caller = channel 0):

| Feature | What it measures |
|---|---|
| `latencia_mediana` | Median time between the agent finishing a turn and the caller starting to respond. |
| `latencia_varianza` | Variance of that response latency. |
| `latencia_cv` | Coefficient of variation (std / median) of latency — captures *consistency* of response timing, not just its average. |
| `pct_respuestas_cortas` | % of responses faster than 0.4s — near-instant replies that would be unusual for a human parsing a live question. |

This is the operationalization of the brief's core behavioral idea: humans respond with variable, sometimes messy timing; a synthetic caller tends toward a narrower, more mechanical latency distribution — which is exactly what `latencia_cv` and `pct_respuestas_cortas` are built to expose.

### Data augmentation — robustness beyond clean audio

For calls in the `train` split, each recording is expanded into **4 variants**: `_orig`, `_agudo` (pitch +2 semitones), `_grave` (pitch −2 semitones), and `_ruido` (gain ×0.85 + Gaussian noise at 28dB SNR). This is what turns the training pool into the larger augmented set used for the stress evaluation below — it specifically probes robustness to pitch shift and additive noise, on top of (not instead of) the clean, unaugmented evaluation.

### Fase 4 — model & ensemble

Three models are trained on the same 31-feature vector (27 acoustic + 4 conversational) and combined by **weighted soft voting** (a straight average of `predict_proba` would treat all three as equally reliable; this doesn't):

| Model | File | Ensemble weight |
|---|---|---|
| CatBoost | `models/modelo_catboost_altur.cbm` | 0.70 |
| Random Forest | `models/modelo_rf_altur.joblib` | 0.15 |
| XGBoost | `models/modelo_xgboost_altur_v4.json` | 0.15 |

Weights were assigned from each model's individual validation performance (CatBoost clearly leads solo, so it dominates the vote) and are **re-normalized automatically** if any model file is missing at load time, so the ensemble degrades gracefully instead of failing.

The decision threshold used for `is_synthetic` is **0.40**, not the default 0.5 — tuned to bias the system toward catching synthetic callers rather than missing them, consistent with a fraud-detection use case where a missed synthetic voice is more costly than a false alarm on a human.

### Fase 5 — API

`POST /detect` wraps the same feature-extraction pipeline (VAD → acoustic features → conversational features → ensemble) behind Altur's exact request/response contract, deployed on a GPU-free Vultr instance — none of the three models need one.

---

## Evaluation scripts

- **Model comparison** (`evaluar_comparativa_completa`): runs the full extraction pipeline over a directory of calls, scores both the standalone CatBoost model and the weighted ensemble against `manifest.csv` ground truth, prints a `classification_report` and ROC-AUC for each, and saves a side-by-side ROC curve (`reports/graficas/comparativa_curvas_roc.png`). It also reports an **error-margin analysis**: for every misclassified call, how far its predicted probability sat from the 0.40 threshold (average distance, and how many were "shallow" errors under 0.15 from the boundary) — this is what the "average error distance" figures below come from.
- **Ensemble breakdown** (`evaluar_ensamble_ponderado`): same pipeline, but prints each individual model's probability and weight per call before the combined verdict, then a full classification report + confusion matrix for the ensemble alone.
- **Single-model check** (`probar_audios_locales`): loads one named model file (any of `.pkl`/`.joblib`/`.json`/`.cbm`) and runs it alone against the manifest — used for debugging or isolating a single model's behavior.

All model *selection* and threshold/weight tuning was done against `train`; the numbers below come from scoring these already-frozen models.

---

## Results

### Clean evaluation — 353 calls

| Model | Accuracy | ROC-AUC | Errors |
|---|---|---|---|
| CatBoost (single model) | 0.99 | 0.9998 | 5 |
| Weighted soft-voting ensemble | **0.99** | **0.9999** | **2** |

| | CatBoost — Human | CatBoost — Synthetic | Ensemble — Human | Ensemble — Synthetic |
|---|---|---|---|---|
| Precision | 0.99 | 0.98 | 1.00 | 0.99 |
| Recall | 0.97 | 1.00 | 0.99 | 1.00 |

### Augmented-pool stress test — 1,128 calls (pitch-shifted + noise-added variants)

Evaluated over the `_orig`/`_agudo`/`_grave`/`_ruido` augmented pool described above, to probe robustness specifically to pitch shift and additive noise rather than to unseen speakers:

| Model | Accuracy | Macro F1 | ROC-AUC | Errors | Avg. error distance from threshold |
|---|---|---|---|---|---|
| CatBoost (single model) | 0.92 | 0.92 | 0.9761 | 88 | 0.2832 |
| Weighted soft-voting ensemble | 0.92 | 0.91 | 0.9753 | 95 | 0.2589 |

Per-class detail — CatBoost (support: 452 human / 676 synthetic):

| | Precision | Recall | F1 |
|---|---|---|---|
| Human | 0.96 | 0.84 | 0.90 |
| Synthetic | 0.90 | 0.97 | 0.94 |

Per-class detail — Ensemble (support: 452 human / 676 synthetic):

| | Precision | Recall | F1 |
|---|---|---|---|
| Human | 0.97 | 0.82 | 0.89 |
| Synthetic | 0.89 | 0.98 | 0.93 |

**Reading these together:** the ensemble buys precision and AUC on clean audio, but the standalone CatBoost model holds up slightly better on human-caller recall once pitch/noise perturbation is introduced. That's consistent with the feature design — heavier noise erodes some of the fine-grained LFCC/CPP detail that the extra ensemble members lean on, while the conversational-timing features (untouched by audio-level perturbation) keep contributing steady signal to whichever model weighs them well.

---

## Mapping to Altur's judging criteria

- **Robustness** — validated on a 1,128-call pool covering pitch shift and additive noise on top of the 353-call clean benchmark.
- **Originality** — 27 acoustic features (LFCC bank, CPP, spectral tilt, vocoder-periodicity) combined with 4 conversational-timing features (latency median/variance/CV, fast-response rate), plus a dedicated Fase 0 audit that catches dataset shortcuts before they reach the model.
- **Technical depth** — a custom-tuned VAD used consistently at train and serve time, an active-frame filter that removes digital-silence/echo contamination before feature extraction, and a two-split (train+val) confirmation rule for every candidate signal.
- **Feasibility** — CatBoost + Random Forest + XGBoost on tabular features run on CPU; no GPU inference required, deployable on Vultr today.
- **Latency** — tree-based ensemble on precomputed features is fast per call compared to running a full deep audio model.

---

## Tech stack & sponsor tools

| Tool | Role |
|---|---|
| **Vultr** | Hosts the `POST /detect` server. CPU-only — none of the three models need a GPU. |
| **ElevenLabs** | Used only as an *adversarial test set* — team-recorded voices, with consent, synthesized as "never seen before" agent voices, to test detection on synthetic sources the models weren't trained on. **Not used for training.** |
| **MongoDB Atlas** *(stretch goal)* | Would store a fingerprint of detected *synthetic* voices only, to flag a repeat cloned voice across calls. Never stores human voice data. |
| CatBoost / Random Forest / XGBoost | Ensemble classifier on the 31-feature vector. |
| Custom Silero-based VAD | Voice activity detection shared across the whole pipeline. |

Tools evaluated and intentionally **not** used: Snowflake, Solana, Tiger Data (the feature set is small/tabular enough that they add infrastructure without improving detection); Gemini (a semantic layer — checking whether the caller says "I don't have that" vs. inventing an answer to a nonexistent detail — was scoped but requires sending audio to a third party, pending Altur's sign-off, and was kept optional/non-blocking for `/detect`).

---

## Repository structure

```
src/
  fase0/
    trampas.py          # data-integrity audit: volume, silence tilt, echo, codec sanity
  fase1/
    audio_base.py        # channel isolation, RMS normalization, resample to 8kHz
  fase2/
    acustica.py           # 27 acoustic features (LFCC, CPP, spectral tilt, vocoder peak)
  fase3/
    conversacional.py     # 4 conversational-timing features
  tools/
    vad.py                 # generar_turnos_vad, fusionar_turnos — custom VAD
models/
  modelo_catboost_altur.cbm
  modelo_rf_altur.joblib
  modelo_xgboost_altur_v4.json
Altur_Data/
  audio/                   # original calls (per manifest)
  audio_augmented/         # _orig/_agudo/_grave/_ruido variants (train split only)
  turns/                   # dataset-provided turn annotations (used as QA reference)
  manifest.csv             # anon_id, label, split
  fase0/                   # cached per-call measurements (not versioned — contains labels)
reports/
  graficas/                # ROC comparisons, Fase 0 spectrum plots
Implementacion_Simple.md   # team-facing roadmap
IMPLEMENTACION1.1.md       # technical rationale per decision
```

## Running it

```bash
pip install -r requirements.txt

# Fase 0 — audit the dataset before trusting any feature
python -m src.fase0.trampas

# Fase 4 — compare CatBoost vs. the weighted ensemble
python -m scripts.evaluar_comparativa_completa

# Fase 5 — run the detection API
python -m src.api.app
```

`POST /detect` expects:

```json
{
  "audio_base64": "<base64-encoded stereo 8kHz WAV, channel 0 = caller, channel 1 = agent>"
}
```

and returns:

```json
{
  "is_synthetic": true,
  "confidence": 0.87
}
```

## What's next

- Close the human-recall gap under noise by making the conversational features noise-robust rather than leaning harder on acoustic ones under degraded audio.
- Extend the stress test to unseen speakers/engines specifically (the current augmented pool tests pitch/noise perturbation of known training calls, not unseen sources — the ElevenLabs adversarial set is the first step toward that).
- Pursue the semantic layer (Phase 3.5-style: caller responses to nonexistent-detail prompts) once data-sharing constraints with Altur are resolved.
- Voice fingerprinting of repeat synthetic callers (MongoDB Atlas) to catch fraud rings reusing the same cloned voice across multiple calls or institutions.

## Team

Fernando García· Carlos Gloria · Andrés Guzmán · Aarón Hernández — HackMTY 2026, Altur Challenge Track.
