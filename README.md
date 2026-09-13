# Yasu Analytics — Voice Deepfake Detection for Call Centers

**HackMTY 2026 · Altur Challenge Track — "Defend the Bank Against Voice Deepfakes"**

Yasu Analytics (internal codename **VoiceGuard**) decides, for a recorded phone call, whether the incoming caller is a real human or a synthetic voice. It's built around Altur's data format: stereo, telephony-grade 8kHz audio, channel 0 = caller, channel 1 = the AI agent, real conversations in Spanish over regular phone lines.

---

## The problem

A few seconds of public audio is enough to clone a voice convincingly. Contact centers — Altur's core business — are now a two-way attack surface: fraudsters impersonate customers to take over accounts, and impersonate banks to social-engineer customers. Most contact centers, human-staffed or AI-powered, have no reliable way to tell who — or what — is actually on the line.

## What it does

Given a call, the pipeline:

1. Isolates and normalizes the caller's channel, resamples to 8kHz.
2. Builds its own voice-activity turns for both channels, rather than trusting the dataset's `turns/` blindly.
3. Extracts **27 acoustic features** from the caller's active voice and **4 conversational-timing features** from the agent↔caller turn-taking.
4. Feeds those 31 features to a trained classifier and returns `{"is_synthetic": bool, "confidence": float}` over `POST /detect`.

Two separate evaluation tracks back this up: a **model bake-off** across 8 classical/deep architectures on precomputed features (`training/`), and an **end-to-end pipeline evaluation** that runs the full raw-audio → VAD → features → inference path against `manifest.csv` ground truth (`tests/`). The results below come from the second track, since it's the one that matches what actually runs at inference time.

---

## Pipeline, as it's implemented

```
src/fase0/trampas.py          Data integrity audit (before trusting any feature)
src/fase1/audio_base.py       Channel isolation, volume normalization, resample to 8kHz
src/tools/vad.py              Custom voice-activity detection (generar_turnos_vad, fusionar_turnos)
src/fase2/acustica.py         27 acoustic features from the caller's voice
src/fase3/conversacional.py   4 conversational-timing features
src/main.py                   Orchestrates fase1→VAD→fase2→fase3 into Altur_Data/features_final_vad.csv
training/                     8-model bake-off + ensembles on the features CSV above
tests/                        End-to-end evaluation: raw audio → pipeline → model, vs. manifest.csv
api.py                        Deployed POST /detect (FastAPI + MongoDB Atlas logging)
```

### Fase 0 — `src/fase0/trampas.py`: auditing the data before trusting it

Before building a single production feature, this script measures **every call** along four trait groups and checks whether each one separates human vs. synthetic calls for reasons that have nothing to do with being AI:

- **Parte 1 — Volume & noise (channel 0):** voice level, background-noise floor, caller SNR, % of exactly-zero samples during silence, % of clipped samples, DC offset, echo level, and the **spectral tilt of the silence itself**.
- **Parte 2 — Spectrum (channel 0):** % energy below 300Hz / above 3400Hz, spectral centroid, for both voice and silence.
- **Parte 3a/3b — Agent leakage (channel 1):** whether the agent's own voice/spectrum or *timing behavior* (turns/minute, median turn length, agent latency, interruption rate) leaks class information it shouldn't.
- **Reference — expected legitimate signal:** caller response latency, overall silence %, call duration.

Two mechanisms make this an actual audit rather than a fishing expedition:

- **Digital silence check.** `inclinacion_db` compares energy in 200–1000Hz vs. 3000–3600Hz during silence. Real background/line noise skews toward low frequencies (positive value); a completely synthetic/injected silence is spectrally flat (near 0). Calls with tilt `< 3 dB` are flagged as "digital silence."
- **Two-split confirmation (`veredicto`).** A feature is only labeled `FUERTE` or `moderada` if its AUC separates the classes **in the same direction on both `train` and `val`** — a feature that only separates on one split is discarded as noise, not used as a real signal.
- **Codec sanity check.** μ-law telephony encoding caps distinct sample values (~256) and peak amplitude (~32124); checked directly as a basic authenticity/format sanity check.
- **Echo detection.** Compares the caller channel's noise floor when *only the agent is speaking* against its normal silence floor — a rise means the agent's audio is bleeding into the caller's channel.

Output: `fase0/resultados/rasgos.csv` (per-feature AUC on train/val) and `espectro.png` (median + interquartile spectrum per class).

### Fase 1 — `src/fase1/audio_base.py` + `src/tools/vad.py`: building our own ears

- `procesar_audio_base`: reads the WAV (mono or stereo), isolates channel 0, resamples to 8kHz, and **normalizes RMS volume to a fixed −20dB target** — directly neutralizing the ~7dB volume gap Fase 0 found between classes.
- `cargar_turnos` / `recortar_voz_activa`: load turn timestamps and concatenate active-voice samples.
- `src/tools/vad.py` (`generar_turnos_vad`, `fusionar_turnos`): generates the team's **own** voice-activity turns from a raw energy threshold (`librosa.effects.split`, 40dB drop, 0.2s minimum), then merges turns from the same channel separated by short pauses (`max_pausa_s=0.5`). This is what both training and serving actually use — the dataset's `turns/` is a reference signal, not a dependency.
- `src/tools/audio.py` is a second, hardened loader (`cargar_llamada`, `normalizar`) built to accept a file path, raw bytes, or base64 directly — closer to what an API needs — with percentile-95-based gain normalization and a capped gain (30dB) so it doesn't blow up near-silent channels. It's a cleaner reimplementation of the same idea as `audio_base.py`; the two aren't currently unified into one shared module (see *Known issues* below).
- `src/main.py` ties fase1 → VAD → fase2 → fase3 together over the whole manifest, caches its own VAD output to `Altur_Data/turns_vad/` (kept separate from the dataset's `turns/`), and writes the final training matrix to `Altur_Data/features_final_vad.csv` — this is the file every model in `training/` is trained and validated on.

### Fase 2 — `src/fase2/acustica.py`: 27 acoustic features

Runs on the caller's voice after a 300–3400Hz Butterworth bandpass filter and an **active-frame filter** that drops any 32ms frame more than 30dB quieter than the loudest 5th percentile of the call — removing residual digital-silence/echo frames before features are computed.

| Feature group | Count | What it captures |
|---|---|---|
| `lfcc_mediana_1..12`, `lfcc_std_1..12` | 24 | Linear-Frequency Cepstral Coefficients (median + std) from a 20-triangular-filter bank spanning 300–3400Hz, DCT of log filterbank energies. LFCC over MFCC because a linear filter bank suits the already-narrow telephony band better than mel spacing. |
| `cpp` | 1 | Cepstral Peak Prominence — periodicity/voicing cleanliness without pitch tracking. |
| `inclinacion_espectral` | 1 | Spectral tilt: energy in 300–1000Hz minus energy in 2000–3400Hz. |
| `vocoder_periodicity_peak` | 1 | Modulation-spectrum prominence (40–200Hz) of the Hilbert envelope — designed to catch frame/block periodicity artifacts a TTS/vocoder can leave in the amplitude envelope. |

### Fase 3 — `src/fase3/conversacional.py`: 4 conversational-timing features

Computed purely from turn timestamps (agent = channel 1, caller = channel 0):

| Feature | What it measures |
|---|---|
| `latencia_mediana` | Median time between the agent finishing a turn and the caller starting to respond. |
| `latencia_varianza` | Variance of that response latency. |
| `latencia_cv` | Coefficient of variation (std / median) — response-timing *consistency*, not just its average. |
| `pct_respuestas_cortas` | % of responses faster than 0.4s — near-instant replies unusual for a human parsing a live question. |

This operationalizes the brief's core behavioral idea: humans respond with variable, messy timing; a synthetic caller tends toward a narrower, more mechanical distribution — exactly what `latencia_cv` and `pct_respuestas_cortas` expose.

### Data augmentation — `scripts/augment_dataset.py`

For calls in the `train` split, each recording expands into **4 variants**: `_orig`, `_agudo` (pitch +2 semitones), `_grave` (pitch −2 semitones), and `_ruido` (gain ×0.85 + Gaussian noise at 28dB SNR). This produces the larger pool used for the stress evaluation below — it specifically probes robustness to pitch shift and additive noise on the training pool, not generalization to unseen speakers.

### Adversarial / demo audio tooling — `src/tools/`

- `estereo.py`: downloads team-recorded ElevenLabs conversations via the ElevenLabs API (with consent, team voices only), forces them to stereo, resamples to 8kHz, and drops them under `tests/ElevenLabs/` — this is the "voice never seen in training" adversarial check.
- `generar_stereo.py`: runs `pyannote/speaker-diarization-3.1` on a mono two-speaker recording to auto-detect turns and rebuild a proper stereo WAV + `turns.json` in Altur's exact format — used to turn arbitrary recordings into pipeline-ready test calls.
- `muestras.py`: stitches two separate audio files (client + agent) into one stereo call with a specified latency gap — handy for building controlled demo calls (e.g. for the judging round).

---

## Model development

### Track A — bake-off on precomputed features (`training/`)

`training/utils.py::cargar_datos()` loads `Altur_Data/features_final_vad.csv`, splits it by the `split` column into `train`/`val`, and carves an additional stratified 15% out of `train` (`X_tr`/`X_es`, seed 42) purely for early stopping — this is a **held-out validation split with an internal early-stopping slice**, not k-fold cross-validation.

Eight models are trained independently against this split, each with its own script:

| Model | Script | Notes |
|---|---|---|
| CatBoost | `train_catboost.py` | 200 iterations, depth 5, lr 0.05, early stopping (10 rounds) |
| XGBoost | `train_xgboost.py` | 200 estimators, depth 4, lr 0.05, subsample/colsample 0.8, early stopping (15 rounds) |
| Random Forest | `train_rf.py` | 200 trees, depth 6 |
| Extra Trees | `train_et.py` | 200 trees, depth 6 |
| SVM (RBF) | `train_svm.py` | C=1.0, `probability=True` |
| LDA | `train_lda.py` | linear discriminant baseline |
| MLP | `train_mlp.py` | (64, 32) hidden layers, alpha 0.01, early stopping |
| 1D CNN (PyTorch) | `train_cnn.py` | 2 conv blocks (32→64 channels) + BatchNorm + adaptive pooling + FC head, trained with BCE loss / Adam, early stopping on a held-out loss |

Every script reports accuracy with a **95% confidence interval computed from the Beta distribution** (`calcular_intervalo_confianza`), a confusion matrix, per-class precision/recall/F1, and a feature-importance chart — native importance for tree models, coefficient magnitude for LDA, and **permutation importance** for MLP, CNN, and the ensembles (since they don't expose native importances). `train_ensemble.py` and `train_hard_ensemble.py` combine **CatBoost + Extra Trees + SVM** by soft voting (mean probability, threshold 0.5) and hard voting (majority of 3), respectively — note this is a *different* trio from the production ensemble below. `compare_models.py` runs all available saved models against the same `val` split and saves side-by-side bar charts (`docs/comparativa_modelos.png`, `docs/f1_por_clase.png`).

This track is what let the team choose CatBoost as the strongest individual model and get a first read on which features mattered before committing to the production ensemble.

### Track B — end-to-end pipeline evaluation (`tests/`)

This is the track whose numbers matter most, because it runs the *actual* inference path — raw WAV → `generar_turnos_vad` → `procesar_audio_base` → bandpass filter → 27 acoustic + 4 conversational features → model — rather than reading precomputed features from a CSV. Ground truth comes straight from `manifest.csv`.

The production ensemble tested here (`tests/evaluar_comparativa.py`, `tests/test_ponderado.py`) is **CatBoost + Random Forest + XGBoost**, combined by **weighted soft voting**:

| Model | File | Ensemble weight |
|---|---|---|
| CatBoost | `models/modelo_catboost_altur.cbm` | 0.70 |
| Random Forest | `models/modelo_rf_altur.joblib` | 0.15 |
| XGBoost | `models/modelo_xgboost_altur_v4.json` | 0.15 |

Weights come from each model's individual validation performance (CatBoost dominates the vote) and are re-normalized automatically if a model file is missing, so the ensemble degrades gracefully. `tests/test_ensemble.py` separately checks **unweighted** soft and hard voting across RF+CatBoost+XGBoost as a sanity comparison against the weighted version. The decision threshold used throughout is **0.40**, not 0.5 — biasing the system toward catching synthetic callers, since a missed synthetic voice is costlier than a false alarm on a human.

`tests/evaluar_comparativa.py` also does the **error-margin analysis**: for every misclassified call, how far its predicted probability sat from the 0.40 threshold, which is where the "average error distance" numbers below come from, plus the ROC comparison plot (`reports/graficas/comparativa_curvas_roc.png`).

---

## Results (Track B — end-to-end pipeline)

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

Evaluated over the `_orig`/`_agudo`/`_grave`/`_ruido` pool from `scripts/augment_dataset.py`, which only covers the `train` split — this tests robustness to pitch shift and additive noise specifically, not generalization to unseen speakers.

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

**Reading these together:** the ensemble buys precision and AUC on clean audio, but the standalone CatBoost model holds up slightly better on human-caller recall once pitch/noise perturbation is introduced — consistent with heavier noise eroding fine-grained LFCC/CPP detail more than it erodes the conversational-timing features.

---

## Deployed API — `api.py`

FastAPI service exposing `POST /detect`, matching Altur's contract (`call_id`, `audio_base64`, `sample_rate`, `channels` in; `is_synthetic`, `confidence` out). At startup it currently loads **XGBoost** into memory (`ml_models["xgboost"]`) — the target for judging is **CatBoost**, the strongest single model in Track B, but that swap hasn't landed in `api.py` yet. Once loaded, it decodes the incoming WAV to a temp file, runs it through the same VAD → acoustic → conversational pipeline used in Track B, applies the **0.40 threshold**, logs every prediction (id, verdict, confidence, latency, vocoder-peak feature) to a **MongoDB Atlas** collection (`proyectoAltur.historial_predicciones`), and — if anything in the pipeline throws (e.g. a silent/corrupt clip) — **falls back to `is_synthetic=False, confidence=0.0`** rather than crashing, so a single bad call doesn't take down the judge's scoring run.

### Known issues to resolve before the judging round

- **The live endpoint still serves standalone XGBoost — the switch to CatBoost is decided but not yet applied.** Track B shows CatBoost alone (0.99 accuracy / 0.9998 ROC-AUC clean, 0.92 / 0.9761 under stress) as the strongest single model, and the plan is to serve CatBoost in `api.py`, not the full ensemble. As of this README, `api.py` still loads `modelo_xgboost_altur_v3.json` (XGBoost) at startup — swapping this to `models/modelo_catboost_altur.cbm` via `CatBoostClassifier().load_model(...)` (see `tests/evaluar_comparativa.py` for the loading pattern) is the pending change before judging.
- **`confidence` semantics likely don't match what `check_endpoint.py` (Altur's own scoring script) expects.** `api.py` always sets `confidence = P(synthetic)`, regardless of the verdict. The judge harness computes AUC/Brier assuming `confidence` means "confidence in whichever verdict was returned" (`confidence` if `is_synthetic` else `1 - confidence`). Under the current implementation, a human verdict with a low raw `P(synthetic)` (e.g. 0.2) gets read by the harness as `1 - 0.2 = 0.8`, i.e. as if the model were fairly sure it was synthetic — the opposite of what happened. Fix: send `confidence = prob_ia if is_synthetic else (1 - prob_ia)`.
- **Import paths in `api.py` don't match the current module layout.** It imports `src.fase1.main`, `src.fase2.fase2_acustica`, and `src.fase3.fase3_conversacional`, while the rest of the repo has settled on `src.fase1.audio_base`, `src.fase2.acustica`, and `src.fase3.conversacional`. Worth reconciling so the deployed service actually starts.
- **Model filename drift.** `train_xgboost.py` saves `modelo_xgboost_altur.json`, while the Track B ensemble scripts load `modelo_xgboost_altur_v4.json`, and `api.py` loads `modelo_xgboost_altur_v3.json`. These look like successive training iterations — confirm which `.json` is actually the latest before the endpoint goes live.

None of these affect the *evaluation* numbers reported above (those come from directly loading the intended model files), but they matter for what the judges will actually hit at `POST /detect`.

---

## How the judges will actually score it — `Altur_Data/scripts/`

Altur ships its own harness in the dataset, and the team can run it locally against their own endpoint before judging:

- **`check_endpoint.py`**: replays calls from `manifest.csv` against a live `/detect` URL exactly like the judge will, and reports `accuracy`, `tpr_synthetic` (recall on synthetic), `tnr_human` (recall on human), `balanced_accuracy`, `mean_latency_s`/`max_latency_s`, and — only if every response includes `confidence` — **ROC-AUC and Brier score** computed from it. This is the concrete mechanism behind Altur's "robustness" and "latency" judging criteria, and the reason the confidence-semantics bug above is worth fixing.
- **`example_server.py`**: a minimal reference `/detect` server with a meaningless placeholder verdict — proves the request/response plumbing works, nothing more.

```bash
python Altur_Data/scripts/example_server.py --port 8000        # sanity-check the contract
python Altur_Data/scripts/check_endpoint.py --url http://localhost:8000/detect --split val --n 50
```

---

## Mapping to Altur's judging criteria

- **Robustness** — validated on a 1,128-call pool covering pitch shift and additive noise on top of the 353-call clean benchmark; `check_endpoint.py` can be re-run against the live endpoint on any split, including a held-out `hidden` split.
- **Originality** — 27 acoustic features (LFCC bank, CPP, spectral tilt, vocoder-periodicity) combined with 4 conversational-timing features, an 8-model bake-off with permutation importance, and a dedicated Fase 0 audit that catches dataset shortcuts before they reach the model.
- **Technical depth** — a custom-tuned VAD used consistently at train and serve time, an active-frame filter removing digital-silence/echo contamination, a two-split confirmation rule for every candidate signal, and confidence-interval reporting on every trained model.
- **Feasibility** — every model in the ensemble runs on CPU; the API degrades gracefully on bad input instead of failing the judge's run.
- **Latency** — tree-based models on precomputed features are fast per call; `check_endpoint.py` reports mean/max latency directly.

---

## Tech stack & sponsor tools

| Tool | Role |
|---|---|
| **Vultr** | Hosts the `POST /detect` server. CPU-only — none of the models need a GPU. |
| **ElevenLabs** | Team-recorded voices, with consent, synthesized as agent voices the model never trained on (`src/tools/estereo.py`) — an adversarial check, not a training source. |
| **MongoDB Atlas** | Every live prediction is logged to `proyectoAltur.historial_predicciones` (`api.py`) — id, verdict, confidence, latency, and vocoder-peak feature per call. A stretch goal (not yet built) is fingerprinting *synthetic* voices specifically to flag repeat cloned voices. |
| CatBoost / Random Forest / XGBoost / Extra Trees / SVM / LDA / MLP / 1D CNN | Full model roster benchmarked in `training/`; CatBoost + RF + XGBoost form the production ensemble validated in `tests/`. |
| Custom VAD (`src/tools/vad.py`) | Voice activity detection shared across the pipeline. |
| pyannote.audio | Speaker diarization used offline to convert arbitrary recordings into Altur's stereo + turns format for adversarial/demo audio. |

Tools evaluated and intentionally **not** used: Snowflake, Solana, Tiger Data (the feature set is small/tabular enough that they add infrastructure without improving detection); Gemini (a semantic layer — checking whether the caller says "I don't have that" vs. inventing an answer to a nonexistent detail — was scoped but requires sending audio to a third party, pending Altur's sign-off).

---

## Repository structure

```
Altur_Data/
  scripts/
    check_endpoint.py       # Altur's judge harness — run this against your own /detect
    example_server.py       # Altur's minimal reference server (plumbing only)
  audio/                    # original calls
  audio_augmented/          # _orig/_agudo/_grave/_ruido variants (train split only)
  turns/                    # dataset-provided turn annotations (reference, not a dependency)
  turns_vad/                # cached output of the team's own VAD (src/main.py)
  manifest.csv              # anon_id, label, split
  features_final_vad.csv    # training matrix built by src/main.py
scripts/
  augment_dataset.py        # pitch/noise augmentation for the train split
src/
  fase0/trampas.py          # data-integrity audit
  fase1/audio_base.py       # channel isolation, RMS normalization, resample
  fase2/acustica.py         # 27 acoustic features
  fase3/conversacional.py   # 4 conversational-timing features
  tools/
    vad.py                  # generar_turnos_vad, fusionar_turnos
    audio.py                 # hardened loader for path/bytes/base64 input
    estereo.py                # ElevenLabs adversarial audio download + prep
    generar_stereo.py         # pyannote diarization → stereo + turns.json
    muestras.py                # synthetic demo-call assembly
  main.py                    # orchestrates fase1→VAD→fase2→fase3 into features_final_vad.csv
tests/
  evaluar_comparativa.py     # Track B: CatBoost vs. weighted ensemble, end-to-end from audio
  test_ensemble.py           # unweighted soft/hard voting sanity check (RF+CatBoost+XGBoost)
  test_ponderado.py          # weighted ensemble only, per-model probability breakdown
  test_1.py                  # single-model check against manifest
  test_proto.py              # early MFCC-based feature prototype (superseded by fase2/acustica.py)
training/
  train_catboost.py, train_xgboost.py, train_rf.py, train_et.py,
  train_svm.py, train_lda.py, train_mlp.py, train_cnn.py   # 8-model bake-off
  train_ensemble.py, train_hard_ensemble.py                  # CatBoost+ExtraTrees+SVM voting
  compare_models.py                                          # side-by-side comparison + charts
  utils.py                                                    # data loading, CI, plotting helpers
models/                      # modelo_catboost_altur.cbm, modelo_rf_altur.joblib,
                              # modelo_xgboost_altur_v4.json, etc.
docs/                         # per-model confusion matrices, class metrics, feature importance
reports/graficas/             # Track B's ROC comparison plot
api.py                        # deployed POST /detect (see "Known issues" above)
Implementacion_Simple.md      # team-facing roadmap
IMPLEMENTACION1.1.md          # technical rationale per decision
```

## Running it

```bash
pip install -r requirements.txt

# Fase 0 — audit the dataset before trusting any feature
python -m src.fase0.trampas

# Build the training feature matrix (fase1 → VAD → fase2 → fase3)
python -m src.main

# Track A — train and compare all 8 models on precomputed features
python -m training.train_catboost   # ...and the rest of training/train_*.py
python -m training.compare_models

# Track B — end-to-end evaluation from raw audio (the numbers in this README)
python -m tests.evaluar_comparativa

# Serve the API
uvicorn api:app --host 0.0.0.0 --port 8000

# Score it the way Altur will
python Altur_Data/scripts/check_endpoint.py --url http://localhost:8000/detect --split val
```

`POST /detect` expects:

```json
{
  "call_id": "anon_123",
  "audio_base64": "<base64-encoded stereo 8kHz WAV, channel 0 = caller, channel 1 = agent>",
  "sample_rate": 8000,
  "channels": 2
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

- Apply the three items in *Known issues* above before the judging window opens: swap `api.py` from XGBoost to CatBoost, fix the `confidence` semantics, and reconcile the import paths.
- Close the human-recall gap under noise by making the conversational features noise-robust rather than leaning harder on acoustic ones under degraded audio.
- Extend the stress test to unseen speakers/engines specifically (the ElevenLabs adversarial set is the first step toward that).
- Pursue the semantic layer (caller responses to nonexistent-detail prompts) once data-sharing constraints with Altur are resolved.
- Voice fingerprinting of repeat synthetic callers in MongoDB Atlas, to catch fraud rings reusing the same cloned voice across multiple calls or institutions.

## Team

Fernando · Carlos · Andrés · Aarón — HackMTY 2026, Altur Challenge Track.
