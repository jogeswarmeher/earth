# Offshore Installation Weather Intelligence
### NVIDIA Earth-2 (FourCastNet + CorrDiff) + Llama-3.1-8B-Instruct + FastAPI + Streamlit

Prototype for a customer that installs offshore jacket/topside structures and needs
accurate 15-day offshore weather intelligence (wind, wave, current, cyclone probability,
visibility) to make **Proceed / Delay / Advance / Reschedule / Suspend** decisions and
avoid vessel + engineer standby cost overruns.

This matches the architecture in the source docs:

```
SQLite (Schedule / Constraints / Activities)
        │
        ▼
   FourCastNet  ──▶  CorrDiff  ──▶  Forecast JSON (15 days)
        │
        ▼
  Prompt Builder  (merges Forecast JSON + SQLite project data)
        │
        ▼
  Llama-3.1-8B-Instruct
        │
   ┌────┴─────────────────────────┐
   ▼                              ▼
Smart Weather Forecast        Free-flow Chat
   Intelligence (UI)             Interface (UI)
```

```
Streamlit (uinvearth.py)  ──HTTP──▶  FastAPI (appnvearth.py)
                                          │
                        ┌─────────────────┼─────────────────┐
                        ▼                 ▼                 ▼
                  FourCastNet         CorrDiff            SQLite
                  (earth2/)           (earth2/)         (db/)
                        └────────┬────────┘
                                 ▼
                        Forecast JSON (15 days)
                                 │
                                 ▼
                        Prompt Builder (llm/)
                                 │
                                 ▼
                        Llama-3.1-8B-Instruct (llm/llm_client.py)
```

---

## 1. Folder structure

```
nvidia-earth2-offshore/
├── README.md
├── requirements.txt
├── .env.example
├── data/                          # offshore_weather_demo.db lives here (git-ignored)
├── db/
│   ├── schema.py                  # CREATE TABLE + seed data (from Steps 1-4 of the schema doc)
│   └── db_utils.py                # read helpers used by the API
├── earth2/
│   ├── synthetic_fallback.py      # deterministic synthetic 15-day forecast generator (demo mode)
│   ├── fourcastnet_engine.py      # loads the FourCastNet *checkpoint* (not the NIM container) via PhysicsNeMo
│   ├── corrdiff_engine.py         # loads the CorrDiff checkpoint, downscales FourCastNet output
│   └── forecast_pipeline.py       # orchestrates FourCastNet → CorrDiff → Forecast JSON, with fallback
├── llm/
│   ├── prompt_builder.py          # builds the system + user prompt from Forecast JSON + SQLite data
│   └── llm_client.py              # Llama-3.1-8B-Instruct client (NVIDIA API catalog or local Ollama)
├── backend/
│   ├── config.py                  # env-driven settings
│   ├── models.py                  # Pydantic request/response schemas
│   └── appnvearth.py              # FastAPI app — GET /forecast/15days, GET /project, POST /chat
├── frontend/
│   └── uinvearth.py               # Streamlit UI — dashboard + free-flow chat
├── scripts/
│   ├── init_db.py                 # creates + seeds the SQLite DB
│   └── download_checkpoints.sh    # `ngc registry model download-version ...` commands
└── tests/
    └── test_api.py
```

---

## 2. Step-by-step setup

### Step 1 — Python environment
```bash
cd nvidia-earth2-offshore
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Configure environment
```bash
cp .env.example .env
# edit .env:
#   USE_REAL_MODEL=false          <- start in synthetic/demo mode (no GPU needed)
#   LLM_PROVIDER=nvidia           <- or "ollama" for a fully local/free LLM
#   NVIDIA_API_KEY=nvapi-...      <- free key from https://build.nvidia.com (Llama-3.1-8B-Instruct)
```

### Step 3 — Initialize the SQLite database
```bash
python scripts/init_db.py
```
This creates `data/offshore_weather_demo.db` with `project_schedule`,
`project_constraints`, and `installation_activity` — exactly the three tables from the
schema doc (weather is intentionally **not** stored here; FourCastNet + CorrDiff
generate it live).

### Step 4 (optional, for real Earth-2 inference) — Download the FourCastNet / CorrDiff checkpoints
This uses the **NGC CLI to download the raw model package**, not the paid NIM
container:
```bash
bash scripts/download_checkpoints.sh
```
This pulls down `training_checkpoints/`, `config.json`, `global_means.npy`,
`global_stds.npy`, `land_mask.nc`, `orography.nc` for FourCastNet, plus the CorrDiff
checkpoint. Point `FOURCASTNET_CHECKPOINT_DIR` / `CORRDIFF_CHECKPOINT_DIR` in `.env` at
the download location, install the optional GPU deps, and set `USE_REAL_MODEL=true`:
```bash
pip install physicsnemo earth2studio torch --extra-index-url https://download.pytorch.org/whl/cu121
```
Without this step, the pipeline automatically runs in **synthetic fallback mode** —
useful for building/demoing the FastAPI + Streamlit + LLM layers before you have GPU
access.

### Step 5 — Run the FastAPI backend
```bash
uvicorn backend.appnvearth:app --reload --port 8000
```
Check it: `curl http://localhost:8000/forecast/15days`

### Step 6 — Run the Streamlit UI
```bash
streamlit run frontend/uinvearth.py
```
Open the URL Streamlit prints (default `http://localhost:8501`).

---

## 3. What "real" vs "demo" mode means

| Mode | `USE_REAL_MODEL` | What runs |
|---|---|---|
| Demo / prototype | `false` | `earth2/synthetic_fallback.py` generates a physically-plausible 15-day forecast (with 2–3 synthetic cyclone spikes), matching the exact Forecast JSON schema the real pipeline would produce. Everything downstream (FastAPI, Streamlit, LLM decisioning) is identical. |
| Production-style | `true` | `earth2/fourcastnet_engine.py` loads the downloaded FourCastNet checkpoint via PhysicsNeMo/Earth2Studio for a medium-range forecast, then `earth2/corrdiff_engine.py` downscales it with CorrDiff to produce the final high-resolution Forecast JSON. If model loading fails for any reason, the pipeline logs a warning and falls back to synthetic mode rather than crashing the demo. |

---

## 4. API reference

### `GET /forecast/15days`
Returns the 15-day Forecast JSON (one entry per day): `day, timestamp, lat, lon,
wind_speed, wave_height, current_speed, cyclone_probability, visibility,
precipitation, pressure, confidence`.

### `GET /project`
Returns `project_schedule`, `project_constraints`, `installation_activities` from
SQLite.

### `POST /chat`
```json
{ "question": "Can Jacket Lift proceed on Day 10?", "day": 10 }
```
Flow: question → latest Forecast JSON → SQLite project data → Prompt Builder →
Llama-3.1-8B-Instruct → free-flow response (Decision, Confidence, Weather Summary,
Activities Impacted, Reasoning, Risk Level, Safe Installation Window, Recommended
Action).

---

## 5. Notes / next steps

- Cost modelling (vessel $/day, engineer standby, penalty risk) is intentionally
  excluded from this baseline, matching the doc's scope ("Cost: Excluded").
- The Prompt Builder's system prompt enforces "use only the supplied forecast and
  project constraints, do not invent weather values" — the LLM never hallucinates
  numbers that aren't in the Forecast JSON / SQLite payload.
- Swap `LLM_PROVIDER=ollama` in `.env` to run Llama-3.1-8B-Instruct fully locally and
  offline (`ollama pull llama3.1:8b`) with zero API cost.
