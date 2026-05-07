<div align="center">

```
╔═══════════════════════════════════════════════════════╗
║           O P T I C A S T   A I                      ║
║   Probabilistic Renewable Generation Forecasting      ║
║                                                       ║
║   Karnataka Grid · KERC DSM Compliance · Air-Gapped  ║
╚═══════════════════════════════════════════════════════╝
```

**AI for Bharat 2026 · Theme 10 · KREDL / KSPDCL**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![TFT](https://img.shields.io/badge/Model-TFT%20%2B%20Conformal-orange?style=flat-square)](https://pytorch-forecasting.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Air-Gap Ready](https://img.shields.io/badge/Deploy-Air--Gap%20%7C%20KSDC-purple?style=flat-square)]()

</div>

---

> Karnataka has **23 GW of installed renewable capacity** and still faced **500+ hours of power shortage in 2024**.
> The problem is not how much it generates — it is that nobody can predict *when*.
> OptiCast fixes that with mathematically guaranteed confidence intervals. Not guesses.

---

## The Problem

Karnataka's grid operators schedule renewable generation in 15-minute blocks under KERC's Deviation Settlement Mechanism (DSM). Miss the band, pay the penalty. The maths are unforgiving:

| Deviation | Penalty Rate | Annual Exposure |
|---|---|---|
| 5–25% off schedule | 12% of APPC (Rs 0.54/unit) | Rs 477 crore |
| 25–35% off schedule | 20% of APPC | |
| > 35% off schedule | 30% of APPC | |

**Three failure modes drive these penalties:**

**1 · The Evening Ramp Problem**
Solar peaks 11 AM–3 PM. Demand peaks 6–9 PM. Without accurate intra-day forecasts, operators discover the 3-hour ramp requirement too late and import emergency Real-Time Market power at a 40–60% premium.

**2 · Monsoon Forecasting Collapse**
June–September cloud chaos causes 20–30% systematic deviations. A P50 point forecast gives operators false confidence. Every deviation triggers DSM penalties.

**3 · Pavagada Voltage Ramp Events**
A 400 MW drop in 8 minutes due to cloud passage creates voltage instability at Pavagada's pooling substations. KSPDCL needs 30-minute advance warning. Without a ramp-aware probabilistic layer, this is impossible.

**Conservative saving with OptiCast: Rs 191 crore / year.**
Methodology: 44,096 MU/year × reducing penalty blocks from 20% → 8% × Rs 0.54/unit. Every number is verifiable from public CEA and KERC data.

---

## The Solution

OptiCast is a **five-layer AI forecasting and compliance system** deployable on Karnataka's own State Data Centre. No vendor lock-in. No cloud dependency. KREDL owns the code permanently.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 5 · Dashboard                                        │
│  Karnataka map · Forecast ribbons · SHAP cards · QCA export │
├─────────────────────────────────────────────────────────────┤
│  LAYER 4 · KERC Compliance Engine                           │
│  96-block QCA schedule · DSM tracker · Ramp alerts · Audit  │
├─────────────────────────────────────────────────────────────┤
│  LAYER 3 · Uncertainty Quantification                       │
│  Quantile regression (P10/P50/P90) + MAPIE Conformal (90%  │
│  mathematically guaranteed coverage band)                   │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2 · AI Core — Temporal Fusion Transformer            │
│  Single model · All Karnataka solar + wind · Static plant   │
│  metadata · Jensen wake correction · Monsoon gating         │
├─────────────────────────────────────────────────────────────┤
│  LAYER 1 · Data Pipeline                                    │
│  ERA5 via Open-Meteo (free, no key) · SCADA connector       │
│  TimescaleDB · Gap filling · Plant features                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Why This Model Stack

### Temporal Fusion Transformer (TFT)

Three independent 2024–2025 research validations confirm TFT outperforms LSTM, CNN-BiLSTM, and Prophet for renewable generation forecasting:

- *Frontiers in Artificial Intelligence*, Feb 2025 — PMC11996805
- *International Journal of Energy Research*, Wiley, Feb 2025
- arXiv:2512.00856 — Robust Probabilistic Load Forecasting (2024)

Five reasons TFT is the right choice for Karnataka specifically:

| Capability | Why It Matters Here |
|---|---|
| All 96 blocks in one pass | No sequential error accumulation across 24-hour horizon |
| Static metadata embedding | One model handles Pavagada (2,050 MW solar) and Koppal (300 MW wind) without separate training |
| Variable Selection Networks | Auto-ranks GHI vs wind speed vs cloud cover per plant per horizon |
| Monsoon gating | Automatically down-weights clear-sky summer patterns when cloud sequences dominate |
| Attention masking | Handles SCADA gaps under 2 hours without imputation that would crash LSTM |

### Conformal Prediction (MAPIE)

Quantile regression outputs P10/P50/P90 bands — but only as accurately as the model's calibration. During distribution shift (monsoon onset, unprecedented heat waves), these can be systematically wrong with no warning.

MAPIE Conformal Prediction adds a **distribution-free mathematical guarantee**:

> When OptiCast outputs a P90 band, the true generation value will fall inside it 90% of the time. Provably. Not by assumption.

This is what lets operators make the scheduling decision that eliminates most DSM penalties: schedule spinning reserve at the P10 lower bound, not P50. With a guarantee, that decision has defined statistical risk. With a guess, it has none.

Research basis: arXiv:2602.02583, arXiv:2510.15780, arXiv:2502.04935.

---

## Performance Targets

| Metric | Persistence Baseline | Climatology Baseline | OptiCast Target |
|---|---|---|---|
| Day-ahead Solar MAPE | 25–35% | 18–22% | **< 12%** |
| Day-ahead Wind MAPE | 30–40% | 20–25% | **< 15%** |
| P90 Conformal Coverage | N/A | N/A | **≥ 90% (guaranteed)** |
| Ramp Detection (>100 MW/hr) | Reactive | Reactive | **≥ 80% with 30 min lead** |
| DSM Penalty Block Rate | ~20% | ~15% | **< 8%** |
| Annual DSM Saving | — | — | **Rs 191 crore (floor)** |

---

## Project Structure

```
Opticast-AI/
│
├── api/
│   └── main.py                  # FastAPI — 9 endpoints, JWT RBAC
│
├── compliance/
│   ├── qca_generator.py         # 96-block KERC QCA schedule CSV
│   ├── deviation_tracker.py     # Real-time actual-vs-scheduled delta
│   ├── ramp_alert.py            # Ramp event detector (>50 MW / 30 min)
│   └── output/                  # Generated QCA CSVs, audit log, DSM projections
│
├── models/
│   ├── tft_trainer.py           # TFT training (PyTorch Forecasting)
│   ├── conformal.py             # MAPIE CP calibration layer
│   ├── shap_explainer.py        # SHAP feature attribution per block
│   ├── tft_best.ckpt            # Trained checkpoint
│   └── master_dataset.parquet  # Training dataset
│
├── data/
│   ├── ingestion_service.py     # ERA5 fetch + SCADA sim + DB write
│   └── generate_scada.py        # Synthetic generation profiles + anomaly injection
│
├── db/
│   └── init.sql                 # TimescaleDB schema + hypertable definitions
│
├── app.py                       # Streamlit dashboard entry point
├── db.py                        # Dashboard DB connection helpers
│
├── Dockerfile                   # API service image
├── Dockerfile.dashboard         # Streamlit image
├── Dockerfile.ingestion         # Ingestion worker image
├── docker-compose.yml           # 5-service orchestration
│
├── requirements.api.txt
├── requirements.dashboard.txt
└── requirements.ingestion.txt
```

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- 8 GB RAM minimum (16 GB recommended for TFT training)
- GPU optional but reduces training time from ~12 hours to ~5 hours

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/Opticast-AI.git
cd Opticast-AI
cp .env .env.local   # edit secrets before first run
```

### 2. Launch the full stack

```bash
docker-compose up --build
```

On first boot, TimescaleDB initialises its schema and TimescaleDB extensions. Allow **60 seconds** before the API becomes reachable — this is normal. All downstream services wait for the DB healthcheck automatically.

| Service | URL | Notes |
|---|---|---|
| Streamlit Dashboard | http://localhost:8501 | Main operator UI |
| FastAPI (Swagger) | http://localhost:8000/docs | REST API docs |
| MLflow Tracking | http://localhost:5000 | Model experiment log |
| TimescaleDB | localhost:5432 | Direct DB access |

### 3. Run ingestion (populates the database)

```bash
# Runs automatically as part of docker-compose, or manually:
docker-compose run --rm ingestion

# Backfill a specific date:
docker-compose run -e RUN_DATE=2024-08-15 --rm ingestion
```

Ingestion fetches 2 years of ERA5 weather data for five Karnataka plant locations via Open-Meteo (free, no API key required), generates synthetic SCADA profiles with injected cloud transients, and writes everything to TimescaleDB.

### 4. Train the TFT model (optional — a checkpoint is included)

```bash
# Inside the api container or locally with pip install -r requirements.api.txt:
python models/tft_trainer.py
# GPU:  ~4–6 hours
# CPU:  ~10–12 hours
# The checkpoint tft_best.ckpt is already included for immediate demo use.
```

### 5. Local development (without Docker)

```bash
pip install -r requirements.api.txt
uvicorn api.main:app --reload --port 8000

pip install -r requirements.dashboard.txt
streamlit run app.py
```

---

## API Endpoints

Full Swagger documentation at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/forecast/{plant_id}/day-ahead` | 96-block P10/P50/P90 + conformal bands |
| `GET` | `/forecast/{plant_id}/intra-day` | Rolling 4-hour updated forecast |
| `GET` | `/forecast/cluster/{id}/summary` | Aggregated cluster-level forecast |
| `GET` | `/compliance/qca/{plant_id}/{date}` | KERC 96-block QCA schedule as CSV |
| `GET` | `/compliance/deviation/{plant_id}` | Live actual vs scheduled delta |
| `GET` | `/compliance/dsm-penalty/running` | Running penalty for current settlement cycle (Rs) |
| `GET` | `/alerts/ramps` | Active ramp alerts with spatial timing estimate |
| `GET` | `/explain/{plant_id}/{block_no}` | SHAP feature attribution for one 15-min block |
| `GET` | `/admin/model-performance` | MAE/RMSE trends and calibration coverage curves |

---

## Technology Stack

| Component | Technology | Why This, Not That |
|---|---|---|
| Forecasting model | PyTorch Forecasting TFT 1.1 | Static covariate support; 3 independent 2024–25 validations. LSTM rejected: sequential error. Prophet rejected: no multivariate static metadata. |
| Uncertainty layer | Quantile Loss + MAPIE 0.8 | Distribution-free coverage guarantee. Bayesian deep learning rejected: GPU cost. MC Dropout rejected: unreliable under distribution shift. |
| Explainability | SHAP DeepExplainer | Operator-readable; VSN-compatible. LIME rejected: unstable for time-series. |
| Weather data | Open-Meteo ERA5 API | Free; no API key; 0.25° grid; 1979–present. IMD API rejected: government agreement required. Tomorrow.io rejected: paid vendor lock-in. |
| Time-series DB | PostgreSQL + TimescaleDB | NIC deployable with existing DBAs; automatic hypertable partitioning. InfluxDB rejected: separate system. |
| Backend | FastAPI 0.111 | Async; auto OpenAPI docs; JWT RBAC; CERT-In compatible; NIC-approved language. |
| Dashboard | Streamlit + Plotly + Leaflet.js | Rapid iteration; KERC report export; PWA-capable. |
| Deployment | Docker Compose | Air-gap capable; single command on KSPDCL SDC; no Kubernetes training required. |
| Model tracking | MLflow 2.13 | One-click rollback; government-auditable versioning. W&B rejected: cloud-only. |

---

## Deployment on Karnataka SDC

OptiCast is designed for **air-gapped deployment** on the Karnataka State Data Centre (KSDC), fully compliant with MeitY data localisation requirements.

```bash
# On the SDC server, after loading the Docker images via tarball:
docker load < opticast_images.tar.gz
docker-compose up -d

# All outbound internet access is needed only for:
#   - First-run image pull (can be pre-loaded from tarball)
#   - ERA5 data fetch during ingestion (can be pre-cached)
# Post-setup: fully air-gapped.
```

**Compliance checklist:**
- ✅ All data stays within KREDL/KSPDCL network
- ✅ CERT-In compatible architecture (FastAPI + PostgreSQL + Docker)
- ✅ MeitY data localisation — no raw generation data leaves Karnataka's servers
- ✅ Cryptographically signed, append-only audit log for every DSM calculation
- ✅ MIT licence — KREDL owns the code permanently at no ongoing cost

---

## Covered Plant Locations

| Plant | Type | Capacity | Coordinates |
|---|---|---|---|
| Pavagada Solar Park | Solar | 2,050 MW | 14.10°N, 77.28°E |
| Chitradurga Wind Cluster | Wind | 500 MW | 14.22°N, 76.39°E |
| Koppal Wind Farm | Wind | 300 MW | 15.35°N, 76.15°E |
| Gadag Wind Cluster | Wind | 250 MW | 15.41°N, 75.63°E |
| Raichur Solar | Solar | 400 MW | 16.20°N, 77.36°E |

The single TFT model covers all five plants simultaneously using static metadata embeddings — no separate model per plant.

---

## Phase 2 Roadmap

| Feature | Technology | Purpose |
|---|---|---|
| ST-GNN Ramp Propagation Oracle | PyTorch Geometric | Predict spatial cloud-front timing between plant clusters with 15-min granularity |
| Federated Learning | Flower (flwr) | BESCOM / HESCOM / GESCOM train a shared model without exchanging raw generation data |
| Online Mondrian CP | Regime-conditional conformal | Recalibrates coverage bands in real-time as monsoon onset changes the data distribution |
| NWP Ensemble Fusion | Multiple ERA5 + IMDAA runs | Ensemble weather inputs for tighter band widths on clear-sky days |

---

## Contributing

This project was built for the AI for Bharat 2026 Hackathon. KREDL and KSPDCL are welcome to fork, modify, and deploy under the MIT licence without restriction or royalty.

For issues, open a GitHub issue with the label `bug` or `enhancement`.

---

## Licence

MIT — see [LICENSE](LICENSE).

KREDL, KSPDCL, and the Government of Karnataka may use, modify, and deploy this codebase permanently at no cost.

---

<div align="center">

*Built for AI for Bharat 2026 · Theme 10 · Karnataka Renewable Energy Development Limited*

*Karnataka's grid. Karnataka's data. Karnataka's code.*

</div>
