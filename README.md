# OptiCast AI — Probabilistic Renewable Generation Forecasting

**AI for Bharat 2026 | Theme 10 | KREDL / KSPDCL**

> Karnataka loses **Rs 191 crore/year** in avoidable DSM penalties due to forecasting failure.  
> OptiCast fixes that with mathematically guaranteed confidence intervals — not guesses.

---

## The Problem

Karnataka has 23+ GW of installed renewable capacity yet faced **500+ hours of power shortage in 2024**.
The cause is not insufficient capacity — it is forecasting failure. Grid operators schedule generation against
a single P50 point forecast and face 20–30% deviations during monsoon and ramp events, triggering
compounding KERC DSM penalties.

**Three failure modes:**
1. **Evening ramp problem** — solar peaks 11 AM–3 PM, demand peaks 6–9 PM. No accurate intra-day forecast = emergency RTM imports at 40–60% premium.
2. **Monsoon forecasting collapse** — chaotic cloud cover causes systematic 20–30% deviations. Each triggers DSM penalties.
3. **Pavagada voltage ramp events** — 400 MW drop in 8 minutes. KSPDCL needs 30-min advance warning. Currently impossible.

---

## The Solution

OptiCast is a **five-layer AI system** delivering probabilistic forecasts with statistically guaranteed confidence intervals and a built-in KERC compliance engine.

| Layer | What It Does |
|---|---|
| **Layer 1 — Data Pipeline** | ERA5/Open-Meteo API (free, no key needed) + synthetic SCADA with TimescaleDB |
| **Layer 2 — AI Core (TFT)** | Single Temporal Fusion Transformer for all solar/wind assets across Karnataka |
| **Layer 3 — Uncertainty** | Quantile regression (P10/P50/P90) + MAPIE Conformal Prediction with 90% coverage guarantee |
| **Layer 4 — KERC Compliance** | Auto-generates 96-block QCA schedule CSV, real-time DSM tracker, ramp alert detector |
| **Layer 5 — Dashboard** | Karnataka Leaflet.js map, forecast ribbons, SHAP explanation cards, one-click KERC export |

---

## Performance Targets

| Metric | Baseline | OptiCast Target |
|---|---|---|
| Day-ahead Solar MAPE | 18–22% (climatology) | **< 12%** |
| Day-ahead Wind MAPE | 20–25% | **< 15%** |
| P90 Conformal Coverage | N/A | **≥ 90% (guaranteed)** |
| Ramp Event Detection | Reactive only | **80%+ detected 30 min ahead** |
| DSM Penalty Block Rate | ~20% of blocks | **< 8% of blocks** |
| Annual DSM Saving | Baseline | **Rs 191 crore/year** |

---

## Why TFT?

Three independent 2024–2025 research validations confirm TFT superiority over LSTM and Prophet:
1. Frontiers in Artificial Intelligence, Feb 2025 (PMC11996805)
2. International Journal of Energy Research, Wiley, Feb 2025
3. arXiv:2512.00856 — Robust Probabilistic Load Forecasting (2024)

**Key advantages for Karnataka's grid:**
- Produces all 96 fifteen-minute blocks simultaneously (no sequential error accumulation)
- Static metadata embedding — one model handles Pavagada 2,050 MW solar AND Koppal 300 MW wind
- Variable Selection Networks (VSN) auto-rank feature importance per plant per horizon
- Native missing data handling via attention masking — SCADA gaps under 2 hours handled gracefully

---

## Why Conformal Prediction?

Quantile regression outputs P10/P50/P90 — but these are only as reliable as the model's calibration.
During distribution shift (monsoon onset, heat waves), quantile outputs can be systematically wrong.

**MAPIE Conformal Prediction adds a distribution-free mathematical guarantee:**
- P90 band contains the true value **90% of the time — provably, not by assumption**
- Band adapts: tight on clear-sky days, wide during monsoon
- Research: arXiv:2602.02583, arXiv:2510.15780, arXiv:2502.04935

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Python 3.10+ (for local development)
- 8GB RAM minimum; GPU recommended for TFT training

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/Opticast_AI.git
cd Opticast_AI
pip install -r requirements.txt
```

### 2. Download ERA5 weather data (free, no API key)
```bash
python data/fetch_era5.py
```
Downloads 2 years of 15-minute weather data for 5 Karnataka plant locations.

### 3. Generate synthetic SCADA profiles
```bash
python data/generate_scada.py
python data/dsm_simulator.py
```

### 4. Train the TFT model
```bash
python models/tft_trainer.py
# GPU: ~4–6 hours | CPU: ~10–12 hours | LightGBM fallback: ~1–2 hours
```

### 5. Calibrate conformal prediction
```bash
python models/conformal.py
# Validates empirical coverage >= 90% on test set
```

### 6. Launch everything with Docker
```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| FastAPI (Swagger docs) | http://localhost:8000/docs |
| MLflow tracking | http://localhost:5000 |
| TimescaleDB | localhost:5432 |

---

## Project Structure

```
Opticast_AI/
├── data/
│   ├── fetch_era5.py          # ERA5 weather download for 5 Karnataka plants
│   ├── generate_scada.py      # Synthetic generation profiles + anomaly injection
│   └── dsm_simulator.py       # KERC DSM penalty simulation
├── models/
│   ├── tft_trainer.py         # TFT training pipeline (PyTorch Forecasting)
│   ├── conformal.py           # MAPIE conformal prediction calibration
│   └── shap_explainer.py      # SHAP DeepExplainer feature attribution
├── compliance/
│   ├── qca_generator.py       # KERC 96-block QCA schedule generator
│   ├── deviation_tracker.py   # Real-time actual vs scheduled DSM monitor
│   └── ramp_alert.py          # Ramp event detection (>50 MW / 30-min)
├── api/
│   └── main.py                # FastAPI backend (9 endpoints)
├── dashboard/
│   └── src/                   # React PWA (Leaflet.js + Recharts)
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /forecast/{plant_id}/day-ahead` | 96-block P10/P50/P90 + conformal bands |
| `GET /forecast/{plant_id}/intra-day` | Rolling 4-hour updated forecast |
| `GET /forecast/cluster/{id}/summary` | Aggregated cluster-level forecast |
| `GET /compliance/qca/{plant_id}/{date}` | Auto-generated KERC schedule CSV |
| `GET /compliance/deviation/{plant_id}` | Real-time actual vs scheduled delta |
| `GET /compliance/dsm-penalty/running` | Running penalty this settlement cycle |
| `GET /alerts/ramps` | Active ramp alerts with spatial timing |
| `GET /explain/{plant_id}/{block_no}` | SHAP explanation for one 15-min block |
| `GET /admin/model-performance` | MAE/RMSE trends and calibration curves |

---

## Financial Impact

**Conservative DSM Penalty Calculation (methodology transparent):**

| Parameter | Value |
|---|---|
| Karnataka RE generation | ~44,096 MU/year (CEA data) |
| Current penalty block rate | ~20% of 15-min blocks |
| Average penalty (15–25% deviation band) | 12% of APPC = Rs 0.54/unit |
| **Current annual burden** | **Rs 477 crore/year** |
| OptiCast target penalty rate | < 8% of blocks |
| **Conservative annual saving** | **Rs 191 crore/year** |

---

## Technology Stack

| Component | Technology | Why |
|---|---|---|
| Forecasting model | PyTorch Forecasting TFT v1.1 | 3 independent 2024–25 validations; static covariate support |
| Uncertainty layer | Quantile Loss + MAPIE 0.8 | Distribution-free coverage guarantee |
| Explainability | SHAP DeepExplainer | Operator-readable; VSN compatible |
| Weather data | Open-Meteo ERA5 API | Free; no key; 0.25° grid; 1979–present |
| Time-series DB | PostgreSQL + TimescaleDB | NIC-deployable; automatic partitioning |
| Backend | FastAPI 0.111 | Async; JWT RBAC; CERT-In compatible |
| Dashboard | React + Recharts + Leaflet.js | Open-source; PWA offline; KERC export |
| Deployment | Docker Compose | Air-gap capable; single-command on KSPDCL SDC |
| Model tracking | MLflow 2.13 | One-click rollback; government auditable |

---

## Deployment Notes

OptiCast is designed for **air-gapped deployment** on Karnataka's State Data Centre (KSDC):
- No outbound internet required post-setup
- CERT-In compliant architecture
- MeitY data localisation compliant — all data stays within Karnataka's network
- KREDL owns the code permanently under MIT licence

---

## Phase 2 Roadmap

| Feature | Technology | Purpose |
|---|---|---|
| ST-GNN Ramp Propagation Oracle | PyTorch Geometric | Spatial cloud-front timing between plant clusters |
| Federated Learning | Flower (flwr) | BESCOM/HESCOM/GESCOM train without sharing raw data |
| Online Mondrian CP | Regime-conditional CP | Adapts calibration to weather shocks in real-time |

---

## Licence

MIT — KREDL/KSPDCL may use, modify, and deploy this codebase permanently at no cost.

---

*Built for AI for Bharat 2026 Hackathon | Theme 10 | Karnataka Renewable Energy Development Limited*
