# TrueProfile AI 🛡️

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20DeepFace-orange.svg)](https://xgboost.readthedocs.io/)
[![Alembic](https://img.shields.io/badge/Migrations-Alembic-purple.svg)](https://alembic.sqlalchemy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TrueProfile AI** is a state-of-the-art multi-modal fake account detection platform. Rather than relying on single-signal heuristics (like pure profile picture search or simple follower ratios), TrueProfile AI fuses **visual biometric signals**, **deepfake artifacts**, **behavioral telemetry**, **metadata consistency**, and **graph network topologies** into a robust, probabilistic risk score.

---

## 🌟 Architecture & Intelligence Pipeline

TrueProfile AI evaluates social media accounts across **5 distinct feature dimensions** and synthesizes them through a meta-classifier fusion engine:

```
                                  ┌───────────────────────────┐
                                  │   Target Profile Input    │
                                  └─────────────┬─────────────┘
                                                │
         ┌──────────────────┬───────────────────┼───────────────────┬──────────────────┐
         ▼                  ▼                   ▼                   ▼                  ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  Face Analysis │  │ Deepfake Check │  │ XGBoost Behav. │  │ Account Meta.  │  │ Network Graph  │
│  (ArcFace/DLib)│  │ (DeepFace Net) │  │  (Classifier)  │  │ (Heuristics)   │  │  (NetworkX)    │
└────────┬───────┘  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘
         │                   │                   │                   │                   │
         │ (Face Score)      │ (Deepfake Score)  │ (Behavior Score)  │ (Metadata Score)  │ (Network Score)
         └───────────────────┴───────────────────┼───────────────────┴───────────────────┘
                                                 ▼
                                  ┌───────────────────────────┐
                                  │   ML Fusion Meta-Engine   │
                                  │  (Logistic Meta-Model)    │
                                  └─────────────┬─────────────┘
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Final Risk Assessment   │
                                  │  • Score: 0.00 – 1.00     │
                                  │  • Risk Level (LOW/HIGH)  │
                                  │  • Action (FLAG/REVIEW)   │
                                  │  • Transparent Evidence   │
                                  └─────────────┬─────────────┘
                                                │
                          ┌─────────────────────┴─────────────────────┐
                          ▼                                           ▼
             ┌─────────────────────────┐                 ┌─────────────────────────┐
             │  Automated Action/Flag  │                 │   Human Review Queue    │
             │   (High/Low Confidence) │                 │  (Borderline Decisions) │
             └─────────────────────────┘                 └─────────────────────────┘
```

### The 5 Core Signal Pillars:

1. **Biometric Face Verification (`face_score`)**:
   - Powered by **ArcFace** deep representations via `DeepFace`.
   - Flags non-human images, stock photos, or duplicated identity vectors.
2. **Deepfake & Synthetic Media Detection (`deepfake_score`)**:
   - Analyzes facial compression anomalies, GAN artifacts, and synthetic generation markers.
3. **Behavioral Telemetry Model (`behavior_score`)**:
   - **XGBoost Classifier** (~92% realistic accuracy with noise tolerance).
   - Evaluates non-linear interaction patterns across account age, post frequency variance, follow burst velocities, and engagement decay.
4. **Metadata & Profile Completeness (`metadata_score`)**:
   - Detects incomplete profiles, default avatars, abnormal account age-to-activity ratios, and bio anomalies.
5. **Network Topology & Graph Centrality (`network_score`)**:
   - Built using **NetworkX** ego-graph simulations.
   - Computes clustering coefficients, reciprocity, follower-following degree imbalances, and bot-cluster centrality.
6. **Probabilistic Fusion Meta-Engine**:
   - Trained meta-classifier (Logistic Regression) that weights each component's predictive power while preserving an empirical weighted average fallback.

---

## 🚀 Key Production Features

- **🛡️ Human-in-the-Loop Review Queue**: Automatically routes ambiguous borderline cases (`0.50`–`0.84` risk) to `/review/queue` where human operators can audit evidence and record `APPROVE`/`FLAG` actions with review notes.
- **⚡ Built-in Rate Limiting**: Powered by `slowapi` to protect expensive inference endpoints (20 req/min for analysis, 60 req/min global).
- **🔑 API Key Authentication**: Integrated `X-API-Key` security scheme with a key generation CLI and optional dev bypass (`BYPASS_AUTH=true`).
- **📊 Observability & Error Monitoring**: Sentry SDK integration ready to track unhandled exceptions and performance bottlenecks in production.
- **🗄️ Database & Schema Migrations**: SQLAlchemy ORM with **Alembic** migration tracking. Runs seamlessly on local **SQLite** and migrates automatically to production **PostgreSQL**.
- **☁️ 1-Click Cloud Deployment**: Pre-configured `render.yaml` and `Procfile` for Render, Heroku, or Railway deployment.

---

## 📁 Repository Structure

```text
TrueProfile-AI/
├── backend/
│   ├── auth.py                  # API Key dependency & authentication logic
│   ├── database.py              # SQLAlchemy engine & session manager
│   ├── main.py                  # FastAPI application & middleware configuration
│   ├── requirements.txt         # Production backend dependencies
│   ├── ml/                      # Machine learning signal modules
│   │   ├── base.py              # BaseModelInterface definition
│   │   ├── face_analysis.py     # ArcFace biometric integration
│   │   ├── deepfake_detector.py # Synthetic media detector
│   │   ├── behavior_model.py    # XGBoost behavioral model wrapper
│   │   ├── metadata_model.py    # Metadata heuristics analyzer
│   │   ├── network_analysis.py  # NetworkX ego-graph centrality analyzer
│   │   └── fusion_engine.py     # ML meta-classifier & weighted baseline
│   ├── models/                  # SQLAlchemy database models (Profile, APIKey)
│   ├── routers/                 # API endpoint routers (profiles, review)
│   ├── schemas/                 # Pydantic validation & response schemas
│   ├── services/                # Orchestration & profile ingestion pipeline
│   └── tests/                   # Pytest automated test suite
├── alembic/                     # Database schema versioning & migration scripts
├── frontend/                    # Web client UI dashboard
├── models/                      # Trained model artifacts (.json, .pkl)
├── scripts/                     # Utility scripts (training, API key generator)
│   ├── train_xgboost.py         # XGBoost model trainer
│   ├── train_fusion.py          # Fusion meta-classifier trainer
│   └── generate_api_key.py      # Secure API key provisioning CLI
├── .env.example                 # Environment configuration template
├── alembic.ini                  # Alembic migration configuration
├── Procfile                     # Heroku/Railway process file
├── render.yaml                  # Render.com infrastructure blueprint
└── README.md                    # Project documentation
```

---

## 🛠️ Local Development & Quickstart

### 1. Prerequisites
- **Python 3.11+**
- **Git**

### 2. Setup Virtual Environment & Install Dependencies

```powershell
# Clone the repository
git clone https://github.com/krishnagupta2107/TrueProfile-AI.git
cd TrueProfile-AI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\activate
# Linux / macOS:
# source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
*(Default settings use SQLite and `BYPASS_AUTH=true` for local development).*

### 4. Train/Initialize ML Models

Train the XGBoost behavioral model and the ML Fusion meta-model (saves artifacts to `models/`):
```powershell
python scripts/train_xgboost.py
python scripts/train_fusion.py
```

### 5. Launch Local Servers

**Terminal 1 — Backend (FastAPI)**:
```powershell
.\venv\Scripts\activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
* Interactive API Documentation (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Health Check: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

**Terminal 2 — Frontend**:
```powershell
python -m http.server 8080 --directory frontend
```
* Dashboard URL: [http://localhost:8080](http://localhost:8080)

---

## 🧪 Running Automated Tests

Run the full pytest suite to verify ML models, API endpoints, and database interactions:

```powershell
pytest backend/tests -v
```

---

## 📡 API Overview

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/` | Service health check and metadata | No |
| `POST` | `/profiles/analyze/manual` | Analyze verified profile data via multipart form & image upload | Yes (`X-API-Key` or bypass) |
| `POST` | `/profiles/analyze` | Run full analysis on a structured JSON feature payload | Yes (`X-API-Key` or bypass) |
| `POST` | `/profiles/analyze/username` | *(Deprecated 501 Stub)* Directs consumers to `/analyze/manual` | No |
| `GET` | `/profiles/` | List all historical profile analyses (paginated) | Yes (`X-API-Key` or bypass) |
| `GET` | `/profiles/{id}` | Get specific analysis report by ID | Yes (`X-API-Key` or bypass) |
| `GET` | `/review/queue` | List all pending borderline profiles awaiting review | Yes (`X-API-Key` or bypass) |
| `GET` | `/review/stats` | Aggregate summary statistics across risk categories | Yes (`X-API-Key` or bypass) |
| `POST` | `/review/{id}/decide` | Submit human review decision (`APPROVE` or `FLAG`) | Yes (`X-API-Key` or bypass) |

### Sample Analysis Response (`POST /profiles/analyze`):
```json
{
  "id": 1,
  "username": "suspicious_bot_01",
  "account_age_days": 5,
  "followers": 12,
  "following": 3500,
  "posts_per_day": 34.2,
  "face_score": 0.85,
  "deepfake_score": 0.72,
  "behavior_score": 0.94,
  "metadata_score": 0.80,
  "network_score": 0.91,
  "risk_score": 0.93,
  "risk_level": "HIGH",
  "recommended_action": "FLAG",
  "evidence": [
    "High post frequency (34.2/day)",
    "Severe follower-to-following imbalance (12/3500)",
    "XGBoost Behavior Model flagged anomalous pattern",
    "High network hub centrality detected"
  ],
  "model_version": "v1.0-ml-fusion",
  "created_at": "2026-08-22T10:30:00Z"
}
```

---

## 🔐 API Key Management

To generate a new API key for external consumers or client dashboards:

```powershell
python scripts/generate_api_key.py --owner "admin@trueprofile.ai"
```

Pass the generated key in your HTTP request headers:
```http
X-API-Key: tp-your_generated_secret_key_here
```

---

## 🚢 Deployment (Render / PostgreSQL)

1. Connect your GitHub repository to [Render.com](https://render.com).
2. Choose **Blueprint** deployment — Render will automatically read `render.yaml`.
3. Render will provision:
   - A managed **PostgreSQL** database.
   - A Python web service running FastAPI with automatic Alembic migrations applied on boot.
4. Set `ALLOWED_ORIGINS` in your environment variables to lock down CORS for your production frontend domain.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
