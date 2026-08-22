# TrueProfile AI 🛡️

**TrueProfile AI** is a multi-signal AI system for detecting fake social media accounts. Instead of relying solely on facial recognition, it fuses image evidence, behavioral analysis, and account metadata to calculate a comprehensive fake-account risk score.

## Architecture & Features

Our architecture implements a vertical slice MVP currently using feature-derived dummy ML models that we will progressively replace with real ML components.

### Core Signals Evaluated:
1. **Face/Image Analysis** *(Pre-integration Phase)*
2. **Behavioral Analysis** (Posts per day, engagement, follow patterns)
3. **Account Metadata** (Account age, completeness)
4. **Network Analysis** (Clustering and follower/following anomalies)

### Project Structure (Backend Scaffolded)
```text
trueprofile-ai/
├── backend/
│   ├── main.py                # FastAPI entry point
│   ├── database.py            # SQLAlchemy setup
│   ├── routers/
│   │   └── profiles.py        # API Endpoints
│   ├── services/
│   │   └── profile_analyzer.py# ML Orchestration Layer
│   ├── models/                # SQLAlchemy Models
│   ├── schemas/               # Pydantic validation schemas
│   ├── ml/                    # AI Components (currently using feature-derived dummy models)
│   │   ├── base.py
│   │   ├── face_analysis.py
│   │   ├── deepfake_detector.py
│   │   ├── behavior_model.py
│   │   ├── metadata_model.py
│   │   ├── network_analysis.py
│   │   └── fusion_engine.py   # Weighted baseline model
│   └── tests/
├── data/
│   └── mock_data_generator.py # Generates realistic correlated profiles
└── frontend/                  # React Dashboard (To be integrated)
```

## Setup Instructions (Local Development)

### 1. Backend Setup
1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # Linux/Mac: source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Generate mock data and initialize the SQLite database:
   ```bash
   python data/mock_data_generator.py
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn backend.main:app --reload
   ```
5. View API docs at: `http://localhost:8000/docs`

## Progress Tracker
- [x] Initial Project Scaffold
- [x] Database & Models Setup
- [x] Mock Data Generator
- [x] FastAPI Endpoints & Services
- [x] Dummy ML Pipeline
- [x] Weighted Fusion Engine Baseline
- [ ] React Dashboard (Deferred)
- [ ] Tests
- [ ] Real ML Model Integration (ArcFace, Deepfake, XGBoost)
- [ ] PostgreSQL Migration
