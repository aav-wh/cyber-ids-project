# AI-Based Intrusion Detection System
**COM668 Final Year Project** | Abdulbosit Abdurazzakov | B00979380

An end-to-end machine learning IDS trained on the CICIDS2017 dataset, combining a supervised Random Forest and a semi-supervised Isolation Forest into a hybrid OR-rule ensemble. Includes a REST API, live dashboard, Docker setup, and CI pipeline.

---

## Project structure

```
cyber-ids-project/
├── src/
│   ├── ids/                        # Python package
│   │   ├── data/loader.py          # CSV loading & cleaning
│   │   ├── features/preprocessing.py  # scaling, encoding, oversampling
│   │   ├── models/train.py         # RF + IF training helpers
│   │   ├── models/predict.py       # classify_flow() inference
│   │   ├── models/evaluate.py      # metrics, confusion matrix, ROC
│   │   └── explainability/shap_analysis.py  # SHAP feature importance
│   ├── api.py                      # Flask REST API + live dashboard
│   └── demo_request.py             # demo script
├── notebooks/                      # Jupyter notebooks (01–09)
├── tests/                          # pytest test suite (33 tests)
├── data/processed/                 # fitted scaler, encoder, feature names
├── models/                         # trained RF + IF .pkl files
├── results/                        # plots and evaluation outputs
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml        # lint → test → Docker build
```

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API + dashboard
python src/api.py
# → http://localhost:5000/dashboard
# → http://localhost:5000/health

# Run tests
pytest tests/ -v

# One-command Docker setup
docker-compose up --build
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/features` | List all 78 required feature names |
| POST | `/predict` | Classify a single network flow |
| POST | `/predict/batch` | Classify up to 1000 flows |
| GET | `/dashboard` | Live detection dashboard |
| GET | `/api/feed` | Last N predictions (JSON) |
| GET | `/api/stats` | Aggregate counts + confidence |
| GET | `/api/shap` | Top feature importances |

## Models

| Model | Type | Training strategy |
|-------|------|-------------------|
| Random Forest | Supervised | 150 trees, `class_weight='balanced'`, 5-fold CV |
| Isolation Forest | Semi-supervised | Trained on BENIGN-only traffic |
| Ensemble | OR rule | ATTACK if either model flags (maximises recall) |

## Dataset

CICIDS2017 — Canadian Institute for Cybersecurity. 8 CSV files covering Monday–Friday traffic with 15 attack types including DDoS, PortScan, Web Attacks, and Infiltration.

> The raw CSV files are not committed to this repository due to size (~2 GB). Download from [the UNB website](https://www.unb.ca/cic/datasets/ids-2017.html) and place in `data/`.
