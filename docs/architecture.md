# AI-IDS System Architecture

COM668 Final Year Project | Abdulbosit Abdurazzakov | B00979380

## Overview

The system follows a layered architecture: raw data flows through preprocessing into a dual-model inference engine exposed via a REST API, with a real-time dashboard for monitoring.

```
┌─────────────────────────────────────────────────────────┐
│                  CICIDS2017 Raw CSVs                    │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              Data Pipeline (ids.data)                   │
│  loader.py → stats.py → validator.py → splitter.py     │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│           Feature Engineering (ids.features)            │
│  preprocessing.py → engineer.py → selector.py          │
│  StandardScaler · LabelEncoder · feature_names.pkl     │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
┌──────────────▼──────────┐  ┌────────────▼───────────────┐
│   Random Forest         │  │   Isolation Forest          │
│   (Supervised)          │  │   (Semi-supervised)         │
│   ids.models.train.py   │  │   Trained on BENIGN only    │
└──────────────┬──────────┘  └────────────┬───────────────┘
               │                          │
┌──────────────▼──────────────────────────▼───────────────┐
│              Ensemble (OR Rule) ids.models.ensemble     │
│    ATTACK if either model flags — maximises recall      │
└─────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              Flask REST API (src/api.py)                │
│  /predict · /predict/batch · /health · /features       │
│  /api/feed · /api/stats · /api/shap · /dashboard       │
└─────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│           Live Detection Dashboard                       │
│  Chart.js · Auto-refresh 3s · Feature importance        │
└─────────────────────────────────────────────────────────┘
```

## Key Design Decisions

**Temporal split** — training on Mon–Wed, testing on Thu–Fri mirrors realistic deployment and avoids data leakage.

**OR ensemble rule** — missing an attack (false negative) is more costly than a false alarm (false positive) in IDS contexts. The OR rule maximises recall at the cost of some precision.

**Semi-supervised IF** — training on BENIGN-only means the system can flag novel attack types it has never seen labelled, unlike pure supervised approaches.

**Stateless API** — each request is self-contained; the API scales horizontally behind a load balancer without shared state.

## Module Map

| Module | Responsibility |
|--------|---------------|
| `ids.data.loader` | CSV loading and cleaning |
| `ids.data.pipeline` | End-to-end pipeline orchestration |
| `ids.features.preprocessing` | Scaling, encoding, feature extraction |
| `ids.features.drift` | Covariate drift detection |
| `ids.models.train` | RF + IF training |
| `ids.models.predict` | Inference (classify_flow) |
| `ids.models.ensemble` | OR/AND/voting combination |
| `ids.models.threshold` | PR-curve threshold optimisation |
| `ids.evaluation.metrics` | Full metrics suite |
| `ids.detection.alerts` | Alert generation and queuing |
| `ids.detection.rules` | Signature rule engine |
| `ids.monitoring.drift_monitor` | Production drift detection |
| `src.api` | Flask REST API |
