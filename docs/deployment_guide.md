# Deployment Guide

COM668 Final Year Project | Abdulbosit Abdurazzakov | B00979380

## Prerequisites

- Python 3.11+
- Docker + Docker Compose (for containerised deployment)
- CICIDS2017 dataset CSVs in `data/`
- Trained model artefacts in `models/` and `data/processed/`

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (first time only)
python src/train_pipeline.py

# Start the API
python src/api.py

# Dashboard: http://localhost:5000/dashboard
```

## Docker Deployment

```bash
# Build and start
docker-compose up --build

# Rebuild without cache
docker-compose build --no-cache && docker-compose up

# Stop
docker-compose down
```

The `docker-compose.yml` mounts `data/`, `models/`, and `results/`
as volumes so trained artefacts persist between container restarts.

## Production (Gunicorn)

```bash
gunicorn 'src.api:app' \
  --workers 4 \
  --bind 0.0.0.0:5000 \
  --timeout 30 \
  --access-logfile logs/access.log
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IDS_DATA_DIR` | `data/` | Raw CSV directory |
| `IDS_MODELS_DIR` | `models/` | Model .pkl directory |
| `IDS_LOG_LEVEL` | `INFO` | Logging level |
| `IDS_API_PORT` | `5000` | API port |

## Running Tests

```bash
# All tests (stubs injected automatically in CI)
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires trained artefacts)
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=src/ids --cov-report=html
```

## Health Monitoring

```bash
# Check API health
curl http://localhost:5000/health

# Expected response
{"status": "healthy", "features": 78, ...}
```

## Troubleshooting

**`FileNotFoundError: Missing artefact 'random_forest'`**
Run `python src/train_pipeline.py` to generate model files.

**`ModuleNotFoundError: No module named 'ids'`**
Run from the project root, or ensure `src/` is on `PYTHONPATH`.

**Port 5000 already in use**
Change `API_PORT` in `config/api_config.yaml` or set `IDS_API_PORT=5001`.
