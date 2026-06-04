# Non-Functional Requirements
## COM668 Final Year Project — AI-Based Intrusion Detection System
**Student:** Abdulbosit Abdurazzakov | **ID:** B00979380

---

Non-functional requirements (NFRs) define the quality constraints the system must satisfy
independently of its specific behaviours. The AT2 feedback identified NFRs as a missing component
of the Software Definition section. The following NFRs are organised by the ISO/IEC 25010 quality
model categories relevant to this project.

---

## NFR-1: Performance

| ID | Requirement | Metric | Target | Verified |
|----|-------------|--------|--------|---------|
| NFR-1.1 | Single-flow inference latency | 95th-percentile response time for `POST /predict` | < 100 ms | ✓ Achieved < 5 ms (see `/health` latency logs) |
| NFR-1.2 | Batch throughput | Flows classified per second | ≥ 500 flows/s | ✓ 1,000-flow batch completes in < 2 s |
| NFR-1.3 | Model loading time | Time from process start to first request served | < 30 s | ✓ ~2 s on standard hardware |
| NFR-1.4 | Memory footprint | Peak RAM during inference | < 2 GB | ✓ RF (~500 MB) + IF (~150 MB) + overhead |

---

## NFR-2: Reproducibility

| ID | Requirement | Implementation |
|----|-------------|----------------|
| NFR-2.1 | All random operations must use a fixed seed | `random_state=42` in `RandomForestClassifier`, `IsolationForest`, `StratifiedKFold`, `StratifiedShuffleSplit`, and `RandomOverSampler` throughout |
| NFR-2.2 | Exact dependency versions must be pinned | `requirements.txt` pins all library versions (e.g. `scikit-learn==1.7.2`) |
| NFR-2.3 | Trained model artefacts must be serialised and version-controlled | `models/random_forest.pkl`, `models/isolation_forest.pkl`, `data/processed/*.pkl` committed to repository |
| NFR-2.4 | Any researcher must be able to reproduce results from the repository | `README.md` provides step-by-step reproduction instructions; `docker-compose up` provides a one-command reproducible environment |

---

## NFR-3: Execution Environment

| ID | Requirement | Specification |
|----|-------------|---------------|
| NFR-3.1 | Python version | Python 3.11.x (tested on 3.11.9) |
| NFR-3.2 | Operating system | Linux (Ubuntu 22.04 in Docker); Windows 10/11 via native Python |
| NFR-3.3 | Minimum hardware | 4-core CPU, 8 GB RAM, 20 GB free disk (for CICIDS2017 CSVs) |
| NFR-3.4 | Container runtime | Docker 24+ and Docker Compose v2+ for containerised deployment |
| NFR-3.5 | No internet required at inference | All models and preprocessing artefacts are bundled locally; no external API calls |

---

## NFR-4: Availability and Reliability

| ID | Requirement | Implementation |
|----|-------------|----------------|
| NFR-4.1 | API must fail fast on startup if any artefact is missing | `load_models()` in `ids/models/predict.py` raises `FileNotFoundError` with a clear message before the server accepts requests |
| NFR-4.2 | Invalid inputs must return structured error responses, not crash | All endpoints return JSON `{"error": "..."}` with appropriate HTTP status codes (400, 404, 405, 500) |
| NFR-4.3 | API must expose a health check endpoint | `GET /health` returns `{"status": "healthy"}` with model metadata |
| NFR-4.4 | Container must restart on failure | `docker-compose.yml` sets `restart: unless-stopped` |

---

## NFR-5: Security

| ID | Requirement | Implementation |
|----|-------------|----------------|
| NFR-5.1 | No re-fitting of models at inference time | `scaler.transform()` is called (never `fit_transform()`) in `classify_flow()`; enforced by code review |
| NFR-5.2 | Input validation before model inference | Feature vectors validated for correct length, numeric type, and absence of NaN/Inf before classification |
| NFR-5.3 | Container runs as non-root user | `Dockerfile` creates `ids` user (UID 1001) and switches to it before starting gunicorn |
| NFR-5.4 | Model files mounted read-only in Docker | `docker-compose.yml` mounts `models/` and `data/processed/` with `:ro` flag |

---

## NFR-6: Maintainability and Testability

| ID | Requirement | Metric | Target | Verified |
|----|-------------|--------|--------|---------|
| NFR-6.1 | All API endpoints must have automated tests | Pytest pass rate | 100% | ✓ 33/33 tests pass |
| NFR-6.2 | Code must pass static analysis | Flake8 exit code | 0 (no errors) | ✓ CI lint step green |
| NFR-6.3 | Inference logic must be isolated from API layer | `classify_flow()` lives in `ids/models/predict.py`, imported by `src/api.py` | Verified by package structure |
| NFR-6.4 | Tests must run without trained models (CI environment) | Stub injection via `unittest.mock` | ✓ CI tests use lightweight stubs |

---

## NFR-7: Scalability

| ID | Requirement | Implementation |
|----|-------------|----------------|
| NFR-7.1 | API must support batch classification | `POST /predict/batch` accepts up to 1,000 flows per request |
| NFR-7.2 | Production server must support concurrent requests | Gunicorn configured with 2 workers × 2 threads (`docker-compose.yml`) |
| NFR-7.3 | Detection feed must be bounded in memory | In-memory `deque(maxlen=200)` prevents unbounded growth |

---

*These NFRs were derived from the IEEE 830 Software Requirements Specification standard and the
ISO/IEC 25010 Systems and Software Quality Requirements and Evaluation (SQuaRE) model.*
