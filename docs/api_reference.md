# AI-IDS REST API Reference

COM668 Final Year Project | Abdulbosit Abdurazzakov | B00979380

Base URL: `http://localhost:5000`

---

## GET /health

Liveness check. Returns model metadata confirming the API is ready.

**Response 200**
```json
{
  "status": "healthy",
  "models": ["random_forest", "isolation_forest", "ensemble"],
  "features": 78,
  "message": "AI-IDS API is running. POST /predict to classify a network flow."
}
```

---

## GET /features

Returns the ordered list of all 78 required feature names.

**Response 200**
```json
{
  "count": 78,
  "features": ["Destination Port", "Flow Duration", ...],
  "note": "All features must be provided in this exact order."
}
```

---

## POST /predict

Classify a single network flow.

**Request body** (named fields):
```json
{
  "Destination Port": 80,
  "Flow Duration": 500000,
  "Total Fwd Packets": 15,
  ...
}
```

**Request body** (positional array):
```json
{
  "features": [80, 500000, 15, ...]
}
```

**Response 200**
```json
{
  "status": "ok",
  "inference_time_ms": 1.23,
  "result": {
    "random_forest": {
      "prediction": "ATTACK",
      "confidence": 0.9342,
      "attack_prob": 0.9342,
      "benign_prob": 0.0658
    },
    "isolation_forest": {
      "prediction": "ATTACK",
      "anomaly_score": -0.0521,
      "note": "More negative score = more anomalous = more likely ATTACK"
    },
    "ensemble": {
      "prediction": "ATTACK",
      "rule": "OR — ATTACK if either model flags (maximises recall)"
    },
    "final_decision": "ATTACK"
  }
}
```

**Error responses**

| Code | Reason |
|------|--------|
| 400  | Missing features, wrong count, non-numeric values, NaN/Inf |
| 405  | GET /predict (method not allowed) |

---

## POST /predict/batch

Classify up to 1000 flows in one request.

**Request body**
```json
{
  "flows": [
    [80, 500000, 15, ...],
    [443, 1000000, 8, ...]
  ]
}
```

**Response 200**
```json
{
  "status": "ok",
  "count": 2,
  "inference_time_ms": 3.45,
  "results": [
    {"flow_index": 0, "result": {...}},
    {"flow_index": 1, "result": {...}}
  ]
}
```

---

## GET /api/feed

Returns the last N predictions for the live dashboard feed.

**Query params:** `n` (default: 50, max: 200)

---

## GET /api/stats

Returns aggregate detection statistics since server startup.

**Response 200**
```json
{
  "total": 1000,
  "attacks": 250,
  "benign": 750,
  "attack_pct": 25.0,
  "avg_confidence": 0.9123,
  "avg_latency_ms": 1.45
}
```

---

## GET /api/shap

Returns top-N feature importances (RF MDI).

**Query params:** `top_n` (default: 15)

---

## GET /dashboard

Serves the live detection dashboard (HTML).
