"""
COM668 Final Year Project -- AI-Based Intrusion Detection System
Flask REST API  (v2 — restructured with ids package)
Author : Abdulbosit Abdurazzakov | B00979380

Endpoints
---------
GET  /health           -- liveness check; confirms models loaded
GET  /features         -- ordered list of required feature names
POST /predict          -- classify a single network flow
POST /predict/batch    -- classify up to 1000 flows in one request

Dashboard data (consumed by /dashboard)
----------------------------------------
GET  /api/feed         -- last N predictions (detection feed)
GET  /api/stats        -- aggregate counts and confidence averages
GET  /api/shap         -- top-N features by mean |SHAP| / MDI importance

Dashboard UI
-------------
GET  /dashboard        -- single-page live detection dashboard

Usage
-----
  python src/api.py              # dev server
  gunicorn 'src.api:app'         # production
  docker-compose up              # one-command setup
"""

import logging
import os
import sys
import time
from collections import deque
from threading import Lock

import numpy as np
from flask import Flask, jsonify, render_template_string, request

# ── Add project root to sys.path so `ids` package is importable ───────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_ROOT, 'src')
for _p in [_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ids.models.predict import classify_flow, load_models

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ids_api")

# ── Load artefacts at startup (fail fast if missing) ──────────────────────────
logger.info("Loading artefacts and models...")
scaler, le, feature_names, rf_model, iso_model = load_models()
logger.info("  RF trees        : %d", rf_model.n_estimators)
logger.info("  IF contamination: %s", iso_model.contamination)
logger.info("  Features        : %d", len(feature_names))

# ── In-memory detection feed (thread-safe, last 200 predictions) ──────────────
_FEED_MAXLEN = 200
_feed: deque = deque(maxlen=_FEED_MAXLEN)
_feed_lock = Lock()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


# ── JSON error handlers ───────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled server error: %s", e)
    return jsonify({"error": "Internal server error."}), 500


# ── Validation helper ─────────────────────────────────────────────────────────
def _parse_feature_vector(data):
    """
    Parse and validate a feature vector from a request dict.

    Returns (feature_vector, None) on success or (None, (error_dict, status)) on failure.
    Accepts either:
      {"features": [v1, ..., v78]}   — positional array
      {"Destination Port": 21, ...}  — named fields
    """
    if "features" in data:
        fv = data["features"]
        if len(fv) != len(feature_names):
            return None, (
                {"error": "Expected {} features, got {}.".format(len(feature_names), len(fv))},
                400,
            )
    else:
        missing = [f for f in feature_names if f not in data]
        if missing:
            return None, (
                {"error": "Missing {} feature(s).".format(len(missing)), "missing": missing[:10]},
                400,
            )
        fv = [data[f] for f in feature_names]

    try:
        fv = [float(v) for v in fv]
    except (ValueError, TypeError) as exc:
        return None, ({"error": "Non-numeric value in features: {}".format(exc)}, 400)

    if any(np.isnan(v) or np.isinf(v) for v in fv):
        return None, ({"error": "Feature vector contains NaN or Inf values."}, 400)

    return fv, None


# ── Core routes ───────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness check — confirms models are loaded and API is ready."""
    return jsonify({
        "status":   "healthy",
        "models":   ["random_forest", "isolation_forest", "ensemble"],
        "features": len(feature_names),
        "message":  "AI-IDS API is running. POST /predict to classify a network flow.",
    }), 200


@app.route("/features", methods=["GET"])
def get_features():
    """Return the ordered list of all required feature names."""
    return jsonify({
        "count":    len(feature_names),
        "features": feature_names,
        "note":     "All features must be provided in this exact order.",
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Classify a single network flow.

    Accepts JSON with either:
      (a) {"features": [val1, ..., val78]}               — raw array
      (b) {"Destination Port": 21, "Flow Duration": ...} — named fields
    """
    start = time.perf_counter()
    data = request.get_json(force=True, silent=True)

    if not data:
        return jsonify({"error": "No valid JSON body received."}), 400

    fv, err = _parse_feature_vector(data)
    if err:
        return jsonify(err[0]), err[1]

    result = classify_flow(fv, scaler, le, rf_model, iso_model)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        "predict | final=%s | rf=%s (%.3f) | if=%s (%.4f) | %.1fms",
        result["final_decision"],
        result["random_forest"]["prediction"],
        result["random_forest"]["confidence"],
        result["isolation_forest"]["prediction"],
        result["isolation_forest"]["anomaly_score"],
        elapsed_ms,
    )

    # Append to detection feed
    feed_entry = {
        "ts":             time.strftime("%H:%M:%S"),
        "final_decision": result["final_decision"],
        "rf_prediction":  result["random_forest"]["prediction"],
        "rf_confidence":  result["random_forest"]["confidence"],
        "attack_prob":    result["random_forest"]["attack_prob"],
        "if_prediction":  result["isolation_forest"]["prediction"],
        "if_score":       result["isolation_forest"]["anomaly_score"],
        "latency_ms":     elapsed_ms,
    }
    with _feed_lock:
        _feed.appendleft(feed_entry)

    return jsonify({
        "status":            "ok",
        "inference_time_ms": elapsed_ms,
        "result":            result,
    }), 200


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Classify multiple network flows in one request.

    Accepts JSON: {"flows": [[val1..val78], [val1..val78], ...]}
    Maximum 1000 flows per request.
    """
    start = time.perf_counter()
    data = request.get_json(force=True, silent=True)

    if not data or "flows" not in data:
        return jsonify({
            "error": "Expected JSON with key 'flows': list of feature arrays."
        }), 400

    flows = data["flows"]
    if not isinstance(flows, list) or len(flows) == 0:
        return jsonify({"error": "'flows' must be a non-empty list."}), 400
    if len(flows) > 1000:
        return jsonify({"error": "Maximum 1000 flows per batch request."}), 400

    results = []
    for i, flow in enumerate(flows):
        if len(flow) != len(feature_names):
            results.append({
                "flow_index": i,
                "error": "Expected {} features, got {}.".format(len(feature_names), len(flow)),
            })
            continue
        try:
            fv = [float(v) for v in flow]
            results.append({
                "flow_index": i,
                "result": classify_flow(fv, scaler, le, rf_model, iso_model),
            })
        except Exception as exc:
            results.append({"flow_index": i, "error": str(exc)})

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info("predict/batch | n=%d | %.1fms", len(flows), elapsed_ms)

    return jsonify({
        "status":            "ok",
        "count":             len(flows),
        "inference_time_ms": elapsed_ms,
        "results":           results,
    }), 200


# ── Dashboard data endpoints ──────────────────────────────────────────────────

@app.route("/api/feed", methods=["GET"])
def api_feed():
    """Return the last N predictions as JSON for the live detection feed."""
    n = min(int(request.args.get("n", 50)), _FEED_MAXLEN)
    with _feed_lock:
        entries = list(_feed)[:n]
    return jsonify({"count": len(entries), "feed": entries}), 200


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return aggregate detection statistics."""
    with _feed_lock:
        entries = list(_feed)

    total   = len(entries)
    attacks = sum(1 for e in entries if e["final_decision"] == "ATTACK")
    benign  = total - attacks

    avg_confidence = (
        round(sum(e["rf_confidence"] for e in entries) / total, 4)
        if total > 0 else 0.0
    )
    avg_latency = (
        round(sum(e["latency_ms"] for e in entries) / total, 2)
        if total > 0 else 0.0
    )

    return jsonify({
        "total":          total,
        "attacks":        attacks,
        "benign":         benign,
        "attack_pct":     round(attacks / total * 100, 1) if total > 0 else 0.0,
        "avg_confidence": avg_confidence,
        "avg_latency_ms": avg_latency,
    }), 200


@app.route("/api/shap", methods=["GET"])
def api_shap():
    """
    Return top-N feature importances.

    Uses RF Mean Decrease in Impurity (MDI) as a fast baseline.
    MDI correlates strongly with SHAP values for Random Forests and
    requires no additional computation beyond what's already in the model.
    """
    top_n = int(request.args.get("top_n", 15))
    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    result = [
        {
            "feature":    feature_names[i],
            "importance": round(float(importances[i]), 6),
        }
        for i in order
    ]

    return jsonify({
        "method":   "MDI (Mean Decrease in Impurity)",
        "top_n":    top_n,
        "features": result,
    }), 200


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Serve the live detection dashboard."""
    # Dashboard HTML is in src/dashboard.html — served as a static string
    return _DASHBOARD_HTML, 200, {"Content-Type": "text/html"}


def _build_dashboard_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI-IDS Live Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#0f1117;--surface:#1a1d27;--border:#2a2d3e;--attack:#e74c3c;--benign:#2ecc71;--accent:#3498db;--text:#e0e0e0;--muted:#8892a4;--radius:8px;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;min-height:100vh;}
header{background:var(--surface);border-bottom:1px solid var(--border);padding:16px 24px;display:flex;align-items:center;justify-content:space-between;}
header h1{font-size:1.1rem;font-weight:600;}
header h1 span{color:var(--accent);}
.badge{font-size:.72rem;padding:3px 10px;border-radius:20px;background:#1e3a1e;color:var(--benign);border:1px solid var(--benign);}
.badge.offline{background:#3a1e1e;color:var(--attack);border-color:var(--attack);}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;padding:20px 24px 0;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px;}
.card .label{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}
.card .value{font-size:2rem;font-weight:700;}
.card .value.attack{color:var(--attack);}
.card .value.benign{color:var(--benign);}
.card .value.accent{color:var(--accent);}
.card .sub{font-size:.8rem;color:var(--muted);margin-top:4px;}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px 24px 0;}
.chart-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px;}
.chart-card h3{font-size:.85rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;}
.feed-section{padding:16px 24px 24px;}
.feed-section h3{font-size:.85rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;}
table{width:100%;border-collapse:collapse;background:var(--surface);border-radius:var(--radius);overflow:hidden;border:1px solid var(--border);font-size:.82rem;}
th{background:#20233a;padding:10px 14px;text-align:left;color:var(--muted);font-weight:500;font-size:.75rem;text-transform:uppercase;letter-spacing:.8px;}
td{padding:9px 14px;border-top:1px solid var(--border);}
tr:hover td{background:#1e2132;}
.pill{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.75rem;font-weight:600;}
.pill.attack{background:rgba(231,76,60,.18);color:var(--attack);border:1px solid rgba(231,76,60,.4);}
.pill.benign{background:rgba(46,204,113,.14);color:var(--benign);border:1px solid rgba(46,204,113,.35);}
.bar-label{font-size:.78rem;color:var(--muted);margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px;display:inline-block;}
.bar-wrap{background:#2a2d3e;border-radius:4px;height:8px;overflow:hidden;margin-bottom:8px;}
.bar-fill{height:100%;background:var(--accent);border-radius:4px;transition:width .4s;}
#shap-container{max-height:320px;overflow-y:auto;}
.conf-bar{display:flex;align-items:center;gap:8px;}
.conf-bar .bar-wrap{flex:1;}
.conf-val{font-size:.8rem;color:var(--muted);width:40px;text-align:right;}
footer{text-align:center;padding:14px;font-size:.72rem;color:var(--muted);border-top:1px solid var(--border);margin-top:8px;}
</style>
</head>
<body>
<header>
  <h1>AI-IDS &nbsp;<span>Live Detection Dashboard</span></h1>
  <span id="sb" class="badge">&#9679; LIVE</span>
</header>
<div class="grid">
  <div class="card"><div class="label">Total Flows</div><div class="value accent" id="s-tot">&#8212;</div><div class="sub">since startup</div></div>
  <div class="card"><div class="label">Attacks</div><div class="value attack" id="s-atk">&#8212;</div><div class="sub" id="s-pct">&#8212;% of traffic</div></div>
  <div class="card"><div class="label">Benign</div><div class="value benign" id="s-ben">&#8212;</div><div class="sub">normal traffic</div></div>
  <div class="card"><div class="label">Avg RF Confidence</div><div class="value accent" id="s-cf">&#8212;</div><div class="sub" id="s-lat">avg latency &#8212;ms</div></div>
</div>
<div class="charts">
  <div class="chart-card"><h3>Attack vs Benign</h3><canvas id="dc" height="180"></canvas></div>
  <div class="chart-card"><h3>Top Feature Importances (RF MDI)</h3><div id="shap-container"><p style="color:var(--muted);font-size:.82rem;margin-top:8px">Loading...</p></div></div>
</div>
<div class="feed-section">
  <h3>Live Detection Feed <span style="color:#3498db;font-size:.72rem;font-weight:400">last 30 results, refreshes every 3s</span></h3>
  <table><thead><tr><th>Time</th><th>Decision</th><th>RF</th><th>Confidence</th><th>Atk Prob</th><th>IF</th><th>IF Score</th><th>Latency</th></tr></thead>
  <tbody id="fb"></tbody></table>
</div>
<footer>COM668 &middot; Abdulbosit Abdurazzakov &middot; B00979380 &nbsp;|&nbsp; Refreshes every 3s</footer>
<script>
var dc=null;
function initChart(){
  dc=new Chart(document.getElementById("dc").getContext("2d"),{type:"doughnut",data:{labels:["ATTACK","BENIGN"],datasets:[{data:[0,0],backgroundColor:["rgba(231,76,60,.85)","rgba(46,204,113,.85)"],borderColor:["#e74c3c","#2ecc71"],borderWidth:2}]},options:{responsive:true,cutout:"68%",plugins:{legend:{labels:{color:"#e0e0e0",font:{size:12}}}}}});
}
function setText(id,v){document.getElementById(id).textContent=v;}
async function loadStats(){
  try{
    var d=await(await fetch("/api/stats")).json();
    setText("s-tot",d.total.toLocaleString());setText("s-atk",d.attacks.toLocaleString());
    setText("s-ben",d.benign.toLocaleString());setText("s-pct",d.attack_pct+"% of traffic");
    setText("s-cf",(d.avg_confidence*100).toFixed(1)+"%");setText("s-lat","avg latency "+d.avg_latency_ms+"ms");
    if(dc){dc.data.datasets[0].data=[d.attacks,d.benign];dc.update("none");}
    document.getElementById("sb").className="badge";document.getElementById("sb").textContent="&#9679; LIVE";
  }catch(e){document.getElementById("sb").className="badge offline";document.getElementById("sb").textContent="&#9679; OFFLINE";}
}
async function loadShap(){
  try{
    var d=await(await fetch("/api/shap?top_n=15")).json();
    var fs=d.features||[];if(!fs.length)return;
    var mx=fs[0].importance||0.0001;
    document.getElementById("shap-container").innerHTML=fs.map(function(f){
      var pct=Math.round((f.importance/mx)*100);
      return '<div><span class="bar-label" title="'+f.feature+'">'+f.feature+'</span><div class="conf-bar"><div class="bar-wrap"><div class="bar-fill" style="width:'+pct+'%"></div></div><span class="conf-val">'+(f.importance*100).toFixed(3)+'%</span></div></div>';
    }).join("");
  }catch(e){}
}
async function loadFeed(){
  try{
    var d=await(await fetch("/api/feed?n=30")).json();
    var rows=(d.feed||[]).map(function(e){
      var fc=e.final_decision==="ATTACK"?"attack":"benign";
      var rc=e.rf_prediction==="ATTACK"?"attack":"benign";
      var ic=e.if_prediction==="ATTACK"?"attack":"benign";
      return '<tr><td>'+e.ts+'</td><td><span class="pill '+fc+'">'+e.final_decision+'</span></td><td><span class="pill '+rc+'">'+e.rf_prediction+'</span></td><td>'+(e.rf_confidence*100).toFixed(1)+'%</td><td>'+(e.attack_prob*100).toFixed(1)+'%</td><td><span class="pill '+ic+'">'+e.if_prediction+'</span></td><td>'+e.if_score+'</td><td>'+e.latency_ms+'ms</td></tr>';
    }).join("");
    document.getElementById("fb").innerHTML=rows||'<tr><td colspan=8 style="text-align:center;color:#8892a4;padding:28px">No predictions yet &mdash; POST to /predict</td></tr>';
  }catch(e){}
}
async function refresh(){await Promise.all([loadStats(),loadFeed()]);}
initChart();loadShap();refresh();
setInterval(refresh,3000);setInterval(loadShap,30000);
</script>
</body>
</html>"""

_DASHBOARD_HTML = _build_dashboard_html()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  AI-Based IDS -- REST API  v2")
    print("  COM668 | Abdulbosit Abdurazzakov | B00979380")
    print("=" * 58)
    print("  GET  http://localhost:5000/health")
    print("  GET  http://localhost:5000/features")
    print("  POST http://localhost:5000/predict")
    print("  POST http://localhost:5000/predict/batch")
    print("  GET  http://localhost:5000/dashboard")
    print("=" * 58 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
