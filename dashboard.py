"""
COM668 - AI-Based Intrusion Detection System
Darktrace-Inspired Professional Dashboard  ·  Excellent Edition
Author: Abdulbosit Abdurazzakov | B00979380
"""

import os, sys, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

ROOT    = os.path.dirname(os.path.abspath(__file__))
SRC     = os.path.join(ROOT, "src")
RESULTS = os.path.join(ROOT, "results")
MODELS  = os.path.join(ROOT, "models")
PROC    = os.path.join(ROOT, "data", "processed")

for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

st.set_page_config(
    page_title="AI-IDS · Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS  ·  Darktrace Excellent Edition
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif !important; }

/* ── Global background with grid ── */
.stApp { background:#03030a; }
.main  {
    background-image:
        linear-gradient(rgba(0,212,255,0.022) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.022) 1px, transparent 1px);
    background-size:52px 52px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div {
    background:linear-gradient(180deg,#010108 0%,#030310 100%) !important;
    border-right:1px solid rgba(0,212,255,0.08) !important;
}
.block-container { padding:1.5rem 2rem 4rem !important; max-width:1440px; }
#MainMenu, footer, header { visibility:hidden; }
.stDeployButton,[data-testid="stSidebarNav"] { display:none; }

/* ── Sidebar logo ── */
.dt-logo { text-align:center; padding:22px 12px 18px; border-bottom:1px solid rgba(0,212,255,0.07); margin-bottom:14px; }
.dt-logo-outer {
    width:64px; height:64px; border-radius:50%;
    background:radial-gradient(circle,rgba(0,212,255,0.18) 0%,rgba(0,212,255,0.04) 55%,transparent 100%);
    border:1px solid rgba(0,212,255,0.22);
    box-shadow:0 0 28px rgba(0,212,255,0.18),0 0 60px rgba(0,212,255,0.06);
    display:flex; align-items:center; justify-content:center;
    margin:0 auto 14px; font-size:28px;
    animation:logo-breathe 4s ease-in-out infinite;
}
@keyframes logo-breathe {
    0%,100%{ box-shadow:0 0 28px rgba(0,212,255,0.18),0 0 60px rgba(0,212,255,0.06); }
    50%    { box-shadow:0 0 40px rgba(0,212,255,0.30),0 0 90px rgba(0,212,255,0.12); }
}
.dt-logo h2 { color:#00d4ff; font-size:14px; font-weight:700; margin:0; letter-spacing:5px; text-transform:uppercase; text-shadow:0 0 20px rgba(0,212,255,0.5); }
.dt-logo p  { color:#1e2348; font-size:10px; margin:4px 0 0; letter-spacing:1.5px; text-transform:uppercase; }

/* ── Nav ── */
div[data-testid="stRadio"] > label { display:none; }
div[data-testid="stRadio"] > div  { gap:1px !important; }
div[data-testid="stRadio"] > div > label {
    background:transparent !important; border:1px solid transparent !important;
    border-radius:4px !important; padding:9px 14px !important; cursor:pointer !important;
    color:#252a52 !important; font-size:10px !important; font-weight:700 !important;
    width:100% !important; letter-spacing:1.5px; text-transform:uppercase;
    font-family:'JetBrains Mono',monospace !important; transition:all .2s;
}
div[data-testid="stRadio"] > div > label:hover {
    background:rgba(0,212,255,0.05) !important;
    border-color:rgba(0,212,255,0.12) !important; color:#00d4ff !important;
    box-shadow:inset 3px 0 0 rgba(0,212,255,0.4);
}
div[data-testid="stRadio"] > div > label[data-baseweb="radio"] > div:first-child { display:none !important; }

/* ── KPI cards ── */
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:18px; }
.kpi-card {
    background:#05050d; border:1px solid rgba(255,255,255,0.04);
    border-radius:8px; padding:18px 20px; position:relative; overflow:hidden;
    transition:transform .2s;
}
.kpi-card:hover { transform:translateY(-2px); }
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    animation:border-glow 3s ease-in-out infinite;
}
.kpi-cyan::before   { background:linear-gradient(90deg,transparent,#00d4ff,transparent); }
.kpi-red::before    { background:linear-gradient(90deg,transparent,#ff2d55,transparent); animation-delay:.8s; }
.kpi-green::before  { background:linear-gradient(90deg,transparent,#00ff88,transparent); animation-delay:1.6s; }
.kpi-purple::before { background:linear-gradient(90deg,transparent,#bf5fff,transparent); animation-delay:2.4s; }
@keyframes border-glow {
    0%,100%{ opacity:.5; } 50%{ opacity:1; }
}
/* Corner decorators */
.kpi-card::after {
    content:''; position:absolute; bottom:8px; right:8px;
    width:14px; height:14px;
    border-right:1px solid rgba(255,255,255,0.05);
    border-bottom:1px solid rgba(255,255,255,0.05);
}
.kpi-label { font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:2px; color:#252a52; margin-bottom:10px; font-family:'JetBrains Mono',monospace; }
.kpi-value { font-size:30px; font-weight:800; line-height:1; margin-bottom:4px; font-family:'JetBrains Mono',monospace; }
.kpi-cyan   .kpi-value { color:#00d4ff; text-shadow:0 0 24px rgba(0,212,255,0.45); }
.kpi-red    .kpi-value { color:#ff2d55; text-shadow:0 0 24px rgba(255,45,85,0.45); }
.kpi-green  .kpi-value { color:#00ff88; text-shadow:0 0 24px rgba(0,255,136,0.45); }
.kpi-purple .kpi-value { color:#bf5fff; text-shadow:0 0 24px rgba(191,95,255,0.45); }
.kpi-sub { font-size:10px; color:#252a52; font-family:'JetBrains Mono',monospace; }

/* ── Threat level banner ── */
.threat-banner {
    display:flex; align-items:center; gap:16px;
    background:#06060f; border:1px solid; border-radius:7px;
    padding:12px 18px; margin-bottom:16px; position:relative; overflow:hidden;
}
.threat-banner::before {
    content:''; position:absolute; top:0; left:0; bottom:0; width:3px;
}
.tb-elevated { border-color:rgba(255,187,0,0.2); }
.tb-elevated::before { background:#ffbb00; box-shadow:0 0 12px rgba(255,187,0,0.6); }
.tb-critical  { border-color:rgba(255,45,85,0.3); }
.tb-critical::before  { background:#ff2d55; box-shadow:0 0 12px rgba(255,45,85,0.7); }
.tb-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.tb-elevated .tb-dot { background:#ffbb00; box-shadow:0 0 8px rgba(255,187,0,0.8); animation:blink 2s infinite; }
.tb-label { font-size:9px; color:#252a52; font-family:'JetBrains Mono',monospace; text-transform:uppercase; letter-spacing:2px; white-space:nowrap; }
.tb-level { font-size:13px; font-weight:800; font-family:'JetBrains Mono',monospace; white-space:nowrap; }
.tb-elevated .tb-level { color:#ffbb00; text-shadow:0 0 12px rgba(255,187,0,0.4); }
.tb-bar-wrap { flex:1; height:4px; background:rgba(255,255,255,0.04); border-radius:2px; overflow:hidden; }
.tb-bar-fill { height:100%; border-radius:2px; }
.tb-elevated .tb-bar-fill { background:linear-gradient(90deg,#00d4ff,#ffbb00); }
.tb-score { font-size:11px; font-weight:700; font-family:'JetBrains Mono',monospace; color:#ffbb00; white-space:nowrap; }
.tb-time  { font-size:9px; color:#252a52; font-family:'JetBrains Mono',monospace; white-space:nowrap; }
.tb-live  { font-size:9px; font-weight:700; color:#ff2d55; font-family:'JetBrains Mono',monospace; animation:blink 1.2s infinite; }

/* ── Section header ── */
.sec-header { display:flex; align-items:center; gap:10px; margin:20px 0 12px; }
.sec-header h3 { font-size:9px; font-weight:700; color:#00d4ff; margin:0; white-space:nowrap; text-transform:uppercase; letter-spacing:2.5px; font-family:'JetBrains Mono',monospace; text-shadow:0 0 10px rgba(0,212,255,0.3); }
.sec-line { flex:1; height:1px; background:linear-gradient(90deg,rgba(0,212,255,0.2),transparent); }

/* ── Threat feed ── */
.threat-item {
    display:flex; gap:10px; align-items:flex-start;
    background:#050510; border:1px solid rgba(0,212,255,0.05);
    border-radius:5px; padding:9px 12px; margin-bottom:5px;
    transition:border-color .2s;
}
.threat-item:hover { border-color:rgba(0,212,255,0.12); }
.threat-dot { width:7px; height:7px; border-radius:50%; margin-top:5px; flex-shrink:0; }
.dot-red   { background:#ff2d55; box-shadow:0 0 8px rgba(255,45,85,0.8); animation:blink 1.5s infinite; }
.dot-amber { background:#ffbb00; box-shadow:0 0 6px rgba(255,187,0,0.7); }
.dot-cyan  { background:#00d4ff; box-shadow:0 0 6px rgba(0,212,255,0.6); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }
.threat-time  { font-size:9px; color:#252a52; font-family:'JetBrains Mono',monospace; white-space:nowrap; padding-top:2px; }
.threat-host  { font-size:12px; font-weight:700; color:#c8caf4; font-family:'JetBrains Mono',monospace; }
.threat-desc  { font-size:10px; color:#363b68; margin-top:1px; }
.threat-score { font-size:11px; font-weight:700; font-family:'JetBrains Mono',monospace; white-space:nowrap; }
.sc-red  { color:#ff2d55; }
.sc-amber{ color:#ffbb00; }
.sc-cyan { color:#00d4ff; }

/* ── Finding cards ── */
.finding { background:#05050d; border-left:2px solid; border-radius:0 6px 6px 0; padding:12px 16px; margin-bottom:8px; }
.f-cyan  { border-color:#00d4ff; }
.f-red   { border-color:#ff2d55; }
.f-green { border-color:#00ff88; }
.f-amber { border-color:#ffbb00; }
.finding-title { font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:5px; font-family:'JetBrains Mono',monospace; }
.f-cyan  .finding-title { color:#00d4ff; }
.f-red   .finding-title { color:#ff2d55; }
.f-green .finding-title { color:#00ff88; }
.f-amber .finding-title { color:#ffbb00; }
.finding-body { font-size:12px; color:#363b68; line-height:1.65; }

/* ── Pred box ── */
.pred-box { border-radius:6px; padding:18px; text-align:center; border:1px solid; margin-bottom:10px; }
.pred-attack {
    background:rgba(255,45,85,0.04); border-color:rgba(255,45,85,0.2);
    animation:attack-anim 2s ease-in-out infinite;
}
.pred-benign { background:rgba(0,255,136,0.03); border-color:rgba(0,255,136,0.18); }
@keyframes attack-anim {
    0%,100%{ box-shadow:0 0 20px rgba(255,45,85,0.05); border-color:rgba(255,45,85,0.2); }
    50%    { box-shadow:0 0 40px rgba(255,45,85,0.12); border-color:rgba(255,45,85,0.4); }
}
.pred-label    { font-size:8px; color:#252a52; font-weight:700; text-transform:uppercase; letter-spacing:2.5px; margin-bottom:8px; font-family:'JetBrains Mono',monospace; }
.pred-decision { font-size:20px; font-weight:800; font-family:'JetBrains Mono',monospace; }
.pred-attack .pred-decision { color:#ff2d55; text-shadow:0 0 20px rgba(255,45,85,0.5); }
.pred-benign .pred-decision { color:#00ff88; text-shadow:0 0 20px rgba(0,255,136,0.5); }
.pred-sub { font-size:10px; color:#252a52; margin-top:5px; font-family:'JetBrains Mono',monospace; }

/* ── Page title ── */
.page-title { margin-bottom:16px; padding-bottom:14px; border-bottom:1px solid rgba(0,212,255,0.07); }
.page-title h1 { font-size:21px; font-weight:800; color:#e8eaf6; line-height:1.2; margin:0; }
.page-title p  { color:#252a52; font-size:11px; margin:5px 0 0; font-family:'JetBrains Mono',monospace; letter-spacing:.5px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap:2px; background:transparent; border-bottom:1px solid rgba(0,212,255,0.07); }
.stTabs [data-baseweb="tab"] {
    background:transparent; border:1px solid transparent; border-bottom:none;
    border-radius:4px 4px 0 0; color:#252a52; font-size:10px; font-weight:700;
    padding:7px 16px; text-transform:uppercase; letter-spacing:1.5px;
    font-family:'JetBrains Mono',monospace !important;
}
.stTabs [aria-selected="true"] {
    background:rgba(0,212,255,0.04) !important;
    border-color:rgba(0,212,255,0.12) !important;
    color:#00d4ff !important; text-shadow:0 0 8px rgba(0,212,255,0.3);
}

/* ── Buttons ── */
.stButton > button {
    background:transparent !important; color:#00d4ff !important;
    border:1px solid rgba(0,212,255,0.22) !important; border-radius:4px !important;
    padding:8px 24px !important; font-weight:700 !important; font-size:11px !important;
    letter-spacing:1.5px !important; text-transform:uppercase !important;
    font-family:'JetBrains Mono',monospace !important;
    position:relative; overflow:hidden;
}
.stButton > button:hover {
    background:rgba(0,212,255,0.06) !important;
    box-shadow:0 0 20px rgba(0,212,255,0.18),inset 0 0 20px rgba(0,212,255,0.04) !important;
    border-color:rgba(0,212,255,0.4) !important;
}

/* ── Metrics ── */
div[data-testid="stMetric"] { background:#05050d; border:1px solid rgba(0,212,255,0.07); border-radius:6px; padding:12px 16px; }
div[data-testid="stMetric"] label { color:#252a52 !important; font-size:9px !important; text-transform:uppercase; letter-spacing:1.5px; font-family:'JetBrains Mono',monospace !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color:#00d4ff !important; font-size:22px !important; font-weight:700 !important; font-family:'JetBrains Mono',monospace !important; text-shadow:0 0 14px rgba(0,212,255,0.3); }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border:1px solid rgba(0,212,255,0.07) !important; border-radius:6px !important; }
.img-caption { text-align:center; color:#252a52; font-size:10px; margin-top:6px; font-family:'JetBrains Mono',monospace; }
hr { border-color:rgba(0,212,255,0.05) !important; margin:14px 0 !important; }
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:#03030a; }
::-webkit-scrollbar-thumb { background:rgba(0,212,255,0.18); border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly config ──────────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(3,3,10,0.97)",
    font=dict(family="'JetBrains Mono','Courier New',monospace", color="#363b68", size=10),
    margin=dict(l=16, r=16, t=36, b=16),
)
DT_AXIS = dict(
    gridcolor="rgba(0,212,255,0.045)",
    linecolor="rgba(0,212,255,0.07)",
    tickcolor="#0d1030",
    tickfont=dict(color="#363b68", size=9),
    zerolinecolor="rgba(0,212,255,0.05)",
)

def _lay(**kw):
    d = dict(PLOTLY_BASE)
    d.update(kw)
    return d


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising threat intelligence engines...")
def load_all():
    try:
        import joblib
        from ids.models.predict import classify_flow
        sc  = joblib.load(os.path.join(PROC,   "scaler.pkl"))
        le  = joblib.load(os.path.join(PROC,   "label_encoder.pkl"))
        fn  = joblib.load(os.path.join(PROC,   "feature_names.pkl"))
        rf  = joblib.load(os.path.join(MODELS, "random_forest.pkl"))
        iso = joblib.load(os.path.join(MODELS, "isolation_forest.pkl"))
        return sc, le, fn, rf, iso, classify_flow, True
    except Exception:
        return None, None, [], None, None, None, False


scaler, le, feature_names, rf_model, iso_model, classify_flow, models_ok = load_all()


@st.cache_data
def load_csv(name):
    p = os.path.join(RESULTS, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def show_img(name, caption=None):
    p = os.path.join(RESULTS, name)
    if os.path.exists(p):
        st.image(Image.open(p), use_container_width=True)
        if caption:
            st.markdown(f'<p class="img-caption">{caption}</p>', unsafe_allow_html=True)
    else:
        st.caption(f"[ {name} ]")


# ── Network topology ───────────────────────────────────────────────────────────
def make_network_graph():
    rng = np.random.default_rng(7)
    nx_, ny_, labels_, threats_, sizes_ = [], [], [], [], []

    nx_.append(0); ny_.append(0); labels_.append("GATEWAY"); threats_.append(0.08); sizes_.append(22)
    for i in range(4):
        a = 2*np.pi*i/4
        nx_.append(2.3*np.cos(a)); ny_.append(2.3*np.sin(a))
        labels_.append(f"SW-{i+1:02d}"); threats_.append(rng.uniform(0.05,0.18)); sizes_.append(15)

    attacked_idx = rng.choice(range(5,23), size=5, replace=False)
    for i in range(18):
        a = 2*np.pi*i/18 + rng.uniform(-0.22,0.22)
        r = rng.uniform(4.5,6.8)
        nx_.append(r*np.cos(a)); ny_.append(r*np.sin(a))
        labels_.append(f"HOST-{i+1:02d}")
        t = rng.uniform(0.78,0.97) if (i+5) in attacked_idx else rng.uniform(0.02,0.22)
        threats_.append(t); sizes_.append(9)

    edges = [(0,i) for i in range(1,5)] + [(i%4+1, h) for i,h in enumerate(range(5,23))]
    ex, ey = [], []
    for a,b in edges:
        ex += [nx_[a], nx_[b], None]; ey += [ny_[a], ny_[b], None]

    colors = []
    borders= []
    for t in threats_:
        if   t > 0.7: colors.append(f"rgba(255,45,85,{min(0.45+t*0.55,1):.2f})");  borders.append("#ff2d55")
        elif t > 0.35: colors.append("rgba(255,187,0,0.75)");                         borders.append("#ffbb00")
        else:          colors.append(f"rgba(0,212,255,{0.2+t:.2f})");                 borders.append("#00d4ff")

    fig = go.Figure()

    # Zone rings
    theta = np.linspace(0, 2*np.pi, 120)
    for r, col, lbl, pos in [
        (3.1,  "rgba(0,212,255,0.06)",  "INTERNAL",  3.3),
        (4.2,  "rgba(255,187,0,0.05)",  "DMZ",        4.4),
        (7.2,  "rgba(255,45,85,0.04)",  "EXTERNAL",   7.4),
    ]:
        fig.add_trace(go.Scatter(
            x=r*np.cos(theta), y=r*np.sin(theta), mode="lines",
            line=dict(color=col, width=1, dash="dot"),
            showlegend=False, hoverinfo="none"))
        fig.add_annotation(x=0, y=r+0.15, text=lbl, showarrow=False,
            font=dict(size=8, color=col.replace("0.0","0.3").replace("0.05","0.3").replace("0.06","0.3"),
                      family="JetBrains Mono"),
            xanchor="center")

    # Threat halos on compromised nodes
    for i, (x,y,t) in enumerate(zip(nx_, ny_, threats_)):
        if t > 0.7:
            fig.add_shape(type="circle",
                x0=x-0.55, y0=y-0.55, x1=x+0.55, y1=y+0.55,
                line=dict(color="rgba(255,45,85,0.35)", width=1),
                fillcolor="rgba(255,45,85,0.06)")

    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
        line=dict(color="rgba(0,212,255,0.07)", width=1),
        showlegend=False, hoverinfo="none"))
    fig.add_trace(go.Scatter(x=nx_, y=ny_, mode="markers+text",
        marker=dict(color=colors, size=sizes_,
                    line=dict(width=1.2, color=borders)),
        text=labels_,
        textposition="bottom center",
        textfont=dict(size=8, color="#363b68", family="JetBrains Mono"),
        hovertemplate="<b>%{text}</b><br>Threat Score: %{customdata:.0%}<extra></extra>",
        customdata=threats_, showlegend=False))

    fig.update_layout(**_lay(
        height=360,
        xaxis=dict(visible=False, range=[-8.2,8.2]),
        yaxis=dict(visible=False, range=[-8.2,8.2]),
        plot_bgcolor="rgba(2,2,8,0.99)",
    ))
    return fig


def make_attack_timeline():
    """24h attack frequency sparkline."""
    rng = np.random.default_rng(42)
    hours = list(range(25))
    # Attacks peak at night and mid-day
    base  = [4,6,8,12,9,6,4,3,5,8,14,18,20,18,15,12,10,9,7,6,8,12,16,20,18]
    noise = rng.integers(-2, 3, size=25)
    vals  = np.clip(np.array(base)+noise, 0, 30).tolist()
    labels= [f"{h:02d}:00" for h in hours]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=vals, mode="lines",
        fill="tozeroy",
        fillcolor="rgba(255,45,85,0.07)",
        line=dict(color="#ff2d55", width=1.5),
        hovertemplate="%{x}<br><b>%{y} attacks</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=vals, mode="markers",
        marker=dict(color="#ff2d55", size=4, line=dict(width=1,color="#ff2d55")),
        showlegend=False, hoverinfo="none",
    ))
    fig.update_layout(**_lay(
        height=160,
        xaxis=dict(**DT_AXIS, dtick=4),
        yaxis=dict(**DT_AXIS, title=dict(text="Attacks/hr",font=dict(size=9,color="#363b68"))),
        margin=dict(l=40,r=12,t=16,b=28),
        showlegend=False,
    ))
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
import datetime
now = datetime.datetime.now().strftime("%H:%M:%S")

with st.sidebar:
    st.markdown("""
    <div class="dt-logo">
        <div class="dt-logo-outer">🛡️</div>
        <h2>AI · IDS</h2>
        <p>Threat Intelligence Platform</p>
        <p style="color:#13163a;font-size:9px;margin-top:2px;">COM668 · B00979380 · QAHE</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("nav", [
        "Threat Visualizer",
        "Model Intelligence",
        "Threat Hunt",
        "Batch Analysis",
        "Live Detection",
        "AI Explainability",
        "Temporal Analysis",
        "Architecture",
    ], label_visibility="collapsed")

    st.markdown("---")
    s_col  = "#00ff88" if models_ok else "#ff2d55"
    s_txt  = "ONLINE" if models_ok else "OFFLINE"
    st.markdown(f"""
    <div style="padding:14px;background:#05050d;border:1px solid rgba(0,212,255,0.07);border-radius:6px;">
        <div style="font-size:9px;color:#1e2248;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;font-family:'JetBrains Mono',monospace;">System Status</div>
        <div style="font-size:10px;color:{s_col};font-family:'JetBrains Mono',monospace;margin-bottom:6px;text-shadow:0 0 8px {s_col};">● MODELS {s_txt}</div>
        <div style="font-size:10px;color:#252a52;font-family:'JetBrains Mono',monospace;margin-bottom:4px;">◈ CICIDS2017 · 2.8M FLOWS</div>
        <div style="font-size:10px;color:#252a52;font-family:'JetBrains Mono',monospace;margin-bottom:4px;">◈ {len(feature_names)} FEATURES</div>
        <div style="font-size:10px;color:#252a52;font-family:'JetBrains Mono',monospace;margin-bottom:4px;">◈ RF · IF · ENSEMBLE</div>
        <div style="height:1px;background:rgba(0,212,255,0.06);margin:10px 0;"></div>
        <div style="font-size:9px;color:#1e2248;font-family:'JetBrains Mono',monospace;">LAST SYNC  <span style="color:#252a52;">{now}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# THREAT VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════
if page == "Threat Visualizer":
    st.markdown('<div class="page-title"><h1>Threat Visualizer</h1><p>Network topology · Threat feed · Intelligence summary — CICIDS2017 dataset</p></div>', unsafe_allow_html=True)

    # Threat level banner
    st.markdown(f"""
    <div class="threat-banner tb-elevated">
        <div class="threat-dot dot-amber" style="margin-top:0;"></div>
        <span class="tb-label">Threat Level</span>
        <span class="tb-level">ELEVATED</span>
        <div class="tb-bar-wrap"><div class="tb-bar-fill" style="width:65%;"></div></div>
        <span class="tb-score">65 / 100</span>
        <span class="tb-time">Assessed {now}</span>
        <span class="tb-live">● LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    st.markdown("""
    <div class="kpi-grid">
        <div class="kpi-card kpi-cyan"><div class="kpi-label">Pipeline Stages</div><div class="kpi-value">9</div><div class="kpi-sub">Notebooks complete</div></div>
        <div class="kpi-card kpi-red"><div class="kpi-label">RF Miss Rate</div><div class="kpi-value">85%</div><div class="kpi-sub">Core challenge</div></div>
        <div class="kpi-card kpi-green"><div class="kpi-label">Peak Recall</div><div class="kpi-value">97%</div><div class="kpi-sub">IF · PR-optimal</div></div>
        <div class="kpi-card kpi-purple"><div class="kpi-label">Unit Tests</div><div class="kpi-value">52</div><div class="kpi-sub">All passing · pytest</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_net, col_feed = st.columns([3,2], gap="large")

    with col_net:
        st.markdown('<div class="sec-header"><h3>Network Topology</h3><div class="sec-line"></div></div>', unsafe_allow_html=True)
        st.plotly_chart(make_network_graph(), use_container_width=True)
        st.markdown("""
        <div style="display:flex;gap:20px;padding:4px 2px;">
            <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:#363b68;">◉ <span style="color:#ff2d55;">COMPROMISED</span></span>
            <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:#363b68;">◉ <span style="color:#ffbb00;">SUSPICIOUS</span></span>
            <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:#363b68;">◉ <span style="color:#00d4ff;">NORMAL</span></span>
            <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:#363b68;">--- INTERNAL / DMZ / EXTERNAL</span>
        </div>
        """, unsafe_allow_html=True)

    with col_feed:
        st.markdown('<div class="sec-header"><h3>Detection Feed</h3><div class="sec-line"></div></div>', unsafe_allow_html=True)
        feed = [
            ("02:14:33","HOST-07","DDoS · High packet rate · 1.2M pkt/s",0.97,"red"),
            ("02:13:58","HOST-12","Port scan · SYN flood · port 22",0.89,"red"),
            ("02:12:41","HOST-03","IF anomaly score exceeded threshold",0.61,"amber"),
            ("02:11:20","HOST-19","Brute-force · SSH login attempts",0.88,"red"),
            ("02:09:55","SW-02",  "Lateral movement · east-west traffic",0.44,"amber"),
            ("02:08:17","HOST-05","Normal HTTP/HTTPS traffic",0.12,"cyan"),
            ("02:07:03","HOST-14","Web request anomaly · unusual UA",0.52,"amber"),
            ("02:05:48","HOST-09","Normal DNS resolution pattern",0.08,"cyan"),
        ]
        for t,h,d,score,lvl in feed:
            st.markdown(f"""
            <div class="threat-item">
                <div class="threat-dot dot-{lvl}"></div>
                <div style="flex:1;min-width:0;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="threat-host">{h}</span>
                        <span class="threat-score sc-{lvl}">{score:.0%}</span>
                    </div>
                    <div class="threat-desc">{d}</div>
                </div>
                <div class="threat-time">{t}</div>
            </div>
            """, unsafe_allow_html=True)

    # Attack timeline + model comparison
    st.markdown('<div class="sec-header"><h3>Attack Frequency · 24h</h3><div class="sec-line"></div></div>', unsafe_allow_html=True)
    st.plotly_chart(make_attack_timeline(), use_container_width=True)

    st.markdown('<div class="sec-header"><h3>Model Comparison</h3><div class="sec-line"></div></div>', unsafe_allow_html=True)
    ens = load_csv("13_ensemble_comparison.csv")
    if ens is not None:
        mc1, mc2 = st.columns([3,2], gap="large")
        with mc1:
            fig = go.Figure()
            ml = ens["Model"].tolist()
            for col_n, col_h, lbl in [("Attack Recall","#00d4ff","Recall"),("Attack Precision","#bf5fff","Precision"),("Attack F1","#00ff88","F1")]:
                v = (ens[col_n]*100).round(1)
                fig.add_trace(go.Bar(name=lbl, x=ml, y=v, marker_color=col_h, marker_line_width=0,
                    text=v.astype(str)+"%", textposition="outside", textfont=dict(color=col_h,size=10)))
            fig.update_layout(**_lay(barmode="group", height=280,
                xaxis=dict(**DT_AXIS),
                yaxis=dict(**DT_AXIS, range=[0,115]),
                legend=dict(orientation="h",y=1.1,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)")))
            st.plotly_chart(fig, use_container_width=True)
        with mc2:
            cats  = ["Recall","Precision","F1","Accuracy","ROC-AUC","Low FPR"]
            rd    = {"Random Forest":[15,99,26,79,84,96],"Isolation Forest":[28,47,35,74,68,28],"IF PR-Optimal":[97,39,55,50,68,10]}
            fcs   = ["rgba(0,212,255,0.1)","rgba(191,95,255,0.1)","rgba(0,255,136,0.1)"]
            lcs   = ["#00d4ff","#bf5fff","#00ff88"]
            fig2  = go.Figure()
            for (m,v),fc,lc in zip(rd.items(),fcs,lcs):
                fig2.add_trace(go.Scatterpolar(r=v+[v[0]],theta=cats+[cats[0]],fill="toself",
                    name=m,line=dict(color=lc,width=1.5),fillcolor=fc,marker=dict(size=4,color=lc)))
            fig2.update_layout(**_lay(height=280,
                polar=dict(bgcolor="rgba(2,2,8,0.99)",
                    radialaxis=dict(visible=True,range=[0,100],gridcolor="rgba(0,212,255,0.06)",linecolor="rgba(0,212,255,0.08)",tickfont=dict(size=8,color="#363b68")),
                    angularaxis=dict(gridcolor="rgba(0,212,255,0.06)",linecolor="rgba(0,212,255,0.08)",tickfont=dict(size=9,color="#7b82b0"))),
                legend=dict(orientation="h",y=-0.18,font=dict(size=9),bgcolor="rgba(0,0,0,0)")))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec-header"><h3>Intelligence Summary</h3><div class="sec-line"></div></div>', unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("""
        <div class="finding f-red"><div class="finding-title">Critical · Class Imbalance</div><div class="finding-body">RF achieves 99.2% precision but only 14.9% recall — missing 85% of attacks. Default threshold biased toward majority BENIGN class.</div></div>
        <div class="finding f-green"><div class="finding-title">Resolution · Threshold Tuning</div><div class="finding-body">PR-optimal threshold on IF at t=−0.125 achieves 97% recall. For IDS, missing an attack is far costlier than a false alarm.</div></div>
        """, unsafe_allow_html=True)
    with fc2:
        st.markdown("""
        <div class="finding f-cyan"><div class="finding-title">Explainability · SHAP</div><div class="finding-body">Top features: Destination Port, Flow Duration, Flow Bytes/s, Packet Length Mean. SHAP values align with domain knowledge.</div></div>
        <div class="finding f-amber"><div class="finding-title">Temporal · Concept Drift</div><div class="finding-body">Performance degrades across CICIDS2017 days. Monday-trained models underperform on Friday — reflecting real-world distribution shift.</div></div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Intelligence":
    st.markdown('<div class="page-title"><h1>Model Intelligence</h1><p>Evaluation metrics · RF · IF · Ensemble · Threshold analysis</p></div>', unsafe_allow_html=True)
    ens = load_csv("13_ensemble_comparison.csv")
    if ens is not None:
        disp = ens[["Model","Attack Recall","Attack Precision","Attack F1","F1-macro","Accuracy"]].copy()
        for c in ["Attack Recall","Attack Precision","Attack F1","F1-macro","Accuracy"]:
            disp[c] = (disp[c]*100).round(1).astype(str)+"%"
        st.dataframe(disp.set_index("Model"), use_container_width=True, height=195)

    tabs = st.tabs(["Random Forest","Isolation Forest","Ensemble","Threshold","Class Balance"])
    with tabs[0]:
        c1,c2=st.columns(2)
        with c1: show_img("05_rf_confusion_matrix.png","RF · Confusion Matrix")
        with c2: show_img("06_rf_roc_curve.png","RF · ROC AUC = 0.8375")
        show_img("03_feature_importance_rf.png","RF · Top 20 Feature Importances (MDI)")
    with tabs[1]:
        c1,c2=st.columns(2)
        with c1: show_img("07_if_confusion_matrix.png","IF · Confusion Matrix")
        with c2: show_img("22_if_score_distributions.png","IF · Anomaly Score Distributions")
        show_img("04_isolation_forest_contamination_tuning.png","IF · Contamination Parameter Sweep")
    with tabs[2]:
        show_img("13_ensemble_comparison.png","Ensemble Methods Comparison")
        show_img("15_ensemble_confusion_matrices.png","Confusion Matrices — All Ensemble Rules")
    with tabs[3]:
        c1,c2=st.columns(2)
        with c1: show_img("23_pr_curve_optimal.png","PR Curve · Optimal Threshold")
        with c2: show_img("24_f1_threshold_curve.png","F1 vs Decision Threshold")
        show_img("26_pr_optimal_confusion_matrices.png","Default vs PR-Optimal")
        thr=load_csv("27_pr_threshold_comparison.csv")
        if thr is not None: st.dataframe(thr.set_index("Method"),use_container_width=True)
    with tabs[4]:
        c1,c2=st.columns(2)
        with c1: show_img("01_class_distribution_before_resampling.png","Before SMOTE")
        with c2: show_img("02_class_distribution_after_resampling.png","After SMOTE")
        st.markdown('<div class="finding f-amber"><div class="finding-title">Data Leakage Prevention</div><div class="finding-body">SMOTE applied AFTER train/test split — exclusively to training set. Test set contains only original data, ensuring metrics reflect true generalisation.</div></div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# THREAT HUNT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Threat Hunt":
    st.markdown('<div class="page-title"><h1>Threat Hunt</h1><p>Submit a network flow · All engines classify in real-time</p></div>',unsafe_allow_html=True)
    if not models_ok: st.error("Detection engines offline."); st.stop()

    PRESETS = {
        "Custom input": {},
        "Benign · HTTP":   {"Destination Port":80,"Flow Duration":120000,"Total Fwd Packets":10,"Total Backward Packets":8,"Total Length of Fwd Packets":1200,"Total Length of Bwd Packets":3400,"Flow Bytes/s":38333.0,"Flow Packets/s":150.0},
        "DDoS flood":      {"Destination Port":80,"Flow Duration":1000,"Total Fwd Packets":500,"Total Backward Packets":0,"Total Length of Fwd Packets":32000,"Total Length of Bwd Packets":0,"Flow Bytes/s":32000000.0,"Flow Packets/s":500000.0},
        "Port scan · SSH": {"Destination Port":22,"Flow Duration":5000,"Total Fwd Packets":1,"Total Backward Packets":1,"Total Length of Fwd Packets":40,"Total Length of Bwd Packets":40,"Flow Bytes/s":16000.0,"Flow Packets/s":400.0},
    }
    preset = st.selectbox("Load preset flow",list(PRESETS.keys()))
    pv = PRESETS[preset]
    st.markdown("---")
    key_feats=["Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets","Total Length of Fwd Packets","Total Length of Bwd Packets","Flow Bytes/s","Flow Packets/s"]
    c1,c2=st.columns(2)
    fv_input={}
    with c1:
        for f in key_feats[:4]: fv_input[f]=st.number_input(f,value=float(pv.get(f,0.0)),step=1.0)
    with c2:
        for f in key_feats[4:]: fv_input[f]=st.number_input(f,value=float(pv.get(f,0.0)),step=1.0)
    st.caption(f"Remaining {len(feature_names)-8} of 78 features default to 0.0")
    fv=[float(fv_input.get(f,0.0)) for f in feature_names]
    st.markdown("---")
    if st.button("Classify Flow"):
        with st.spinner("Running detection engines..."):
            time.sleep(0.15)
            try:
                r=classify_flow(fv,scaler,le,rf_model,iso_model)
                final=r["final_decision"]; rf_pred=r["random_forest"]["prediction"]
                rf_conf=r["random_forest"]["confidence"]; rf_atk=r["random_forest"]["attack_prob"]
                if_pred=r["isolation_forest"]["prediction"]; if_scr=r["isolation_forest"]["anomaly_score"]
                ep=r["ensemble"]["prediction"]
                box="pred-attack" if final=="ATTACK" else "pred-benign"
                icon="▲ THREAT DETECTED" if final=="ATTACK" else "✓ TRAFFIC NORMAL"
                st.markdown(f'<div class="pred-box {box}"><div class="pred-label">Final Decision · Ensemble OR Rule</div><div class="pred-decision">{icon}</div><div class="pred-sub">{"Alert — investigate immediately" if final=="ATTACK" else "No malicious signature detected"}</div></div>',unsafe_allow_html=True)
                mc1,mc2,mc3=st.columns(3)
                for col,lbl,pred,sub in [(mc1,"Random Forest",rf_pred,f"Conf {rf_conf*100:.1f}% · Atk {rf_atk*100:.1f}%"),(mc2,"Isolation Forest",if_pred,f"Score {if_scr:.4f}"),(mc3,"Ensemble OR",ep,"Attack if either engine flags")]:
                    c_="pred-attack" if pred=="ATTACK" else "pred-benign"
                    l_="▲ ATTACK" if pred=="ATTACK" else "✓ BENIGN"
                    col.markdown(f'<div class="pred-box {c_}" style="padding:12px;"><div class="pred-label">{lbl}</div><div class="pred-decision" style="font-size:15px;">{l_}</div><div class="pred-sub">{sub}</div></div>',unsafe_allow_html=True)
                fig_g=go.Figure(go.Indicator(mode="gauge+number",value=rf_atk*100,
                    title={"text":"ATTACK PROBABILITY","font":{"color":"#363b68","size":11,"family":"JetBrains Mono"}},
                    number={"suffix":"%","font":{"color":"#e8eaf6","size":28,"family":"JetBrains Mono"}},
                    gauge={"axis":{"range":[0,100],"tickcolor":"#363b68","tickfont":{"color":"#363b68","size":9}},
                           "bar":{"color":"#ff2d55" if rf_atk>0.5 else "#00ff88","thickness":0.22},
                           "bgcolor":"#05050d","bordercolor":"rgba(0,212,255,0.07)",
                           "steps":[{"range":[0,40],"color":"rgba(0,255,136,0.05)"},{"range":[40,70],"color":"rgba(255,187,0,0.05)"},{"range":[70,100],"color":"rgba(255,45,85,0.05)"}],
                           "threshold":{"line":{"color":"#ffbb00","width":1.5},"value":50}}))
                fig_g.update_layout(**_lay(height=190,margin=dict(l=32,r=32,t=48,b=8)))
                st.plotly_chart(fig_g,use_container_width=True)
            except Exception as e: st.error(f"Engine error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Batch Analysis":
    st.markdown('<div class="page-title"><h1>Batch Analysis</h1><p>Upload a CSV · All rows classified by all engines simultaneously · Max 5,000 rows</p></div>',unsafe_allow_html=True)
    if not models_ok: st.error("Detection engines offline."); st.stop()
    st.markdown('<div class="finding f-cyan"><div class="finding-title">Input Format</div><div class="finding-body">CSV must have column headers matching the 78 CICIDS2017 feature names. Missing columns default to 0. Extra columns ignored.</div></div>',unsafe_allow_html=True)
    uploaded=st.file_uploader("Drop CSV here",type=["csv"])
    if uploaded:
        try:
            df_up=pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_up):,} rows × {len(df_up.columns)} columns")
            if len(df_up)>5000: st.warning("Truncating to 5,000 rows."); df_up=df_up.head(5000)
            with st.spinner(f"Analysing {len(df_up):,} flows..."):
                rows=[]
                for _,row in df_up.iterrows():
                    fv=[float(row.get(f,0.0)) for f in feature_names]
                    try:
                        r=classify_flow(fv,scaler,le,rf_model,iso_model)
                        rows.append({"RF":r["random_forest"]["prediction"],"Conf":round(r["random_forest"]["confidence"],4),"Atk%":round(r["random_forest"]["attack_prob"]*100,1),"IF":r["isolation_forest"]["prediction"],"IF Score":round(r["isolation_forest"]["anomaly_score"],4),"Decision":r["final_decision"]})
                    except: rows.append({"RF":"ERR","Conf":0,"Atk%":0,"IF":"ERR","IF Score":0,"Decision":"ERR"})
            df_res=pd.concat([df_up.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
            total=len(df_res); attacks=(df_res["Decision"]=="ATTACK").sum(); benign=total-attacks
            atk_pct=round(attacks/total*100,1) if total else 0
            st.markdown(f'<div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);"><div class="kpi-card kpi-cyan"><div class="kpi-label">Total Flows</div><div class="kpi-value">{total:,}</div><div class="kpi-sub">Rows analysed</div></div><div class="kpi-card kpi-red"><div class="kpi-label">Threats</div><div class="kpi-value">{attacks:,}</div><div class="kpi-sub">{atk_pct}% of traffic</div></div><div class="kpi-card kpi-green"><div class="kpi-label">Clean</div><div class="kpi-value">{benign:,}</div><div class="kpi-sub">Normal flows</div></div></div>',unsafe_allow_html=True)
            fig_p=go.Figure(go.Pie(labels=["ATTACK","BENIGN"],values=[attacks,benign],hole=0.65,
                marker=dict(colors=["#ff2d55","#00d4ff"],line=dict(color="#03030a",width=2)),
                textinfo="label+percent",textfont=dict(color="#c8caf4",size=11,family="JetBrains Mono")))
            fig_p.update_layout(**_lay(height=230,showlegend=False))
            st.plotly_chart(fig_p,use_container_width=True)
            st.dataframe(df_res[["RF","Conf","Atk%","IF","IF Score","Decision"]].head(100),use_container_width=True,height=280)
            st.download_button("Export Full Results CSV",df_res.to_csv(index=False).encode(),"ids_batch_results.csv","text/csv")
        except Exception as e: st.error(f"Error: {e}")
    else:
        st.info("Upload a CSV file above to begin batch analysis.")
        if feature_names: st.caption(f"Expected columns (first 10 of {len(feature_names)}): {', '.join(feature_names[:10])} ···")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Detection":
    st.markdown('<div class="page-title"><h1>Live Detection</h1><p>Synthetic flow stream · Real-time classification · Simulates active IDS deployment</p></div>',unsafe_allow_html=True)
    if not models_ok: st.error("Detection engines offline."); st.stop()

    def gen_flow(rng, attack=False):
        fv=rng.standard_normal(len(feature_names)).tolist()
        idx={n:i for i,n in enumerate(feature_names)}
        if attack:
            if "Flow Packets/s" in idx:         fv[idx["Flow Packets/s"]]=float(rng.uniform(100000,1000000))
            if "Destination Port" in idx:        fv[idx["Destination Port"]]=float(rng.choice([22,23,3389,445]))
            if "Total Backward Packets" in idx:  fv[idx["Total Backward Packets"]]=0.0
        else:
            if "Destination Port" in idx: fv[idx["Destination Port"]]=float(rng.choice([80,443,8080]))
            if "Flow Packets/s" in idx:   fv[idx["Flow Packets/s"]]=float(rng.uniform(10,500))
        return fv

    c1,c2,c3=st.columns(3)
    with c1: n_flows=st.slider("Flows",20,200,60,step=10)
    with c2: atk_ratio=st.slider("Attack ratio (%)",5,60,20,step=5)
    with c3:
        st.markdown("<br>",unsafe_allow_html=True)
        run=st.button("Run Detection Stream",use_container_width=True)

    if run:
        rng=np.random.default_rng(42)
        n_atk=int(n_flows*atk_ratio/100)
        gt=["ATTACK"]*n_atk+["BENIGN"]*(n_flows-n_atk); rng.shuffle(gt)
        prog=st.progress(0,text="Initialising detection stream...")
        feed=[]
        for i,g in enumerate(gt):
            fv=gen_flow(rng,attack=(g=="ATTACK"))
            r=classify_flow(fv,scaler,le,rf_model,iso_model)
            feed.append({"Flow #":i+1,"Ground Truth":g,"Decision":r["final_decision"],"RF":r["random_forest"]["prediction"],"Atk%":round(r["random_forest"]["attack_prob"]*100,1),"IF":r["isolation_forest"]["prediction"],"IF Score":round(r["isolation_forest"]["anomaly_score"],4),"Correct":r["final_decision"]==g})
            prog.progress((i+1)/n_flows,text=f"Flow {i+1}/{n_flows} · {r['final_decision']}")
        prog.empty()
        df_sim=pd.DataFrame(feed)
        tp=int(((df_sim["Ground Truth"]=="ATTACK")&(df_sim["Decision"]=="ATTACK")).sum())
        fp=int(((df_sim["Ground Truth"]=="BENIGN")&(df_sim["Decision"]=="ATTACK")).sum())
        fn=int(((df_sim["Ground Truth"]=="ATTACK")&(df_sim["Decision"]=="BENIGN")).sum())
        recall=round(tp/(tp+fn)*100,1) if (tp+fn) else 0
        precision=round(tp/(tp+fp)*100,1) if (tp+fp) else 0
        st.markdown(f'<div class="kpi-grid"><div class="kpi-card kpi-green"><div class="kpi-label">True Positives</div><div class="kpi-value">{tp}</div><div class="kpi-sub">Attacks caught</div></div><div class="kpi-card kpi-red"><div class="kpi-label">Missed</div><div class="kpi-value">{fn}</div><div class="kpi-sub">False negatives</div></div><div class="kpi-card kpi-cyan"><div class="kpi-label">Recall</div><div class="kpi-value">{recall}%</div><div class="kpi-sub">Detection rate</div></div><div class="kpi-card kpi-purple"><div class="kpi-label">Precision</div><div class="kpi-value">{precision}%</div><div class="kpi-sub">Alert accuracy</div></div></div>',unsafe_allow_html=True)
        fig_tl=go.Figure()
        for lbl,col,sym in [("ATTACK","#ff2d55","x"),("BENIGN","#00d4ff","circle")]:
            mask=df_sim["Decision"]==lbl
            fig_tl.add_trace(go.Scatter(x=df_sim[mask]["Flow #"],y=df_sim[mask]["Atk%"],mode="markers",name=lbl,marker=dict(color=col,size=8,symbol=sym,line=dict(width=1,color=col),opacity=0.85)))
        fig_tl.add_hline(y=50,line_dash="dot",line_color="#ffbb00",line_width=1,annotation_text="Decision boundary",annotation_font_color="#ffbb00",annotation_font_size=9)
        fig_tl.update_layout(**_lay(height=270,xaxis=dict(**DT_AXIS,title=dict(text="Flow #",font=dict(size=9,color="#363b68"))),yaxis=dict(**DT_AXIS,title=dict(text="Attack Probability (%)",font=dict(size=9,color="#363b68"))),legend=dict(orientation="h",y=1.1,bgcolor="rgba(0,0,0,0)",font=dict(size=10))))
        st.plotly_chart(fig_tl,use_container_width=True)
        st.dataframe(df_sim.drop(columns=["Correct"]),use_container_width=True,height=280)
        missed=df_sim[(df_sim["Ground Truth"]=="ATTACK")&(df_sim["Decision"]=="BENIGN")]
        if not missed.empty:
            st.markdown(f'<div class="finding f-red"><div class="finding-title">Missed Attacks · {len(missed)} Flows</div><div class="finding-body">These {len(missed)} flows were ground-truth ATTACK but classified BENIGN. PR-optimal threshold tuning reduces this to ~3% miss rate. See Notebook 09.</div></div>',unsafe_allow_html=True)
    else:
        show_img("21_live_sim_final.png","Previous simulation · 200-flow detection stream")
        st.info("Configure parameters and click Run Detection Stream.")


# ══════════════════════════════════════════════════════════════════════════════
# AI EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "AI Explainability":
    st.markdown('<div class="page-title"><h1>AI Explainability</h1><p>SHAP · SHapley Additive exPlanations · Feature attribution · Decision transparency</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="finding f-cyan"><div class="finding-title">Why Explainability Matters in IDS</div><div class="finding-body">SHAP reveals why a model flags a flow — not just that it does. SOC analysts need feature-level justification to act on alerts, audit model bias, and validate that the model has learned real attack signatures rather than dataset artefacts. Without explainability, an IDS is a black box that cannot be trusted in production.</div></div>',unsafe_allow_html=True)
    tabs=st.tabs(["Summary","Waterfall · Attack","Waterfall · Benign","Dependence","SHAP vs MDI"])
    with tabs[0]:
        show_img("16_shap_summary.png","SHAP Summary · Feature importance and directional impact on attack probability")
        st.markdown('<div class="finding f-green"><div class="finding-title">Key Finding</div><div class="finding-body">Destination Port has highest mean |SHAP|. Ports 22 (SSH), 23 (Telnet), 3389 (RDP), 445 (SMB) strongly associated with attack traffic. Flow Duration and Flow Bytes/s are secondary discriminators.</div></div>',unsafe_allow_html=True)
    with tabs[1]: show_img("18_shap_waterfall_attack.png","Waterfall · Single ATTACK prediction fully decomposed into feature contributions")
    with tabs[2]: show_img("19_shap_waterfall_benign.png","Waterfall · Single BENIGN prediction fully decomposed")
    with tabs[3]: show_img("20_shap_dependence.png","Dependence · Feature value vs SHAP contribution coloured by interaction feature")
    with tabs[4]:
        show_img("17_shap_vs_mdi.png","SHAP vs RF MDI · Cross-validation of feature importance rankings")
        st.markdown('<div class="finding f-amber"><div class="finding-title">MDI Validated by SHAP</div><div class="finding-body">Strong correlation between SHAP and MDI rankings validates MDI as a fast reliable proxy for the /api/shap Flask endpoint, avoiding full SHAP computation at inference time.</div></div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Temporal Analysis":
    st.markdown('<div class="page-title"><h1>Temporal Analysis</h1><p>Concept drift detection · Cross-day generalisation · Distribution shift over time</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="finding f-amber"><div class="finding-title">Concept Drift in Cybersecurity</div><div class="finding-body">Attack patterns evolve rapidly. A model trained on Monday traffic may miss variants that emerge on Friday. This temporal generalisation gap is the central challenge in production IDS deployment and a key limitation of all static-dataset evaluations including CICIDS2017.</div></div>',unsafe_allow_html=True)
    show_img("11_concept_drift_analysis.png","Cross-day F1 · Model trained Monday, evaluated on subsequent days")
    drift=load_csv("11_concept_drift_table.csv")
    if drift is not None:
        st.markdown('<div class="sec-header"><h3>Cross-Day Performance</h3><div class="sec-line"></div></div>',unsafe_allow_html=True)
        st.dataframe(drift,use_container_width=True)

    with c1: st.markdown('<div class="finding f-red"><div class="finding-title">CICIDS2017 Limitations</div><div class="finding-body">Generated in 2017 in a controlled lab. Does not include post-2017 attack types: supply-chain, AI-generated phishing, cloud-native threats, or LLM-assisted reconnaissance.</div></div>',unsafe_allow_html=True)
    with c2: st.markdown('<div class="finding f-cyan"><div class="finding-title">Mitigation Strategies</div><div class="finding-body">Periodic retraining on fresh samples. Online learning for incremental updates. IF is more adaptable to drift than supervised RF. Ensemble diversity provides robustness buffer.</div></div>',unsafe_allow_html=True)


# ============================================================
# ARCHITECTURE
# ============================================================
elif page == "Architecture":
    st.markdown('<div class="page-title"><h1>Architecture</h1><p>System design - Pipeline - UML diagrams - Non-functional requirements</p></div>',unsafe_allow_html=True)
    tabs=st.tabs(["Pipeline","UML Diagrams","NFR - MoSCoW","Risk Register"])

    with tabs[0]:
        steps=[
            ("01","Data Ingestion",    "CICIDS2017 CSVs - 8 capture days - ~2.8M rows - 78 features. Merged, cleaned, duplicates removed, infinite values capped."),
            ("02","Preprocessing",     "Binary label: BENIGN vs ATTACK. StandardScaler fitted on training set only - never re-fitted on test data."),
            ("03","Class Balancing",   "RandomOverSampler applied AFTER train/test split to training set only. Prevents data leakage."),
            ("04","Model Training",    "Random Forest: 100 trees, class_weight=balanced, random_state=42. Isolation Forest: contamination=0.10."),
            ("05","Ensemble Design",   "OR rule: flag ATTACK if either model detects. Maximises recall for IDS context."),
            ("06","Evaluation",        "Precision, Recall, F1, ROC-AUC, confusion matrices. PR-curve: t=-0.125 achieves 97% recall on IF."),
            ("07","Explainability",    "SHAP TreeExplainer on RF. Waterfall, summary, dependence plots. /api/shap Flask endpoint."),
            ("08","Deployment",        "Flask REST API: /predict, /predict/batch, /api/feed, /api/stats, /api/shap. Docker containerised."),
        ]
        for num,title,desc in steps:
            st.markdown(f'<div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:6px;"><div style="min-width:34px;height:34px;border-radius:5px;background:rgba(0,212,255,0.07);border:1px solid rgba(0,212,255,0.14);display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:#00d4ff;font-family:JetBrains Mono,monospace;">{num}</div><div style="flex:1;background:#05050d;border:1px solid rgba(0,212,255,0.06);border-radius:6px;padding:9px 14px;border-left:2px solid rgba(0,212,255,0.12);"><div style="font-size:12px;font-weight:700;color:#c8caf4;margin-bottom:3px;">{title}</div><div style="font-size:11px;color:#363b68;line-height:1.55;">{desc}</div></div></div>',unsafe_allow_html=True)

    with tabs[1]:
        c1,c2,c3=st.columns(3)
        with c1: show_img("UML_01_use_case.png","Use Case Diagram")
        with c2: show_img("UML_02_activity.png","Activity Diagram")
        with c3: show_img("UML_03_sequence.png","Sequence Diagram")

    with tabs[2]:
        nfrs=[
            ("Performance",    "MUST",   "Single-flow inference < 100ms - Measured avg 2-8ms"),
            ("Reproducibility","MUST",   "Full pipeline trainable from notebooks - random_state=42"),
            ("Scalability",    "SHOULD", "Batch API handles up to 1,000 flows per request"),
            ("Portability",    "SHOULD", "Fully containerised via Docker"),
            ("Maintainability","SHOULD", "Modular src/ids/ package"),
            ("Explainability", "SHOULD", "SHAP values via Notebook 07 and /api/shap endpoint"),
            ("Testability",    "COULD",  "52 pytest tests across preprocessing, predict, and API"),
            ("Real-time pcap", "WONT",   "Full live packet-capture deployment out of scope"),
        ]
        badge={"MUST":"#ff2d55","SHOULD":"#ffbb00","COULD":"#00d4ff","WONT":"#363b68"}
        for req,pri,desc in nfrs:
            bc=badge.get(pri,"#363b68")
            st.markdown(f'<div style="display:flex;gap:12px;align-items:center;background:#05050d;border:1px solid rgba(0,212,255,0.06);border-radius:6px;padding:10px 14px;margin-bottom:5px;"><div style="min-width:130px;font-size:11px;font-weight:700;color:#c8caf4;font-family:JetBrains Mono,monospace;">{req}</div><span style="padding:2px 10px;border-radius:3px;font-size:9px;font-weight:700;background:{bc}18;color:{bc};border:1px solid {bc}40;min-width:55px;text-align:center;font-family:JetBrains Mono,monospace;">{pri}</span><div style="font-size:11px;color:#363b68;">{desc}</div></div>',unsafe_allow_html=True)

    with tabs[3]:
        show_img("32_risk_register.png","Risk Register")
