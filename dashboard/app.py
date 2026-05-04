"""
OptiCast AI — KERC DSM Compliance Dashboard
Single-file Streamlit application | Frontend only (mock data)
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OptiCast AI · KERC DSM Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — dark utility-grid aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@400;700;900&display=swap');

/* ── Root tokens ── */
:root {
    --bg-base:      #080c14;
    --bg-panel:     #0d1520;
    --bg-card:      #111d2e;
    --border:       #1e3a55;
    --accent-cyan:  #00d4ff;
    --accent-amber: #f5a623;
    --accent-red:   #ff4d4d;
    --accent-green: #00e676;
    --text-primary: #e8f4fd;
    --text-muted:   #6b8fa8;
    --font-mono:    'Share Tech Mono', monospace;
    --font-body:    'Barlow', sans-serif;
    --font-cond:    'Barlow Condensed', sans-serif;
}

/* ── Base reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background-color: var(--bg-panel) !important; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* ── Hide default decoration ── */
#MainMenu, footer { visibility: hidden; }
.block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 100% !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ─────────────── HEADER ─────────────── */
.oc-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 1.8rem;
    background: linear-gradient(135deg, #0d1a2d 0%, #091523 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent-cyan);
    border-radius: 6px;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.oc-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 39px,
        rgba(0,212,255,0.03) 39px,
        rgba(0,212,255,0.03) 40px
    );
    pointer-events: none;
}
.oc-logo-group { display: flex; align-items: center; gap: 1rem; }
.oc-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #003c57, #00d4ff22);
    border: 1.5px solid var(--accent-cyan);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
}
.oc-title {
    font-family: var(--font-cond);
    font-weight: 900;
    font-size: 1.55rem;
    letter-spacing: 0.06em;
    color: var(--text-primary);
    line-height: 1.1;
    text-transform: uppercase;
}
.oc-subtitle {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent-cyan);
    letter-spacing: 0.12em;
    margin-top: 2px;
    opacity: 0.85;
}
.oc-meta {
    text-align: right;
    font-family: var(--font-mono);
}
.oc-live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0, 230, 118, 0.12);
    border: 1px solid rgba(0, 230, 118, 0.35);
    color: var(--accent-green);
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}
.oc-live-dot {
    width: 7px; height: 7px;
    background: var(--accent-green);
    border-radius: 50%;
    animation: pulse 1.6s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,230,118,0.5); }
    50%       { opacity: .7; box-shadow: 0 0 0 5px rgba(0,230,118,0); }
}
.oc-timestamp {
    color: var(--text-muted);
    font-size: 0.7rem;
    letter-spacing: 0.08em;
}

/* ─────────────── KPI CARDS ─────────────── */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.1rem 1.4rem;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
}
.kpi-card.cyan::after  { background: linear-gradient(90deg, var(--accent-cyan), transparent); }
.kpi-card.amber::after { background: linear-gradient(90deg, var(--accent-amber), transparent); }
.kpi-card.red::after   { background: linear-gradient(90deg, var(--accent-red), transparent); }
.kpi-card.green::after { background: linear-gradient(90deg, var(--accent-green), transparent); }

.kpi-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}
.kpi-value {
    font-family: var(--font-cond);
    font-weight: 700;
    font-size: 2rem;
    line-height: 1;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}
.kpi-value.cyan  { color: var(--accent-cyan); }
.kpi-value.amber { color: var(--accent-amber); }
.kpi-value.red   { color: var(--accent-red); }
.kpi-value.green { color: var(--accent-green); }

.kpi-delta {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
}
.kpi-delta.up   { color: var(--accent-green); }
.kpi-delta.down { color: var(--accent-red); }
.kpi-icon {
    position: absolute;
    top: 1rem; right: 1rem;
    font-size: 1.6rem;
    opacity: 0.18;
}

/* ─────────────── SECTION TITLES ─────────────── */
.section-title {
    font-family: var(--font-cond);
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title .dot {
    width: 6px; height: 6px;
    background: var(--accent-cyan);
    border-radius: 50%;
}

/* ─────────────── ALERT CARDS ─────────────── */
.alert-ramp {
    background: rgba(255, 77, 77, 0.08);
    border: 1px solid rgba(255,77,77,0.35);
    border-left: 3px solid var(--accent-red);
    border-radius: 4px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.7rem;
}
.alert-ramp .alert-header {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent-red);
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.alert-ramp .alert-body {
    font-family: var(--font-body);
    font-size: 0.88rem;
    color: var(--text-primary);
    font-weight: 400;
    line-height: 1.4;
}
.alert-ramp .alert-tag {
    display: inline-block;
    background: rgba(255,77,77,0.15);
    color: var(--accent-red);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 3px;
    margin-top: 0.5rem;
    letter-spacing: 0.08em;
}

/* ─────────────── SHAP CARD ─────────────── */
.shap-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
}
.shap-title {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--accent-cyan);
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
}
.shap-prediction {
    font-family: var(--font-cond);
    font-weight: 700;
    font-size: 1.35rem;
    color: var(--text-primary);
    margin-bottom: 0.7rem;
}
.shap-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    font-family: var(--font-body);
    font-size: 0.83rem;
}
.shap-bar-wrap {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    overflow: hidden;
}
.shap-bar {
    height: 100%;
    border-radius: 3px;
}
.shap-label { color: var(--text-muted); min-width: 130px; }
.shap-val   { font-family: var(--font-mono); font-size: 0.75rem; min-width: 48px; text-align: right; }
.shap-pos { background: linear-gradient(90deg, #00e676, #00b050); }
.shap-neg { background: linear-gradient(90deg, #ff4d4d, #c0392b); }

/* ─────────────── TABS ─────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-radius: 6px 6px 0 0;
    gap: 0;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-cond) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    padding: 0.6rem 1.4rem !important;
    border-radius: 0 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
    background: rgba(0,212,255,0.05) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 1.2rem !important;
}

/* ─────────────── DATA TABLE ─────────────── */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
[data-testid="stDataFrame"] th {
    background: #0a1929 !important;
    color: var(--accent-cyan) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stDataFrame"] td {
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    border-color: var(--border) !important;
}
[data-testid="stDataFrame"] tr:hover td { background: rgba(0,212,255,0.04) !important; }

/* ─────────────── BUTTONS ─────────────── */
.stButton > button {
    background: linear-gradient(135deg, #003c57 0%, #005a7a 100%) !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-family: var(--font-cond) !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.6rem !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #005a7a 0%, #007ca8 100%) !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.25) !important;
    transform: translateY(-1px) !important;
}

/* ─────────────── EXPANDER ─────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: var(--font-cond) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ─────────────── SUCCESS / INFO ─────────────── */
.stSuccess {
    background: rgba(0,230,118,0.08) !important;
    border: 1px solid rgba(0,230,118,0.3) !important;
    color: var(--accent-green) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

/* ─────────────── DIVIDER ─────────────── */
hr { border-color: var(--border) !important; }

/* ─────────────── INFO BAR ─────────────── */
.info-bar {
    display: flex;
    gap: 1.5rem;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.5rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.info-bar span { display: flex; align-items: center; gap: 5px; }
.info-bar .hi  { color: var(--accent-cyan); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MOCK DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def generate_forecast_data():
    """24-hour Pavagada Solar Park forecast (MW)"""
    hours = pd.date_range("2025-06-20 00:00", periods=97, freq="15min")
    t = np.linspace(0, 24, 97)

    # Solar profile: zero at night, Gaussian peak at noon
    p50_raw = np.where(
        (t >= 6) & (t <= 18.5),
        420 * np.exp(-0.5 * ((t - 12.8) / 2.8) ** 2) + np.random.normal(0, 8, 97),
        np.random.normal(0, 1.5, 97)
    )
    p50 = np.clip(p50_raw, 0, None)

    # Uncertainty bands widen mid-day
    uncertainty = 18 + 32 * np.exp(-0.5 * ((t - 12.8) / 3.5) ** 2)
    p10 = np.clip(p50 - uncertainty * 1.4, 0, None)
    p90 = p50 + uncertainty * 1.1

    df = pd.DataFrame({"timestamp": hours, "P10": p10, "P50": p50, "P90": p90})
    df["P10"] = df["P10"].clip(lower=0)
    return df

@st.cache_data
def generate_qca_schedule():
    """96-block KERC QCA schedule"""
    blocks = list(range(1, 97))
    base_time = datetime(2025, 6, 20, 0, 0)
    times = [(base_time + timedelta(minutes=15 * i)).strftime("%H:%M") for i in range(96)]
    t = np.linspace(0, 24, 96)
    declared = np.where(
        (t >= 6) & (t <= 18.5),
        np.clip(410 * np.exp(-0.5 * ((t - 12.8) / 2.8) ** 2) + np.random.normal(0, 5, 96), 0, None),
        np.zeros(96)
    )
    actual = declared + np.random.normal(0, 12, 96)
    actual = np.clip(actual, 0, None)
    deviation = actual - declared
    dsm_charge = np.where(np.abs(deviation) > 15, np.abs(deviation) * 0.02, 0.0)

    df = pd.DataFrame({
        "Block": blocks,
        "Time Slot": times,
        "Declared (MW)": np.round(declared, 1),
        "Actual (MW)": np.round(actual, 1),
        "Deviation (MW)": np.round(deviation, 1),
        "DSM Charge (₹L)": np.round(dsm_charge, 3),
        "Status": ["⚠️ Ramp" if 13.5 <= t[i] <= 14.5 else ("✅ Normal" if abs(deviation[i]) < 15 else "🔴 Deviated") for i in range(96)]
    })
    return df

forecast_df = generate_forecast_data()
qca_df = generate_qca_schedule()
now_str = datetime.now().strftime("%d %b %Y  •  %H:%M:%S IST")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="oc-header">
    <div class="oc-logo-group">
        <div class="oc-icon">⚡</div>
        <div>
            <div class="oc-title">OptiCast AI</div>
            <div class="oc-subtitle">KERC DSM Compliance Dashboard · KSPDCL Renewable Division</div>
        </div>
    </div>
    <div class="oc-meta">
        <div class="oc-live-badge"><span class="oc-live-dot"></span>LIVE TELEMETRY</div>
        <div class="oc-timestamp">{now_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Info bar
st.markdown("""
<div class="info-bar">
    <span>🗓 Forecast Date: <span class="hi">20 Jun 2025</span></span>
    <span>🏭 Assets Online: <span class="hi">4 / 4</span></span>
    <span>📡 Data Latency: <span class="hi">38 ms</span></span>
    <span>🔁 Model: <span class="hi">OptiCast-v2.1-XGB</span></span>
    <span>🌐 Grid: <span class="hi">Southern Regional Grid · 220kV</span></span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TOP KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown("""
    <div class="kpi-card cyan">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-label">Current Karnataka Demand</div>
        <div class="kpi-value cyan">11.2 GW</div>
        <div class="kpi-delta up">▲ +0.4 GW vs yesterday 15:30</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown("""
    <div class="kpi-card red">
        <div class="kpi-icon">💸</div>
        <div class="kpi-label">Active DSM Penalty Risk</div>
        <div class="kpi-value red" style="font-size:1.5rem;">HIGH · ₹12L/hr</div>
        <div class="kpi-delta down">▼ Deviation &gt;15MW · Block 58</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div class="kpi-card amber">
        <div class="kpi-icon">🔔</div>
        <div class="kpi-label">Active Ramp Alerts</div>
        <div class="kpi-value amber">1 Active</div>
        <div class="kpi-delta">Pavagada · 14:00 IST window</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown("""
    <div class="kpi-card green">
        <div class="kpi-icon">☀️</div>
        <div class="kpi-label">Pavagada Capacity Factor</div>
        <div class="kpi-value green">38.7%</div>
        <div class="kpi-delta up">▲ +2.1% vs 30-day avg</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈  24-HR FORECAST RIBBON",
    "🗂  QCA SCHEDULE & COMPLIANCE",
    "🔬  MODEL DIAGNOSTICS",
])


# ══════════════════════════════════════════════
# TAB 1 — Forecast Ribbon + Alerts
# ══════════════════════════════════════════════
with tab1:
    col_chart, col_alerts = st.columns([3, 1], gap="medium")

    # ── LEFT: Forecast Ribbon ──
    with col_chart:
        st.markdown('<div class="section-title"><span class="dot"></span>Pavagada Solar Park · P10 / P50 / P90 Probabilistic Forecast</div>', unsafe_allow_html=True)

        fig = go.Figure()

        # P90 upper fill (positive)
        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["P90"],
            mode="lines",
            line=dict(width=0),
            name="P90",
            showlegend=False,
            hovertemplate="%{x|%H:%M}  P90: %{y:.1f} MW<extra></extra>",
        ))

        # P10 lower fill
        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["P10"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0,212,255,0.10)",
            name="P10–P90 Band",
            hovertemplate="%{x|%H:%M}  P10: %{y:.1f} MW<extra></extra>",
        ))

        # P50 central line
        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["P50"],
            mode="lines",
            line=dict(color="#00d4ff", width=2.5),
            name="P50 (Median Forecast)",
            hovertemplate="%{x|%H:%M}  P50: %{y:.1f} MW<extra></extra>",
        ))

        # Ramp alert vertical annotation at 14:00
        # Pass as string to avoid pandas Timestamp integer-arithmetic error
        # Convert the specific time to Unix milliseconds (Plotly's preferred numeric format for time)
        ramp_time_ms = datetime(2025, 6, 20, 14, 0).timestamp() * 1000

        # Plot the vertical line using the millisecond value
        fig.add_vline(
            x=ramp_time_ms,
            line_dash="dash",
            line_color="#ff4d4d",
            line_width=1.5,
            annotation_text="⚠ RAMP EVENT",
            annotation_position="top",
            annotation_font=dict(color="#ff4d4d", size=11, family="Share Tech Mono"),
        )
        # Curtailment zone annotation
        fig.add_vrect(
            x0="2025-06-20 13:45",
            x1="2025-06-20 15:00",
            fillcolor="rgba(255,77,77,0.07)",
            layer="below",
            line_width=0,
        )

        fig.update_layout(
            plot_bgcolor="#0d1520",
            paper_bgcolor="#0d1520",
            font=dict(family="Barlow, sans-serif", color="#6b8fa8", size=11),
            height=360,
            margin=dict(l=8, r=8, t=18, b=8),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8f4fd", size=11),
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(30,58,85,0.6)",
                gridwidth=1,
                zeroline=False,
                tickformat="%H:%M",
                tickfont=dict(family="Share Tech Mono", size=10),
                tickcolor="#1e3a55",
                linecolor="#1e3a55",
                tickmode="linear",
                dtick=3 * 3600000,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(30,58,85,0.6)",
                gridwidth=1,
                zeroline=False,
                title=dict(text="Generation (MW)", font=dict(size=11)),
                tickfont=dict(family="Share Tech Mono", size=10),
                tickcolor="#1e3a55",
                linecolor="#1e3a55",
            ),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#111d2e",
                bordercolor="#1e3a55",
                font=dict(family="Share Tech Mono", size=11, color="#e8f4fd"),
            ),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Mini stat row below chart
        s1, s2, s3, s4 = st.columns(4)
        for col, label, val, color in [
            (s1, "Peak P50 Forecast", "431.2 MW", "#00d4ff"),
            (s2, "Ramp Magnitude",    "−50 MW",   "#ff4d4d"),
            (s3, "Sunrise / Sunset",  "06:07 / 18:43", "#f5a623"),
            (s4, "GHI (avg today)",   "624 W/m²", "#00e676"),
        ]:
            col.markdown(f"""
            <div style="background:#111d2e;border:1px solid #1e3a55;border-radius:5px;
                         padding:.7rem 1rem;text-align:center">
                <div style="font-family:'Share Tech Mono';font-size:.65rem;color:#6b8fa8;
                             letter-spacing:.1em;text-transform:uppercase;margin-bottom:.3rem">{label}</div>
                <div style="font-family:'Barlow Condensed';font-weight:700;font-size:1.15rem;
                             color:{color}">{val}</div>
            </div>""", unsafe_allow_html=True)

    # ── RIGHT: Alerts & SHAP ──
    with col_alerts:
        # Ramp Alert Panel
        st.markdown('<div class="section-title"><span class="dot"></span>Ramp Alerts</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-ramp">
            <div class="alert-header">🔴 SEVERITY: HIGH · ACTIVE</div>
            <div class="alert-body">
                <strong>Pavagada Solar Park</strong><br>
                50 MW drop predicted at <strong>14:00 IST</strong> due to incoming cloud front.<br><br>
                Recommend dispatchable reserve activation from <strong>Sharavathi HEP</strong>.
            </div>
            <div class="alert-tag">CLOUD-FRONT · BLOCK 56–58</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-ramp" style="border-left-color:#f5a623;border-color:rgba(245,166,35,0.35);
              background:rgba(245,166,35,0.06)">
            <div class="alert-header" style="color:#f5a623">🟡 SEVERITY: MEDIUM · MONITORING</div>
            <div class="alert-body">
                Potential 25 MW over-generation at <strong>09:15 IST</strong>. Cloud clearance earlier than forecast.
            </div>
            <div class="alert-tag" style="background:rgba(245,166,35,0.15);color:#f5a623">IRRADIANCE SPIKE · BLOCK 37</div>
        </div>
        """, unsafe_allow_html=True)

        # SHAP Explanation Card
        st.markdown('<div class="section-title" style="margin-top:.8rem"><span class="dot"></span>XAI · Forecast Explainability</div>', unsafe_allow_html=True)
        
        # Ensure there are NO leading spaces on the lines below
        shap_html = """
<div class="shap-card">
    <div class="shap-title">SHAP Breakdown · Block 58 · 14:30 IST</div>
    <div class="shap-prediction">38.2 MW</div>
    <div class="shap-row">
        <span class="shap-label">GHI (irradiance)</span>
        <div class="shap-bar-wrap"><div class="shap-bar shap-pos" style="width:78%"></div></div>
        <span class="shap-val" style="color:#00e676">+12.4</span>
    </div>
    <div class="shap-row">
        <span class="shap-label">Temp Derating</span>
        <div class="shap-bar-wrap"><div class="shap-bar shap-neg" style="width:28%"></div></div>
        <span class="shap-val" style="color:#ff4d4d">−2.3</span>
    </div>
    <div class="shap-row">
        <span class="shap-label">Cloud Proxy (sat)</span>
        <div class="shap-bar-wrap"><div class="shap-bar shap-neg" style="width:19%"></div></div>
        <span class="shap-val" style="color:#ff4d4d">−1.5</span>
    </div>
    <div class="shap-row">
        <span class="shap-label">Wind Speed</span>
        <div class="shap-bar-wrap"><div class="shap-bar shap-pos" style="width:14%"></div></div>
        <span class="shap-val" style="color:#00e676">+0.8</span>
    </div>
    <div class="shap-row">
        <span class="shap-label">Panel Soiling</span>
        <div class="shap-bar-wrap"><div class="shap-bar shap-neg" style="width:9%"></div></div>
        <span class="shap-val" style="color:#ff4d4d">−0.4</span>
    </div>
    <div style="margin-top:.75rem;padding-top:.6rem;border-top:1px solid #1e3a55; font-family:'Share Tech Mono';font-size:.65rem;color:#6b8fa8">
        BASE VALUE: 30.2 MW &nbsp;|&nbsp; BIAS: −0.6 &nbsp;|&nbsp; RESIDUAL: −0.4
    </div>
</div>
"""
        st.markdown(shap_html, unsafe_allow_html=True)

        # Asset status mini-panel
        st.markdown('<div class="section-title" style="margin-top:.8rem"><span class="dot"></span>Asset Health</div>', unsafe_allow_html=True)
        assets = [
            ("Pavagada Solar",   "🟢 ONLINE",  "#00e676"),
            ("Chitradurga Wind", "🟢 ONLINE",  "#00e676"),
            ("Sharavathi HEP",   "🟡 STANDBY", "#f5a623"),
            ("BESS Unit 1",      "🟢 ONLINE",  "#00e676"),
        ]
        for name, status, color in assets:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                         padding:.4rem .7rem;background:#111d2e;border:1px solid #1e3a55;
                         border-radius:4px;margin-bottom:5px">
                <span style="font-family:'Barlow';font-size:.82rem;color:#e8f4fd">{name}</span>
                <span style="font-family:'Share Tech Mono';font-size:.68rem;
                              color:{color};letter-spacing:.07em">{status}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — QCA Schedule
# ══════════════════════════════════════════════
with tab2:
    col_left, col_right = st.columns([2, 1], gap="medium")

    with col_left:
        st.markdown('<div class="section-title"><span class="dot"></span>KERC QCA 96-Block Daily Schedule</div>', unsafe_allow_html=True)

        generate_btn = st.button("⬇  Generate 96-Block KERC QCA CSV", use_container_width=False)

        if generate_btn or st.session_state.get("qca_generated"):
            st.session_state["qca_generated"] = True
            st.success("✅  QCA schedule generated successfully — Pavagada Solar Park · 20 Jun 2025 · Ready for KERC submission")
            st.dataframe(
                qca_df,
                use_container_width=True,
                height=420,
                hide_index=True,
                column_config={
                    "Block":          st.column_config.NumberColumn("Block", width="small"),
                    "Time Slot":      st.column_config.TextColumn("Slot", width="small"),
                    "Declared (MW)":  st.column_config.NumberColumn("Declared MW", format="%.1f"),
                    "Actual (MW)":    st.column_config.NumberColumn("Actual MW",   format="%.1f"),
                    "Deviation (MW)": st.column_config.NumberColumn("Deviation MW",format="%.1f"),
                    "DSM Charge (₹L)":st.column_config.NumberColumn("DSM ₹L",     format="%.3f"),
                    "Status":         st.column_config.TextColumn("Status"),
                },
            )
        else:
            st.markdown("""
            <div style="background:#111d2e;border:1px dashed #1e3a55;border-radius:6px;
                         padding:2.5rem;text-align:center;margin-top:1rem">
                <div style="font-size:2rem;margin-bottom:.5rem">📄</div>
                <div style="font-family:'Barlow Condensed';font-size:1rem;color:#6b8fa8;
                             letter-spacing:.1em">Click the button above to generate the 96-block QCA schedule</div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title"><span class="dot"></span>DSM Compliance Summary</div>', unsafe_allow_html=True)

        compliance_metrics = [
            ("Total Declared Energy", "4,312 MWh", "#00d4ff"),
            ("Total Actual Generation", "4,289 MWh", "#00d4ff"),
            ("Net Deviation", "−23 MWh", "#ff4d4d"),
            ("Blocks with Deviation >15MW", "8 blocks", "#f5a623"),
            ("Estimated DSM Charge", "₹1.84 Lakh", "#ff4d4d"),
            ("Compliance Score", "91.7%", "#00e676"),
        ]
        for label, val, color in compliance_metrics:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                         padding:.6rem .9rem;background:#111d2e;border:1px solid #1e3a55;
                         border-radius:4px;margin-bottom:6px">
                <span style="font-family:'Barlow';font-size:.83rem;color:#6b8fa8">{label}</span>
                <span style="font-family:'Share Tech Mono';font-size:.88rem;
                              color:{color};font-weight:600">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:1rem"><span class="dot"></span>KERC Submission Checklist</div>', unsafe_allow_html=True)
        checklist = [
            ("✅", "QCA format validation",        "PASSED"),
            ("✅", "Digital signature",             "APPLIED"),
            ("✅", "Plant identifier (GTIS code)",  "6KAR-PAV-01"),
            ("🔄", "SLDC portal upload",            "PENDING"),
            ("⏳", "Submission deadline",           "21 Jun 09:00"),
        ]
        for icon, item, status in checklist:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:.45rem .7rem;
                         font-family:'Barlow';font-size:.8rem;
                         border-bottom:1px solid #1e3a5544">
                <span>{icon} {item}</span>
                <span style="font-family:'Share Tech Mono';font-size:.72rem;color:#6b8fa8">{status}</span>
            </div>""", unsafe_allow_html=True)

        with st.expander("📋  KERC DSM Regulations Reference"):
            st.markdown("""
            <div style="font-family:'Barlow';font-size:.82rem;color:#6b8fa8;line-height:1.7">
            <b style="color:#00d4ff">KERC DSM Regulations 2014 (Amended 2021)</b><br><br>
            • <b>Tolerance band:</b> ±15% of declared schedule or ±150 MW, whichever is lower<br>
            • <b>Over-drawal surcharge:</b> ₹0.50–₹1.50/unit (zone-dependent)<br>
            • <b>Under-injection penalty:</b> ₹0.25/unit below tolerance<br>
            • <b>Frequency band:</b> 49.95–50.05 Hz (no penalty zone)<br>
            • <b>Reporting frequency:</b> 15-minute blocks (96/day)<br>
            • <b>Submission window:</b> D+1 by 09:00 IST to SLDC portal
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — Model Diagnostics
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title"><span class="dot"></span>Model Performance Metrics · Last 30 Days</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)

    perf = [
        (mc1, "RMSE",        "14.3 MW",  "#00d4ff"),
        (mc2, "MAE",         "10.8 MW",  "#00d4ff"),
        (mc3, "MAPE",        "4.2%",     "#00e676"),
        (mc4, "nRMSE",       "3.1%",     "#00e676"),
        (mc5, "Pinball Loss","8.72",     "#f5a623"),
    ]
    for col, label, val, color in perf:
        col.markdown(f"""
        <div style="background:#111d2e;border:1px solid #1e3a55;border-radius:5px;
                     padding:.9rem;text-align:center">
            <div style="font-family:'Share Tech Mono';font-size:.65rem;color:#6b8fa8;
                         letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem">{label}</div>
            <div style="font-family:'Barlow Condensed';font-weight:700;font-size:1.4rem;color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

    # Feature importance mini bar chart
    di1, di2 = st.columns([1, 1], gap="medium")

    with di1:
        st.markdown('<div class="section-title"><span class="dot"></span>Feature Importance (SHAP · Mean |φ|)</div>', unsafe_allow_html=True)
        features = {
            "GHI (NWP)":             0.92,
            "Cloud Fraction (INSAT)": 0.71,
            "Temperature":            0.44,
            "Wind Speed":             0.28,
            "Humidity":               0.19,
            "Aerosol Optical Depth":  0.14,
            "Panel Temp Derating":    0.11,
            "Soiling Index":          0.07,
        }
        feat_fig = go.Figure(go.Bar(
            x=list(features.values()),
            y=list(features.keys()),
            orientation="h",
            marker=dict(
                color=list(features.values()),
                colorscale=[[0, "#003c57"], [0.5, "#0077a8"], [1, "#00d4ff"]],
                showscale=False,
            ),
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        ))
        feat_fig.update_layout(
            plot_bgcolor="#0d1520", paper_bgcolor="#0d1520",
            height=280, margin=dict(l=4, r=4, t=4, b=4),
            xaxis=dict(gridcolor="#1e3a55", tickfont=dict(family="Share Tech Mono", size=9), zeroline=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(family="Barlow", size=11, color="#e8f4fd")),
            font=dict(color="#6b8fa8"),
        )
        st.plotly_chart(feat_fig, use_container_width=True, config={"displayModeBar": False})

    with di2:
        st.markdown('<div class="section-title"><span class="dot"></span>Model Run Log</div>', unsafe_allow_html=True)
        logs = [
            ("15:30:02", "inference", "Block 62-96 inference complete · 34ms", "#00e676"),
            ("15:15:01", "inference", "Block 58-61 inference complete · 31ms", "#00e676"),
            ("15:00:10", "alert",     "Ramp alert triggered · Pavagada · Block 56", "#ff4d4d"),
            ("14:45:00", "retrain",   "Incremental retrain with latest obs data", "#f5a623"),
            ("14:30:05", "inference", "Block 54-57 inference complete · 29ms", "#00e676"),
            ("14:15:02", "warning",   "P10/P90 band widened: ↑ cloud uncertainty", "#f5a623"),
            ("14:00:01", "inference", "Block 50-53 inference complete · 33ms", "#00e676"),
            ("13:45:00", "data",      "INSAT-3DR cloud mask refreshed", "#00d4ff"),
        ]
        for ts, typ, msg, color in logs:
            st.markdown(f"""
            <div style="display:flex;gap:.75rem;padding:.4rem .5rem;
                         border-bottom:1px solid #1e3a5533;align-items:flex-start">
                <span style="font-family:'Share Tech Mono';font-size:.68rem;
                              color:#1e3a55;min-width:60px">{ts}</span>
                <span style="font-family:'Share Tech Mono';font-size:.65rem;
                              color:{color};text-transform:uppercase;min-width:65px">{typ}</span>
                <span style="font-family:'Barlow';font-size:.8rem;color:#6b8fa8">{msg}</span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
             padding:.7rem 1rem;background:#0d1520;border:1px solid #1e3a55;
             border-radius:4px;font-family:'Share Tech Mono';font-size:.65rem;
             color:#1e3a55;flex-wrap:wrap;gap:.5rem">
    <span>OPTICAST AI &nbsp;·&nbsp; v2.1.4 &nbsp;·&nbsp; © 2025 KSPDCL Renewable Division</span>
    <span>KERC DSM REGULATIONS 2014 (AMENDED 2021) &nbsp;·&nbsp; QCA COMPLIANCE MODULE</span>
    <span style="color:#00d4ff">SYSTEM NOMINAL &nbsp;·&nbsp; ALL SERVICES OPERATIONAL</span>
</div>
""", unsafe_allow_html=True)