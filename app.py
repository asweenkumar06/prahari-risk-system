import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import os, sys

st.set_page_config(
    page_title="Industrial Accident Risk System",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #8b949e !important;
    font-size: 13px;
    padding: 6px 0;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #8b949e !important;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 20px;
}

/* Main background */
.main .block-container {
    background: #f6f8fa;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Remove default streamlit padding weirdness */
.stApp { background: #f6f8fa; }

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric-card .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 32px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-card .lbl {
    font-size: 12px;
    color: #57606a;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Risk badges */
.badge-critical { background:#ff000015; color:#cf222e; border:1px solid #ff818266; border-radius:4px; padding:2px 10px; font-size:12px; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.badge-moderate { background:#fb8f0015; color:#9a6700; border:1px solid #e3b34166; border-radius:4px; padding:2px 10px; font-size:12px; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.badge-safe     { background:#1a7f3715; color:#1a7f37; border:1px solid #56d36466; border-radius:4px; padding:2px 10px; font-size:12px; font-weight:600; font-family:'IBM Plex Mono',monospace; }

/* Factory card */
.factory-card {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-left: 4px solid #cf222e;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.factory-card.moderate { border-left-color: #9a6700; }
.factory-card.safe     { border-left-color: #1a7f37; }
.factory-card h4 { margin:0 0 4px; font-size:15px; font-weight:600; color:#24292f; }
.factory-card .meta { font-size:12px; color:#57606a; margin-bottom:10px; }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: #57606a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 24px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #d0d7de;
}

/* Page title */
.page-title {
    font-size: 24px;
    font-weight: 700;
    color: #24292f;
    margin-bottom: 4px;
}
.page-sub {
    font-size: 14px;
    color: #57606a;
    margin-bottom: 20px;
}

/* Gap flags */
.gap-flag {
    background: #fff8c5;
    border: 1px solid #d4a72c66;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
    color: #7d4e00;
    margin: 4px 0;
}
.gap-flag-critical {
    background: #ffebe9;
    border: 1px solid #ff818266;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
    color: #82071e;
    margin: 4px 0;
}

/* Recommendation card */
.rec-card {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.rec-card h4 { margin:0 0 12px; font-size:15px; font-weight:600; color:#24292f; }
.rec-item {
    font-size: 13px;
    color: #24292f;
    padding: 7px 0;
    border-bottom: 1px solid #f0f0f0;
    line-height: 1.6;
}
.rec-item:last-child { border-bottom: none; }

/* Score gauge bar */
.score-bar-wrap { background:#e8eaed; border-radius:4px; height:8px; margin:8px 0; }
.score-bar-inner { height:8px; border-radius:4px; }

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stSlider"] label { font-size:13px; font-weight:500; color:#24292f; }

.stDataFrame { border: 1px solid #d0d7de; border-radius:6px; }
</style>
""", unsafe_allow_html=True)


# ─── Data loading (cached) ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data and running risk engine…")
def load_data():
    sys.path.insert(0, os.path.dirname(__file__))
    from risk_engine import run_risk_engine
    return run_risk_engine()


factories, hospitals, fire_st, amb_depots, gaps_df, recs_df = load_data()


# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚠️ IARES")
    st.markdown("Industrial Accident Risk &\nEmergency Supply System")
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio(
        "",
        [
            "🏠  Overview Dashboard",
            "🏭  Factory Risk Explorer",
            "📦  Supply Gap Analyzer",
            "📋  Pre-Positioning Recs",
            "🎛️  Risk Trend Simulator",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Data snapshot**")
    critical_count  = int((factories["risk_label"] == "Critical").sum())
    moderate_count  = int((factories["risk_label"] == "Moderate").sum())
    safe_count      = int((factories["risk_label"] == "Safe").sum())
    st.markdown(f"""
<div style='font-size:12px;line-height:2;color:#8b949e;'>
🔴 Critical: <b style='color:#ff7b72;'>{critical_count}</b><br>
🟡 Moderate: <b style='color:#e3b341;'>{moderate_count}</b><br>
🟢 Safe: <b style='color:#56d364;'>{safe_count}</b><br>
🏭 Total: <b style='color:#c9d1d9;'>{len(factories)}</b>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
<div style='font-size:10px;color:#484f58;line-height:1.8;'>
Model: Random Forest<br>
Features: 5 risk indicators<br>
Districts: 10 (Rajasthan)<br>
Data: Synthetic (demo)
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════
if page == "🏠  Overview Dashboard":
    st.markdown('<div class="page-title">Overview Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Real-time risk snapshot across all 500 factories in 10 Rajasthan districts</div>', unsafe_allow_html=True)

    # Top metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="val" style="color:#24292f">{len(factories)}</div><div class="lbl">Total Factories</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="val" style="color:#cf222e">{critical_count}</div><div class="lbl">Critical Risk</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="val" style="color:#9a6700">{moderate_count}</div><div class="lbl">Moderate Risk</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="val" style="color:#1a7f37">{safe_count}</div><div class="lbl">Safe</div></div>', unsafe_allow_html=True)
    with c5:
        gap_count = int((gaps_df["gap_count"] > 0).sum()) if len(gaps_df) else 0
        st.markdown(f'<div class="metric-card"><div class="val" style="color:#0969da">{gap_count}</div><div class="lbl">Supply Gaps</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # District risk summary
    st.markdown('<div class="section-header">District-wise Risk Breakdown</div>', unsafe_allow_html=True)
    dist_summary = (
        factories.groupby(["district", "risk_label"])
        .size().unstack(fill_value=0)
        .reset_index()
    )
    for col in ["Safe", "Moderate", "Critical"]:
        if col not in dist_summary.columns:
            dist_summary[col] = 0

    fig = go.Figure()
    fig.add_bar(name="Safe",     x=dist_summary["district"], y=dist_summary["Safe"],     marker_color="#1a7f37")
    fig.add_bar(name="Moderate", x=dist_summary["district"], y=dist_summary["Moderate"], marker_color="#e3b341")
    fig.add_bar(name="Critical", x=dist_summary["district"], y=dist_summary["Critical"], marker_color="#cf222e")
    fig.update_layout(
        barmode="stack", height=320, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#f6f8fa", paper_bgcolor="#f6f8fa",
        font=dict(family="IBM Plex Sans", size=12, color="#24292f"),
        xaxis=dict(gridcolor="#e8eaed"), yaxis=dict(gridcolor="#e8eaed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # District cards
    cols = st.columns(5)
    for i, (_, row) in enumerate(dist_summary.iterrows()):
        total = row["Safe"] + row["Moderate"] + row["Critical"]
        crit_pct = round(row["Critical"] / total * 100) if total else 0
        color = "#cf222e" if crit_pct > 30 else ("#9a6700" if crit_pct > 15 else "#1a7f37")
        with cols[i % 5]:
            st.markdown(f"""
<div class="metric-card" style="margin-bottom:10px;">
  <div style="font-size:13px;font-weight:600;color:#24292f;margin-bottom:6px;">{row['district']}</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;color:{color};">{crit_pct}%</div>
  <div style="font-size:11px;color:#57606a;">critical rate</div>
  <div style="font-size:11px;color:#57606a;margin-top:4px;">🔴{int(row['Critical'])} 🟡{int(row['Moderate'])} 🟢{int(row['Safe'])}</div>
</div>
""", unsafe_allow_html=True)

    # Folium map
    st.markdown('<div class="section-header">Factory Risk Map — Rajasthan</div>', unsafe_allow_html=True)
    m = folium.Map(location=[26.5, 74.5], zoom_start=7, tiles="CartoDB positron")

    color_map = {"Critical": "#cf222e", "Moderate": "#e3b341", "Safe": "#1a7f37"}
    radius_map = {"Critical": 9, "Moderate": 7, "Safe": 5}

    for _, row in factories.iterrows():
        clr = color_map.get(row["risk_label"], "#888")
        rad = radius_map.get(row["risk_label"], 5)
        popup_html = f"""
        <div style='font-family:sans-serif;font-size:12px;min-width:180px'>
          <b>{row['name']}</b><br>
          District: {row['district']}<br>
          Chemical: {row['chemical_type']}<br>
          Risk Score: <b>{row['risk_score']}</b><br>
          Label: <b style='color:{clr}'>{row['risk_label']}</b><br>
          Violations: {row['violation_count']} | Complaints: {row['worker_complaints']}
        </div>"""
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=rad, color=clr, fill=True, fill_color=clr, fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{row['name']} [{row['risk_label']}]",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
         padding:12px 16px;border-radius:6px;border:1px solid #d0d7de;font-family:sans-serif;font-size:12px;'>
      <b>Risk Level</b><br>
      <span style='color:#cf222e'>●</span> Critical &nbsp;
      <span style='color:#e3b341'>●</span> Moderate &nbsp;
      <span style='color:#1a7f37'>●</span> Safe
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=None, height=500, returned_objects=[])


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — FACTORY RISK EXPLORER
# ═══════════════════════════════════════════════════════════════════
elif page == "🏭  Factory Risk Explorer":
    st.markdown('<div class="page-title">Factory Risk Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Filter and inspect individual factory risk profiles</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        district_filter = st.multiselect("District", sorted(factories["district"].unique()), default=[])
    with f2:
        risk_filter = st.multiselect("Risk Level", ["Critical", "Moderate", "Safe"], default=["Critical", "Moderate"])
    with f3:
        chem_filter = st.multiselect("Chemical Type", sorted(factories["chemical_type"].unique()), default=[])

    filtered = factories.copy()
    if district_filter: filtered = filtered[filtered["district"].isin(district_filter)]
    if risk_filter:     filtered = filtered[filtered["risk_label"].isin(risk_filter)]
    if chem_filter:     filtered = filtered[filtered["chemical_type"].isin(chem_filter)]
    filtered = filtered.sort_values("risk_score", ascending=False)

    st.markdown(f'<div class="section-header">{len(filtered)} factories matching filters</div>', unsafe_allow_html=True)

    for _, row in filtered.head(50).iterrows():
        lbl   = row["risk_label"]
        score = row["risk_score"]
        card_class = lbl.lower()
        bar_color  = "#cf222e" if lbl == "Critical" else ("#e3b341" if lbl == "Moderate" else "#1a7f37")
        badge_html = f'<span class="badge-{lbl.lower()}">{lbl}</span>'

        overflow_note = ""
        if row["storage_overflow_pct"] > 0:
            overflow_note = f"⚠️ {row['storage_overflow_pct']:.1f}% over licensed limit"
        elif row["storage_overflow_pct"] < 0:
            overflow_note = f"✅ {abs(row['storage_overflow_pct']):.1f}% under capacity"

        st.markdown(f"""
<div class="factory-card {card_class}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div>
      <h4>{row['name']}</h4>
      <div class="meta">📍 {row['district']} &nbsp;|&nbsp; ☣️ {row['chemical_type']} &nbsp;|&nbsp; ID #{int(row['factory_id'])}</div>
    </div>
    <div style="text-align:right;">{badge_html}<br>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:600;color:{bar_color};">{score}</span>
      <span style="font-size:11px;color:#57606a;">/100</span>
    </div>
  </div>
  <div class="score-bar-wrap"><div class="score-bar-inner" style="width:{score}%;background:{bar_color};"></div></div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;font-size:12px;color:#57606a;margin-top:8px;">
    <div>🚨 Violations<br><b style="color:#24292f;font-size:14px;">{int(row['violation_count'])}</b></div>
    <div>📅 Last Insp.<br><b style="color:#24292f;font-size:14px;">{int(row['months_since_inspection'])}mo</b></div>
    <div>👷 Complaints<br><b style="color:#24292f;font-size:14px;">{int(row['worker_complaints'])}</b></div>
    <div>💥 Accidents<br><b style="color:#24292f;font-size:14px;">{int(row['accident_history'])}</b></div>
    <div>🛢️ Storage<br><b style="color:{'#cf222e' if row['storage_overflow_pct']>0 else '#1a7f37'};font-size:12px;">{overflow_note}</b></div>
  </div>
</div>
""", unsafe_allow_html=True)

    if len(filtered) > 50:
        st.info(f"Showing top 50 of {len(filtered)} results. Apply more filters to narrow down.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — SUPPLY GAP ANALYZER
# ═══════════════════════════════════════════════════════════════════
elif page == "📦  Supply Gap Analyzer":
    st.markdown('<div class="page-title">Supply Gap Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Emergency resource adequacy for all Critical factories</div>', unsafe_allow_html=True)

    if gaps_df.empty:
        st.warning("No critical factories found.")
    else:
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown(f'<div class="metric-card"><div class="val" style="color:#cf222e">{len(gaps_df)}</div><div class="lbl">Critical Factories</div></div>', unsafe_allow_html=True)
        with g2:
            factories_with_gaps = int((gaps_df["gap_count"] > 0).sum())
            st.markdown(f'<div class="metric-card"><div class="val" style="color:#9a6700">{factories_with_gaps}</div><div class="lbl">Have Supply Gaps</div></div>', unsafe_allow_html=True)
        with g3:
            total_gaps = int(gaps_df["gap_count"].sum())
            st.markdown(f'<div class="metric-card"><div class="val" style="color:#0969da">{total_gaps}</div><div class="lbl">Total Gap Flags</div></div>', unsafe_allow_html=True)

        st.markdown("")
        dist_filter_g = st.selectbox("Filter by District", ["All"] + sorted(gaps_df["district"].unique().tolist()))
        show_df = gaps_df if dist_filter_g == "All" else gaps_df[gaps_df["district"] == dist_filter_g]

        st.markdown('<div class="section-header">Detailed Gap Table</div>', unsafe_allow_html=True)

        for _, row in show_df.sort_values("gap_count", ascending=False).iterrows():
            gap_items = [g for g in row["gaps"].split(" | ") if g and g != "No critical gaps"]
            gap_html = ""
            for g in gap_items:
                cls = "gap-flag-critical" if "No " in g or "insufficient" in g or "HAZMAT" in g else "gap-flag"
                gap_html += f'<div class="{cls}">⚠️ {g}</div>'
            if not gap_items:
                gap_html = '<div style="font-size:12px;color:#1a7f37;">✅ No critical supply gaps detected</div>'

            st.markdown(f"""
<div style="background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:16px 20px;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div>
      <span style="font-size:15px;font-weight:600;color:#24292f;">{row['factory_name']}</span>
      <span style="font-size:12px;color:#57606a;margin-left:10px;">📍{row['district']} | ☣️{row['chemical_type']}</span>
    </div>
    <span class="badge-critical">Score: {row['risk_score']}</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;font-size:12px;color:#57606a;margin-bottom:10px;">
    <div>🏥 <b>{row['nearest_hospital']}</b><br>{row['hospital_dist_km']}km away | {row['hospital_beds']} beds</div>
    <div>🚒 <b>{row['nearest_fire_stn']}</b><br>{row['fire_dist_km']}km away | {row['fire_foam_kl']}kL foam | HAZMAT: {'✅' if row['hazmat_available'] else '❌'}</div>
    <div>🚑 <b>{row['nearest_ambulance']}</b><br>{row['ambulance_dist_km']}km away | {row['ambulance_fleet']} vehicles</div>
  </div>
  {gap_html}
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — PRE-POSITIONING RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "📋  Pre-Positioning Recs":
    st.markdown('<div class="page-title">Pre-Positioning Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Actionable deployment instructions for every Critical factory</div>', unsafe_allow_html=True)

    if recs_df.empty:
        st.warning("No critical factories found.")
    else:
        district_filter_r = st.selectbox("Filter by District", ["All"] + sorted(recs_df["district"].unique().tolist()))
        show_recs = recs_df if district_filter_r == "All" else recs_df[recs_df["district"] == district_filter_r]
        show_recs = show_recs.sort_values("risk_score", ascending=False)

        st.markdown(f'<div class="section-header">{len(show_recs)} critical factories requiring pre-positioning</div>', unsafe_allow_html=True)

        for _, row in show_recs.iterrows():
            rec_lines = [r for r in row["recommendations"].split("\n") if r.strip()]
            rec_items_html = "".join(f'<div class="rec-item">{line}</div>' for line in rec_lines)

            st.markdown(f"""
<div class="rec-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <h4>🏭 {row['factory_name']}</h4>
    <div>
      <span class="badge-critical">CRITICAL</span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:16px;font-weight:600;color:#cf222e;margin-left:10px;">{row['risk_score']}/100</span>
    </div>
  </div>
  <div style="font-size:12px;color:#57606a;margin-bottom:12px;">
    📍 {row['district']} &nbsp;|&nbsp; ☣️ {row['chemical_type']} &nbsp;|&nbsp; ID #{int(row['factory_id'])}
  </div>
  <div style="background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px 14px;">
    <div style="font-size:11px;font-weight:600;color:#57606a;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">Action Items</div>
    {rec_items_html}
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 5 — RISK TREND SIMULATOR
# ═══════════════════════════════════════════════════════════════════
elif page == "🎛️  Risk Trend Simulator":
    st.markdown('<div class="page-title">Risk Trend Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Adjust risk parameters for any factory and see the predicted risk score update live</div>', unsafe_allow_html=True)

    from model import load_model, predict_score, score_to_label

    col_sel, col_sliders = st.columns([1, 2])

    with col_sel:
        st.markdown('<div class="section-header">Select Factory</div>', unsafe_allow_html=True)
        district_s = st.selectbox("District", sorted(factories["district"].unique()))
        factory_options = factories[factories["district"] == district_s]["name"].tolist()
        selected_name = st.selectbox("Factory", factory_options)
        frow = factories[factories["name"] == selected_name].iloc[0]

        st.markdown('<div class="section-header">Baseline Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="font-size:13px;line-height:2.2;color:#24292f;">
☣️ Chemical: <b>{frow['chemical_type']}</b><br>
🛢️ Licensed: <b>{frow['licensed_capacity_kl']}kL</b><br>
🛢️ Used: <b>{frow['storage_used_kl']}kL</b><br>
📊 Overflow: <b>{frow['storage_overflow_pct']:.1f}%</b>
</div>
""", unsafe_allow_html=True)

    with col_sliders:
        st.markdown('<div class="section-header">Adjust Risk Parameters</div>', unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            violations = st.slider("Violation Count",      0, 20,  int(frow["violation_count"]))
            complaints = st.slider("Worker Complaints",    0, 30,  int(frow["worker_complaints"]))
            accidents  = st.slider("Accident History",     0, 10,  int(frow["accident_history"]))
        with s2:
            overflow   = st.slider("Storage Overflow %", -30.0, 60.0, float(frow["storage_overflow_pct"]), step=0.5)
            months_ins = st.slider("Months Since Inspection", 1, 60, int(frow["months_since_inspection"]))

        clf, scaler = load_model()
        feature_dict = {
            "violation_count":          violations,
            "storage_overflow_pct":     overflow,
            "months_since_inspection":  months_ins,
            "worker_complaints":        complaints,
            "accident_history":         accidents,
        }
        live_score = predict_score(clf, scaler, feature_dict)
        live_label = score_to_label(live_score)
        bar_color  = "#cf222e" if live_label == "Critical" else ("#e3b341" if live_label == "Moderate" else "#1a7f37")
        badge_cls  = f"badge-{live_label.lower()}"

        st.markdown('<div class="section-header">Live Risk Score</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:#ffffff;border:1px solid #d0d7de;border-radius:8px;padding:24px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.06);">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:56px;font-weight:700;color:{bar_color};line-height:1;">{live_score}</div>
  <div style="font-size:14px;color:#57606a;margin:4px 0 12px;">out of 100</div>
  <span class="{badge_cls}" style="font-size:14px;padding:4px 18px;">{live_label}</span>
  <div class="score-bar-wrap" style="margin-top:16px;">
    <div class="score-bar-inner" style="width:{live_score}%;background:{bar_color};height:10px;border-radius:5px;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Sensitivity chart
    st.markdown('<div class="section-header">Sensitivity Analysis — How Each Factor Affects Score</div>', unsafe_allow_html=True)
    param_ranges = {
        "violation_count":         (0, 20),
        "storage_overflow_pct":    (-30, 60),
        "months_since_inspection": (1, 60),
        "worker_complaints":       (0, 30),
        "accident_history":        (0, 10),
    }
    sensitivity_data = []
    for param, (lo, hi) in param_ranges.items():
        test_vals = np.linspace(lo, hi, 20)
        scores = []
        for val in test_vals:
            fd = feature_dict.copy()
            fd[param] = val
            scores.append(predict_score(clf, scaler, fd))
        sensitivity_data.append({
            "param": param.replace("_", " ").title(),
            "delta": max(scores) - min(scores),
        })
    sens_df = pd.DataFrame(sensitivity_data).sort_values("delta", ascending=True)
    fig_s = px.bar(
        sens_df, x="delta", y="param", orientation="h",
        color="delta", color_continuous_scale=["#1a7f37", "#e3b341", "#cf222e"],
        labels={"delta": "Score Impact Range", "param": ""},
    )
    fig_s.update_layout(
        height=260, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#f6f8fa", paper_bgcolor="#f6f8fa",
        font=dict(family="IBM Plex Sans", size=12, color="#24292f"),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#e8eaed"),
        yaxis=dict(gridcolor="#e8eaed"),
    )
    st.plotly_chart(fig_s, use_container_width=True)

    # Score trajectory
    st.markdown('<div class="section-header">Score Trajectory vs Violation Count (all else equal)</div>', unsafe_allow_html=True)
    viol_vals = list(range(0, 21))
    traj_scores = []
    for v in viol_vals:
        fd = feature_dict.copy(); fd["violation_count"] = v
        traj_scores.append(predict_score(clf, scaler, fd))

    fig_t = go.Figure()
    fig_t.add_scatter(x=viol_vals, y=traj_scores, mode="lines+markers",
                      line=dict(color="#cf222e", width=2),
                      marker=dict(size=6, color="#cf222e"))
    fig_t.add_hline(y=70, line_dash="dash", line_color="#9a6700",
                    annotation_text="Critical threshold (70)", annotation_position="right")
    fig_t.add_hline(y=40, line_dash="dash", line_color="#1a7f37",
                    annotation_text="Moderate threshold (40)", annotation_position="right")
    fig_t.update_layout(
        height=260, margin=dict(l=0, r=30, t=10, b=0),
        plot_bgcolor="#f6f8fa", paper_bgcolor="#f6f8fa",
        font=dict(family="IBM Plex Sans", size=12, color="#24292f"),
        xaxis=dict(title="Violation Count", gridcolor="#e8eaed"),
        yaxis=dict(title="Risk Score", gridcolor="#e8eaed", range=[0, 100]),
    )
    st.plotly_chart(fig_t, use_container_width=True)
