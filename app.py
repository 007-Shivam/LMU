import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gender Wage Gap Analysis",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Global backgrounds ── */
.stApp, .main, .main .block-container,
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: #f5f0e8 !important;
    color: #6b0f1a;
}
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ── */

[data-testid="stSidebar"] > div:first-child {
    background-color: #ede8df !important;
    border-right: 1px solid #c8bfaf;
}
/* Sidebar text only — do NOT touch widget internals */
/* ── Sidebar collapse toggle button ── */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] {
    background-color: #ede8df !important;
    border: 1px solid #c8bfaf !important;
    color: #7a1a2a !important;
}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg {
    fill: #7a1a2a !important;
    color: #7a1a2a !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([data-baseweb]),
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: #7a1a2a !important;
}


/* Title block */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: #6b0f1a;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1.05rem;
    color: #9a4a55;
    font-weight: 300;
    margin-top: 0.3rem;
}
.chapter-badge {
    display: inline-block;
    background: #e8e0d0;
    color: #8b1a2a;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid #c8bfaf;
    margin-bottom: 0.6rem;
}

/* Metric cards */
.metric-card {
    background: #ede8df;
    border: 1px solid #c8bfaf;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-card .metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9a4a55;
    margin-bottom: 0.3rem;
}
.metric-card .metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #6b0f1a;
}
.metric-card .metric-delta {
    font-size: 0.8rem;
    margin-top: 0.2rem;
}
.metric-card .neg { color: #c0392b; }
.metric-card .pos { color: #4caf7d; }

/* Chart container */
.chart-card {
    background: #ede8df;
    border: 1px solid #c8bfaf;
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.chart-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #6b0f1a;
    margin-bottom: 0.2rem;
}
.chart-desc {
    font-size: 0.78rem;
    color: #9a4a55;
    margin-bottom: 0.8rem;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #c8bfaf;
    margin: 1.5rem 0;
}

/* Insight box */
.insight-box {
    background: #e8e0d0;
    border-left: 3px solid #8b1a2a;
    padding: 0.9rem 1.1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #7a1a2a;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Colours (matches original charts) ─────────────────────────────────────────
C_MALE   = "#4c8bbe"
C_FEMALE = "#e07b3a"
C_NEG    = "#c0392b"
C_NEG_LT = "#e57373"
C_POS    = "#4caf7d"
PLOT_BG  = "#ede8df"
GRID_CLR = "#d9d0c0"
TEXT_CLR = "#7a1a2a"

LAYOUT_BASE = dict(
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family="DM Sans", color=TEXT_CLR, size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
    yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    kde   = pd.read_csv("story_01_kde_density.csv")
    forest = pd.read_csv("story_02_forest_plot.csv")
    waterfall = pd.read_csv("story_03_waterfall_steps.csv")
    did   = pd.read_csv("story_04_did_line_chart.csv")
    dumb  = pd.read_csv("story_05_dumbbell_plot.csv")
    fore  = pd.read_csv("story_06_forecast_data.csv")
    return kde, forest, waterfall, did, dumb, fore

kde_df, forest_df, waterfall_df, did_df, dumb_df, fore_df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ Navigation")
    chapter = st.radio(
        "Jump to chapter",
        [
            "📖 Full Story",
            "1 · Inciting Incident",
            "2 · The Math",
            "3 · The Mechanism",
            "4 · Policy Experiment",
            "5 · Hidden Factors",
            "6 · The Forecast",
        ],
        index=0,
    )
    st.markdown("<hr style='border-color:#c8bfaf'>", unsafe_allow_html=True)
    st.markdown("**Dataset**")
    st.caption("NLSY79 · Graduate cohort")
    st.caption("Years: 1997 – 2023")

    show_raw = st.checkbox("Show raw data tables", value=False)

# ── Helper: section header ────────────────────────────────────────────────────
def section_header(badge, title, desc):
    st.markdown(f'<div class="chapter-badge">{badge}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{desc}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO metrics (always visible)
# ══════════════════════════════════════════════════════════════════════════════
female_coef = forest_df.loc[forest_df["term"] == "Female", "effect_pct"].values[0]
anchor_penalty = waterfall_df.iloc[2]["display_value"]
final_level = waterfall_df.iloc[3]["display_value"]

m1, m2, m3, m4 = st.columns(4)
for col, label, val, delta_cls in [
    (m1, "Gender Wage Penalty",   f"{female_coef:.1f}%", "neg"),
    (m2, "Anchoring Penalty",      f"{anchor_penalty:.1f}%", "neg"),
    (m3, "Female/Male Pay Index",  f"{final_level:.1f}",  "neg"),
    (m4, "Married Gap vs Single", "+42%", "neg"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{val}</div>
      <div class="metric-delta {delta_cls}">{'▼ significant' if delta_cls=='neg' else '▲ above'}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 1 – KDE
# ══════════════════════════════════════════════════════════════════════════════
show_1 = chapter in ("📖 Full Story", "1 · Inciting Incident")
if show_1:
    section_header("Chapter 01", "The Inciting Incident",
                   "Male vs Female log-wage density distributions reveal a persistent, unexplained gap.")

    male_df  = kde_df[kde_df["sex"] == "Male"]
    fem_df   = kde_df[kde_df["sex"] == "Female"]
    med_male = male_df["median_log_wage"].iloc[0]
    med_fem  = fem_df["median_log_wage"].iloc[0]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=male_df["x_log_wage"], y=male_df["density"],
        fill="tozeroy", fillcolor="rgba(76,139,190,0.18)",
        line=dict(color=C_MALE, width=2), name="Male"))
    fig1.add_trace(go.Scatter(
        x=fem_df["x_log_wage"], y=fem_df["density"],
        fill="tozeroy", fillcolor="rgba(224,123,58,0.18)",
        line=dict(color=C_FEMALE, width=2), name="Female"))
    for x, c, label in [(med_fem, C_FEMALE, "Female median"), (med_male, C_MALE, "Male median")]:
        fig1.add_vline(x=x, line=dict(color=c, dash="dash", width=1.5))
        fig1.add_annotation(x=x, y=0.82, text=label, showarrow=False,
                            font=dict(color=c, size=10), xanchor="center")
    fig1.add_annotation(
        x=(med_fem + med_male) / 2, y=0.78,
        text=f"Gap = {med_male - med_fem:.3f} log pts",
        showarrow=True, arrowhead=2, arrowcolor="#8b1a2a",
        font=dict(color="#6b0f1a", size=11), bgcolor="#e8e0d0",
        bordercolor="#8b1a2a", borderwidth=1, borderpad=4,
    )
    fig1.update_layout(**LAYOUT_BASE, height=380,
        xaxis_title="Log Hourly Wage", yaxis_title="Density",
        legend=dict(bgcolor="#ede8df", bordercolor="#c8bfaf", borderwidth=1))
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('<div class="insight-box">Both distributions peak near the same modal wage, yet the male distribution carries significantly more mass in the upper tail — driving a persistent median gap even after accounting for part-time workers in the lower spike.</div>', unsafe_allow_html=True)

    if show_raw:
        st.dataframe(kde_df, use_container_width=True, height=200)

# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 2 – Forest plot
# ══════════════════════════════════════════════════════════════════════════════
show_2 = chapter in ("📖 Full Story", "2 · The Math")
if show_2:
    if show_1: st.markdown("<br>", unsafe_allow_html=True)
    section_header("Chapter 02", "The Math",
                   "Clustered OLS coefficients with 95 % confidence intervals.")

    fp = forest_df.copy()
    fp["color"] = fp["effect_pct"].apply(lambda v: C_NEG if v < 0 else C_POS)

    fig2 = go.Figure()
    for _, row in fp.iterrows():
        fig2.add_shape(type="line",
            x0=row["ci_low_pct"], x1=row["ci_high_pct"],
            y0=row["label"], y1=row["label"],
            line=dict(color="#a09080", width=1.5))
    fig2.add_trace(go.Scatter(
        x=fp["effect_pct"], y=fp["label"],
        mode="markers",
        marker=dict(color=fp["color"], size=10, line=dict(color="#f5f0e8", width=1.5)),
        error_x=dict(
            type="data",
            symmetric=False,
            array=fp["ci_high_pct"] - fp["effect_pct"],
            arrayminus=fp["effect_pct"] - fp["ci_low_pct"],
            color="#a09080", thickness=1.5, width=6
        ),
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>Effect: %{x:.2f}%<extra></extra>",
    ))
    fig2.add_vline(x=0, line=dict(color="#9a4a55", dash="dash", width=1.5))
    fig2.update_layout(**LAYOUT_BASE, height=340,
        xaxis_title="Impact on Current Pay (%)",
        yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<div class="insight-box">The Female coefficient (≈ −14.3%) dwarfs all other predictors and remains statistically significant with clustered standard errors. Education delivers the largest positive premium (~+6%), while age and tenure effects are modest.</div>', unsafe_allow_html=True)

    if show_raw:
        st.dataframe(forest_df, use_container_width=True, height=200)

# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 3 – Waterfall
# ══════════════════════════════════════════════════════════════════════════════
show_3 = chapter in ("📖 Full Story", "3 · The Mechanism")
if show_3:
    if show_1 or show_2: st.markdown("<br>", unsafe_allow_html=True)
    section_header("Chapter 03", "The Mechanism",
                   "Stepwise decomposition from baseline male salary to final female salary.")

    wf = waterfall_df.copy()
    colors = wf["color"].tolist()

    bar_bases   = [0, wf.iloc[1]["bar_start"], wf.iloc[2]["bar_start"], 0]
    bar_heights = wf["display_value"].tolist()

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=wf["label"],
        y=bar_heights,
        base=bar_bases,
        marker_color=colors,
        text=[f"{v:+.1f}%" if i in [1, 2] else f"{v:.1f}" for i, v in enumerate(bar_heights)],
        textposition="outside",
        textfont=dict(color="#6b0f1a", size=12),
        hovertemplate="<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    # Dotted connector lines between bar tops
    ends = wf["bar_end"].tolist()
    for i in range(len(ends) - 1):
        fig3.add_shape(type="line",
            x0=i, x1=i + 1, y0=ends[i], y1=ends[i],
            line=dict(color="#a09080", width=1, dash="dot"))
    fig3.update_layout(**LAYOUT_BASE, height=420,
        yaxis_title="Indexed Salary Level",
        yaxis_range=[0, 115],
        showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="insight-box">Two compounding penalties carve 16% from the female wage: a structural gender discount (−14.3%) applied by employers, followed by a prior-pay anchoring effect (−1.9%) that locks earlier lower wages into future offers.</div>', unsafe_allow_html=True)

    if show_raw:
        st.dataframe(waterfall_df, use_container_width=True, height=150)

# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 4 – DiD line chart
# ══════════════════════════════════════════════════════════════════════════════
show_4 = chapter in ("📖 Full Story", "4 · Policy Experiment")
if show_4:
    if any([show_1, show_2, show_3]): st.markdown("<br>", unsafe_allow_html=True)
    section_header("Chapter 04", "The Policy Experiment",
                   "Difference-in-Differences: West Region salary-history ban vs Rest of US (1997–2023).")

    ctrl = did_df[did_df["Group"] == "Rest of US (Control)"]
    trt  = did_df[did_df["Group"] == "West Region (Treatment)"]

    fig4 = go.Figure()
    for df_, name, color in [(ctrl, "Rest of US (Control)", C_MALE),
                              (trt,  "West Region (Treatment)", C_FEMALE)]:
        fig4.add_trace(go.Scatter(
            x=df_["Interview_Year"], y=df_["avg_log_wage_growth"],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            name=name,
            hovertemplate=f"<b>{name}</b><br>Year: %{{x}}<br>Growth: %{{y:.3f}}<extra></extra>",
        ))
    fig4.add_vline(x=2018, line=dict(color="#6b0f1a", dash="dash", width=1.5))
    fig4.add_annotation(x=2018.3, y=0.155, text="2018 Policy<br>Intervention",
                        showarrow=False, font=dict(color="#6b0f1a", size=10), xanchor="left")
    # shade post-2018
    fig4.add_vrect(x0=2018, x1=did_df["Interview_Year"].max(),
                   fillcolor="rgba(91,155,213,0.06)", layer="below", line_width=0)
    fig4.update_layout(**LAYOUT_BASE, height=380,
        xaxis_title="Year", yaxis_title="Avg Log Wage Growth (Job Switchers)",
        legend=dict(bgcolor="#ede8df", bordercolor="#c8bfaf", borderwidth=1))
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('<div class="insight-box">Pre-2018 the two groups track closely (parallel trends). Post-ban, the West Region line diverges upward — consistent with salary-history bans reducing the anchoring penalty for job-switching women.</div>', unsafe_allow_html=True)

    if show_raw:
        st.dataframe(did_df, use_container_width=True, height=200)

# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 5 – Dumbbell
# ══════════════════════════════════════════════════════════════════════════════
show_5 = chapter in ("📖 Full Story", "5 · Hidden Factors")
if show_5:
    if any([show_1, show_2, show_3, show_4]): st.markdown("<br>", unsafe_allow_html=True)
    section_header("Chapter 05", "The Hidden Factors",
                   "Wage gaps by marital status — the marriage premium is starkly gendered.")

    fig5 = go.Figure()
    for _, row in dumb_df.iterrows():
        fig5.add_shape(type="line",
            x0=row["female_avg_wage"], x1=row["male_avg_wage"],
            y0=row["Marital_Group"], y1=row["Marital_Group"],
            line=dict(color="#b0a090", width=2))
        fig5.add_annotation(
            x=(row["female_avg_wage"] + row["male_avg_wage"]) / 2,
            y=row["Marital_Group"],
            text=f"Gap: ${row['gap_male_minus_female']:.2f}",
            showarrow=False, yshift=18,
            font=dict(color="#7a1a2a", size=11))

    fig5.add_trace(go.Scatter(
        x=dumb_df["female_avg_wage"], y=dumb_df["Marital_Group"],
        mode="markers", marker=dict(color=C_FEMALE, size=16),
        name="Women", hovertemplate="Women · %{y}<br>$%{x:.2f}/hr<extra></extra>"))
    fig5.add_trace(go.Scatter(
        x=dumb_df["male_avg_wage"], y=dumb_df["Marital_Group"],
        mode="markers", marker=dict(color=C_MALE, size=16),
        name="Men", hovertemplate="Men · %{y}<br>$%{x:.2f}/hr<extra></extra>"))
    fig5.update_layout(**LAYOUT_BASE, height=300,
        xaxis_title="Average Hourly Wage ($)",
        yaxis_title="",
        legend=dict(bgcolor="#ede8df", bordercolor="#c8bfaf", borderwidth=1))
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown('<div class="insight-box">Marriage widens the gap by 42% relative to single workers. Men receive a marriage premium; women experience a "motherhood penalty." This interaction is a major unobserved confounder in simple regressions.</div>', unsafe_allow_html=True)

    if show_raw:
        st.dataframe(dumb_df, use_container_width=True, height=150)

# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 6 – Forecast
# ══════════════════════════════════════════════════════════════════════════════
show_6 = chapter in ("📖 Full Story", "6 · The Forecast")
if show_6:
    if any([show_1, show_2, show_3, show_4, show_5]): st.markdown("<br>", unsafe_allow_html=True)
    section_header("Chapter 06", "The Forecast",
                   "Prophet trend extrapolation of average log wages through 2030.")

    male_f = fore_df[fore_df["sex"] == "Male"]
    fem_f  = fore_df[fore_df["sex"] == "Female"]

    fig6 = go.Figure()
    for df_, color, name in [(male_f, C_MALE, "Male"), (fem_f, C_FEMALE, "Female")]:
        hist = df_[df_["is_forecast"] == 0]
        fore = df_[df_["is_forecast"] == 1]
        # CI band
        fig6.add_trace(go.Scatter(
            x=pd.concat([fore["year"], fore["year"].iloc[::-1]]),
            y=pd.concat([fore["yhat_upper"], fore["yhat_lower"].iloc[::-1]]),
            fill="toself", fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba") if color.startswith("rgb") else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        # Historical
        fig6.add_trace(go.Scatter(
            x=hist["year"], y=hist["historical_y"],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            name=f"{name} Historical"))
        # Forecast
        fig6.add_trace(go.Scatter(
            x=fore["year"], y=fore["yhat"],
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            name=f"{name} Forecast"))

    fig6.add_vline(x=2021, line=dict(color="#9a4a55", dash="dot", width=1.5))
    fig6.add_annotation(x=2021.3, y=3.72, text="Forecast starts",
                        showarrow=False, font=dict(color="#9a4a55", size=10), xanchor="left")
    fig6.update_layout(**LAYOUT_BASE, height=400,
        xaxis_title="Year", yaxis_title="Average Log Wage",
        legend=dict(bgcolor="#ede8df", bordercolor="#c8bfaf", borderwidth=1, orientation="h", y=1.08))
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown('<div class="insight-box">Under current trends, the log-wage gap persists through 2030. The female trajectory rises but remains structurally below the male line — the gap closes marginally without policy intervention, but does not converge within the forecast horizon.</div>', unsafe_allow_html=True)

    if show_raw:
        st.dataframe(fore_df, use_container_width=True, height=200)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#9a4a55; font-size:0.75rem; font-family:DM Sans;'>
    Gender Wage Gap Analysis Dashboard · NLSY79 Graduate Cohort · Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)