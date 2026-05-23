"""
Streamlit dashboard for ChurnRadar.

Single-page layout, dark theme, custom CSS for a modern look. The dashboard
is read-only on top of the artifacts produced by main.py:

    python main.py            # produces best_model.joblib and reports/*
    streamlit run dashboard.py

If you just want to poke at it without retraining, the page degrades
gracefully when artifacts are missing and tells you what to run.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import data_loader, economic_impact, features, preprocessing, segmentation


# ---------- Page config + custom styling ----------

st.set_page_config(
    page_title="ChurnRadar",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A compact, deliberate stylesheet. Goals: airy spacing, soft elevated cards,
# numeric metrics that read like a finance dashboard rather than a Jupyter
# notebook. The colour story is purple-on-charcoal with red as the risk accent.
st.markdown(
    """
    <style>
    /* Base */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    h1, h2, h3, h4 {
        letter-spacing: -0.01em;
    }
    /* Hide the default Streamlit chrome that adds noise */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Hero band */
    .hero {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d1f4f 60%, #4a2160 100%);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 28px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(124, 92, 255, 0.12);
    }
    .hero h1 {
        margin: 0 0 6px 0;
        font-size: 28px;
        font-weight: 700;
        color: #f5f5fa;
    }
    .hero p {
        margin: 0;
        color: #a8aac0;
        font-size: 14px;
    }
    .hero .tag {
        display: inline-block;
        margin-top: 14px;
        padding: 4px 12px;
        background: rgba(124, 92, 255, 0.18);
        border: 1px solid rgba(124, 92, 255, 0.4);
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
        color: #c8b8ff;
    }

    /* KPI cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 16px;
        margin-bottom: 28px;
    }
    .kpi-card {
        background: #161b22;
        border: 1px solid #232936;
        border-radius: 14px;
        padding: 18px 20px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #3a3f4b;
    }
    .kpi-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8b8f9c;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #f5f5fa;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 12px;
        color: #6b7080;
        margin-top: 4px;
    }
    .kpi-card.risk .kpi-value { color: #ff5470; }
    .kpi-card.ok   .kpi-value { color: #4ade80; }
    .kpi-card.warn .kpi-value { color: #fbbf24; }

    /* Section header */
    .section-h {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b8f9c;
        margin: 28px 0 12px 0;
    }

    /* Risk pill */
    .pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
    }
    .pill.high { background: rgba(255,84,112,0.15); color: #ff5470; border: 1px solid rgba(255,84,112,0.3); }
    .pill.med  { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
    .pill.low  { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        padding: 10px 18px;
        color: #8b8f9c;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124, 92, 255, 0.15);
        color: #c8b8ff;
    }

    /* Tighten dataframes */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Plotly theme helpers ----------

PLOT_BG = "#161b22"
PAPER_BG = "#0e1117"
GRID = "#232936"
ACCENT = "#7c5cff"
DANGER = "#ff5470"
OK = "#4ade80"
SEQ = ["#7c5cff", "#a78bfa", "#c4b5fd", "#fbbf24", "#fb923c", "#ff5470"]


def _style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color="#e6e8ee", family="Inter, Segoe UI, sans-serif", size=13),
        margin=dict(l=20, r=20, t=40, b=40),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    if height:
        fig.update_layout(height=height)
    return fig


# ---------- Data loading (cached) ----------

@st.cache_data(show_spinner=False)
def _load_pipeline():
    df = data_loader.load_data()
    df = preprocessing.clean(df)
    df = features.add_features(df)
    seg = segmentation.segment_customers(df, n_clusters=4)
    df = df.assign(segment=seg.labels)
    return df, seg


@st.cache_resource(show_spinner=False)
def _load_model(path: str = "best_model.joblib"):
    if not Path(path).exists():
        return None
    return joblib.load(path)


def _score_all(model, df: pd.DataFrame) -> np.ndarray:
    encoded = preprocessing.encode_for_model(df.drop(columns=["AgeGroup", "segment"]))
    encoded = encoded.drop(columns=["Exited"], errors="ignore")
    return model.predict_proba(encoded)[:, 1]


# ---------- Hero ----------

st.markdown(
    """
    <div class="hero">
        <h1>🏦 ChurnRadar</h1>
        <p>Retail banking customer intelligence. Predictive churn, behavioural segments, and revenue at risk.</p>
        <span class="tag">Kaggle · Churn Modelling · 10,000 customers</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Boot the pipeline
with st.spinner("Loading data and segmenting customers..."):
    df, seg_result = _load_pipeline()
    bundle = _load_model()

if bundle is None:
    st.error(
        "No trained model on disk yet. Run **`python main.py`** first to "
        "fit a model and write the artifacts, then come back here."
    )
    st.stop()

model = bundle["model"]
model_name = bundle["name"]
probas = _score_all(model, df)
risk = economic_impact.summarise_risk(
    probas, segments=seg_result.labels, high_risk_threshold=0.5
)


# ---------- KPI strip ----------

avg_p = float(probas.mean())
hist_rate = float(df["Exited"].mean())
high_share = risk.high_risk_count / len(df)

st.markdown(
    f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Customers</div>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-sub">across France, Germany, Spain</div>
        </div>
        <div class="kpi-card warn">
            <div class="kpi-label">Historical churn</div>
            <div class="kpi-value">{hist_rate:.1%}</div>
            <div class="kpi-sub">share who actually left</div>
        </div>
        <div class="kpi-card risk">
            <div class="kpi-label">Revenue at risk</div>
            <div class="kpi-value">{economic_impact.format_currency(risk.total_expected_loss)}</div>
            <div class="kpi-sub">{risk.high_risk_count:,} customers in high-risk band</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Model in use</div>
            <div class="kpi-value">{model_name.replace('_', ' ').title()}</div>
            <div class="kpi-sub">avg predicted P = {avg_p:.1%}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Tabs ----------

overview_tab, segment_tab, lookup_tab, about_tab = st.tabs(
    ["📊 Overview", "👥 Segments", "🔎 Customer lookup", "ℹ️ About"]
)


# =================== OVERVIEW ===================
with overview_tab:
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown('<div class="section-h">Where the risk is concentrated</div>',
                    unsafe_allow_html=True)

        by_seg = risk.by_segment.reset_index().merge(
            seg_result.profile[["label"]].reset_index(),
            on="segment", how="left",
        )
        by_seg["display"] = by_seg["label"].fillna(
            by_seg["segment"].astype(str)
        ) + "  ·  #" + by_seg["segment"].astype(str)
        by_seg = by_seg.sort_values("expected_loss", ascending=True)

        fig = go.Figure(go.Bar(
            x=by_seg["expected_loss"],
            y=by_seg["display"],
            orientation="h",
            marker=dict(
                color=by_seg["avg_probability"],
                colorscale=[[0, "#3b82f6"], [0.5, "#a855f7"], [1, "#ff5470"]],
                showscale=True,
                colorbar=dict(
                    title=dict(text="P(churn)", side="right"),
                    thickness=12, len=0.7,
                ),
            ),
            hovertemplate="<b>%{y}</b><br>"
                          "Expected loss: $%{x:,.0f}<br>"
                          "Avg probability: %{marker.color:.1%}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Expected loss (USD)",
            yaxis_title="",
            title="Segment risk ranking",
        )
        st.plotly_chart(_style_fig(fig, height=380), use_container_width=True)

    with right:
        st.markdown('<div class="section-h">Churn probability distribution</div>',
                    unsafe_allow_html=True)

        # Histogram of predicted probabilities. Helps the user see whether
        # the model produces a confident bimodal split or a long tail.
        fig = go.Figure(go.Histogram(
            x=probas, nbinsx=40,
            marker=dict(color=ACCENT, line=dict(color=PLOT_BG, width=1)),
        ))
        fig.add_vline(
            x=0.5, line_color=DANGER, line_dash="dash",
            annotation_text="High-risk threshold",
            annotation_font_color=DANGER,
            annotation_position="top",
        )
        fig.update_layout(
            xaxis_title="Predicted P(churn)",
            yaxis_title="Customers",
            showlegend=False,
        )
        st.plotly_chart(_style_fig(fig, height=380), use_container_width=True)

    # Second row: geography vs age churn rate
    st.markdown('<div class="section-h">Churn by geography and age band</div>',
                unsafe_allow_html=True)
    g1, g2 = st.columns(2)

    with g1:
        geo = df.groupby("Geography")["Exited"].mean().reset_index()
        geo["Exited"] *= 100
        geo = geo.sort_values("Exited", ascending=False)
        fig = px.bar(
            geo, x="Geography", y="Exited",
            color="Exited", color_continuous_scale=["#3b82f6", "#a855f7", "#ff5470"],
            labels={"Exited": "Churn rate (%)"},
        )
        fig.update_traces(
            text=geo["Exited"].round(1).astype(str) + "%",
            textposition="outside", cliponaxis=False,
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(_style_fig(fig, height=340), use_container_width=True)

    with g2:
        if "AgeGroup" in df.columns:
            age = (
                df.groupby("AgeGroup", observed=True)["Exited"].mean().reset_index()
            )
            age["Exited"] *= 100
            fig = px.bar(
                age, x="AgeGroup", y="Exited",
                color="Exited", color_continuous_scale=["#3b82f6", "#a855f7", "#ff5470"],
                labels={"Exited": "Churn rate (%)", "AgeGroup": "Age band"},
            )
            fig.update_traces(
                text=age["Exited"].round(1).astype(str) + "%",
                textposition="outside", cliponaxis=False,
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(_style_fig(fig, height=340), use_container_width=True)


# =================== SEGMENTS ===================
with segment_tab:
    st.markdown('<div class="section-h">Behavioural segments</div>',
                unsafe_allow_html=True)
    st.write(
        "KMeans groups customers into four behavioural clusters. Labels are "
        "generated from centroid statistics, so sanity-check them against the "
        "raw numbers before acting."
    )

    profile = seg_result.profile.copy()
    if "churn_rate" in profile.columns:
        profile = profile.rename(columns={"churn_rate": "churn_rate_%"})

    st.dataframe(
        profile.style
            .format({
                "avg_age": "{:.1f}",
                "avg_balance": "€{:,.0f}",
                "avg_salary": "€{:,.0f}",
                "avg_products": "{:.1f}",
                "active_share": "{:.0%}",
                "churn_rate_%": "{:.1f}%",
            })
            .background_gradient(subset=["churn_rate_%"], cmap="Reds")
            .background_gradient(subset=["avg_balance"], cmap="Purples"),
        use_container_width=True,
    )

    st.markdown('<div class="section-h">Historical vs predicted churn by segment</div>',
                unsafe_allow_html=True)

    rate_df = (
        df.assign(probability=probas)
        .groupby("segment")
        .agg(
            customers=("Age", "size"),
            historical=("Exited", "mean"),
            predicted=("probability", "mean"),
        )
        .reset_index()
    )
    rate_df[["historical", "predicted"]] *= 100
    rate_df = rate_df.merge(
        seg_result.profile[["label"]].reset_index(), on="segment", how="left"
    )
    rate_df["display"] = rate_df["label"] + "  ·  #" + rate_df["segment"].astype(str)

    fig = go.Figure()
    fig.add_bar(
        name="Historical", x=rate_df["display"], y=rate_df["historical"],
        marker_color="#a78bfa",
        hovertemplate="Historical: %{y:.1f}%<extra></extra>",
    )
    fig.add_bar(
        name="Predicted (avg)", x=rate_df["display"], y=rate_df["predicted"],
        marker_color="#ff5470",
        hovertemplate="Predicted avg: %{y:.1f}%<extra></extra>",
    )
    fig.update_layout(
        barmode="group", yaxis_title="Churn rate (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(_style_fig(fig, height=380), use_container_width=True)


# =================== CUSTOMER LOOKUP ===================
with lookup_tab:
    st.markdown('<div class="section-h">Single-customer score</div>',
                unsafe_allow_html=True)
    st.write("Enter customer attributes on the left to score them in real time.")

    left, right = st.columns([1, 1.2])

    with left:
        c1, c2 = st.columns(2)
        with c1:
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.slider("Age", 18, 92, 38)
            credit_score = st.slider("Credit score", 350, 850, 650)
        with c2:
            tenure = st.slider("Tenure (years)", 0, 10, 5)
            num_products = st.selectbox("Products held", [1, 2, 3, 4])
            has_card = st.checkbox("Has credit card", True)
            is_active = st.checkbox("Active member", True)

        balance = st.number_input(
            "Balance (EUR)", 0.0, 300_000.0, 75_000.0, step=1000.0
        )
        salary = st.number_input(
            "Estimated salary (EUR)", 0.0, 300_000.0, 100_000.0, step=1000.0
        )

    # Build a one-row frame matching training schema
    row = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_card),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": salary,
        "Exited": 0,
    }])
    row = features.add_features(row)
    encoded = preprocessing.encode_for_model(row.drop(columns=["AgeGroup"]))
    encoded = encoded.drop(columns=["Exited"], errors="ignore")

    training_cols = list(
        preprocessing.encode_for_model(
            df.drop(columns=["AgeGroup", "segment"])
        ).drop(columns=["Exited"], errors="ignore").columns
    )
    for c in training_cols:
        if c not in encoded.columns:
            encoded[c] = 0
    encoded = encoded[training_cols]

    p = float(model.predict_proba(encoded)[0, 1])
    expected = economic_impact.expected_loss_per_customer(np.array([p]))[0]

    with right:
        # Gauge chart for the probability
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number=dict(suffix="%", font=dict(size=42, color="#f5f5fa")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#3a3f4b"),
                bar=dict(color=ACCENT, thickness=0.3),
                bgcolor=PLOT_BG,
                borderwidth=0,
                steps=[
                    dict(range=[0, 25], color="#1e3a2f"),
                    dict(range=[25, 50], color="#3a2f1e"),
                    dict(range=[50, 100], color="#3a1e2a"),
                ],
                threshold=dict(
                    line=dict(color="white", width=3),
                    thickness=0.75, value=50,
                ),
            ),
            title=dict(text="Predicted churn probability", font=dict(size=14, color="#8b8f9c")),
        ))
        st.plotly_chart(_style_fig(gauge, height=320), use_container_width=True)

        # Risk pill + dollar figure
        if p >= 0.5:
            pill_html = '<span class="pill high">HIGH RISK</span>'
            note = "Worth a proactive retention contact."
        elif p >= 0.25:
            pill_html = '<span class="pill med">MODERATE RISK</span>'
            note = "Watch for further signal before acting."
        else:
            pill_html = '<span class="pill low">LOW RISK</span>'
            note = "Customer profile looks stable."

        st.markdown(
            f"""
            <div class="kpi-card" style="margin-top:8px">
                <div class="kpi-label">Risk assessment</div>
                <div style="margin: 8px 0">{pill_html}</div>
                <div class="kpi-value" style="font-size:22px">
                    {economic_impact.format_currency(expected)} expected loss
                </div>
                <div class="kpi-sub">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =================== ABOUT ===================
with about_tab:
    st.markdown('<div class="section-h">About this project</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
**ChurnRadar** is a small portfolio project. It pairs a churn classifier with
KMeans customer segments, and then translates the model output into a rough
revenue-at-risk number that a non-technical reader can react to. Everything
runs on top of the Kaggle *Churn Modelling* dataset (10,000 customers from
a European bank across France, Germany and Spain).

Three models get trained and compared on each pipeline run: logistic
regression, random forest, and gradient boosting. The dashboard uses
whichever one scored highest on ROC-AUC during the most recent run.

The segmentation is KMeans with k=4. Labels are heuristic, generated from
the centroid statistics, and stable enough between runs to be worth showing
in a dashboard. The economic layer is a single multiplication: predicted
churn probability times an assumed customer lifetime value of \\$2,000.
That CLV assumption is the obvious place to attack the numbers if you want
to make them more realistic.

The dollar figures here are illustrative. The point of the project is the
method, not the magnitudes. A longer narrative lives in
`reports/findings.md` in the project repo.
        """
    )
