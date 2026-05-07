"""
app.py — Seafood Cold-Chain Spoilage Risk Predictor
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="ColdChain Risk AI",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        color: #0ff;
        letter-spacing: -1px;
        text-shadow: 0 0 30px rgba(0,255,255,0.3);
        margin-bottom: 0;
    }
    .sub-title {
        font-family: 'DM Sans', sans-serif;
        font-weight: 300;
        color: #aaa;
        font-size: 1rem;
        margin-top: 4px;
    }
    .risk-card {
        border-radius: 16px;
        padding: 28px 32px;
        margin: 16px 0;
        text-align: center;
    }
    .risk-low    { background: linear-gradient(135deg, #003d1a, #006b2d); border: 1px solid #00ff6a44; }
    .risk-medium { background: linear-gradient(135deg, #3d2d00, #7a5800); border: 1px solid #ffd70044; }
    .risk-high   { background: linear-gradient(135deg, #3d0000, #8b0000); border: 1px solid #ff000044; }

    .risk-score {
        font-family: 'Space Mono', monospace;
        font-size: 4rem;
        font-weight: 700;
        line-height: 1;
    }
    .risk-label {
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 8px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #0ff;
        border-bottom: 1px solid #0ff3;
        padding-bottom: 8px;
        margin: 24px 0 16px;
    }
    .about-hero {
        background: linear-gradient(135deg, #0a0a1a, #0d1b2a);
        border: 1px solid #0ff3;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 24px;
    }
    .about-hero h1 {
        font-family: 'Space Mono', monospace;
        color: #0ff;
        font-size: 2rem;
        margin-bottom: 16px;
    }
    .about-hero p {
        color: #ccc;
        font-size: 1rem;
        line-height: 2;
        max-width: 650px;
        margin: 0 auto;
    }
    .about-card {
        background: #0d1b2a;
        border: 1px solid #0ff2;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 16px;
    }
    .about-card h3 {
        color: #0ff;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .about-card p {
        color: #ccc;
        line-height: 1.9;
        font-size: 0.95rem;
        margin: 0;
    }
    .dev-card {
        background: linear-gradient(135deg, #0a1628, #0d2137);
        border: 1px solid #0ff3;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin-top: 8px;
    }
    .dev-card h3 {
        color: #0ff;
        font-family: 'Space Mono', monospace;
        font-size: 1.2rem;
        margin-bottom: 4px;
    }
    .dev-card .role {
        color: #aaa;
        font-size: 0.9rem;
        margin-bottom: 20px;
        letter-spacing: 1px;
    }
    .dev-links a {
        display: inline-block;
        margin: 6px 5px;
        padding: 10px 22px;
        border-radius: 8px;
        background: #0ff1;
        border: 1px solid #0ff4;
        color: #0ff !important;
        text-decoration: none;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# Load Model
@st.cache_resource
def load_model():
    try:
        with open("model/linear_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("model/features.pkl", "rb") as f:
            features = pickle.load(f)
        return model, scaler, features
    except FileNotFoundError:
        return None, None, None


model, scaler, MODEL_FEATURES = load_model()


# Helper Functions
def get_risk_level(score):
    if score < 30:
        return "LOW", "🟢", "risk-low", "#00ff6a"
    elif score < 60:
        return "MEDIUM", "🟡", "risk-medium", "#ffd700"
    else:
        return "HIGH", "🔴", "risk-high", "#ff4444"


def engineer_features(inputs: dict) -> dict:
    d = inputs.copy()
    d['Temp_Abuse_Index'] = d['Avg_Storage_Temp_C'] * d['Temp_Excursion_Hours']
    d['Cooling_Protection_Ratio'] = (
        d['Packaging_Quality_Score'] + d['Ice_Replacement_Count']
    ) / max(d['Transit_Duration_Hours'], 0.1)
    d['Load_Stress_Index'] = (d['Vehicle_Load_Pct'] * d['Transit_Duration_Hours']) / 100
    return d


def predict_risk(inputs: dict):
    features_engineered = engineer_features(inputs)
    row = pd.DataFrame([{f: features_engineered[f] for f in MODEL_FEATURES}])
    row_scaled = scaler.transform(row)
    score = model.predict(row_scaled)[0]
    return float(np.clip(score, 0, 100))


# Header
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<div class="main-title">🐟 ColdChain Risk AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Seafood Spoilage Prediction System — Cold-Chain Logistics Intelligence</div>', unsafe_allow_html=True)

st.divider()

if model is None:
    st.error("Model not found! Run: python train.py")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown('<div class="section-header">Environmental Conditions</div>', unsafe_allow_html=True)
    avg_storage_temp = st.slider("Avg Storage Temperature (C)", -5.0, 25.0, 4.0, 0.5)
    ambient_temp     = st.slider("Ambient Temperature (C)", 10.0, 45.0, 28.0, 0.5)
    humidity         = st.slider("Relative Humidity (%)", 20.0, 100.0, 70.0, 1.0)
    temp_excursion   = st.slider("Temp Excursion Hours", 0.0, 24.0, 2.0, 0.5)

    st.markdown('<div class="section-header">Logistics Factors</div>', unsafe_allow_html=True)
    transit_duration = st.slider("Transit Duration (Hours)", 1.0, 72.0, 12.0, 0.5)
    distance_km      = st.slider("Distance (KM)", 10.0, 1000.0, 150.0, 10.0)
    fuel_use         = st.slider("Fuel Used (Liters)", 5.0, 200.0, 40.0, 1.0)
    vehicle_load     = st.slider("Vehicle Load (%)", 10.0, 100.0, 70.0, 1.0)

    st.markdown('<div class="section-header">Operational Factors</div>', unsafe_allow_html=True)
    packaging_quality = st.slider("Packaging Quality Score (0-10)", 0.0, 10.0, 7.0, 0.5)
    door_opens        = st.number_input("Door Open Events", min_value=0, max_value=50, value=5)
    ice_replacement   = st.number_input("Ice Replacement Count", min_value=0, max_value=20, value=2)
    hygiene_score     = st.slider("Hygiene Inspection Score (0-10)", 0.0, 10.0, 7.5, 0.5)

    st.button("Predict Spoilage Risk", type="primary", use_container_width=True)


inputs = {
    'Avg_Storage_Temp_C':       avg_storage_temp,
    'Temp_Excursion_Hours':     temp_excursion,
    'Relative_Humidity_Pct':    humidity,
    'Transit_Duration_Hours':   transit_duration,
    'Distance_KM':              distance_km,
    'Packaging_Quality_Score':  packaging_quality,
    'Vehicle_Load_Pct':         vehicle_load,
    'Door_Open_Events':         float(door_opens),
    'Ice_Replacement_Count':    float(ice_replacement),
    'Ambient_Temp_C':           ambient_temp,
    'Inspection_Hygiene_Score': hygiene_score,
    'Fuel_Use_Liters':          fuel_use,
}

score = predict_risk(inputs)
level, emoji, css_class, color = get_risk_level(score)

tab1, tab2, tab3 = st.tabs(["Prediction", "Feature Analysis", "About"])

with tab1:
    col_result, col_gauge = st.columns([1, 1])

    with col_result:
        st.markdown(f"""
        <div class="risk-card {css_class}">
            <div class="risk-score" style="color:{color}">{score:.1f}</div>
            <div class="risk-label" style="color:{color}">{emoji} {level} RISK</div>
            <div style="color:#aaa; font-size:0.85rem; margin-top:12px;">Spoilage Risk Score / 100</div>
        </div>
        """, unsafe_allow_html=True)

        fe = engineer_features(inputs)
        st.markdown('<div class="section-header">Derived Risk Indicators</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Temp Abuse Index",  f"{fe['Temp_Abuse_Index']:.1f}")
        m2.metric("Cooling Ratio",     f"{fe['Cooling_Protection_Ratio']:.2f}")
        m3.metric("Load Stress",       f"{fe['Load_Stress_Index']:.1f}")

        st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
        recs = []
        if avg_storage_temp > 6:
            recs.append("Lower storage temperature — current temp is above safe threshold (>6C)")
        if temp_excursion > 4:
            recs.append("Reduce temperature excursions — prolonged exposure increases spoilage rapidly")
        if door_opens > 8:
            recs.append("Minimize door openings — each opening disrupts internal temperature")
        if transit_duration > 24:
            recs.append("Optimize route — long transit duration compounds risk factors")
        if packaging_quality < 5:
            recs.append("Upgrade packaging — low quality provides insufficient thermal protection")
        if not recs:
            recs.append("Shipment conditions look good! Maintain current standards.")
        for r in recs:
            st.markdown(r)

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 16, 'color': '#ccc'}},
            number={'font': {'size': 48, 'color': color}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#555'},
                'bar': {'color': color, 'thickness': 0.25},
                'bgcolor': '#111',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30],   'color': '#003d1a'},
                    {'range': [30, 60],  'color': '#3d2d00'},
                    {'range': [60, 100], 'color': '#3d0000'},
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 3},
                    'thickness': 0.8,
                    'value': score
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=320,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("""
        | Zone | Range | Action |
        |------|-------|--------|
        | Low | 0-30 | Normal shipment |
        | Medium | 30-60 | Monitor closely |
        | High | 60-100 | Intervention needed |
        """)

with tab2:
    st.markdown('<div class="section-header">How Each Factor Affects Risk</div>', unsafe_allow_html=True)

    feature_labels = {
        'Avg_Storage_Temp_C':       'Storage Temp',
        'Temp_Excursion_Hours':     'Temp Excursion',
        'Relative_Humidity_Pct':    'Humidity',
        'Transit_Duration_Hours':   'Transit Duration',
        'Distance_KM':              'Distance',
        'Packaging_Quality_Score':  'Packaging Quality',
        'Vehicle_Load_Pct':         'Vehicle Load',
        'Door_Open_Events':         'Door Openings',
        'Ice_Replacement_Count':    'Ice Replacements',
        'Ambient_Temp_C':           'Ambient Temp',
        'Inspection_Hygiene_Score': 'Hygiene Score',
        'Fuel_Use_Liters':          'Fuel Used',
    }

    sensitivity = []
    for feat, label in feature_labels.items():
        base_val = inputs[feat]
        delta = max(abs(base_val) * 0.3, 1.0)
        score_up = predict_risk({**inputs, feat: base_val + delta})
        score_dn = predict_risk({**inputs, feat: base_val - delta})
        sensitivity.append({'Feature': label, 'Impact': round(score_up - score_dn, 2)})

    sens_df = pd.DataFrame(sensitivity).sort_values('Impact', ascending=True)

    fig_sens = px.bar(
        sens_df, x='Impact', y='Feature', orientation='h',
        color='Impact',
        color_continuous_scale=['#00ff6a', '#ffd700', '#ff4444'],
        title='Feature Sensitivity (higher = more impact on risk)',
    )
    fig_sens.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,30,1)',
        font=dict(color='#ccc'),
        height=420,
        coloraxis_showscale=False,
        margin=dict(t=50, b=20)
    )
    fig_sens.update_xaxes(gridcolor='#2a2a4a', title='Risk Impact (points)')
    fig_sens.update_yaxes(gridcolor='#2a2a4a', title='')
    st.plotly_chart(fig_sens, use_container_width=True)

    st.markdown('<div class="section-header">Current Input Summary</div>', unsafe_allow_html=True)
    summary_data = {k: [round(v, 2)] for k, v in inputs.items()}
    st.dataframe(pd.DataFrame(summary_data).T.rename(columns={0: 'Value'}), use_container_width=True)

with tab3:
    st.markdown("""
    <div class="about-hero">
        <h1>🐟 ColdChain Risk AI</h1>
        <p>
            An intelligent prediction system designed to protect seafood quality during transportation.
            <br><br>
            By analyzing shipment conditions in real-time, it detects potential spoilage risks before they happen.
            <br><br>
            Helping logistics teams make faster, smarter decisions and reduce product loss across the cold-chain.
        </p>
    </div>

    <div class="about-card">
        <h3>🌡️ How It Works</h3>
        <p>
            Enter your shipment conditions on the left panel — temperature, transit duration,
            packaging quality, and more. The system instantly calculates a
            <strong style="color:#0ff">Spoilage Risk Score</strong> from 0 to 100,
            giving you a clear signal:
            <strong style="color:#00ff6a">Low</strong>,
            <strong style="color:#ffd700">Medium</strong>, or
            <strong style="color:#ff4444">High</strong> risk.
        </p>
    </div>

    <div class="about-card">
        <h3>💡 Why It Matters</h3>
        <p>
            Spoilage in cold-chain logistics causes massive financial losses every year.
            Early detection means early action — saving products, reducing waste, and protecting revenue.
        </p>
    </div>

    <div class="dev-card">
        <h3>👨‍💻 Eslam Hassan Abdelzaher</h3>
        <p class="role">AI Engineer</p>
        <div class="dev-links">
            <a href="https://github.com/eh1322005" target="_blank">⚡ GitHub</a>
            <a href="https://www.linkedin.com/feed/" target="_blank">💼 LinkedIn</a>
            <a href="https://wa.me/201020121479" target="_blank">📱 WhatsApp</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
