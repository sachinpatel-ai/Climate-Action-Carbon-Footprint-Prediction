import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import ast
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Carbon Emission Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.5rem; margin: 0; }
    .main-header p  { font-size: 1.1rem; opacity: 0.9; margin: 0.5rem 0 0; }

    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-low    { background: #d4edda; border: 2px solid #28a745; color: #155724; }
    .result-medium { background: #fff3cd; border: 2px solid #ffc107; color: #856404; }
    .result-high   { background: #f8d7da; border: 2px solid #dc3545; color: #721c24; }

    .tip-card {
        background: #f0f7f0;
        border-left: 4px solid #40916c;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.95rem;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d6a4f;
        margin: 1.2rem 0 0.5rem;
        border-bottom: 2px solid #d4edda;
        padding-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_PATH    = "carbon_model.joblib"
ENCODERS_PATH = "carbon_encoders.joblib"

BODY_TYPES      = ["normal", "overweight", "obese", "underweight"]
SEX_TYPES       = ["female", "male"]
DIET_TYPES      = ["omnivore", "vegetarian", "vegan", "pescatarian"]
SHOWER_TYPES    = ["daily", "less frequently", "more frequently", "twice a day"]
HEATING_TYPES   = ["coal", "natural gas", "wood", "electricity"]
TRANSPORT_TYPES = ["private", "public", "walk/bicycle"]
VEHICLE_TYPES   = ["petrol", "diesel", "hybrid", "lpg", "electric", "non vehicle"]
ACTIVITY_TYPES  = ["often", "never", "sometimes"]
AIR_FREQ_TYPES  = ["frequently", "rarely", "never", "very frequently"]
WASTE_SIZES     = ["large", "extra large", "small", "medium"]
ENERGY_EFF      = ["No", "Sometimes", "Yes"]
RECYCLING_OPTS  = ["Paper", "Plastic", "Glass", "Metal"]
COOKING_OPTS    = ["Stove", "Oven", "Microwave", "Grill", "Airfryer"]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def clean_list_column(col):
    def clean_value(x):
        try:
            items = ast.literal_eval(x)
            items = [i.strip().lower() for i in items]
            return "_".join(sorted(items))
        except:
            return str(x).lower()
    return col.apply(clean_value)

def encode_multiselect(selected_list):
    cleaned = [i.strip().lower() for i in selected_list]
    return "_".join(sorted(cleaned)) if cleaned else "none"

def get_emission_level(emission):
    if emission < 6000:
        return "LOW", "result-low", "🌱"
    elif emission < 12000:
        return "MEDIUM", "result-medium", "⚠️"
    else:
        return "HIGH", "result-high", "🔴"

def generate_suggestions(user, emission):
    level, _, _ = get_emission_level(emission)
    tips = []
    if level == "HIGH":
        tips.append("Your carbon footprint is HIGH. Major lifestyle changes are recommended.")
    elif level == "MEDIUM":
        tips.append("Your carbon footprint is MODERATE. Some improvements can help significantly.")
    else:
        tips.append("Great! Your carbon footprint is LOW. Keep it up!")

    if user.get("Transport") == "private":
        tips.append("🚌 Use public transport, carpool, or cycling to reduce transport emissions.")
    if user.get("Vehicle Type") in ["petrol", "diesel"]:
        tips.append("🚗 Consider switching to an electric or hybrid vehicle.")
    if user.get("Vehicle Monthly Distance Km", 0) > 300:
        tips.append("📍 Try reducing monthly travel distance or use remote-work options.")
    if user.get("Frequency of Traveling by Air") in ["frequently", "very frequently"]:
        tips.append("✈️ Air travel has high emissions. Try reducing flights or offset your carbon.")
    if user.get("Diet") == "omnivore":
        tips.append("🥗 Reduce meat consumption. A plant-based diet can cut emissions by up to 50%.")
    if user.get("How Many New Clothes Monthly", 0) > 10:
        tips.append("👕 Fast fashion increases emissions. Buy fewer, higher-quality clothes.")
    if user.get("How Long TV PC Daily Hour", 0) > 8:
        tips.append("💻 Reduce screen time to save electricity.")
    if str(user.get("Energy efficiency", "")).lower() == "no":
        tips.append("⚡ Use energy-efficient appliances (LED bulbs, inverter AC, star-rated devices).")
    if user.get("Waste Bag Weekly Count", 0) > 4:
        tips.append("🗑️ Try reducing household waste and compost organic material.")
    if "microwave" not in str(user.get("Cooking_With", "")).lower():
        tips.append("🍳 Microwaves and air-fryers use less energy than stoves/ovens.")
    return tips

# ─────────────────────────────────────────────
# Model Training / Loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading / training model…")
def load_or_train_model(csv_path=None):
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        model    = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        return model, encoders, None, None

    if csv_path is None:
        return None, None, None, None

    df = pd.read_csv(csv_path)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.lower().str.strip()

    df["Vehicle Type"] = df["Vehicle Type"].fillna("non vehicle")
    df["Cooking_With"] = clean_list_column(df["Cooking_With"])
    df["Recycling"]    = clean_list_column(df["Recycling"])

    X = df.drop("CarbonEmission", axis=1)
    y = df["CarbonEmission"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    encoders = {}
    for col in X_train.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col]  = le.transform(X_test[col])
        encoders[col] = le

    model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    joblib.dump(model,    MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)

    return model, encoders, r2, mae

def predict_emission(model, encoders, feature_columns, user_data):
    df_user = pd.DataFrame([user_data])[feature_columns]
    for col in encoders:
        df_user[col] = df_user[col].astype(str).str.lower().str.strip()
        known = list(encoders[col].classes_)
        df_user[col] = df_user[col].apply(lambda x: x if x in known else known[0])
        df_user[col] = encoders[col].transform(df_user[col])
    return model.predict(df_user)[0]

FEATURE_COLUMNS = [
    "Body Type", "Sex", "Diet", "How Often Shower",
    "Heating Energy Source", "Transport", "Vehicle Type",
    "Social Activity", "Monthly Grocery Bill",
    "Frequency of Traveling by Air", "Vehicle Monthly Distance Km",
    "Waste Bag Size", "Waste Bag Weekly Count",
    "How Long TV PC Daily Hour", "How Many New Clothes Monthly",
    "How Long Internet Daily Hour", "Energy efficiency",
    "Recycling", "Cooking_With"
]

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌿 Carbon Emission Predictor</h1>
    <p>Estimate your annual carbon footprint and get personalised eco-tips</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar – model setup
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Setup")

    model_ready = os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH)
    if model_ready:
        st.success("✅ Model loaded from disk")
        if st.button("🔄 Retrain Model", use_container_width=True):
            for f in [MODEL_PATH, ENCODERS_PATH]:
                if os.path.exists(f):
                    os.remove(f)
            st.cache_resource.clear()
            st.rerun()
    else:
        st.info("Upload your dataset to train the model.")
        uploaded = st.file_uploader("Upload Carbon Emission.csv", type=["csv"])
        if uploaded:
            with open("Carbon Emission.csv", "wb") as f:
                f.write(uploaded.read())
            st.cache_resource.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "Uses a Gradient Boosting Regressor (500 trees). "
        "Model is saved via **joblib** so it loads instantly after the first run."
    )

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
csv_exists = os.path.exists("Carbon Emission.csv")
model, encoders, r2, mae = load_or_train_model("Carbon Emission.csv" if csv_exists else None)

if model is None:
    st.warning("⚠️ No trained model found. Please upload **Carbon Emission.csv** in the sidebar to train the model.")
    st.stop()

# Show training metrics if just trained
if r2 is not None:
    st.success(f"✅ Model trained successfully! R² = **{r2*100:.2f}%** | MAE = **{mae:.2f} kg CO₂**")

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
st.subheader("📋 Enter Your Lifestyle Details")

with st.form("prediction_form"):
    # ── Personal ──────────────────────────────
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    body_type = c1.selectbox("Body Type",  BODY_TYPES)
    sex        = c2.selectbox("Sex",        SEX_TYPES)
    diet       = c3.selectbox("Diet",       DIET_TYPES)

    # ── Home ──────────────────────────────────
    st.markdown('<div class="section-title">🏠 Home & Energy</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    shower         = c1.selectbox("How Often Shower",       SHOWER_TYPES)
    heating        = c2.selectbox("Heating Energy Source",  HEATING_TYPES)
    energy_eff     = c3.selectbox("Energy Efficiency",      ENERGY_EFF)

    c1, c2 = st.columns(2)
    tv_hours       = c1.slider("TV / PC Daily Hours",    0, 24, 4)
    internet_hours = c2.slider("Internet Daily Hours",   0, 24, 3)

    # ── Transport ─────────────────────────────
    st.markdown('<div class="section-title">🚗 Transport</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    transport     = c1.selectbox("Transport Mode",              TRANSPORT_TYPES)
    vehicle_type  = c2.selectbox("Vehicle Type",                VEHICLE_TYPES)
    air_travel    = c3.selectbox("Air Travel Frequency",        AIR_FREQ_TYPES)
    vehicle_km    = st.slider("Vehicle Monthly Distance (km)", 0, 9999, 500, step=50)

    # ── Consumption ───────────────────────────
    st.markdown('<div class="section-title">🛒 Consumption & Lifestyle</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    grocery_bill   = c1.slider("Monthly Grocery Bill ($)",    50, 299, 150)
    new_clothes    = c2.slider("New Clothes / Month",          0,  50,   5)
    social_act     = c3.selectbox("Social Activity",         ACTIVITY_TYPES)

    # ── Waste ─────────────────────────────────
    st.markdown('<div class="section-title">♻️ Waste & Recycling</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    waste_size  = c1.selectbox("Waste Bag Size",        WASTE_SIZES)
    waste_count = c2.slider("Waste Bags / Week",  1, 7, 3)

    c1, c2 = st.columns(2)
    recycling_sel = c1.multiselect("What do you recycle?",  RECYCLING_OPTS, default=["Paper", "Plastic"])
    cooking_sel   = c2.multiselect("Cooking Methods Used",  COOKING_OPTS,   default=["Stove"])

    submitted = st.form_submit_button("🌍 Predict My Carbon Footprint", use_container_width=True)

# ─────────────────────────────────────────────
# Prediction & Results
# ─────────────────────────────────────────────
if submitted:
    recycling_encoded = encode_multiselect(recycling_sel) if recycling_sel else "none"
    cooking_encoded   = encode_multiselect(cooking_sel)   if cooking_sel   else "none"

    user_data = {
        "Body Type":                       body_type.lower(),
        "Sex":                             sex.lower(),
        "Diet":                            diet.lower(),
        "How Often Shower":                shower.lower(),
        "Heating Energy Source":           heating.lower(),
        "Transport":                       transport.lower(),
        "Vehicle Type":                    vehicle_type.lower(),
        "Social Activity":                 social_act.lower(),
        "Monthly Grocery Bill":            grocery_bill,
        "Frequency of Traveling by Air":   air_travel.lower(),
        "Vehicle Monthly Distance Km":     vehicle_km,
        "Waste Bag Size":                  waste_size.lower(),
        "Waste Bag Weekly Count":          waste_count,
        "How Long TV PC Daily Hour":       tv_hours,
        "How Many New Clothes Monthly":    new_clothes,
        "How Long Internet Daily Hour":    internet_hours,
        "Energy efficiency":               energy_eff.lower(),
        "Recycling":                       recycling_encoded,
        "Cooking_With":                    cooking_encoded,
    }

    with st.spinner("Calculating your carbon footprint…"):
        emission = predict_emission(model, encoders, FEATURE_COLUMNS, user_data)

    level, css_class, icon = get_emission_level(emission)

    st.markdown("---")
    st.subheader("📊 Your Results")

    # ── Main result card ──────────────────────
    st.markdown(f"""
    <div class="result-card {css_class}">
        <div style="font-size:3rem">{icon}</div>
        <div style="font-size:2.2rem; font-weight:700">{emission:,.0f} kg CO₂ / year</div>
        <div style="font-size:1.3rem; font-weight:600; margin-top:0.3rem">Carbon Level: {level}</div>
        <div style="font-size:0.95rem; margin-top:0.3rem">
            {'Below 6,000 kg – excellent!' if level=='LOW' else
             'Between 6,000–12,000 kg – room for improvement.' if level=='MEDIUM' else
             'Above 12,000 kg – action needed!'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gauge / context metrics ────────────────
    col1, col2, col3 = st.columns(3)
    avg_world = 4800
    avg_india = 1900
    col1.metric("🌍 World Avg",   f"{avg_world:,} kg", delta=f"{emission - avg_world:+,.0f} kg vs you", delta_color="inverse")
    col2.metric("🇮🇳 India Avg",  f"{avg_india:,} kg", delta=f"{emission - avg_india:+,.0f} kg vs you", delta_color="inverse")
    col3.metric("🌲 Trees Needed", f"{max(1, int(emission / 21)):,}", help="Approx. trees required to offset your annual footprint (1 tree ≈ 21 kg CO₂/yr)")

    # ── Progress bar relative to scale ────────
    st.markdown("**Emission Scale**")
    capped = min(emission, 20000)
    st.progress(int(capped / 200))
    st.caption(f"Scale: 0 → 20,000 kg CO₂/year  |  Your value: **{emission:,.0f} kg**")

    # ── Tips ──────────────────────────────────
    st.subheader("💡 Personalised Eco Tips")
    suggestions = generate_suggestions(user_data, emission)
    for tip in suggestions:
        st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)

    # ── Summary table ─────────────────────────
    with st.expander("📄 View Your Input Summary"):
        summary = {k: v for k, v in user_data.items()}
        st.dataframe(pd.DataFrame(list(summary.items()), columns=["Feature", "Your Value"]), use_container_width=True)
