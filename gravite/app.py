import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request, os

st.set_page_config(page_title="Accident Severity", layout="centered")

# Model loading------------------------------------------------------
MODEL_URL = "https://github.com/pratapsingh23/uk-road-accident-severity-prediction/releases/download/v1.0/model_bundle.pkl"

@st.cache_resource
def load_bundle():
    if not os.path.exists("model_bundle.pkl"):
        with st.spinner("Downloading model...."):
            urllib.request.urlretrieve(MODEL_URL, 'model_bundle.pkl')
    return joblib.load("model_bundle.pkl")

bundle = load_bundle()
model = bundle["model"]
THRESHOLD = bundle["threshold"]

# Label------------------------------------------------------
AREA = {
    'Urban': 1.0, 'Rural':2.0, 'Unknown': np.nan
}

ROAD_TYPE = {
    'Single carriageway': 6.0, 'Dual carriageway': 3.0, 'Roundabout': 1.0,
    'One way street': 2.0, 'Slip road': 7.0, 'Unknown': np.nan
}

LIGHT ={
    'Daylight': 1.0, 'Darkness - light lit': 4.0, 'Darkness - light unlit': 5.0, 
    'Darkness - no lighting': 6.0, 'Darkness - lighting unknown': 7.0, 'Unknown': np.nan
}

WEATHER = {
    'Fine, no high winds': 1.0, 'Raining + high winds': 5.0, 'Snowing + high winds': 6.0,
    'Fog or mist': 7.0, 'Other': 8.0, 'Unknown': np.nan
}

SURFACE = {
    'Dry': 1.0, 'Wet or dump': 2.0, 'Snow':3.0, 'Frost or ice': 4.0, 
    'Flood over 3cm': 5.0, 'Unknown': np.nan
}

JUNCTION = {
    'Not a junction': 0.0, 'T / staggered': 13.0, 'Crossroads': 16.0,
    '>4 Arms (not roundabout)': 17.0, 'Private drive': 18.0, 'Other junction': 19.0,
    'Unknown': np.nan
}



CONTROL ={
    'Give way / Uncontrolled': 4.0, 'Traffic signal': 2.0, 'Stop sign': 3.0,
    'Authorised person': 1.0, 'Unknown': np.nan 
}

DAY = {
    'Sunday': 1.0, 'Monday': 2.0, 'Tuesday': 3.0, 'Wednesday': 4.0, 
    'Thursday': 5.0, 'Friday': 6.0, 'Saturaday': 7.0
}

# Navigation------------------------------------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go(page):
    st.session_state.page = page
    st.rerun()

# Fragmented title-----------------------------

lines = [
    "Where you wanna go?",
    "Explore 500,000+ UK road collisions...",
    "Predict accident severity with Gravite.",
    "Uncover hidden patterns in road safety.",
    "Analyze high-risk junctions & speed zones.",
    "Evaluate vulnerable road user risks.",
    "Compare urban vs rural crash dynamics.",
    "Inspect Random Forest feature importance.",
    "Test real-time weather & light conditions.",
    "Turn raw collision data into life-saving insights."
]
if 'line_idx' not in st.session_state:
    st.session_state.line_idx = 0
@st.fragment(run_every=3)
def rotating_title():
    idx = st.session_state.line_idx
    st.title(linesp[idx], text_alignment='center')
    st.session_state.line_idx = (idx + 1) % len(lines)
    

# home------------------------------------------------------
if st.session_state.page == 'home':
    st.write("")
    rotating_title()
    st.subheader("UK Road Accident Severity Analysis & Prediction", text_alignment='center')
    st.write("")

    if st.button("1. EDA", use_container_width=True):
        go("eda")
    st.write("")

    if st.button("2. Predict Severity", use_container_width=True):
        go("predict")
    st.write("")

    if st.button("3. Model Details", use_container_width=True):
        go("model_detail")


# EDA---------------------------------------
elif st.session_state.page == 'eda':
    if st.button('<- Back'):
        go('home')
    st.title('EDA')
    st.info("Coming Soon....")


# predict------------------------------------
elif st.session_state.page == 'predict':
    if st.button('<- Back'):
        go('home')
    st.title("KSI vs Slight")
    st.write("Predict accident severity by **Gravite**")

    with st.form('predict_form'):
        a, b = st.columns(2)

        with a:
            st.subheader("Numbers")
            speed_limit = st.selectbox("Speed Limit (mph)", [20, 30, 40, 50, 60, 70], index=2)
            n_vehicles = st.number_input("Number of Vehicles", 1, 30, 1)
            min_age = st.number_input("Min driver age", 16, 100, 28)
            max_age = st.number_input("Max driver age", 16, 100, 32)
            engine_cc = st.number_input("Max engine CC", 0, 15000, 1200, step=50)
            male_share = st.slider("Male driver share", 0.0, 1.0, 0.1)
            imd = st.slider("Avg casualty IMD (1=most deprived)", 1.0, 10.0, 5.0, 0.5)

            st.subheader("Road users involved")
            cc = st.columns(2)
            has_cycle = cc[0].checkbox("Cycle")
            has_motorcycle = cc[1].checkbox("Motorcycle")
            has_goods = cc[0].checkbox("Goods vehicle")
            has_bus = cc[1].checkbox("Bus")
            has_vul_veh = cc[0].checkbox("Vulnerable vehicle")
            has_vul_cas = cc[1].checkbox("Vulnerable casualty")
            has_turn = cc[0].checkbox("Turning")
            has_escoot = cc[1].checkbox("E-scooter casualty")
            has_overt = cc[0].checkbox("Overtaking")
            has_peds = cc[1].checkbox("Pedestrian")

        with b:
            st.subheader("Conditions")
            area = st.selectbox("Area", list(AREA))
            road = st.selectbox("Road type", list(ROAD_TYPE))
            light = st.selectbox("Light conditions", list(LIGHT))
            weather = st.selectbox("Weather", list(WEATHER))
            surface = st.selectbox("Road surface", list(SURFACE))
            junction = st.selectbox("Junction detail", list(JUNCTION))
            control = st.selectbox("Junction control", list(CONTROL))
            day = st.selectbox("Day of week", list(DAY))

        submitted = st.form_submit_button("Predict", use_container_width=True)

    # Prediction------------------------------------------------------

    if submitted:
        row = {
            # numeric
            'speed_limit': float(speed_limit),
            'number_of_vehicles': n_vehicles,
            'min_driver_age': float(min_age),
            'max_driver_age': float(max_age),
            'max_engine_cc': float(engine_cc),
            'male_driver_share': male_share,
            'avg_casualty_imd': imd,

            # binary
            "has_cycle": int(has_cycle),
            "has_motorcycle": int(has_motorcycle),
            "has_goods_vehicle": int(has_goods),
            "has_bus": int(has_bus),
            "has_vulnerable_vehicle": int(has_vul_veh),
            "has_overtaking": int(has_overt),
            "has_turning": int(has_turn),
            "has_vulnerable_casualty": int(has_vul_cas),
            "has_escooter_casualty": int(has_escoot),
            "has_pedestrian_movement": int(has_peds),

            # categorical
            "urban_or_rural_area": AREA[area],
            "road_type": ROAD_TYPE[road],
            "light_conditions": LIGHT[light],
            "weather_conditions": WEATHER[weather],
            "road_surface_conditions": SURFACE[surface],
            "junction_detail": JUNCTION[junction],
            "junction_control": CONTROL[control],
            "day_of_week": DAY[day],
        }

        X_new  = pd.DataFrame([row])
        proba = float(model.predict_proba(X_new)[0, 1])
        pred  =int(proba >= THRESHOLD)

        st.progress(proba, text=f"KSI (Killed or Seriously Injured) probability: {proba * 100 :.1f}%")
        if pred == 1:
            st.error(F"**KSI** (Killed or Seriously Injured)")
        else:
            st.success(f"**Slight** injuries")
        st.caption("Predictions may be inaccurate for now.", text_alignment='center')


# model details-------------------------------------
if st.session_state.page == 'model_detail':
    if st.button('<- Back'):
        go('home')
    st.title("ML Model details")
    st.info("Coming Soon....")
