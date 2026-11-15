import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request, os
import streamlit.components.v1 as components

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

def chart(name):
    return os.path.join(ASSETS_DIR, name)

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
    
# home------------------------------------------------------
if st.session_state.page == 'home':
    st.write("")
    st.title("Where would you like to go?", text_alignment='center')
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

    st.title('EDA - UK Road Accidents')
    st.caption("5,13,801 accidents records")

    # KPI row---------
    k1, k2, k3, k4  = st.columns(4)
    k1.metric("Total accidents", "5,13,801")
    k2.metric("Slight", "75.8%")
    k3.metric("Serious", "22.7%")
    k4.metric("Fatal", "1.5%")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Severity", "Urban vs Rura;", "Time Patterns", "Pedestrain", "Deprivation", "KSI Hotspots"]
    )

    with tab1:
        st.image(chart("distribution_severity.png"), use_container_width=True)
        st.info(
            "**Insights:** About 75 in 100 accidents cause only minor injuries -"
            "but 22.7% are serious, i.e. **1 an 4** accidents is serious or fatal!"
        )
    with tab2:
        st.image(chart("urban_vs_rural.png"), use_container_width=True)
        st.warning(
            "**Insights:** The rural fatal rate is **2.8%** - **3.5x higher**"
            "than urban (0.8%)!"
        )

    with tab3:
        st.image(chart("season_wise_acc.png"), use_container_width=True)
        st.info(
            "**Insight:** Severity stays **almost flat** across all seasons - "
            "no single season is more dangerous."
        )

        st.image(chart("weekdays_acc.png"), use_container_width=True)
        st.info(
        "**Insight:** Weekday evening rush hours see significantly more accidents "
        "(**59.6%**) compared to morning rush hours (**40.4%**)."
)

    with tab4:
        st.image(chart("severity_pedestrian.png"), use_container_width=True)
        st.warning(
            "**Insight:** Pedestrian involved = **33.6% KSI** vs 22.8% without"
            "(+10.8 points) - also the model's #1 feature!"
        )

    with tab5:
        st.image(chart("severity_deprived.png"), use_container_width=True)
        st.info(
            "**Insight:** KSI share is nearly flat across deprivation bands "
            "(23.0% -> 25.4%) - IMD shows no strong effect on severity."
        )

    with tab6:
        st.markdown(
            "**KSI rate by 1 km grid** - darker red = higher share of "
            "killed/seriously-injured accidents (only grids with >= 50 accidents)."
        )
        try:
            with open(chart("ksi_rate_map.html"), "r", encoding="utf-8") as f:
                map_html = f.read()
            components.html(map_html, height=800, scrolling=True)
        except FileNotFoundError:
            st.error("ksi_rate_map.html not found in assets/folder")

    st.divider()
    st.caption("Charts: eda.ipynb | Gravite")


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
    st.caption("Gravite - How KSI predict by Random Forest")


    # quick KPIs-----------------
    k1, k2, k3, k4  = st.columns(4)
    k1.metric('KSI Recall', '0.74')
    k2.metric('KSI F1', '0.464')
    k3.metric('Threshold', '0.45')
    k4.metric('Test size', '1,02,761')

    st.divider()

    tab1, tab2, tab3 = st.tabs(['Specs', 'Feature Importance', 'Pipeline & Journey'])

    # tab 1: space------------------------------------------
    with tab1:
        st.markdown("""
        | Item | Value |
        |---|---|
        | Algorithm | **Random Forest Classifier** |
        | Trees (`n_estimators`) | 100 |
        | `max_depth` | 15 |
        | `class_weight` | `balanced` |
        | `random_state` | 42 |
        | Features | 25 |
        | Training rows | 4,11,040 (80%) |
        | Test rows | 1,02,761 (20%, stratified) |
        | Dataset | 5,13,801 UK road accident records |
        | Target | **KSI** (fatal + serious) vs **Slight** |
        """)

    # tab 2: Feature Importance--------------------
    with tab2:
        importance = pd.Series({
            'has_vulnerable': 0.094,
            'number_of_vehicles': 0.081,
            'max_driver_age': 0.078,
            'speed_limit': 0.064,
            'min_driver_age': 0.059,
            'max_engine_cc': 0.058,
            'avg_casualty_imd': 0.041,
            'male_driver_share': 0.041,
            'has_vulnerable_vehicle': 0.039,
            'has_motorcycle': 0.034,
        }).sort_values()
        st.bar_chart(importance, height=400)
        st.caption("Top 10 feature - RF impurity-based importance. Vulnerable road users + speed + driver age dominate")

    # tab 3: Pipeline * journey-----------------
    with tab3:
        st.markdown("**Pipeline**")
        st.code("""
raw inputs (25 features)
   → ColumnTransformer
       ├─ numeric  : passthrough (NaN kept as-is)
       ├─ categorical : OneHotEncoder (raw codes, NaN = own category)
       └─ binary flags : passthrough
   → RandomForest (100 trees, depth 15, balanced)
   → P(KSI) ≥ 0.45 ?  KSI 🚨 : Slight ✅
        """, language=None)

        st.markdown("**Training journey (why each step mattered):**")
        st.markdown("""
        - **Baseline RF** -> KSI recall **0.15** -model almost guessed "slight"
        - **`class_weight='balanced'`** -> recall **0.62** (4.2x more serious cases caught)
        - **Threshold sweep (0.45)** -> recall **0.74**, F1 **0.464**
        - **Accuracy trap:** 65% looks low, but 76% of data is "slight" - a model guessing *always slight* scores 75.8%! Accuracy was deprioritized for **recall/F1 on KSI**
        - **NaN stratagy:** no imputation - RF handles missing values natively (junction_control was 44.6% missing!)

        """)

        with st.expander("Why threshold 0.45?"):
            st.write("Precision-recall sweep on test set: 0.45 maximized KSI F1 (0.464) while pushing recall to 0.74 - safety-first traadeoff: false alarm < missed serious accident.")
