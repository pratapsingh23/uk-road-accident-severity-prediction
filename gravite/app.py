import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# Page Config & Paths
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Gravite | UK Road Accident Severity",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
GRID_PATH = os.path.join(BASE_DIR, "hot_grid.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_bundle.pkl")

GRID_URL = "https://github.com/pratapsingh23/uk-road-accident-severity-prediction/releases/download/v1.0/hot_grid.csv"
MODEL_URL = "https://github.com/pratapsingh23/uk-road-accident-severity-prediction/releases/download/v1.0/model_bundle.pkl"


def chart(name: str) -> str:
    return os.path.join(ASSETS_DIR, name)


def show_chart(filename: str):
    path = chart(filename)
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning(f"Chart asset not found: `{filename}`")


# ------------------------------------------------------------------
# Data & Model Loading
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_grid():
    if not os.path.exists(GRID_PATH):
        urllib.request.urlretrieve(GRID_URL, GRID_PATH)
    return pd.read_csv(GRID_PATH)


@st.cache_resource(show_spinner="Downloading model bundle...")
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return joblib.load(MODEL_PATH)


try:
    bundle = load_bundle()
    model = bundle["model"]
    THRESHOLD = float(bundle["threshold"])
    MODEL_LOAD_ERROR = None
except Exception as e:
    model = None
    THRESHOLD = 0.45
    MODEL_LOAD_ERROR = str(e)


# ------------------------------------------------------------------
# Label Encodings
# ------------------------------------------------------------------
AREA = {"Urban": 1.0, "Rural": 2.0, "Unknown": np.nan}

ROAD_TYPE = {
    "Single carriageway": 6.0,
    "Dual carriageway": 3.0,
    "Roundabout": 1.0,
    "One way street": 2.0,
    "Slip road": 7.0,
    "Unknown": np.nan,
}

LIGHT = {
    "Daylight": 1.0,
    "Darkness - light lit": 4.0,
    "Darkness - light unlit": 5.0,
    "Darkness - no lighting": 6.0,
    "Darkness - lighting unknown": 7.0,
    "Unknown": np.nan,
}

WEATHER = {
    "Fine, no high winds": 1.0,
    "Raining + high winds": 5.0,
    "Snowing + high winds": 6.0,
    "Fog or mist": 7.0,
    "Other": 8.0,
    "Unknown": np.nan,
}

SURFACE = {
    "Dry": 1.0,
    "Wet or damp": 2.0,
    "Snow": 3.0,
    "Frost or ice": 4.0,
    "Flood over 3cm": 5.0,
    "Unknown": np.nan,
}

JUNCTION = {
    "Not a junction": 0.0,
    "T / staggered": 13.0,
    "Crossroads": 16.0,
    ">4 Arms (not roundabout)": 17.0,
    "Private drive": 18.0,
    "Other junction": 19.0,
    "Unknown": np.nan,
}

CONTROL = {
    "Give way / Uncontrolled": 4.0,
    "Traffic signal": 2.0,
    "Stop sign": 3.0,
    "Authorised person": 1.0,
    "Unknown": np.nan,
}

DAY = {
    "Sunday": 1.0,
    "Monday": 2.0,
    "Tuesday": 3.0,
    "Wednesday": 4.0,
    "Thursday": 5.0,
    "Friday": 6.0,
    "Saturday": 7.0,
}


# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------
PAGES = {
    "home": "🏠 Home",
    "eda": "📊 Exploratory Analysis",
    "predict": "🎯 Predict Severity",
    "model_detail": "🧠 Model Details",
}

if "page" not in st.session_state:
    st.session_state.page = "home"


def go(page: str):
    st.session_state.page = page
    st.rerun()


with st.sidebar:
    st.title("🚦 Gravite")
    st.caption("UK Road Accident Severity Analysis & ML Prediction")
    st.divider()

    for key, label in PAGES.items():
        btn_type = "primary" if st.session_state.page == key else "secondary"
        if st.button(label, key=f"side_{key}", use_container_width=True, type=btn_type):
            go(key)

    st.divider()
    st.caption("Dataset: 5,13,801 UK Road Accidents")
    st.caption(f"Active KSI Threshold: `{THRESHOLD:.2f}`")


# ------------------------------------------------------------------
# Page: Home
# ------------------------------------------------------------------
if st.session_state.page == "home":
    with st.container(border=True):
        st.title("🚦 Gravite")
        st.subheader("UK Road Accident Severity Analysis & KSI Prediction Platform")
        st.write(
            "Analyze 5,13,801 historical UK road accidents, explore geographic hotspots, "
            "and predict Killed or Seriously Injured (KSI) risk using Random Forest."
        )

    st.write("")
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        with st.container(border=True):
            st.subheader("📊 1. Exploratory Analysis")
            st.write(
                "Explore severity splits, urban vs rural fatality rates, "
                "rush-hour trends, and 1 km KSI hotspots."
            )
            if st.button("Open EDA Dashboard", use_container_width=True, type="primary"):
                go("eda")

    with c2:
        with st.container(border=True):
            st.subheader("🎯 2. Predict Severity")
            st.write(
                "Enter road, weather, vehicle, and driver details to estimate "
                "whether an accident is KSI or Slight."
            )
            if st.button("Launch Predictor", use_container_width=True, type="primary"):
                go("predict")

    with c3:
        with st.container(border=True):
            st.subheader("🧠 3. Model Details")
            st.write(
                "Inspect Random Forest specs, top feature importances, "
                "and threshold calibration."
            )
            if st.button("View Model Specs", use_container_width=True, type="primary"):
                go("model_detail")


# ------------------------------------------------------------------
# Page: EDA
# ------------------------------------------------------------------
elif st.session_state.page == "eda":
    st.title("📊 Exploratory Data Analysis")
    st.caption("Key patterns, high-risk hours, and geographic hotspots behind UK road accidents.")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        with st.container(border=True):
            st.metric("Total Accidents", "5,13,801")
    with k2:
        with st.container(border=True):
            st.metric("Slight", "75.8%")
    with k3:
        with st.container(border=True):
            st.metric("Serious", "22.7%")
    with k4:
        with st.container(border=True):
            st.metric("Fatal", "1.5%")

    st.write("")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Severity",
            "Urban vs Rural",
            "Time Patterns",
            "Pedestrian",
            "Deprivation",
            "KSI Hotspots",
        ]
    )

    with tab1:
        show_chart("distribution_severity.png")
        st.info(
            "About 75 in 100 accidents cause only minor injuries — "
            "but 24.2% (nearly 1 in 4) are serious or fatal (KSI)."
        )

    with tab2:
        show_chart("urban_vs_rural.png")
        st.warning(
            "The rural fatal rate is 2.8% — 3.5x higher than urban areas (0.8%)."
        )

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            show_chart("season_wise_acc.png")
            st.info(
                "Severity stays almost flat across all seasons — "
                "no single season is significantly more dangerous."
            )
        with col_b:
            show_chart("weekdays_acc.png")
            st.info(
                "Weekday evening rush hours see significantly more accidents "
                "(59.6%) compared to morning rush hours (40.4%)."
            )

    with tab4:
        show_chart("severity_pedestrian.png")
        st.warning(
            "Pedestrian involved = 33.6% KSI vs 22.8% without pedestrian involvement."
        )

    with tab5:
        show_chart("severity_deprived.png")
        st.info(
            "KSI share is nearly flat across deprivation bands (23.0% → 25.4%) — "
            "IMD shows no strong effect on severity."
        )

    with tab6:
        st.write(
            "KSI rate by 1 km grid — darker red indicates a higher share of "
            "killed/seriously-injured accidents (grids with ≥50 accidents)."
        )
        try:
            hot = load_grid()
            fig = px.scatter_map(
                hot,
                lat="lat_g",
                lon="long_g",
                color="ksi_rate_capped",
                size="total",
                size_max=18,
                color_continuous_scale="YlOrRd",
                hover_data={
                    "lat_g": False,
                    "long_g": False,
                    "total": True,
                    "ksi_rate_capped": ":.1f",
                    "urban_share": ":.0f",
                    "avg_speed_limit": ":.0f",
                },
                zoom=5,
                center={"lat": 55.3781, "lon": -3.4360},
                map_style="carto-positron",
                height=600,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Interactive map could not be loaded: {e}")
            show_chart("map_plot.png")

    st.divider()
    st.caption("Source: eda.ipynb | Gravite")


# ------------------------------------------------------------------
# Page: Predict Severity
# ------------------------------------------------------------------
elif st.session_state.page == "predict":
    st.title("🎯 Accident Severity Predictor")
    st.caption(
        "Estimate whether an accident is likely to result in KSI (Killed or Seriously Injured) or Slight injury."
    )

    if MODEL_LOAD_ERROR:
        st.error(f"Model could not be loaded: {MODEL_LOAD_ERROR}")
        st.stop()

    with st.form("predict_form"):
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            with st.container(border=True):
                st.subheader("Vehicle & Driver Metrics")
                c_num1, c_num2 = st.columns(2)
                with c_num1:
                    speed_limit = st.selectbox("Speed Limit (mph)", [20, 30, 40, 50, 60, 70], index=2)
                    min_age = st.number_input("Min Driver Age", 16, 100, 28)
                    engine_cc = st.number_input("Max Engine CC", 0, 15000, 1200, step=50)
                with c_num2:
                    n_vehicles = st.number_input("Number of Vehicles", 1, 30, 1)
                    max_age = st.number_input("Max Driver Age", 16, 100, 32)
                    male_share = st.slider("Male Driver Share", 0.0, 1.0, 0.5, 0.05)

                imd = st.slider("Avg Casualty IMD (1 = most deprived)", 1.0, 10.0, 5.0, 0.5)

            with st.container(border=True):
                st.subheader("Road Users & Maneuvers Involved")
                cc1, cc2 = st.columns(2)
                has_cycle = cc1.checkbox("Cycle")
                has_motorcycle = cc2.checkbox("Motorcycle")
                has_goods = cc1.checkbox("Goods vehicle")
                has_bus = cc2.checkbox("Bus")
                has_vul_veh = cc1.checkbox("Vulnerable vehicle")
                has_vul_cas = cc2.checkbox("Vulnerable casualty")
                has_turn = cc1.checkbox("Turning")
                has_escoot = cc2.checkbox("E-scooter casualty")
                has_overt = cc1.checkbox("Overtaking")
                has_peds = cc2.checkbox("Pedestrian")

        with col_right:
            with st.container(border=True):
                st.subheader("Road & Environmental Conditions")
                area = st.selectbox("Area", list(AREA.keys()))
                road = st.selectbox("Road Type", list(ROAD_TYPE.keys()))
                light = st.selectbox("Light Conditions", list(LIGHT.keys()))
                weather = st.selectbox("Weather", list(WEATHER.keys()))
                surface = st.selectbox("Road Surface", list(SURFACE.keys()))
                junction = st.selectbox("Junction Detail", list(JUNCTION.keys()))
                control = st.selectbox("Junction Control", list(CONTROL.keys()))
                day = st.selectbox("Day of Week", list(DAY.keys()))

        submitted = st.form_submit_button("Predict Severity", use_container_width=True, type="primary")

    if submitted:
        if min_age > max_age:
            st.warning("Min Driver Age cannot be greater than Max Driver Age. Please adjust the ages and try again.")
        else:
            row = {
                "speed_limit": float(speed_limit),
                "number_of_vehicles": int(n_vehicles),
                "min_driver_age": float(min_age),
                "max_driver_age": float(max_age),
                "max_engine_cc": float(engine_cc),
                "male_driver_share": float(male_share),
                "avg_casualty_imd": float(imd),
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
                "urban_or_rural_area": AREA[area],
                "road_type": ROAD_TYPE[road],
                "light_conditions": LIGHT[light],
                "weather_conditions": WEATHER[weather],
                "road_surface_conditions": SURFACE[surface],
                "junction_detail": JUNCTION[junction],
                "junction_control": CONTROL[control],
                "day_of_week": DAY[day],
            }

            X_new = pd.DataFrame([row])
            proba = float(np.clip(model.predict_proba(X_new)[0, 1], 0.0, 1.0))
            pred = int(proba >= THRESHOLD)

            with st.container(border=True):
                res_col1, res_col2 = st.columns([2, 1])

                with res_col1:
                    st.progress(proba, text=f"KSI Probability: {proba * 100:.1f}% (Threshold: {THRESHOLD * 100:.0f}%)")
                    if pred == 1:
                        st.error("🚨 Prediction: **KSI (Killed or Seriously Injured)**")
                    else:
                        st.success("✅ Prediction: **Slight Injuries**")

                with res_col2:
                    st.metric("Predicted KSI Risk", f"{proba * 100:.1f}%", delta=f"Threshold {THRESHOLD * 100:.0f}%")

            st.caption("Disclaimer: Predictions are statistical estimates from historical UK accident data.")


# ------------------------------------------------------------------
# Page: Model Details
# ------------------------------------------------------------------
elif st.session_state.page == "model_detail":
    st.title("🧠 ML Model Details")
    st.caption("Gravite — Random Forest KSI Classification Architecture & Training Notes")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        with st.container(border=True):
            st.metric("KSI Recall", "0.74")
    with k2:
        with st.container(border=True):
            st.metric("KSI F1 Score", "0.464")
    with k3:
        with st.container(border=True):
            st.metric("Decision Threshold", f"{THRESHOLD:.2f}")
    with k4:
        with st.container(border=True):
            st.metric("Test Set Size", "1,02,761")

    st.write("")
    tab1, tab2, tab3 = st.tabs(["Specifications", "Feature Importance", "Pipeline & Journey"])

    with tab1:
        st.markdown(
            """
            | Parameter | Value |
            |---|---|
            | Algorithm | Random Forest Classifier |
            | Trees (`n_estimators`) | 100 |
            | `max_depth` | 15 |
            | `class_weight` | `balanced` |
            | `random_state` | 42 |
            | Input Features | 25 |
            | Training Rows | 4,11,040 (80%) |
            | Test Rows | 1,02,761 (20%, stratified) |
            | Full Dataset | 5,13,801 UK road accident records |
            | Target | KSI (Fatal + Serious) vs Slight |
            """
        )

    with tab2:
        importance_df = pd.DataFrame(
            {
                "Feature": [
                    "has_vulnerable_casualty",
                    "number_of_vehicles",
                    "max_driver_age",
                    "speed_limit",
                    "min_driver_age",
                    "max_engine_cc",
                    "avg_casualty_imd",
                    "male_driver_share",
                    "has_vulnerable_vehicle",
                    "has_motorcycle",
                ],
                "Importance": [0.094, 0.081, 0.078, 0.064, 0.059, 0.058, 0.041, 0.041, 0.039, 0.034],
            }
        ).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            color="Importance",
            color_continuous_scale="Reds",
            height=420,
        )
        fig_imp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_imp.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=10, r=30, t=10, b=10),
            xaxis_title="Impurity-Based Importance",
            yaxis_title="",
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Top 10 features — vulnerable road users, speed limit, and driver age dominate.")

    with tab3:
        st.subheader("Inference Pipeline")
        st.code(
            f"""raw inputs (25 features)
   → ColumnTransformer
       ├─ numeric      : passthrough (NaN kept as-is)
       ├─ categorical  : OneHotEncoder (raw codes, NaN = own category)
       └─ binary flags : passthrough
   → RandomForest (100 trees, depth 15, balanced)
   → P(KSI) ≥ {THRESHOLD:.2f} ?  KSI 🚨 : Slight ✅""",
            language=None,
        )

        st.subheader("Training Journey")
        st.markdown(
            """
            - **Baseline RF** → KSI recall `0.15` (model mostly predicted *Slight*).
            - **`class_weight='balanced'`** → Recall jumped to `0.62` (4.2x more serious cases caught).
            - **Threshold sweep (`0.45`)** → Recall `0.74`, KSI F1 `0.464`.
            - **Accuracy trap** → 65% accuracy looks low, but 75.8% of data is *Slight* — a dummy model always guessing *Slight* scores 75.8%. Recall and F1 on KSI were prioritized.
            - **NaN strategy** → No imputation; Random Forest handles missing values natively (`junction_control` was 44.6% missing).
            """
        )

        with st.expander("Why threshold 0.45?"):
            st.write(
                "Precision-recall sweep on the test set showed 0.45 maximized KSI F1 (0.464) "
                "while pushing recall to 0.74 — a safety-first tradeoff where a false alarm is "
                "much less costly than missing a serious accident."
            )
