import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# Page Config & Paths --------------------------------------------------

st.set_page_config(
    page_title="UK Road Accident Severity Dashboard",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
GRID_PATH = os.path.join(BASE_DIR, "hot_grid.csv")
MODEL_PATH = os.path.join(BASE_DIR, "gravite_hgbc_bundle.joblib")

GRID_URL = "https://github.com/pratapsingh23/uk-road-accident-severity-prediction/releases/download/v1.0/hot_grid.csv"
MODEL_URL = "https://github.com/pratapsingh23/uk-road-accident-severity-prediction/releases/download/v1.0/gravite_hgbc_bundle.joblib"


def chart(name):
    return os.path.join(ASSETS_DIR, name)


def show_chart(filename, width=720):
    path = chart(filename)
    if os.path.exists(path):
        st.image(path, width=width)
    else:
        st.warning(f"Chart not found: {filename}")


# Data & Model Loading -------------------------------------------------

@st.cache_data
def load_grid():
    if not os.path.exists(GRID_PATH):
        urllib.request.urlretrieve(GRID_URL, GRID_PATH)
    return pd.read_csv(GRID_PATH)


@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
model = bundle["model"]
features = bundle["features"]
cat_dtypes = bundle["cat_dtypes"]
THRESHOLD = bundle.get("threshold", 0.50)


# Label Encoding -------------------------------------------------------

AREA = {
    "Urban": 1.0,
    "Rural": 2.0,
    "Unknown": np.nan,
}

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

HOUR = {
    "12:00 AM (Midnight)": 0,
    "01:00 AM": 1,
    "02:00 AM": 2,
    "03:00 AM": 3,
    "04:00 AM": 4,
    "05:00 AM": 5,
    "06:00 AM": 6,
    "07:00 AM": 7,
    "08:00 AM": 8,
    "09:00 AM": 9,
    "10:00 AM": 10,
    "11:00 AM": 11,
    "12:00 PM (Noon)": 12,
    "01:00 PM": 13,
    "02:00 PM": 14,
    "03:00 PM": 15,
    "04:00 PM": 16,
    "05:00 PM": 17,
    "06:00 PM": 18,
    "07:00 PM": 19,
    "08:00 PM": 20,
    "09:00 PM": 21,
    "10:00 PM": 22,
    "11:00 PM": 23,
}

SPECIAL_CONDITIONS = {
    "None": 0.0,
    "Auto traffic signal - out": 1.0,
    "Auto signal part defective": 2.0,
    "Road sign or marking defective": 3.0,
    "Roadworks": 4.0,
    "Road surface defective": 5.0,
    "Oil or diesel": 6.0,
    "Mud": 7.0,
    "Unknown": np.nan,
}

CARRIAGEWAY_HAZARDS = {
    "None": 0.0,
    "Vehicle load on road": 1.0,
    "Other object on road": 2.0,
    "Previous accident": 3.0,
    "Dog on road": 4.0,
    "Other animal on road": 5.0,
    "Pedestrian in carriageway - not injured": 6.0,
    "Any animal in carriageway (except ridden horse)": 7.0,
    "Unknown": np.nan,
}


# Sidebar --------------------------------------------------------------

with st.sidebar:
    st.header("Accident Dashboard")
    st.caption("UK Road Accident Severity Analysis & Prediction")
    st.divider()

    st.subheader("Dataset Summary")
    st.write("- **Records:** 5,13,801")
    st.write("- **Target:** KSI vs Slight")
    st.write("- **Features:** 30")
    st.divider()

    st.subheader("Model Summary")
    st.write("- **Algorithm:** HistGradientBoostingClassifier")
    st.write(f"- **Threshold:** {THRESHOLD:.2f}")
    st.write("- **KSI Recall:** 0.66")
    st.write("- **KSI F1:** 0.48")


# Dashboard Header & Top KPIs ------------------------------------------

st.title("UK Road Accident Severity Dashboard")
st.caption("Exploratory analysis of 5,13,801 UK accidents and KSI prediction powered by the Gravite model.")

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    with st.container(border=True):
        st.metric("Total Accidents", "5,13,801")
with k2:
    with st.container(border=True):
        st.metric("Slight Injuries", "75.8%")
with k3:
    with st.container(border=True):
        st.metric("Serious Injuries", "22.7%")
with k4:
    with st.container(border=True):
        st.metric("Fatal Injuries", "1.5%")
with k5:
    with st.container(border=True):
        st.metric("KSI (Fatal + Serious)", "24.2%")

st.write("")


# Main Dashboard Tabs --------------------------------------------------

main_eda, main_predict, main_model = st.tabs(
    ["EDA & Hotspots", "Predict Severity (By Gravite)", "Model Details"]
)


# TAB 1: EDA & HOTSPOTS ------------------------------------------------

with main_eda:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Severity", "Urban vs Rural", "Time Patterns", "Pedestrian", "Deprivation", "KSI Hotspots"]
    )

    with tab1:
        show_chart("distribution_severity.png")
        st.info(
            "About 75 in 100 accidents cause only minor injuries — "
            "but 22.7% are serious (nearly 1 in 4 accidents is serious or fatal)."
        )

    with tab2:
        show_chart("urban_vs_rural.png")
        st.warning("The rural fatal rate is 2.8% — 3.5x higher than urban (0.8%).")

    with tab3:
            show_chart("season_wise_acc.png", width="stretch")
            st.info(
                "Severity stays almost flat across all seasons — "
                "no single season is more dangerous."
            )
            st.divider()
            show_chart("weekdays_acc.png", width="stretch")
            st.info(
                "Weekday evening rush hours see significantly more accidents "
                "(59.6%) compared to morning rush hours (40.4%)."
            )

    with tab4:
        show_chart("severity_pedestrian.png")
        st.warning("Pedestrian involved = 33.6% KSI vs 22.8% without.")

    with tab5:
        show_chart("severity_deprived.png")
        st.info(
            "KSI share is nearly flat across deprivation bands (23.0% → 25.4%) — "
            "IMD shows no strong effect on severity."
        )

    with tab6:
        st.write(
            "KSI rate by 1 km grid — darker red = higher share of "
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
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"Map is not loaded: {e}")
            show_chart("map_plot.png")


# TAB 2: PREDICT SEVERITY ----------------------------------------------

with main_predict:
    st.subheader("Predict Accident Severity using Gravite")
    st.caption("Adjust the accident parameters below to estimate KSI (Killed or Seriously Injured) risk.")

    with st.form("predict_form"):
        a, b = st.columns(2, gap="large")

        with a:
            with st.container(border=True):
                st.markdown("**Vehicle & Driver Numbers**")
                n1, n2 = st.columns(2)
                with n1:
                    speed_limit = st.selectbox("Speed Limit (mph)", [20, 30, 40, 50, 60, 70], index=2)
                    min_age = st.number_input("Min driver age", 16, 100, 28)
                    engine_cc = st.number_input("Max engine CC", 0, 15000, 1200, step=50)
                    imd = st.slider("Avg casualty IMD (1 = most deprived)", 1.0, 10.0, 5.0, 0.5)
                with n2:
                    n_vehicles = st.number_input("Number of Vehicles", 1, 30, 1)
                    max_age = st.number_input("Max driver age", 16, 100, 32)
                    max_vehicle_age = st.number_input("Max vehicle age", 0, 100, 10)
                    male_share = st.slider("Male driver share", 0.0, 1.0, 0.1)

            with st.container(border=True):
                st.markdown("**Road Users Involved & Road Flag**")
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
                trunk_road_f = cc[0].checkbox("Trunk road")

        with b:
            with st.container(border=True):
                st.markdown("**Road & Weather Conditions**")
                c1, c2 = st.columns(2)
                with c1:
                    area = st.selectbox("Area", list(AREA))
                    road = st.selectbox("Road type", list(ROAD_TYPE))
                    light = st.selectbox("Light conditions", list(LIGHT))
                    weather = st.selectbox("Weather", list(WEATHER))
                    surface = st.selectbox("Road surface", list(SURFACE))
                with c2:
                    junction = st.selectbox("Junction detail", list(JUNCTION))
                    control = st.selectbox("Junction control", list(CONTROL))
                    special_cond = st.selectbox("Special conditions at site", list(SPECIAL_CONDITIONS))
                    hazards = st.selectbox("Carriageway hazards", list(CARRIAGEWAY_HAZARDS))

                t1, t2 = st.columns(2)
                with t1:
                    day = st.selectbox("Day of week", list(DAY))
                with t2:
                    hour = st.selectbox("Collision hour", list(HOUR), index=14)

        submitted = st.form_submit_button("Predict Severity", width="stretch", type="primary")

    if submitted:
        if min_age > max_age:
            st.warning("Min driver age cannot be greater than Max driver age.")
        else:
            row = {
                "speed_limit": float(speed_limit),
                "number_of_vehicles": n_vehicles,
                "min_driver_age": float(min_age),
                "max_driver_age": float(max_age),
                "max_vehicle_age": float(max_vehicle_age),
                "max_engine_cc": float(engine_cc),
                "male_driver_share": male_share,
                "avg_casualty_imd": imd,
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
                "trunk_road_flag": 1.0 if trunk_road_f else 2.0,
                "urban_or_rural_area": AREA[area],
                "road_type": ROAD_TYPE[road],
                "light_conditions": LIGHT[light],
                "weather_conditions": WEATHER[weather],
                "road_surface_conditions": SURFACE[surface],
                "special_conditions_at_site": SPECIAL_CONDITIONS[special_cond],
                "carriageway_hazards": CARRIAGEWAY_HAZARDS[hazards],
                "junction_detail": JUNCTION[junction],
                "junction_control": CONTROL[control],
                "day_of_week": DAY[day],
                "hour": HOUR[hour],
            }

            X_new = pd.DataFrame([row])[features].copy()

            for col, dtype in cat_dtypes.items():
                X_new[col] = X_new[col].astype(dtype)

            proba = float(np.clip(model.predict_proba(X_new)[0, 1], 0.0, 1.0))
            pred = int(proba >= THRESHOLD)

            with st.container(border=True):
                r1, r2 = st.columns([3, 1])
                with r1:
                    st.progress(proba, text=f"KSI Probability: {proba * 100:.1f}%")
                    if pred == 1:
                        st.error("Prediction: KSI (Killed or Seriously Injured)")
                    else:
                        st.success("Prediction: Slight injuries")
                with r2:
                    st.metric("KSI Risk", f"{proba * 100:.1f}%", delta=f"Threshold {THRESHOLD * 100:.0f}%")

            st.caption("Predictions may be inaccurate for now.")


# TAB 3: GRAVITE MODEL DETAILS -----------------------------------------

with main_model:
    st.subheader("Gravite — HistGradientBoostingClassifier Model Details")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            st.metric("KSI Recall", "0.66")
    with m2:
        with st.container(border=True):
            st.metric("KSI F1", "0.48")
    with m3:
        with st.container(border=True):
            st.metric("Threshold", f"{THRESHOLD:.2f}")
    with m4:
        with st.container(border=True):
            st.metric("Features", f"{len(features)}")

    st.write("")
    sub1, sub2 = st.tabs(["Specs", "Working"])

    with sub1:
        st.markdown(
            """
            | Item | Value |
            |---|---|
            | Model Name | Gravite (`HistGradientBoostingClassifier`) |
            | Boosting Iterations | 167 (Early stopped out of `max_iter=600`) |
            | `max_leaf_nodes` | 255 |
            | `min_samples_leaf` | 100 |
            | `learning_rate` | 0.05 |
            | `l2_regularization` | 1.0 |
            | `class_weight` | balanced |
            | Categorical Handling | Native (`categorical_features='from_dtype'`) |
            | Features | 30 (8 numeric, 11 binary, 11 categorical) |
            | Validation Split | 80/20 Stratified (4,11,040 train / 1,02,761 test) |
            | Production Training | 100% dataset (5,13,801 UK road accident records) |
            | Target | KSI (Fatal + Serious) vs Slight |
            """
        )

    with sub2:
        st.markdown("**How Gravite Works (Step-by-Step)**")
        st.markdown(
            f"""
1. **Input Preparation:** Takes 30 accident parameters (8 numeric, 11 binary flags, and 11 categorical factors) and aligns them to the exact schema used during training.
2. **Pattern Analysis:** Uses `HistGradientBoostingClassifier` to evaluate risk interactions across speed, road users, vehicle profiles, weather, and time of day.
3. **Risk Scoring:** Calculates the probability of a severe collision and flags **KSI** if the risk is **≥ {THRESHOLD:.2f}** (otherwise **Slight**).
            """
        )

        st.markdown("**Key Modeling Decisions:**")
        st.markdown(
            """
- **Handling Class Imbalance:** Since 75.8% of accidents are slight injuries, balanced class weights were used so the model prioritizes catching severe (KSI) cases (`0.66` recall).
- **Direct Categorical Learning:** Categorical features like `hour`, `road_type`, and `junction_detail` are handled natively without complex encoding.
- **No Artificial Data Filling:** Missing values in road and junction records are left as-is, allowing the model to learn real-world missing patterns directly.
- **Preventing Overfitting:** Tree size (`max_leaf_nodes=255`), leaf limits (`min_samples_leaf=100`), and early stopping ensure reliable predictions across 5.13 lakh records.
- **Full-Data Final Training:** After validating performance (`0.48` KSI F1) on an 80/20 split, the production model was trained on 100% of the dataset (converging at 167 iterations).
            """
        )

        with st.expander(f"Why threshold {THRESHOLD:.2f}?"):
            st.write(
                f"Testing multiple probability cutoffs on the 1,02,761 test records showed that {THRESHOLD:.2f} gives the best balance "
                "between KSI F1 (0.48) and Recall (0.66) — catching most serious collisions without triggering excessive false alarms."
            )
