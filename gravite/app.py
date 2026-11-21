import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

RELEASE = "https://github.com/pratapsingh23/uk-road-accident-severity-prediction/releases/download/v1.0"
GRID_URL = f"{RELEASE}/hot_grid.csv"
MODEL_URL = f"{RELEASE}/model_bundle.pkl"
GRID_PATH = os.path.join(BASE_DIR, "hot_grid.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_bundle.pkl")

st.set_page_config(page_title="Gravite | UK Road Accident Severity", page_icon="🛣️", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:wght@600;700&display=swap');

    :root {
        --sign: #00468b;
        --sign-dark: #003469;
        --marking: #f2c230;
        --asphalt: #23272e;
        --muted: #5b6573;
        --paper: #eceff3;
        --line: #d3d9e1;
        --danger: #c0301f;
        --safe: #1f7a4d;
    }
    html, body, [class*="css"], .stApp { font-family: 'Barlow', sans-serif; }
    .stApp { background: var(--paper); }
    .block-container { max-width: 1100px; padding-top: 1.8rem; padding-bottom: 3rem; }
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

    h1, h2, h3 { color: var(--asphalt); font-family: 'Barlow Condensed', 'Barlow', sans-serif; letter-spacing: 0; }
    h1 { font-weight: 700; font-size: 2.4rem; }
    h2, h3 { font-weight: 600; }

    /* Home: motorway-sign hero */
    .sign {
        background: var(--sign); border-radius: 14px; padding: 8px;
        box-shadow: 0 2px 0 var(--sign-dark);
    }
    .sign-inner {
        border: 3px solid #fff; border-radius: 9px; padding: 2.2rem 2.4rem;
    }
    .sign-inner h1 { color: #fff; font-size: 3.6rem; margin: 0; line-height: 1; }
    .sign-inner p { color: #dbe8f7; font-size: 1.2rem; margin: .6rem 0 0; }
    .marking {
        height: 6px; margin: 1.1rem 0 1.8rem;
        background: repeating-linear-gradient(90deg, var(--marking) 0 46px, transparent 46px 66px);
    }

    .nav-card {
        background: #fff; border: 1px solid var(--line); border-left: 6px solid var(--sign);
        border-radius: 8px; padding: 1.1rem 1.3rem .8rem; min-height: 140px;
    }
    .nav-card h3 { margin: 0 0 .3rem; font-size: 1.5rem; }
    .nav-card p { margin: 0; color: var(--muted); font-size: 1rem; }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #fff; border: 1px solid var(--line); border-top: 4px solid var(--sign);
        border-radius: 8px; padding: .8rem 1rem;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; color: var(--asphalt); }

    /* Tabs */
    button[data-baseweb="tab"] { font-weight: 600; }
    div[data-baseweb="tab-highlight"] { background: var(--sign) !important; height: 3px; }

    /* Buttons */
    .stButton > button {
        border-radius: 6px; font-weight: 600; background: #fff;
        border: 1.5px solid var(--sign); color: var(--sign);
    }
    .stButton > button:hover { background: var(--sign); color: #fff; border-color: var(--sign); }
    .stFormSubmitButton > button {
        border-radius: 6px; font-weight: 700; font-size: 1.05rem;
        background: var(--sign); color: #fff; border: 3px solid #fff;
        outline: 2px solid var(--sign);
    }
    .stFormSubmitButton > button:hover { background: var(--sign-dark); color: #fff; }

    /* Form panel */
    div[data-testid="stForm"] { background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 1.4rem; }

    /* Result */
    .verdict {
        border-radius: 8px; padding: 1.1rem 1.4rem; font-weight: 600; font-size: 1.25rem;
        margin-top: .8rem; background: #fff; border: 1px solid var(--line); border-left-width: 10px;
    }
    .verdict.ksi { color: var(--danger); border-left-color: var(--danger); }
    .verdict.slight { color: var(--safe); border-left-color: var(--safe); }
    .foot { text-align: center; color: var(--muted); font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def centered(tag: str, text: str):
    st.markdown(f"<{tag} style='text-align:center'>{text}</{tag}>", unsafe_allow_html=True)


def chart(name: str) -> str:
    return os.path.join(ASSETS_DIR, name)


def show_chart(name: str):
    path = chart(name)
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning(f"Chart not found: assets/{name}")


# ---------------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------------
def download(url: str, dest: str):
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)  # avoids a half-downloaded file being treated as valid


@st.cache_data(show_spinner=False)
def load_grid() -> pd.DataFrame:
    if not os.path.exists(GRID_PATH):
        download(GRID_URL, GRID_PATH)
    return pd.read_csv(GRID_PATH)


@st.cache_resource(show_spinner=False)
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        download(MODEL_URL, MODEL_PATH)
    return joblib.load(MODEL_PATH)


try:
    with st.spinner("Loading model..."):
        bundle = load_bundle()
    model = bundle["model"]
    THRESHOLD = float(bundle["threshold"])
except Exception as exc:
    st.error(f"The model could not be loaded. Check your internet connection and try again.\n\n{exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Label encodings (must match training)
# ---------------------------------------------------------------------------
AREA = {"Urban": 1.0, "Rural": 2.0, "Unknown": np.nan}

ROAD_TYPE = {
    "Single carriageway": 6.0, "Dual carriageway": 3.0, "Roundabout": 1.0,
    "One way street": 2.0, "Slip road": 7.0, "Unknown": np.nan,
}

LIGHT = {
    "Daylight": 1.0, "Darkness - lights lit": 4.0, "Darkness - lights unlit": 5.0,
    "Darkness - no lighting": 6.0, "Darkness - lighting unknown": 7.0, "Unknown": np.nan,
}

WEATHER = {
    "Fine, no high winds": 1.0, "Raining + high winds": 5.0, "Snowing + high winds": 6.0,
    "Fog or mist": 7.0, "Other": 8.0, "Unknown": np.nan,
}

SURFACE = {
    "Dry": 1.0, "Wet or damp": 2.0, "Snow": 3.0, "Frost or ice": 4.0,
    "Flood over 3cm": 5.0, "Unknown": np.nan,
}

JUNCTION = {
    "Not a junction": 0.0, "T / staggered": 13.0, "Crossroads": 16.0,
    "More than 4 arms (not roundabout)": 17.0, "Private drive": 18.0,
    "Other junction": 19.0, "Unknown": np.nan,
}

CONTROL = {
    "Give way / uncontrolled": 4.0, "Traffic signal": 2.0, "Stop sign": 3.0,
    "Authorised person": 1.0, "Unknown": np.nan,
}

DAY = {
    "Sunday": 1.0, "Monday": 2.0, "Tuesday": 3.0, "Wednesday": 4.0,
    "Thursday": 5.0, "Friday": 6.0, "Saturday": 7.0,
}

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"


def go(page: str):
    st.session_state.page = page
    st.rerun()


def back_button():
    if st.button("← Back to home"):
        go("home")


page = st.session_state.page

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
if page == "home":
    st.markdown(
        """
        <div class="sign"><div class="sign-inner">
                <h1>Gravite</h1>
                <p>UK road accident severity analysis and prediction</p>
            </div></div>
            <div class="marking"></div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="large")
    cards = [
        (c1, "Explore the data", "Charts and maps showing when, where and why serious accidents happen.", "Open analysis", "eda"),
        (c2, "Predict severity", "Enter the conditions of an accident and get the chance it is serious or fatal.", "Start prediction", "predict"),
        (c3, "Model details", "How the Random Forest was built, tuned and evaluated.", "View details", "model_detail"),
    ]
    for col, title, desc, label, target in cards:
        with col:
            st.markdown(f"<div class='nav-card'><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)
            if st.button(label, key=f"nav_{target}", use_container_width=True):
                go(target)

# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
elif page == "eda":
    back_button()
    st.title("Exploring UK road accidents")
    st.caption("Key patterns, high-risk times and the factors behind accident severity.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total accidents", "513,801")
    k2.metric("Slight", "75.8%")
    k3.metric("Serious", "22.7%")
    k4.metric("Fatal", "1.5%")

    st.write("")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Severity", "Urban vs rural", "Time patterns", "Pedestrians", "Deprivation", "KSI hotspots"]
    )

    with tab1:
        show_chart("distribution_severity.png")
        st.info(
            "About 76 in 100 accidents cause only slight injuries, but around 1 in 4 "
            "is serious or fatal."
        )

    with tab2:
        show_chart("urban_vs_rural.png")
        st.warning("The rural fatal rate is 2.8%, about 3.5 times the urban rate (0.8%).")

    with tab3:
        show_chart("season_wise_acc.png")
        st.info("Severity stays almost flat across seasons, so no single season is clearly more dangerous.")
        show_chart("weekdays_acc.png")
        st.info("Weekday evening rush hours see more accidents (59.6%) than morning rush hours (40.4%).")

    with tab4:
        show_chart("severity_pedestrian.png")
        st.warning("When a pedestrian is involved, 33.6% of accidents are KSI, against 22.8% without one.")

    with tab5:
        show_chart("severity_deprived.png")
        st.info(
            "The KSI share is nearly flat across deprivation bands (23.0% to 25.4%), "
            "so deprivation shows no strong effect on severity."
        )

    with tab6:
        st.markdown(
            "KSI rate by 1 km grid square. Darker red means a higher share of killed or "
            "seriously injured casualties (squares with at least 50 accidents)."
        )
        try:
            hot = load_grid()
            fig = px.scatter_map(
                hot, lat="lat_g", lon="long_g",
                color="ksi_rate_capped", size="total", size_max=18,
                color_continuous_scale="YlOrRd",
                labels={"ksi_rate_capped": "KSI rate (%)"},
                hover_data={
                    "lat_g": False, "long_g": False, "total": True,
                    "ksi_rate_capped": ":.1f", "urban_share": ":.0f",
                    "avg_speed_limit": ":.0f",
                },
                zoom=5, center={"lat": 55.3781, "lon": -3.4360},
                map_style="carto-positron", height=600,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"The interactive map could not be loaded ({exc}). Showing the static map instead.")
            show_chart("map_plot.png")

    st.divider()
    st.markdown("<div class='foot'>Charts from eda.ipynb · Gravite</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
elif page == "predict":
    back_button()
    st.title("Predict accident severity")
    st.caption("KSI means killed or seriously injured. Slight means minor injuries only.")

    with st.form("predict_form"):
        a, b = st.columns(2, gap="large")

        with a:
            st.subheader("Vehicles and people")
            speed_limit = st.selectbox("Speed limit (mph)", [20, 30, 40, 50, 60, 70], index=2)
            n_vehicles = st.number_input("Number of vehicles", 1, 30, 1)
            min_age = st.number_input("Youngest driver age", 16, 100, 28)
            max_age = st.number_input("Oldest driver age", 16, 100, 32)
            engine_cc = st.number_input("Largest engine size (cc)", 0, 15000, 1200, step=50)
            male_share = st.slider("Share of male drivers", 0.0, 1.0, 0.5)
            imd = st.slider("Average casualty deprivation (1 = most deprived)", 1.0, 10.0, 5.0, 0.5)

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
            st.subheader("Road and weather conditions")
            area = st.selectbox("Area", list(AREA))
            road = st.selectbox("Road type", list(ROAD_TYPE))
            light = st.selectbox("Light conditions", list(LIGHT))
            weather = st.selectbox("Weather", list(WEATHER))
            surface = st.selectbox("Road surface", list(SURFACE))
            junction = st.selectbox("Junction detail", list(JUNCTION))
            control = st.selectbox("Junction control", list(CONTROL))
            day = st.selectbox("Day of week", list(DAY))

        submitted = st.form_submit_button("Predict severity", use_container_width=True)

    if submitted:
        if min_age > max_age:
            st.error("The youngest driver age cannot be higher than the oldest driver age.")
            st.stop()

        row = {
            # numeric
            "speed_limit": float(speed_limit),
            "number_of_vehicles": int(n_vehicles),
            "min_driver_age": float(min_age),
            "max_driver_age": float(max_age),
            "max_engine_cc": float(engine_cc),
            "male_driver_share": float(male_share),
            "avg_casualty_imd": float(imd),
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

        X_new = pd.DataFrame([row])
        # Make sure the columns are in the same order the model was trained with
        if hasattr(model, "feature_names_in_"):
            X_new = X_new.reindex(columns=list(model.feature_names_in_))

        try:
            proba = float(model.predict_proba(X_new)[0, 1])
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        is_ksi = proba >= THRESHOLD

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("KSI probability", f"{proba * 100:.1f}%")
        m2.metric("Decision threshold", f"{THRESHOLD * 100:.0f}%")
        st.progress(min(max(proba, 0.0), 1.0))

        if is_ksi:
            st.markdown(
                "<div class='verdict ksi'>High risk: likely killed or seriously injured (KSI)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='verdict slight'>Lower risk: likely slight injuries</div>",
                unsafe_allow_html=True,
            )
        st.caption("This is a statistical estimate and can be wrong. Do not use it for safety-critical decisions.")

# ---------------------------------------------------------------------------
# Model details
# ---------------------------------------------------------------------------
elif page == "model_detail":
    back_button()
    st.title("Model details")
    st.caption("How Gravite predicts KSI with a Random Forest.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("KSI recall", "0.74")
    k2.metric("KSI F1", "0.464")
    k3.metric("Threshold", "0.45")
    k4.metric("Test size", "102,761")

    st.write("")
    tab1, tab2, tab3 = st.tabs(["Specifications", "Feature importance", "Pipeline and approach"])

    with tab1:
        st.markdown(
            """
| Item | Value |
|---|---|
| Algorithm | Random Forest classifier |
| Trees (n_estimators) | 100 |
| max_depth | 15 |
| class_weight | balanced |
| random_state | 42 |
| Features | 25 |
| Training rows | 411,040 (80%) |
| Test rows | 102,761 (20%, stratified) |
| Dataset | 513,801 UK road accident records |
| Target | KSI (fatal + serious) vs slight |
"""
        )

    with tab2:
        importance = pd.Series(
            {
                "has_vulnerable": 0.094,
                "number_of_vehicles": 0.081,
                "max_driver_age": 0.078,
                "speed_limit": 0.064,
                "min_driver_age": 0.059,
                "max_engine_cc": 0.058,
                "avg_casualty_imd": 0.041,
                "male_driver_share": 0.041,
                "has_vulnerable_vehicle": 0.039,
                "has_motorcycle": 0.034,
            }
        ).sort_values()
        fig = px.bar(
            importance, orientation="h", height=420,
            labels={"value": "Importance", "index": ""},
        )
        fig.update_traces(marker_color="#00468b")
        fig.update_layout(showlegend=False, margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Top 10 features by impurity-based importance. Vulnerable road users, "
            "speed and driver age matter most."
        )

    with tab3:
        st.subheader("Pipeline")
        st.code(
            """raw inputs (25 features)
   -> ColumnTransformer
        |- numeric       : passthrough (NaN kept as-is)
        |- categorical   : OneHotEncoder (raw codes, NaN = its own category)
        |- binary flags  : passthrough
   -> RandomForest (100 trees, depth 15, balanced)
   -> P(KSI) >= 0.45 ?  KSI : Slight""",
            language=None,
        )

        st.subheader("Training journey")
        st.markdown(
            """
- **Baseline:** KSI recall was only 0.15, so the model almost always guessed "slight".
- **Class weights:** `class_weight='balanced'` raised recall to 0.62, catching about 4.2 times more serious cases.
- **Threshold tuning:** a sweep settled on 0.45, giving recall 0.74 and F1 0.464.
- **Accuracy trap:** 65% accuracy looks low, but always guessing "slight" scores 75.8%, so recall and F1 on KSI were prioritised.
- **Missing values:** no imputation was used. Random Forest handles NaN natively, and `junction_control` was 44.6% missing.
"""
        )

        with st.expander("Why a threshold of 0.45?"):
            st.write(
                "A precision-recall sweep on the test set showed 0.45 maximised KSI F1 (0.464) "
                "while lifting recall to 0.74. This is a safety-first trade-off: a false alarm "
                "costs less than a missed serious accident."
            )
