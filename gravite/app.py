import os
import tempfile
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / 'assets'
DATA_DIR = BASE_DIR / 'data'
GRID_URL = 'https://github.com/pratapsingh23/uk-road-accident-severity-prediction/releases/download/v1.0/hot_grid.csv'
MODEL_URL = 'https://github.com/pratapsingh23/uk-road-accident-severity-prediction/releases/download/v1.0/model_bundle.pkl'

st.set_page_config(page_title='Gravite | Road Safety', page_icon='🚦', layout='wide')
st.markdown('''<style>
.stApp {background: #f6f8fc; color: #17233a;}
.block-container {max-width: 1150px; padding-top: 2.5rem; padding-bottom: 3rem;}
h1,h2,h3 {color: #142541; letter-spacing: -0.025em;}
.hero {background: linear-gradient(125deg,#102a50,#205b81); color:white; padding: 2.2rem; border-radius: 18px; margin-bottom: 1.5rem;}
.hero h1 {color:white; margin: 0 0 .4rem; font-size: 2.3rem;}
.hero p {color:#dbe9f5; margin:0; font-size:1.05rem;}
.tag {color:#a5e3da; font-size:.8rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase;}
div[data-testid="stMetric"] {background:white; border:1px solid #e4eaf2; border-radius:14px; padding: 16px;}
div.stButton > button, div.stFormSubmitButton > button {border-radius: 9px; font-weight:600;}
</style>''', unsafe_allow_html=True)

# Codes are retained from the supplied app; they must match the training pipeline.
AREA = {'Urban': 1.0, 'Rural': 2.0, 'Unknown': np.nan}
ROAD_TYPE = {'Single carriageway': 6.0, 'Dual carriageway': 3.0, 'Roundabout': 1.0, 'One way street': 2.0, 'Slip road': 7.0, 'Unknown': np.nan}
LIGHT = {'Daylight': 1.0, 'Darkness - light lit': 4.0, 'Darkness - light unlit': 5.0, 'Darkness - no lighting': 6.0, 'Darkness - lighting unknown': 7.0, 'Unknown': np.nan}
WEATHER = {'Fine, no high winds': 1.0, 'Raining + high winds': 5.0, 'Snowing + high winds': 6.0, 'Fog or mist': 7.0, 'Other': 8.0, 'Unknown': np.nan}
SURFACE = {'Dry': 1.0, 'Wet or damp': 2.0, 'Snow': 3.0, 'Frost or ice': 4.0, 'Flood over 3cm': 5.0, 'Unknown': np.nan}
JUNCTION = {'Not a junction': 0.0, 'T / staggered': 13.0, 'Crossroads': 16.0, '>4 Arms (not roundabout)': 17.0, 'Private drive': 18.0, 'Other junction': 19.0, 'Unknown': np.nan}
CONTROL = {'Give way / Uncontrolled': 4.0, 'Traffic signal': 2.0, 'Stop sign': 3.0, 'Authorised person': 1.0, 'Unknown': np.nan}
DAY = {'Sunday': 1.0, 'Monday': 2.0, 'Tuesday': 3.0, 'Wednesday': 4.0, 'Thursday': 5.0, 'Friday': 6.0, 'Saturday': 7.0}


def download_if_missing(url, name):
    """Download atomically so an interrupted download is never cached as complete."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / name
    # Also support files already placed alongside the original app.
    local = BASE_DIR / name
    if local.is_file():
        return local
    if not target.is_file():
        fd, temp_name = tempfile.mkstemp(dir=DATA_DIR, prefix='.download-')
        os.close(fd)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(temp_name, 'wb') as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return target


@st.cache_resource
def load_bundle():
    # Only load a bundle from a source you trust: pickle can execute arbitrary code.
    bundle = joblib.load(download_if_missing(MODEL_URL, 'model_bundle.pkl'))
    if not isinstance(bundle, dict) or 'model' not in bundle or 'threshold' not in bundle:
        raise ValueError('Model bundle must contain model and threshold.')
    threshold = float(bundle['threshold'])
    if not 0 <= threshold <= 1:
        raise ValueError('Model threshold must be between 0 and 1.')
    return bundle


@st.cache_data
def load_grid():
    grid = pd.read_csv(download_if_missing(GRID_URL, 'hot_grid.csv'))
    required = {'lat_g', 'long_g', 'ksi_rate_capped', 'total', 'urban_share', 'avg_speed_limit'}
    missing = required - set(grid.columns)
    if missing:
        raise ValueError('Missing map columns: ' + ', '.join(sorted(missing)))
    return grid


def show_chart(name):
    path = ASSETS_DIR / name
    if path.is_file():
        st.image(str(path), use_container_width=True)
    else:
        st.info(f'Chart unavailable. Add {name} to the assets folder.')


def home():
    st.markdown('<div class="hero"><div class="tag">Road safety intelligence</div><h1>Better insights. Safer roads.</h1><p>Explore UK road accident patterns and estimate injury severity with Gravite.</p></div>', unsafe_allow_html=True)
    st.subheader('Explore the platform')
    cols = st.columns(3)
    for col, title, desc, page in [
        (cols[0], 'Accident insights', 'Explore severity, road conditions and hotspots.', 'Insights'),
        (cols[1], 'Severity prediction', 'Estimate KSI probability for an accident.', 'Predict'),
        (cols[2], 'Model details', 'Understand the model and its limitations.', 'Model details'),
    ]:
        with col:
            st.markdown(f'### {title}\n{desc}')
            if st.button('Open →', key=page, use_container_width=True):
                st.session_state.navigation = page
                st.rerun()
    st.divider()
    st.caption('Research and educational use only. Predictions are not a substitute for emergency response or professional judgment.')


def insights():
    st.title('Accident insights')
    st.caption('Explore patterns in the supplied UK road accident dataset.')
    for col, label, value in zip(st.columns(4), ['Total accidents', 'Slight', 'Serious', 'Fatal'], ['513,801', '75.8%', '22.7%', '1.5%']):
        col.metric(label, value)
    st.caption('Figures and chart interpretations are taken from the original analysis.')
    tabs = st.tabs(['Severity', 'Urban vs rural', 'Time patterns', 'Pedestrians', 'Deprivation', 'KSI hotspots'])
    with tabs[0]:
        show_chart('distribution_severity.png')
        st.info('Approximately 24.2% of recorded accidents were serious or fatal.')
    with tabs[1]:
        show_chart('urban_vs_rural.png')
        st.info('The reported fatal share was 2.8% in rural areas versus 0.8% in urban areas.')
    with tabs[2]:
        show_chart('season_wise_acc.png')
        st.info('The original analysis reported a broadly similar severity distribution across seasons.')
        show_chart('weekdays_acc.png')
        st.info('In the original rush-hour comparison, 59.6% occurred in the evening and 40.4% in the morning.')
    with tabs[3]:
        show_chart('severity_pedestrian.png')
        st.info('The reported KSI share was 33.6% with pedestrian involvement versus 22.8% without.')
    with tabs[4]:
        show_chart('severity_deprived.png')
        st.info('The reported KSI share across deprivation bands ranged from 23.0% to 25.4%.')
    with tabs[5]:
        st.markdown('KSI share by 1 km grid, for grids with at least 50 accidents.')
        try:
            hot = load_grid()
            fig = px.scatter_map(hot, lat='lat_g', lon='long_g', color='ksi_rate_capped', size='total', size_max=18,
                color_continuous_scale='YlOrRd', hover_data={'lat_g': False, 'long_g': False, 'total': True,
                'ksi_rate_capped': ':.1f', 'urban_share': ':.0f', 'avg_speed_limit': ':.0f'},
                zoom=5, center={'lat': 55.3781, 'lon': -3.4360}, map_style='carto-positron', height=600)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f'Interactive map unavailable: {exc}')
            show_chart('map_plot.png')


def predict():
    st.title('Severity prediction')
    st.caption('Estimate the probability of a killed or seriously injured (KSI) outcome.')
    with st.form('predict_form'):
        a, b = st.columns(2)
        with a:
            st.subheader('Accident details')
            speed = st.selectbox('Speed limit (mph)', [20, 30, 40, 50, 60, 70], index=2)
            vehicles = st.number_input('Number of vehicles', min_value=1, max_value=30, value=1)
            min_age = st.number_input('Minimum driver age', min_value=16, max_value=100, value=28)
            max_age = st.number_input('Maximum driver age', min_value=16, max_value=100, value=32)
            engine = st.number_input('Maximum engine capacity (cc)', min_value=0, max_value=15000, value=1200, step=50)
            male = st.slider('Male driver share', 0.0, 1.0, 0.1, help='Fraction of drivers who are male, from 0 to 1.')
            imd = st.slider('Average casualty IMD decile', 1.0, 10.0, 5.0, 0.5, help='1 = most deprived; 10 = least deprived.')
        with b:
            st.subheader('Road & conditions')
            area = st.selectbox('Area', list(AREA))
            road = st.selectbox('Road type', list(ROAD_TYPE))
            light = st.selectbox('Light conditions', list(LIGHT))
            weather = st.selectbox('Weather', list(WEATHER))
            surface = st.selectbox('Road surface', list(SURFACE))
            junction = st.selectbox('Junction detail', list(JUNCTION))
            control = st.selectbox('Junction control', list(CONTROL))
            day = st.selectbox('Day of week', list(DAY))
        st.subheader('Road users & manoeuvres')
        c1, c2, c3 = st.columns(3)
        cycle = c1.checkbox('Cycle'); motorcycle = c2.checkbox('Motorcycle'); goods = c3.checkbox('Goods vehicle')
        bus = c1.checkbox('Bus'); vul_veh = c2.checkbox('Vulnerable vehicle'); vul_cas = c3.checkbox('Vulnerable casualty')
        turning = c1.checkbox('Turning'); escooter = c2.checkbox('E-scooter casualty'); overtaking = c3.checkbox('Overtaking')
        pedestrian = c1.checkbox('Pedestrian')
        submitted = st.form_submit_button('Calculate prediction', type='primary', use_container_width=True)
    if not submitted:
        return
    if min_age > max_age:
        st.error('Minimum driver age cannot be greater than maximum driver age.')
        return
    row = {'speed_limit': float(speed), 'number_of_vehicles': vehicles, 'min_driver_age': float(min_age),
        'max_driver_age': float(max_age), 'max_engine_cc': float(engine), 'male_driver_share': male,
        'avg_casualty_imd': imd, 'has_cycle': int(cycle), 'has_motorcycle': int(motorcycle),
        'has_goods_vehicle': int(goods), 'has_bus': int(bus), 'has_vulnerable_vehicle': int(vul_veh),
        'has_overtaking': int(overtaking), 'has_turning': int(turning), 'has_vulnerable_casualty': int(vul_cas),
        'has_escooter_casualty': int(escooter), 'has_pedestrian_movement': int(pedestrian),
        'urban_or_rural_area': AREA[area], 'road_type': ROAD_TYPE[road], 'light_conditions': LIGHT[light],
        'weather_conditions': WEATHER[weather], 'road_surface_conditions': SURFACE[surface],
        'junction_detail': JUNCTION[junction], 'junction_control': CONTROL[control], 'day_of_week': DAY[day]}
    try:
        with st.spinner('Loading model and calculating prediction…'):
            bundle = load_bundle()
            model = bundle['model']
            X = pd.DataFrame([row])
            # Respect the feature order if the fitted pipeline exposes it.
            if hasattr(model, 'feature_names_in_'):
                expected = list(model.feature_names_in_)
                if set(expected) != set(X.columns):
                    raise ValueError(f'Feature mismatch: missing {sorted(set(expected)-set(X.columns))}; unexpected {sorted(set(X.columns)-set(expected))}')
                X = X[expected]
            classes = list(model.classes_)
            if 1 not in classes:
                raise ValueError('Expected KSI to be encoded as class 1 in the model.')
            probability = float(model.predict_proba(X)[0, classes.index(1)])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Model returned an invalid probability.')
            threshold = float(bundle['threshold'])
        st.divider()
        st.subheader('Prediction result')
        st.metric('Estimated KSI probability', f'{probability:.1%}')
        st.progress(probability)
        if probability >= threshold:
            st.error('Classification: KSI (killed or seriously injured)')
        else:
            st.success('Classification: Slight injury')
        st.caption(f'Model decision threshold: {threshold:.1%}. This is an estimate, not a guarantee; it is not intended for emergency or safety-critical decisions.')
    except Exception as exc:
        st.error(f'Prediction could not be completed: {exc}')
        st.info('Check the model file, internet connection, training feature schema and package versions.')


def details():
    st.title('Model details')
    st.caption('A closer look at the supplied Random Forest model and its reported evaluation.')
    threshold_text = 'See model bundle'
    # Avoid a download on this page; report the original documented threshold.
    for col, label, value in zip(st.columns(4), ['KSI recall', 'KSI F1', 'Documented threshold', 'Test records'], ['0.74', '0.464', '0.45', '102,761']):
        col.metric(label, value)
    st.caption('Metrics and specifications below are from the original code; they have not been independently verified.')
    specs, importance, pipeline = st.tabs(['Specifications', 'Feature importance', 'Pipeline & limitations'])
    with specs:
        st.markdown('''| Item | Reported value |
|---|---|
| Algorithm | Random Forest classifier |
| Trees | 100 |
| Maximum depth | 15 |
| Class weight | balanced |
| Random state | 42 |
| Features | 25 |
| Train / test records | 411,040 / 102,761 |
| Target | KSI (fatal or serious) vs slight |''')
    with importance:
        importance = pd.Series({'has_vulnerable': 0.094, 'number_of_vehicles': 0.081, 'max_driver_age': 0.078,
            'speed_limit': 0.064, 'min_driver_age': 0.059, 'max_engine_cc': 0.058, 'avg_casualty_imd': 0.041,
            'male_driver_share': 0.041, 'has_vulnerable_vehicle': 0.039, 'has_motorcycle': 0.034}).sort_values()
        st.bar_chart(importance, height=400)
        st.caption('Reported impurity-based importances, reproduced from the original code. Verify feature names against the trained model.')
    with pipeline:
        st.markdown('''The original description reports a column transformer, one-hot encoding for categorical inputs and a balanced random forest. Predictions use the threshold stored in the model bundle.

- Reported KSI recall: **0.74**; reported F1: **0.464**.
- These historical metrics do not guarantee performance on new accidents.
- If the threshold was selected on the test set, the reported test performance may be optimistic. Use a separate validation set for threshold tuning and an untouched test set for final evaluation.
- Predictions depend on the training schema, coding, input quality and model calibration.''')


st.sidebar.markdown('## 🚦 Gravite')
st.sidebar.caption('UK road safety intelligence')
page = st.sidebar.radio('Navigation', ['Home', 'Insights', 'Predict', 'Model details'], key='navigation')
st.sidebar.divider()
st.sidebar.caption('Explore data responsibly. KSI means killed or seriously injured.')
{'Home': home, 'Insights': insights, 'Predict': predict, 'Model details': details}[page]()
