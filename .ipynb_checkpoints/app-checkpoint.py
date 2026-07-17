"""
Ford GoBike (Bay Wheels) 2018 — Interactive Ridership Dashboard
Built from the EDA in Ford_GoBike_2018_EDA.ipynb (P Suman Sangeet, LABMENTIX Data
Analytics and AI )

Run:  streamlit run app.py
"""
import io
import zipfile
import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page config & style
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ford GoBike 2018 — Ridership Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0B6E4F"
ACCENT = "#F2A007"
SECONDARY = "#3B6E8F"
MUTED = "#8A9BA8"
PALETTE_SEQ = [PRIMARY, ACCENT, SECONDARY, "#B23A48", "#6C4A9C", "#4C9F70"]

st.markdown(f"""
<style>
    .stMetric {{
        background-color: rgba(11,110,79,0.06);
        border: 1px solid rgba(11,110,79,0.15);
        border-radius: 10px;
        padding: 10px 14px;
    }}
    .insight-box {{
        background-color: rgba(242,160,7,0.10);
        border-left: 4px solid {ACCENT};
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0 18px 0;
        font-size: 0.95rem;
    }}
    .rec-box {{
        background-color: rgba(11,110,79,0.07);
        border-left: 4px solid {PRIMARY};
        border-radius: 6px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.95rem;
    }}
</style>
""", unsafe_allow_html=True)

MONTH_ORDER = ['January','February','March','April','May','June','July',
               'August','September','October','November','December']
DOW_ORDER = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

RAW_COLUMN_MAP = {
    'duration_sec': 'duration_sec',
    'start_time': 'start_time',
    'end_time': 'end_time',
    'start_station_id': 'start_station_id',
    'start_station_name': 'start_station_name',
    'start_station_latitude': 'start_station_latitude',
    'start_station_longitude': 'start_station_longitude',
    'end_station_id': 'end_station_id',
    'end_station_name': 'end_station_name',
    'end_station_latitude': 'end_station_latitude',
    'end_station_longitude': 'end_station_longitude',
    'bike_id': 'bike_id',
    'user_type': 'user_type',
    'member_birth_year': 'member_birth_year',
    'member_gender': 'member_gender',
}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_demo_data():
    path = os.path.join(os.path.dirname(__file__), "demo_gobike_2018.csv.gz")
    df = pd.read_csv(path, parse_dates=['start_time', 'end_time'])
    df['start_month_name'] = pd.Categorical(df['start_month_name'], categories=MONTH_ORDER, ordered=True)
    df['start_day_of_week'] = pd.Categorical(df['start_day_of_week'], categories=DOW_ORDER, ordered=True)
    return df


def wrangle_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates the notebook's Data Wrangling section on a raw uploaded file."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    if 'end_time' in df.columns:
        df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

    df = df.dropna(subset=[c for c in ['start_station_id', 'end_station_id'] if c in df.columns]).copy()

    df['duration_min'] = df['duration_sec'] / 60
    if 'start_time' in df.columns:
        df['start_month'] = df['start_time'].dt.month
        df['start_month_name'] = pd.Categorical(
            df['start_time'].dt.month_name(), categories=MONTH_ORDER, ordered=True)
        df['start_day_of_week'] = pd.Categorical(
            df['start_time'].dt.day_name(), categories=DOW_ORDER, ordered=True)
        df['start_hour'] = df['start_time'].dt.hour
        df['is_weekend'] = df['start_day_of_week'].isin(['Saturday', 'Sunday'])
        if 'member_birth_year' in df.columns:
            df['age'] = df['start_time'].dt.year - df['member_birth_year']

    if 'start_station_name' in df.columns and 'end_station_name' in df.columns:
        df['same_station'] = df['start_station_name'] == df['end_station_name']
        df['route'] = df['start_station_name'] + ' \u2192 ' + df['end_station_name']

    return df


@st.cache_data(show_spinner=True)
def load_uploaded(files) -> pd.DataFrame:
    frames = []
    for f in files:
        name = f.name.lower()
        if name.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(f.read())) as z:
                for inner in z.namelist():
                    if inner.lower().endswith('.csv'):
                        with z.open(inner) as fh:
                            frames.append(pd.read_csv(fh))
        elif name.endswith('.csv'):
            frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    return wrangle_raw(raw)


# ----------------------------------------------------------------------------
# Sidebar — data source & filters
# ----------------------------------------------------------------------------
st.sidebar.title("🚲 Ford GoBike 2018")
st.sidebar.caption("Bay Wheels ridership explorer")

st.sidebar.markdown("### Data source")
uploaded = st.sidebar.file_uploader(
    "Upload real Ford GoBike monthly CSV(s) or a data.zip",
    type=['csv', 'zip'], accept_multiple_files=True,
    help="Optional — without this, the dashboard runs on a demo sample "
         "statistically matched to the full 2018 dataset (1,863,721 trips)."
)

if uploaded:
    df_full = load_uploaded(uploaded)
    if df_full.empty:
        st.sidebar.error("Couldn't parse any valid CSV rows — falling back to demo data.")
        df_full = load_demo_data()
        st.sidebar.info("📊 Running on demo data (statistically matched sample)")
    else:
        st.sidebar.success(f"✅ Loaded {len(df_full):,} real trips")
else:
    df_full = load_demo_data()
    st.sidebar.info("📊 Running on demo data (statistically matched sample of the "
                     "real 1,863,721-trip 2018 dataset). Upload real CSVs above for "
                     "full-fidelity numbers.")

st.sidebar.markdown("### Filters")

months_present = [m for m in MONTH_ORDER if m in df_full['start_month_name'].unique().tolist()]
sel_months = st.sidebar.multiselect("Month", months_present, default=months_present)

user_types_present = sorted(df_full['user_type'].dropna().unique().tolist())
sel_user_types = st.sidebar.multiselect("User type", user_types_present, default=user_types_present)

day_type = st.sidebar.radio("Day type", ["All", "Weekday only", "Weekend only"], horizontal=False)

if 'member_gender' in df_full.columns:
    gender_present = sorted(df_full['member_gender'].dropna().unique().tolist())
    sel_gender = st.sidebar.multiselect("Gender (reported)", gender_present, default=gender_present)
else:
    sel_gender = None

max_dur = int(np.nanpercentile(df_full['duration_min'], 99.5)) if len(df_full) else 60
dur_range = st.sidebar.slider("Trip duration (minutes)", 0, max(max_dur, 5), (0, max(max_dur, 5)))

# Apply filters
df = df_full[df_full['start_month_name'].isin(sel_months) & df_full['user_type'].isin(sel_user_types)]
if day_type == "Weekday only":
    df = df[~df['is_weekend']]
elif day_type == "Weekend only":
    df = df[df['is_weekend']]
if sel_gender is not None:
    df = df[df['member_gender'].isin(sel_gender) | df['member_gender'].isna()] if len(sel_gender) == len(gender_present) else df[df['member_gender'].isin(sel_gender)]
df = df[(df['duration_min'] >= dur_range[0]) & (df['duration_min'] <= dur_range[1])]

st.sidebar.markdown("---")
st.sidebar.caption(f"**{len(df):,}** trips match current filters "
                    f"(of {len(df_full):,} loaded)")
st.sidebar.caption("Built by P Suman Sangeet · LABMENTIX Data Analytics and AI")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🚲 Ford GoBike (Bay Wheels) — 2018 Ridership Dashboard")
st.caption(
    "Interactive deployment of the 2018 exploratory data analysis — San Francisco, "
    "Oakland, Berkeley & San Jose bike-share trips. Use the sidebar to filter and "
    "explore; every chart below reacts live."
)

if df.empty:
    st.warning("No trips match the current filter combination — widen a filter in the sidebar.")
    st.stop()

tabs = st.tabs([
    "📌 Overview", "🕒 Time Patterns", "👥 Rider Segments",
    "📍 Stations & Routes", "🔗 Correlations", "💡 Recommendations",
])

# ----------------------------------------------------------------------------
# TAB 1 — Overview
# ----------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total trips", f"{len(df):,}")
    c2.metric("Median duration", f"{df['duration_min'].median():.1f} min")
    c3.metric("Mean duration", f"{df['duration_min'].mean():.1f} min")
    sub_share = (df['user_type'] == 'Subscriber').mean() * 100
    c4.metric("Subscriber share", f"{sub_share:.1f}%")
    c5.metric("Unique stations", f"{df['start_station_name'].nunique():,}")

    st.markdown("")
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Trip duration distribution")
        fig = px.histogram(df[df['duration_min'] <= 60], x='duration_min', nbins=80,
                            color_discrete_sequence=[PRIMARY])
        fig.add_vline(x=df['duration_min'].median(), line_dash='dash', line_color=ACCENT,
                      annotation_text=f"Median {df['duration_min'].median():.1f} min")
        fig.add_vline(x=df['duration_min'].mean(), line_dash='dash', line_color=SECONDARY,
                      annotation_text=f"Mean {df['duration_min'].mean():.1f} min")
        fig.update_layout(xaxis_title="Duration (minutes, capped at 60 for readability)",
                           yaxis_title="Number of trips", showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">📈 Trip duration is strongly right-skewed — '
                    'the median is well below the mean because of a long tail of very long '
                    'rides. The median, not the mean, is the representative "typical trip".</div>',
                    unsafe_allow_html=True)

    with right:
        st.subheader("Rider composition")
        ut_counts = df['user_type'].value_counts()
        fig = px.pie(values=ut_counts.values, names=ut_counts.index,
                     color=ut_counts.index,
                     color_discrete_map={'Subscriber': PRIMARY, 'Customer': ACCENT}, hole=0.45)
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">🎫 The recurring, commuter Subscriber base '
                    'dominates ridership — the core membership revenue line to protect and grow.</div>',
                    unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 2 — Time patterns
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Trips by hour of day, split by rider type")
    hourly = df.groupby(['start_hour', 'user_type'], observed=True).size().reset_index(name='count')
    fig = px.line(hourly, x='start_hour', y='count', color='user_type', markers=True,
                  color_discrete_map={'Subscriber': PRIMARY, 'Customer': ACCENT})
    fig.update_layout(xaxis_title="Hour of day (24h)", yaxis_title="Number of trips", height=380,
                       xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight-box">⏰ Subscribers show a sharp AM/PM double-peak — '
                'classic commuting. Customers ride in a flatter single hump through the '
                'afternoon — spontaneous/leisure use.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Trips by day of week")
        dow_counts = df['start_day_of_week'].value_counts().reindex(DOW_ORDER)
        colors = [PRIMARY if d not in ('Saturday', 'Sunday') else ACCENT for d in DOW_ORDER]
        fig = px.bar(x=dow_counts.index, y=dow_counts.values, color=DOW_ORDER,
                     color_discrete_sequence=colors)
        fig.update_layout(xaxis_title="", yaxis_title="Number of trips", showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Monthly ridership trend")
        month_counts = df['start_month_name'].value_counts().reindex(
            [m for m in MONTH_ORDER if m in df['start_month_name'].unique()])
        fig = px.area(x=month_counts.index, y=month_counts.values,
                      color_discrete_sequence=[SECONDARY])
        fig.update_traces(line_color=SECONDARY)
        fig.update_layout(xaxis_title="", yaxis_title="Number of trips", height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">📅 Ridership is clearly weekday-dominated (roughly '
                '2x weekend volume) and seasonal, climbing through spring/summer toward an '
                'autumn peak before dropping off in winter.</div>', unsafe_allow_html=True)

    st.subheader("Trip volume heatmap — day of week × hour of day")
    pivot = df.pivot_table(index='start_day_of_week', columns='start_hour',
                            values='duration_sec', aggfunc='count', observed=True).reindex(DOW_ORDER)
    fig = px.imshow(pivot, color_continuous_scale='YlGnBu', aspect='auto',
                    labels=dict(x="Hour of day", y="", color="Trips"))
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight-box">🗺️ The two commute peaks appear as dark vertical '
                'bands only Monday–Friday; weekends show one broad warm midday band instead — '
                'the clearest single confirmation of the commute-vs-leisure split.</div>',
                unsafe_allow_html=True)

    st.subheader("Weekday vs weekend, by hour")
    hr_wk = df.groupby(['start_hour', 'is_weekend'], observed=True).size().reset_index(name='count')
    hr_wk['Day type'] = hr_wk['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
    fig = px.line(hr_wk, x='start_hour', y='count', color='Day type', markers=True,
                  color_discrete_map={'Weekday': PRIMARY, 'Weekend': ACCENT})
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Number of trips", height=360,
                       xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — Rider segments
# ----------------------------------------------------------------------------
with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Trip duration by user type")
        fig = px.box(df, x='user_type', y='duration_min', color='user_type',
                     category_orders={'user_type': ['Subscriber', 'Customer']},
                     color_discrete_map={'Subscriber': PRIMARY, 'Customer': ACCENT},
                     points=False)
        fig.update_layout(showlegend=False, yaxis_title="Duration (minutes)", xaxis_title="",
                           height=380)
        st.plotly_chart(fig, use_container_width=True)
        stats = df.groupby('user_type', observed=True)['duration_min'].agg(['mean', 'median']).round(1)
        st.dataframe(stats, use_container_width=True)

    with c2:
        if 'member_gender' in df.columns and df['member_gender'].notna().any():
            st.subheader("Gender distribution")
            gender_counts = df['member_gender'].value_counts()
            fig = px.bar(x=gender_counts.index, y=gender_counts.values,
                        color=gender_counts.index, color_discrete_sequence=PALETTE_SEQ)
            pct = (gender_counts / gender_counts.sum() * 100).round(1)
            fig.update_traces(text=[f"{p}%" for p in pct], textposition='outside')
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of trips", height=380)
            st.plotly_chart(fig, use_container_width=True)

    if 'age' in df.columns and df['age'].notna().any():
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Rider age distribution")
            age_df = df.dropna(subset=['age'])
            fig = px.histogram(age_df, x='age', nbins=50, color_discrete_sequence=[SECONDARY])
            fig.add_vline(x=age_df['age'].median(), line_dash='dash', line_color=ACCENT,
                          annotation_text=f"Median age {age_df['age'].median():.0f}")
            fig.update_layout(xaxis_title="Age (years)", yaxis_title="Number of trips", height=360)
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            st.subheader("Age vs. trip duration")
            sample = age_df.sample(n=min(8000, len(age_df)), random_state=42)
            fig = px.scatter(sample, x='age', y='duration_min', opacity=0.15,
                             color_discrete_sequence=[PRIMARY], trendline=None)
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(xaxis_title="Age (years)", yaxis_title="Duration (minutes)",
                               yaxis_range=[0, 60], height=360)
            st.plotly_chart(fig, use_container_width=True)
            corr = age_df[['age', 'duration_min']].corr().iloc[0, 1]
            st.caption(f"Correlation (age vs. duration): **{corr:.3f}** — essentially no "
                       "linear relationship.")

    st.markdown('<div class="insight-box">👥 Customers ride roughly 2x longer than '
                'Subscribers on average — a clear commute-vs-leisure behavioral split. '
                'Age carries almost no predictive signal for trip duration.</div>',
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 4 — Stations & routes
# ----------------------------------------------------------------------------
with tabs[3]:
    top_n = st.slider("Number of top stations/routes to show", 5, 20, 10)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Top {top_n} busiest start stations")
        top_stations = df['start_station_name'].value_counts().head(top_n).sort_values()
        fig = px.bar(x=top_stations.values, y=top_stations.index, orientation='h',
                     color_discrete_sequence=[PRIMARY])
        fig.update_layout(xaxis_title="Number of trips", yaxis_title="", height=max(360, top_n*28))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader(f"Top {top_n} routes (excluding round-trips)")
        routed = df[~df['same_station']] if 'same_station' in df.columns else df
        top_routes = routed['route'].value_counts().head(top_n).sort_values()
        fig = px.bar(x=top_routes.values, y=[r[:45] for r in top_routes.index], orientation='h',
                     color_discrete_sequence=[SECONDARY])
        fig.update_layout(xaxis_title="Number of trips", yaxis_title="", height=max(360, top_n*28))
        st.plotly_chart(fig, use_container_width=True)

    if {'start_station_latitude', 'start_station_longitude'}.issubset(df.columns):
        st.subheader("Station map — bubble size = trip volume")
        geo = df.groupby(['start_station_name', 'start_station_latitude', 'start_station_longitude'],
                         observed=True).size().reset_index(name='trips')
        fig = px.scatter_mapbox(geo, lat='start_station_latitude', lon='start_station_longitude',
                                size='trips', color='trips', hover_name='start_station_name',
                                color_continuous_scale='YlGnBu', size_max=40, zoom=10.5,
                                mapbox_style='carto-positron')
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">📍 A handful of major transit-hub stations '
                '(Ferry Building, Caltrain, Market St corridor) capture a disproportionate '
                'share of demand — the top priority list for rebalancing and dock maintenance.'
                '</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 5 — Correlations
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Correlation heatmap of numeric features")
    num_cols = [c for c in ['duration_sec', 'age', 'start_hour', 'start_month'] if c in df.columns]
    corr = df[num_cols].dropna().corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    aspect='auto')
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight-box">🔗 All pairwise correlations among duration, age, '
                'hour, and month sit close to zero — none of these numeric variables linearly '
                'predict trip duration. The dominant driver is the categorical <b>user_type</b> '
                'split instead. A useful negative result: future duration models should '
                'prioritize categorical/behavioral features over these numeric ones.</div>',
                unsafe_allow_html=True)

    if 'age' in df.columns and df['age'].notna().any():
        st.subheader("Pairwise relationships (sampled)")
        pp_sample = df.dropna(subset=['age']).sample(n=min(3000, df['age'].notna().sum()), random_state=42)
        fig = px.scatter_matrix(
            pp_sample, dimensions=['duration_min', 'age', 'start_hour'], color='user_type',
            color_discrete_map={'Subscriber': PRIMARY, 'Customer': ACCENT}, opacity=0.3)
        fig.update_layout(height=650)
        fig.update_traces(diagonal_visible=False, marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 6 — Recommendations
# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("💡 Business recommendations (from the notebook's Solution section)")
    recs = [
        ("Rebalance on a two-peak weekday schedule, not a flat plan",
         "Concentrate rebalancing trucks/staff around the 8 AM and 5 PM weekday windows; "
         "scale weekend ops down to a single mid-day push."),
        ("Prioritize the top transit-hub stations",
         "Ferry Building, Caltrain, and the Market St corridor drive a disproportionate "
         "share of demand — give them replenishment and maintenance priority, and merge "
         "the split Caltrain station IDs into one canonical station."),
        ("Differentiate pricing/marketing by user type, not age",
         "Subscriber vs. Customer is the strongest behavioral signal in the data (duration, "
         "timing, usage pattern); age shows almost no predictive value and shouldn't drive "
         "pricing tiers."),
        ("Grow the Customer→Subscriber conversion funnel",
         "Only ~15% of trips are casual Customers — in-app prompts after N rides in a month "
         "could convert habitual casual riders into higher-LTV Subscribers."),
        ("Target the weekend and off-season gap",
         "Weekend volume runs roughly half of weekday volume, and Nov–Dec trails the Oct "
         "peak — weekend/winter promotions could grow usage without adding weekday fleet "
         "pressure."),
        ("Invest in high-traffic corridors",
         "The top routes are short, fixed downtown/waterfront hops — strong, data-backed "
         "candidates for dedicated bike-lane advocacy with city partners."),
    ]
    for title, body in recs:
        st.markdown(f'<div class="rec-box"><b>{title}.</b> {body}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Headline numbers (current filter)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Trips analyzed", f"{len(df):,}")
    c2.metric("Median trip", f"{df['duration_min'].median():.1f} min")
    c3.metric("Subscriber share", f"{(df['user_type']=='Subscriber').mean()*100:.1f}%")

    st.caption(
        "Dashboard built from the Ford GoBike (Bay Wheels) 2018 EDA capstone — "
        "P Suman Sangeet, LABMENTIX , Data Analytics & AI Internship ."
    )