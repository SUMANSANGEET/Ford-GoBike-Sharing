# Ford GoBike (Bay Wheels) 2018 — Interactive Streamlit Dashboard

An interactive deployment of the `Ford_GoBike_2018_EDA.ipynb` capstone (P Suman,
LABMENTIX Bold Analytics Cohort 2025 — INNOVEXIS). Turns the notebook's 15 static
charts into a live, filterable Streamlit dashboard.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Data

- **Demo mode (default):** `demo_gobike_2018.csv.gz` — a 60,000-row synthetic
  sample statistically matched to the notebook's published 2018 stats (median
  trip 9.25 min, 85/15 Subscriber/Customer split, Oct ridership peak, etc.),
  so the app runs immediately with no setup.
- **Real data mode:** use the sidebar uploader to drop in the actual monthly
  Ford GoBike/Bay Wheels 2018 CSVs (or a `data.zip` containing them). The app
  re-runs the notebook's wrangling steps (drop rows missing station IDs,
  engineer duration/age/hour/day/route features) and every chart recalculates
  on the real numbers.

## What's inside

- **Overview** — KPI cards, duration distribution, rider composition
- **Time Patterns** — hourly commute curves, day-of-week, monthly seasonality,
  day×hour heatmap, weekday vs. weekend
- **Rider Segments** — duration by user type, gender split, age distribution,
  age-vs-duration
- **Stations & Routes** — top stations/routes, geographic bubble map
- **Correlations** — numeric correlation heatmap, sampled pair plot
- **Recommendations** — the notebook's business recommendations, restated
  live against whatever filter you currently have applied

All filters (month, user type, weekday/weekend, gender, duration range) apply
across every tab.

## Deploy

Push this folder to a GitHub repo and deploy free on
[Streamlit Community Cloud](https://streamlit.io/cloud), pointing it at
`app.py`. No secrets or API keys required.
