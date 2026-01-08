import os
import joblib
import pandas as pd
import streamlit as st

from src.planner import latest_supervised_rows, forecast_table, scenario_table, add_deltas

DATA_PATH = "data/processed/ads_daily.parquet"
MODEL_PATH = "artifacts/models/forecaster_clicks.joblib"

st.set_page_config(page_title="Ads Planning Forecaster", layout="wide")
st.title("Ads Planning Forecaster")
st.caption("Next-day clicks forecasting + scenario planning (campaign level)")

# --- Load artifacts ---
if not os.path.exists(DATA_PATH):
    st.error("Missing data. Run: python -m scripts.build_dataset")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("Missing model. Run: python -m scripts.train_forecaster")
    st.stop()

df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

bundle = joblib.load(MODEL_PATH)

# --- Sidebar controls ---
with st.sidebar:
    st.header("Scenarios")
    st.write("Adjust assumptions and compare against the baseline forecast.")

    impressions_mult = st.slider("Impressions multiplier", 0.5, 1.5, 1.0, 0.05)
    ctr_mult = st.slider("CTR multiplier", 0.5, 1.5, 1.0, 0.05)

    custom_name = st.text_input("Scenario name", "Custom Scenario")

    run_btn = st.button("Run Forecast", type="primary")

# --- Build base prediction rows (latest day context) ---
base = latest_supervised_rows(df, target="clicks")

if base.empty:
    st.error("Not enough data to build lag/rolling features yet (need at least ~14 days).")
    st.stop()

latest_context_date = pd.to_datetime(base["date"].max()).date()

st.subheader("Baseline Forecast (Next-day Clicks)")
st.write(f"Model context date: **{latest_context_date}** (forecasting next day)")

baseline = forecast_table(bundle, base)
st.dataframe(baseline, use_container_width=True)

# --- Run scenarios ---
if run_btn:
    scenarios = {
        "Impressions +20%": {"impressions": 1.2},
        "Impressions -20%": {"impressions": 0.8},
        "CTR +10%": {"ctr": 1.1},
        "CTR -10%": {"ctr": 0.9},
        custom_name: {"impressions": impressions_mult, "ctr": ctr_mult},
    }

    st.subheader("Scenario Comparison")
    scen = scenario_table(bundle, base, scenarios)
    scen = add_deltas(scen)

    # Display tidy table
    st.dataframe(
        scen.sort_values(["scenario", "pred_clicks"], ascending=[True, False]),
        use_container_width=True
    )

    # Download results
    csv = scen.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scenario results (CSV)",
        data=csv,
        file_name="scenario_results.csv",
        mime="text/csv",
    )

    # Simple chart: scenario totals
    st.subheader("Total Predicted Clicks by Scenario")
    totals = scen.groupby("scenario", as_index=False)["pred_clicks"].sum()
    st.bar_chart(totals.set_index("scenario")["pred_clicks"])
