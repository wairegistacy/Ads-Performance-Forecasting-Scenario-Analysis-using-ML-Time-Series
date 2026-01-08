import os
import joblib
import pandas as pd
import numpy as np

from src.features import make_supervised_frame

DATA_PATH = "data/processed/ads_daily.parquet"
MODEL_PATH = "artifacts/models/forecaster_clicks.joblib"


def apply_scenario(
    df: pd.DataFrame,
    scenario: dict,
) -> pd.DataFrame:
    """
    Apply a scenario by modifying selected feature columns.
    Example:
      {"impressions": 1.2}  -> +20% impressions
      {"ctr": 0.9}          -> -10% CTR
    """
    out = df.copy()
    for col, multiplier in scenario.items():
        if col not in out.columns:
            raise ValueError(f"Scenario column '{col}' not found in dataframe.")
        out[col] = out[col] * multiplier
    return out


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run build_dataset first.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Run train_forecaster first.")

    # Load data + model
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    bundle = joblib.load(MODEL_PATH)
    target = bundle.target

    # Build supervised frame
    sup = make_supervised_frame(df, target=target, group_col="campaign")

    # Use the most recent available day as baseline
    latest_date = sup["date"].max()
    base = sup[sup["date"] == latest_date].copy()

    if len(base) == 0:
        raise RuntimeError("No rows available for scenario analysis.")

    # Baseline prediction
    base_pred = bundle.predict(base)
    base["pred_clicks"] = base_pred

    print("\nBaseline forecast (next-day clicks):")
    print(base[["campaign", "pred_clicks"]])

    # -----------------------------
    # Define scenarios
    # -----------------------------
    scenarios = {
        "Impressions +20%": {"impressions": 1.2},
        "Impressions -20%": {"impressions": 0.8},
        "CTR -10%": {"ctr": 0.9},
        "CTR +10%": {"ctr": 1.1},
    }

    results = []

    for name, changes in scenarios.items():
        scen_df = apply_scenario(base, changes)
        scen_pred = bundle.predict(scen_df)

        tmp = base[["campaign"]].copy()
        tmp["scenario"] = name
        tmp["pred_clicks"] = scen_pred
        results.append(tmp)

    res = pd.concat(results, ignore_index=True)

    print("\nScenario analysis results:")
    print(res)

    # Save results
    os.makedirs("artifacts/forecasts", exist_ok=True)
    out_csv = "artifacts/forecasts/scenario_analysis_clicks.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nSaved scenario results to: {out_csv}")


if __name__ == "__main__":
    main()
