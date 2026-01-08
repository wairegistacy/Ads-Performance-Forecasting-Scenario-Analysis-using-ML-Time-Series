import numpy as np
import pandas as pd

from src.features import make_supervised_frame


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def latest_supervised_rows(df_daily: pd.DataFrame, target: str = "clicks") -> pd.DataFrame:
    """
    Build supervised frame and return rows for the latest available date
    (these rows represent the next-day prediction context).
    """
    sup = make_supervised_frame(df_daily, target=target, group_col="campaign")
    latest_date = sup["date"].max()
    base = sup[sup["date"] == latest_date].copy()
    return base


def apply_scenario(base_df: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    """
    Apply a scenario by multiplying selected columns (e.g., impressions, ctr).
    scenario example: {"impressions": 1.2, "ctr": 0.9}
    """
    out = base_df.copy()
    for col, mult in scenario.items():
        if col not in out.columns:
            raise ValueError(f"Scenario column '{col}' not found in model features.")
        out[col] = out[col] * float(mult)
    return out


def forecast_table(bundle, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return predictions per campaign from a ForecastBundle loaded via joblib.
    """
    preds = bundle.predict(base_df)
    out = pd.DataFrame({
        "campaign": base_df["campaign"].astype(str).values,
        "pred_clicks": preds,
    })
    out["pred_clicks"] = out["pred_clicks"].clip(lower=0)
    return out.sort_values("pred_clicks", ascending=False).reset_index(drop=True)


def scenario_table(bundle, base_df: pd.DataFrame, scenarios: dict) -> pd.DataFrame:
    """
    Run multiple scenarios and return a tidy table.
    """
    base_pred = bundle.predict(base_df)
    rows = []

    # Baseline
    for camp, val in zip(base_df["campaign"].astype(str).values, base_pred):
        rows.append({"scenario": "Baseline", "campaign": camp, "pred_clicks": max(0.0, float(val))})

    # Scenarios
    for name, changes in scenarios.items():
        scen_df = apply_scenario(base_df, changes)
        scen_pred = bundle.predict(scen_df)
        for camp, val in zip(base_df["campaign"].astype(str).values, scen_pred):
            rows.append({"scenario": name, "campaign": camp, "pred_clicks": max(0.0, float(val))})

    out = pd.DataFrame(rows)
    return out.sort_values(["scenario", "pred_clicks"], ascending=[True, False]).reset_index(drop=True)


def add_deltas(scen_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds delta vs baseline per campaign.
    """
    base = scen_df[scen_df["scenario"] == "Baseline"][["campaign", "pred_clicks"]].rename(
        columns={"pred_clicks": "baseline_clicks"}
    )
    out = scen_df.merge(base, on="campaign", how="left")
    out["delta_clicks"] = out["pred_clicks"] - out["baseline_clicks"]
    out["delta_pct"] = np.where(
        out["baseline_clicks"] > 0,
        out["delta_clicks"] / out["baseline_clicks"],
        np.nan
    )
    return out
