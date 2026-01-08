import os
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/ads.csv"
OUT_PATH = "data/processed/ads_daily.parquet"

# Your file only has month + day (no year). Choose a year for the time series.
# You can change this to 2023/2025 etc. It won't affect modeling logic.
YEAR = 2024

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def parse_month_day_to_date(df: pd.DataFrame) -> pd.Series:
    if "month" not in df.columns or "day" not in df.columns:
        raise ValueError("Expected columns 'month' and 'day' in the raw dataset.")

    m = df["month"].astype(str).str.strip().str.lower()
    d = pd.to_numeric(df["day"], errors="coerce")

    # Map month names -> month number
    m_num = m.map(MONTH_MAP)

    bad_month = m_num.isna()
    bad_day = d.isna()

    if bad_month.any():
        samples = df.loc[bad_month, "month"].head(10).tolist()
        raise ValueError(f"Unrecognized month values (sample): {samples}")

    if bad_day.any():
        samples = df.loc[bad_day, "day"].head(10).tolist()
        raise ValueError(f"Unrecognized day values (sample): {samples}")

    # Build YYYY-MM-DD
    date_str = (
        pd.Series([YEAR] * len(df)).astype(str)
        + "-"
        + m_num.astype(int).astype(str).str.zfill(2)
        + "-"
        + d.astype(int).astype(str).str.zfill(2)
    )

    dt = pd.to_datetime(date_str, errors="coerce")
    if dt.isna().any():
        bad = dt.isna()
        samples = date_str[bad].head(10).tolist()
        raise ValueError(f"Failed to parse some dates (sample): {samples}")

    return dt


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing {RAW_PATH}. Put your dataset CSV at data/raw/ads.csv")

    df = pd.read_csv(RAW_PATH)

    # Drop empty unnamed columns
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]

    print("Raw columns:", df.columns.tolist())
    print("Raw preview:\n", df.head(3))

    # Build a real date
    dt = parse_month_day_to_date(df)

    # Map your dataset columns -> standard schema
    # campaign: campaign_number
    # impressions: displays
    # clicks: clicks
    # cost: cost
    # conversions: post_click_conversions
    col_campaign = "campaign_number"
    col_impr = "displays"
    col_clicks = "clicks"
    col_cost = "cost"
    col_conv = "post_click_conversions"

    for c in [col_campaign, col_impr, col_clicks, col_cost, col_conv]:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' not found. Available: {df.columns.tolist()}")

    out = pd.DataFrame({
        "date": dt.dt.date,
        "campaign": df[col_campaign].astype(str).str.strip(),
        "impressions": pd.to_numeric(df[col_impr], errors="coerce"),
        "clicks": pd.to_numeric(df[col_clicks], errors="coerce"),
        "cost": pd.to_numeric(df[col_cost], errors="coerce"),
        "conversions": pd.to_numeric(df[col_conv], errors="coerce"),
    }).fillna(0)

    # Clean weird negatives
    for c in ["impressions", "clicks", "cost", "conversions"]:
        out[c] = out[c].clip(lower=0)

    # Aggregate to daily per campaign
    out = (
        out.groupby(["date", "campaign"], as_index=False)
           .agg({"impressions": "sum", "clicks": "sum", "cost": "sum", "conversions": "sum"})
    )

    # Derived KPIs (avoid divide by zero)
    out["ctr"] = out["clicks"] / out["impressions"].replace(0, 1)
    out["cpc"] = out["cost"] / out["clicks"].replace(0, 1)

    # Optional extra KPI: conversion rate
    out["cvr"] = out["conversions"] / out["clicks"].replace(0, 1)

    # Sanity checks
    out["date"] = pd.to_datetime(out["date"])
    print("\nSanity checks:")
    print("Rows:", len(out))
    print("Date range:", out["date"].min().date(), "→", out["date"].max().date())
    print("Unique days:", out["date"].nunique())
    print("Unique campaigns:", out["campaign"].nunique())
    print("\nPreview:\n", out.head(5))

    if out["date"].nunique() <= 1:
        raise ValueError(
            "Processed dataset has <= 1 unique day. "
            "Your raw dataset might contain only one day OR month/day parsing failed."
        )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(out):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
