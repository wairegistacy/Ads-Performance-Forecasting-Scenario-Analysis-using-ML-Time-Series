import pandas as pd

TARGETS = ["clicks", "cost", "conversions", "revenue"]  # revenue exists in raw, but not saved yet unless you add it
NUMERIC_COLS = ["impressions", "clicks", "cost", "conversions", "ctr", "cpc", "cvr"]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["dow"] = out["date"].dt.dayofweek  # 0=Mon
    out["dom"] = out["date"].dt.day
    out["week"] = out["date"].dt.isocalendar().week.astype(int)
    return out

def add_lag_features(df: pd.DataFrame, group_col: str, cols, lags=(1, 7, 14)) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([group_col, "date"])
    for c in cols:
        for l in lags:
            out[f"{c}_lag{l}"] = out.groupby(group_col)[c].shift(l)
    return out

def add_rolling_features(df: pd.DataFrame, group_col: str, cols, windows=(7, 14)) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([group_col, "date"])
    for c in cols:
        for w in windows:
            out[f"{c}_roll{w}_mean"] = (
                out.groupby(group_col)[c].shift(1).rolling(w).mean().reset_index(level=0, drop=True)
            )
            out[f"{c}_roll{w}_std"] = (
                out.groupby(group_col)[c].shift(1).rolling(w).std().reset_index(level=0, drop=True)
            )
    return out

def make_supervised_frame(df: pd.DataFrame, target: str, group_col: str = "campaign") -> pd.DataFrame:
    """
    Build a supervised learning frame for next-day prediction:
      y_t = target at date t
      X_t = features derived from data up to t-1 (lags/rolling)
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    out = add_time_features(out)
    out = add_lag_features(out, group_col=group_col, cols=[target, "impressions", "ctr", "cpc", "cvr"], lags=(1, 7, 14))
    out = add_rolling_features(out, group_col=group_col, cols=[target, "clicks", "cost", "conversions"], windows=(7, 14))

    # Drop rows with NA created by lag/rolling
    out = out.dropna().reset_index(drop=True)
    return out
