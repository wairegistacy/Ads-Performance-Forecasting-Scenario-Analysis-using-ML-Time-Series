import os
import joblib
import pandas as pd

from src.features import make_supervised_frame
from src.model import train_forecaster

DATA_PATH = "data/processed/ads_daily.parquet"
OUT_PATH = "artifacts/models/forecaster_clicks.joblib"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run: python -m scripts.build_dataset")

    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    target = "clicks"  # start here
    sup = make_supervised_frame(df, target=target, group_col="campaign")

    # Time split (last 14 days as validation)
    max_date = sup["date"].max()
    cutoff = max_date - pd.Timedelta(days=14)

    train_df = sup[sup["date"] <= cutoff].copy()
    val_df = sup[sup["date"] > cutoff].copy()

    bundle = train_forecaster(train_df, target=target)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump(bundle, OUT_PATH)

    print(f"Saved model to: {OUT_PATH}")
    print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")
    print(f"Feature count: {len(bundle.feature_cols)}")
    print("Date range:", sup["date"].min().date(), "→", sup["date"].max().date())

if __name__ == "__main__":
    main()
