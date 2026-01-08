import os
import numpy as np
import pandas as pd

from src.features import make_supervised_frame
from src.model import train_forecaster
from src.metrics import mae, rmse, mape

DATA_PATH = "data/processed/ads_daily.parquet"

def rolling_backtest(
    sup: pd.DataFrame,
    target: str,
    horizon_days: int = 1,
    min_train_days: int = 45,
    step_days: int = 7,
) -> pd.DataFrame:
    sup = sup.sort_values("date").reset_index(drop=True)

    start_date = sup["date"].min() + pd.Timedelta(days=min_train_days)
    end_date = sup["date"].max() - pd.Timedelta(days=horizon_days)

    results = []

    cut = start_date
    while cut <= end_date:
        train_df = sup[sup["date"] < cut].copy()
        test_df = sup[(sup["date"] >= cut) & (sup["date"] < cut + pd.Timedelta(days=step_days))].copy()

        if len(test_df) == 0 or len(train_df) == 0:
            cut += pd.Timedelta(days=step_days)
            continue

        bundle = train_forecaster(train_df, target=target)
        preds = bundle.predict(test_df)

        y_true = test_df[target].to_numpy()
        y_pred = preds

        results.append({
            "cutoff": cut.date(),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mape": mape(y_true, y_pred),
        })

        cut += pd.Timedelta(days=step_days)

    return pd.DataFrame(results)

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run: python -m scripts.build_dataset")

    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    target = "clicks"
    sup = make_supervised_frame(df, target=target, group_col="campaign")

    bt = rolling_backtest(sup, target=target, horizon_days=1, min_train_days=45, step_days=7)
    print("\nBacktest summary:")
    print(bt.describe(include="all"))

    os.makedirs("artifacts/forecasts", exist_ok=True)
    out_csv = "artifacts/forecasts/backtest_clicks.csv"
    bt.to_csv(out_csv, index=False)
    print(f"\nSaved backtest results to: {out_csv}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(bt["cutoff"].astype(str), bt["mae"])
        plt.xticks(rotation=30, ha="right")
        plt.title("Rolling Backtest MAE (Clicks)")
        plt.tight_layout()
        plt.savefig("artifacts/forecasts/backtest_mae_clicks.png", dpi=200)
        plt.close()
        print("Saved plot to: artifacts/forecasts/backtest_mae_clicks.png")
    except Exception as e:
        print("Plot skipped:", e)

if __name__ == "__main__":
    main()
