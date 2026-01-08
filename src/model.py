from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class ForecastBundle:
    model: HistGradientBoostingRegressor
    feature_cols: List[str]
    target: str

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].to_numpy()
        return self.model.predict(X)


def train_forecaster(df: pd.DataFrame, target: str) -> ForecastBundle:
    feature_cols = [c for c in df.columns if c not in {"date", "campaign", target}]

    X = df[feature_cols].to_numpy()
    y = df[target].to_numpy()

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=500,
        random_state=42,
    )
    model.fit(X, y)

    return ForecastBundle(model=model, feature_cols=feature_cols, target=target)
