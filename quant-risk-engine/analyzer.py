"""Quantitative risk statistics for a numeric series."""

from __future__ import annotations

import numpy as np
import pandas as pd


class QuantitativeRiskAnalyzer:
    """Mean, sample standard deviation, and historical Value at Risk."""

    def __init__(self, data):
        series = pd.Series(list(data), dtype="float64")
        if series.empty:
            raise ValueError("data must contain at least one observation")
        if series.isna().any():
            raise ValueError("data must not contain missing values")
        self.data = series

    def calculate_statistics(self) -> dict[str, float]:
        std_dev = self.data.std(ddof=1) if len(self.data) > 1 else 0.0
        return {
            "mean": float(self.data.mean()),
            "std_dev": float(std_dev),
        }

    def calculate_var(self, confidence_level: float) -> float:
        """Historical VaR of simple returns, returned as a positive loss."""
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1")
        if len(self.data) < 2:
            raise ValueError("at least two observations are required to estimate VaR")

        returns = self.data.pct_change().dropna().to_numpy()
        tail = float(np.quantile(returns, 1.0 - confidence_level))
        return float(-tail)
