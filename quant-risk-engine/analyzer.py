"""Quantitative risk statistics for a return series."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_ACKLAM_P_LOW = 0.02425


def _horner(coefficients: tuple[float, ...], variable: float) -> float:
    """Evaluate a polynomial with Horner's method."""
    total = 0.0
    for coefficient in coefficients:
        total = total * variable + coefficient
    return total


def _standard_normal_ppf(probability: float) -> float:
    """Inverse CDF of the standard normal distribution.

    Rational approximation by Peter J. Acklam, accurate to about 1e-9.
    SciPy is intentionally not required.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1")

    if probability < _ACKLAM_P_LOW:
        q = math.sqrt(-2.0 * math.log(probability))
        return _horner(_ACKLAM_C, q) / (_horner(_ACKLAM_D, q) * q + 1.0)

    if probability > 1.0 - _ACKLAM_P_LOW:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        numerator = _horner(_ACKLAM_C, q)
        denominator = _horner(_ACKLAM_D, q) * q + 1.0
        return -(numerator / denominator)

    q = probability - 0.5
    r = q * q
    numerator = _horner(_ACKLAM_A, r) * q
    denominator = _horner(_ACKLAM_B, r) * r + 1.0
    return numerator / denominator


class QuantitativeRiskAnalyzer:
    """Mean, sample standard deviation, and Gaussian parametric VaR.

    ``data`` is a series of returns or P&L, not price levels. Parametric VaR
    is computed on that series directly.
    """

    def __init__(self, data: Sequence[float]) -> None:
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
        """One-period Gaussian parametric VaR, as a loss relative to zero.

        VaR_c = -mean + Phi^{-1}(c) * sample_std.
        A positive result is a loss threshold at the requested confidence.
        """
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1")
        if len(self.data) < 2:
            raise ValueError(
                "at least two observations are required to estimate VaR"
            )

        stats = self.calculate_statistics()
        z_score = _standard_normal_ppf(confidence_level)
        return float(-stats["mean"] + z_score * stats["std_dev"])
