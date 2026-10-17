"""Simulate a daily return series and print a parametric risk report."""

from __future__ import annotations

import numpy as np

from analyzer import QuantitativeRiskAnalyzer

ANNUAL_DRIFT = 0.08
ANNUAL_VOLATILITY = 0.20
TRADING_DAYS = 252
SEED = 42
CONFIDENCE_LEVEL = 0.95


def simulate_daily_returns(
    n_days: int = TRADING_DAYS,
    annual_drift: float = ANNUAL_DRIFT,
    annual_volatility: float = ANNUAL_VOLATILITY,
    seed: int = SEED,
) -> np.ndarray:
    """Draw i.i.d. Gaussian simple returns for one trading year.

    Drift scales with 1/252 and volatility with 1/sqrt(252), which matches
    a constant annual mean and an i.i.d. daily variance.
    """
    if n_days < 2:
        raise ValueError("n_days must be at least 2")

    daily_drift = annual_drift / TRADING_DAYS
    daily_volatility = annual_volatility / np.sqrt(TRADING_DAYS)
    generator = np.random.default_rng(seed)
    return generator.normal(
        loc=daily_drift,
        scale=daily_volatility,
        size=n_days,
    )


def build_report(
    returns: np.ndarray,
    stats: dict[str, float],
    var_95: float,
) -> str:
    """Format mean, sample standard deviation, and 95% parametric VaR."""
    mean = stats["mean"]
    std_dev = stats["std_dev"]
    rows = (
        ("Observações", f"{returns.size}"),
        ("Drift anual", f"{ANNUAL_DRIFT:.2%}"),
        ("Volatilidade anual", f"{ANNUAL_VOLATILITY:.2%}"),
        ("Semente", f"{SEED}"),
        ("Média diária", f"{mean:.6f} ({mean:.4%})"),
        ("Desvio padrão diário", f"{std_dev:.6f} ({std_dev:.4%})"),
        ("Confiança", f"{CONFIDENCE_LEVEL:.0%}"),
        ("VaR paramétrico (1 dia)", f"{var_95:.6f} ({var_95:.4%})"),
    )
    label_width = max(len(label) for label, _ in rows)

    lines = [
        "Relatório de risco quantitativo",
        "=" * 40,
        "",
        "Série: retornos diários simples ~ N(μ, σ²)",
        "",
    ]
    lines.extend(f"{label:<{label_width}}  {value}" for label, value in rows)
    lines.extend(
        [
            "",
            "Sob a hipótese normal, a perda diária supera esse VaR",
            "em cerca de 5% dos pregões.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    returns = simulate_daily_returns()
    analyzer = QuantitativeRiskAnalyzer(returns)
    stats = analyzer.calculate_statistics()
    var_95 = analyzer.calculate_var(CONFIDENCE_LEVEL)
    print(build_report(returns, stats, var_95))


if __name__ == "__main__":
    main()
