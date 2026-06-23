"""
portfolio_metrics.py
====================

Wiederverwendbare Performance-Metriken für Portfolio-Backtests.

Verwendet in den Backtests der Notebooks 03_CAPM und 05_Modifikationen.

Enthält:
  - annualized_return     : Arithmetisch annualisierte Durchschnittsrendite
  - annualized_volatility : Annualisierte Standardabweichung
  - sharpe_ratio          : Sharpe Ratio (Excess Returns, rf=0)
  - max_drawdown          : Maximaler kumulierter Verlust vom Hoch
  - cumulative_returns    : Kumulierte Rendite-Zeitreihe (Basis 1)
  - performance_summary   : Alle Metriken als DataFrame für mehrere Portfolios

Konvention: Alle Renditen als Dezimalzahlen (0.01 = 1%).
            Für Sharpe werden Excess Returns erwartet (rf bereits abgezogen).

Autor: Lucas Posern, Bachelorarbeit "Kritik des CAPM und Erweiterung
       durch das Black-Litterman-Model".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Einzelne Metriken

def annualized_return(returns: pd.Series, frequency: int = 12) -> float:
    """Arithmetisch annualisierte Durchschnittsrendite.

    Parameters
    ----------
    returns   : monatliche (oder andere Frequenz) Renditen als pd.Series
    frequency : Perioden pro Jahr (12 für monatlich, 252 für täglich)
    """
    return float(returns.mean() * frequency)


def annualized_volatility(returns: pd.Series, frequency: int = 12) -> float:
    """Annualisierte Standardabweichung der Renditen."""
    return float(returns.std(ddof=1) * np.sqrt(frequency))


def sharpe_ratio(returns: pd.Series, frequency: int = 12) -> float:
    """Sharpe Ratio (setzt voraus: returns sind bereits Excess Returns, rf=0).

    Sharpe = E[r - rf] / σ(r - rf)  =  annualized_return / annualized_vol
    """
    vol = annualized_volatility(returns, frequency)
    if vol < 1e-12:
        return np.nan
    return annualized_return(returns, frequency) / vol


def max_drawdown(returns: pd.Series) -> float:
    """Maximaler Drawdown: größter kumulierter Verlust vom vorherigen Hoch.

    Gibt einen negativen Wert zurück (z.B. -0.35 = -35%).
    """
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Kumulierte Rendite-Zeitreihe, normiert auf 1.0 am Startpunkt.

    Beispiel: [0.01, -0.02, 0.03] -> [1.01, 0.9898, 1.0195]
    """
    return (1.0 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    """Gesamtrendite über den gesamten Zeitraum (nicht annualisiert)."""
    return float(cumulative_returns(returns).iloc[-1] - 1.0)


# Zusammenfassung für mehrere Portfolios

def performance_summary(
    portfolio_returns: dict[str, pd.Series],
    frequency: int = 12,
) -> pd.DataFrame:
    """Erstellt eine Vergleichstabelle mit allen Performance-Metriken.

    Parameters
    ----------
    portfolio_returns : dict  {Portfolio-Name -> pd.Series monatlicher Returns}
                        Die Renditen sollten Excess Returns sein (rf abgezogen),
                        damit Sharpe korrekt berechnet wird.
    frequency        : Perioden pro Jahr (12 für monatlich)

    Returns
    -------
    pd.DataFrame
        Zeilen = Portfolios, Spalten = Metriken (Return, Vol, Sharpe, MDD, Total)
    """
    rows = []
    for name, ret in portfolio_returns.items():
        rows.append({
            "Portfolio":        name,
            "Ann. Return":      annualized_return(ret, frequency),
            "Ann. Volatility":  annualized_volatility(ret, frequency),
            "Sharpe Ratio":     sharpe_ratio(ret, frequency),
            "Max Drawdown":     max_drawdown(ret),
            "Total Return":     total_return(ret),
        })
    df = pd.DataFrame(rows).set_index("Portfolio")
    return df


def portfolio_performance(returns: pd.Series, rf: pd.Series, frequency: int = 12) -> dict:
    """Metriken für ein einzelnes Portfolio als dict (für Notebook-Verwendung).

    Parameters
    ----------
    returns   : Brutto-Monatsrenditen
    rf        : Risikofreie Rate (gleiche Länge wie returns)
    frequency : Perioden pro Jahr (12 für monatlich)
    """
    excess = returns - rf.reindex(returns.index).fillna(0)
    return {
        "Ann. Return":     annualized_return(returns, frequency),
        "Ann. Volatility": annualized_volatility(returns, frequency),
        "Sharpe Ratio":    sharpe_ratio(excess, frequency),
        "Max Drawdown":    max_drawdown(returns),
        "Total Return":    total_return(returns),
    }
