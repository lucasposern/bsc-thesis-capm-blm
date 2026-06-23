"""
covariance.py
=============

Kovarianzmatrix \\Sigma für das Black-Litterman-Modell.

Wird importiert in black_litterman.py:
    from covariance import sample_cov, get_estimator

Alle Funktionen:
    Eingabe : pd.DataFrame mit Renditen (T x n), decimal-Format (0.01 = 1 %)
    Ausgabe : annualisierter pd.DataFrame (n x n)

Verfügbare Schätzer
---------------------
    sample_cov               klassische Sample-Kovarianz
    ledoit_wolf_cov          Ledoit-Wolf Shrinkage
    ewma_cov                 Exponentially Weighted Moving Average
    constant_correlation_cov Konstante-Korrelation Shrinkage (Elton-Gruber)

Autor: Lucas Posern, Bachelorarbeit "Kritik des CAPM und Erweiterung
       durch das Black-Litterman-Model".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Hilfsfunktion

def _ann_factor(frequency: str) -> int:
    """Annualisierungsfaktor: M=12, D=252, W=52."""
    return {"M": 12, "D": 252, "W": 52}[frequency]


# Sample-Kovarianz

def sample_cov(
    returns: pd.DataFrame,
    annualize: bool = True,
    frequency: str = "M",
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Klassische Sample-Kovarianzmatrix.

    Formel:
        \\hat{\\Sigma} = (1 / (T-1)) * \\sum_t (r_t - \\bar{r})(r_t - \\bar{r})^T

    Parameter
    ---------
    returns   : Rendite-DataFrame (T x n)
    annualize : Auf Jahresbasis hochrechnen (True = Standard)
    frequency : Datenfrequenz für Annualisierungsfaktor ("M", "D", "W")
    ddof      : Freiheitsgrade (1 = unbiased, Standard)
    """
    cov = returns.cov(ddof=ddof)
    if annualize:
        cov = cov * _ann_factor(frequency)
    return cov


# Ledoit-Wolf Shrinkage

def ledoit_wolf_cov(
    returns: pd.DataFrame,
    annualize: bool = True,
    frequency: str = "M",
) -> pd.DataFrame:
    """
    Ledoit-Wolf Shrinkage-Kovarianz.

    Reduziert Schätzfehler bei grossem n/T-Verhältnis durch Schrumpfung
    der Sample-Kovarianz auf einen skalierten Identitäts-Target.

    Quelle: Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix".

    Erfordert: scikit-learn  (pip install scikit-learn)
    """
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as e:
        raise ImportError(
            "scikit-learn nicht installiert. Bitte 'pip install scikit-learn' ausführen."
        ) from e
    X = returns.dropna().values
    lw = LedoitWolf().fit(X)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    if annualize:
        cov = cov * _ann_factor(frequency)
    return cov


# EWMA-Kovarianz

def ewma_cov(
    returns: pd.DataFrame,
    halflife: float = 60,
    annualize: bool = True,
    frequency: str = "M",
) -> pd.DataFrame:
    """
    EWMA-Kovarianz (Exponentially Weighted Moving Average).

    Gewichtet juengere Beobachtungen stärker. Bei halflife=60 Monaten
    sind Datenpunkte älter als ca. 5 Jahre nahezu irrelevant.

    Parameter
    ---------
    halflife : Halbwertszeit in Perioden (z.B. 60 Monate = 5 Jahre)
    """
    cov = returns.ewm(halflife=halflife, min_periods=12).cov().dropna()
    last_idx = cov.index.get_level_values(0).unique().max()
    cov = cov.loc[last_idx]
    if annualize:
        cov = cov * _ann_factor(frequency)
    return cov


# Konstante-Korrelation Shrinkage

def constant_correlation_cov(
    returns: pd.DataFrame,
    shrinkage: float = 0.5,
    annualize: bool = True,
    frequency: str = "M",
) -> pd.DataFrame:
    """
    Konstante-Korrelation Shrinkage (Elton-Gruber-Target).

    Formel:
        \\hat{\\Sigma} = alpha * \\Sigma_{target} + (1-alpha) * \\Sigma_{sample}

    Dabei ist \\Sigma_{target} = D * \\bar{\\rho} * D, mit
        D           = diag(\\sigma_i)   (Standardabweichungen)
        \\bar{\\rho} = mittlere paarweise Korrelation aus \\Sigma_{sample}

    Parameter
    ---------
    shrinkage : Gewicht des Targets (0 = nur Sample, 1 = nur Target)
    """
    sample = returns.cov()
    sigma_vec = np.sqrt(np.diag(sample))
    corr = sample.values / np.outer(sigma_vec, sigma_vec)
    np.fill_diagonal(corr, np.nan)
    rbar = np.nanmean(corr)
    target_corr = np.full_like(corr, rbar)
    np.fill_diagonal(target_corr, 1.0)
    target = pd.DataFrame(
        np.outer(sigma_vec, sigma_vec) * target_corr,
        index=sample.index, columns=sample.columns,
    )
    cov = shrinkage * target + (1 - shrinkage) * sample
    if annualize:
        cov = cov * _ann_factor(frequency)
    return cov


# Registry: Schätzer per Name abrufen

ESTIMATORS: dict = {
    "sample":         sample_cov,
    "ledoit_wolf":    ledoit_wolf_cov,
    "ewma":           ewma_cov,
    "constant_corr":  constant_correlation_cov,
}


def get_estimator(name: str):
    """
    Liefert einen Kovarianz-Schätzer per Name.

    Verfügbare Namen: "sample", "ledoit_wolf", "ewma", "constant_corr"
    """
    if name not in ESTIMATORS:
        raise KeyError(
            f"Unbekannter Schätzer '{name}'. Verfügbar: {list(ESTIMATORS)}"
        )
    return ESTIMATORS[name]


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "sample",
    **kwargs,
) -> pd.DataFrame:
    """
    Schätzt die Kovarianzmatrix mit dem angegebenen Verfahren.

    Parameter
    ---------
    returns : Rendite-DataFrame (T x n)
    method  : "sample", "ledoit_wolf", "ewma", "constant_corr"
    **kwargs: Weitergabe an den jeweiligen Schätzer (z.B. annualize, frequency)
    """
    return get_estimator(method)(returns, **kwargs)
