"""
data_loader.py
==============

Laden der Renditezeitreihen für die Black-Litterman / CAPM Tests aus den
lokalen CSVs im Ordner ``data/``. Kein Internet erforderlich.

Datengrundlage (Idzorek-ETF-Universum, 2008-2024)
-------------------------------------------------
- ``etf_returns_monthly.csv``  : Monatsrenditen der 8 Idzorek-ETFs
- ``rf_monthly.csv``           : monatlicher risikofreier Zins (^IRX)
- ``spx_returns_monthly.csv``  : Marktüberschussrendite des S&P 500 (^GSPC)
- ``eq_weights_train_start.csv``: Idzorek-Marktkapitalisierungsgewichte (Prior-Anker)

Konvention
----------
- Alle Renditen als Dezimalzahlen (0.01 = 1%).
- Index ist ein DatetimeIndex auf Monatsende.
- Spalten sind die ETF-Tickers.
- Risk-free Rate ``rf`` wird separat als pd.Series zurückgegeben.

Autor: Lucas Posern, Bachelorarbeit "Kritik des CAPM und Erweiterung
durch das Black-Litterman-Modell".
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd


# Datencontainer
@dataclass
class ReturnPanel:
    """Container für Renditezeitreihen + Risk-Free + (optional) Marktrendite.

    Attribute
    ---------
    returns : pd.DataFrame
        Asset-Renditen (Dezimal).
    rf : pd.Series
        Risk-Free Rate, gleiche Frequenz wie `returns`.
    market_excess : pd.Series | None
        Excess-Return des Marktes (Mkt-RF), hier der S&P 500.
    market_caps : pd.Series | None
        Gleichgewichtsgewichte w_eq (normiert). Optional.
    frequency : str
        "M" (monatlich), "D" (täglich) oder "W" (wöchentlich).
    """

    returns: pd.DataFrame
    rf: pd.Series
    market_excess: Optional[pd.Series] = None
    market_caps: Optional[pd.Series] = None
    frequency: str = "M"

    def excess_returns(self) -> pd.DataFrame:
        """Excess Returns r_i,t - r_f,t."""
        return self.returns.sub(self.rf, axis=0)

    def annualization_factor(self) -> int:
        """Faktor zum Hochrechnen auf Jahresbasis (12 monatlich, 252 täglich, 52 wöchentlich)."""
        return {"M": 12, "D": 252, "W": 52}[self.frequency]

    def __repr__(self) -> str:
        n_obs, n_assets = self.returns.shape
        return (
            f"ReturnPanel(freq={self.frequency}, n_obs={n_obs}, "
            f"n_assets={n_assets}, "
            f"period={self.returns.index.min().date()}..{self.returns.index.max().date()})"
        )


# Idzorek-ETF-Universum

#: Idzorek (2004, S. 14) Gleichgewichtsgewichte - institutionelle Referenzallokation
IDZOREK_W_EQ: dict[str, float] = {
    "AGG": 0.1934,
    "BWX": 0.2613,
    "IWF": 0.1209,
    "IWD": 0.1209,
    "IWO": 0.0134,
    "IWN": 0.0134,
    "EFA": 0.2418,
    "EEM": 0.0349,
}

#: Kurzbeschriftungen für Plots
IDZOREK_LABELS: dict[str, str] = {
    "AGG": "US Bonds (AGG)",
    "BWX": "Intl Bonds (BWX)",
    "IWF": "US Lg Growth (IWF)",
    "IWD": "US Lg Value (IWD)",
    "IWO": "US Sm Growth (IWO)",
    "IWN": "US Sm Value (IWN)",
    "EFA": "Intl Dev (EFA)",
    "EEM": "Emg Mkts (EEM)",
}


def _data_dir() -> str:
    """Pfad zum Ordner ``data/`` neben ``core/``."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "data"))


def load_idzorek_etf() -> ReturnPanel:
    """Lädt das 8-ETF-Idzorek-Universum aus den lokalen CSVs als ReturnPanel.

    Liest ``etf_returns_monthly.csv``, ``rf_monthly.csv`` und
    ``spx_returns_monthly.csv`` aus ``data/``. Die normierten
    Gleichgewichtsgewichte liegen im Attribut ``panel.market_caps`` (Name 'w_eq').

    Returns
    -------
    ReturnPanel
        returns       : monatliche ETF-Renditen (8 Spalten)
        rf            : monatlicher Risk-Free
        market_excess : S&P-500-Überschussrendite (Marktproxy)
        market_caps   : Idzorek-Gleichgewichtsgewichte (normiert, Name='w_eq')
        frequency     : "M"
    """
    data_dir = _data_dir()

    ret = pd.read_csv(os.path.join(data_dir, "etf_returns_monthly.csv"),
                      index_col=0, parse_dates=True).sort_index()
    tickers = [t for t in IDZOREK_W_EQ if t in ret.columns]
    ret = ret[tickers]
    ret.index = ret.index.to_period("M").to_timestamp("M")  # Monatsende normieren

    rf = pd.read_csv(os.path.join(data_dir, "rf_monthly.csv"),
                     index_col=0, parse_dates=True).iloc[:, 0].sort_index()
    rf.index = rf.index.to_period("M").to_timestamp("M")

    mkt = pd.read_csv(os.path.join(data_dir, "spx_returns_monthly.csv"),
                      index_col=0, parse_dates=True).iloc[:, 0].sort_index()
    mkt.index = mkt.index.to_period("M").to_timestamp("M")

    common = ret.index.intersection(rf.index).intersection(mkt.index)
    ret, rf, mkt = ret.loc[common], rf.loc[common], mkt.loc[common]

    return ReturnPanel(
        returns=ret,
        rf=rf,
        market_excess=mkt,
        market_caps=_normalized_w_eq(tickers),
        frequency="M",
    )


def load_local_idzorek(
    tickers: Optional[list] = None,
    start: str = "2008-01-01",
    end: str = "2024-12-31",
) -> ReturnPanel:
    """Lädt das Idzorek-ETF-Universum und filtert auf den Zeitraum [start, end].

    Parameters
    ----------
    tickers : list | None
        Teilmenge der 8 Ticker. Bei None alle 8.
    start, end : str
        Zeitraum (inklusive), Format 'YYYY-MM-DD'.

    Returns
    -------
    ReturnPanel  (frequency="M", market_caps = normierte Idzorek-Gewichte)
    """
    panel = load_idzorek_etf()

    mask = (panel.returns.index >= start) & (panel.returns.index <= end)
    if tickers is not None:
        cols = [t for t in tickers if t in panel.returns.columns]
    else:
        cols = list(panel.returns.columns)

    return ReturnPanel(
        returns=panel.returns.loc[mask, cols],
        rf=panel.rf.loc[mask],
        market_excess=panel.market_excess.loc[mask],
        market_caps=_normalized_w_eq(cols),
        frequency="M",
    )


def _normalized_w_eq(tickers: list[str]) -> pd.Series:
    """Normierte Idzorek-Gleichgewichtsgewichte für die gegebenen Ticker (Summe 1)."""
    w_raw = {t: IDZOREK_W_EQ[t] for t in tickers if t in IDZOREK_W_EQ}
    w_sum = sum(w_raw.values())
    return pd.Series({t: w_raw[t] / w_sum for t in w_raw}, name="w_eq")
