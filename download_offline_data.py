"""
download_offline_data.py  (Idzorek-Universum)
=============================================
Erzeugt die Rohdaten der Notebooks für das Idzorek-Universum (mit Internet).

Lokal ausführen:
    pip install yfinance pandas
    python download_offline_data.py

Schreibt direkt nach  data/  (die Notebooks lesen ausschliesslich data/):
    etf_returns_monthly.csv     -- Renditen der 8 Idzorek-ETFs, Jan 2008 - Dez 2024
    rf_monthly.csv              -- Risk-Free Rate (^IRX, monatlich)
    spx_returns_monthly.csv     -- S&P-500-Excess-Returns (^GSPC - rf)
    eq_weights_train_start.csv  -- w_eq = Idzorek (2004) Gleichgewichtsgewichte

Universum (Idzorek 2004, "A Step-by-Step Guide to the Black-Litterman Model")
-----------------------------------------------------------------------------
    ETF   Anlageklasse              Auflage   w_eq (Idzorek)
    ----  ------------------------  --------  --------------
    AGG   US Bonds                  2003-09   0.1934
    BWX   Intl Bonds                2007-10   0.2613
    IWF   US Large Growth           2000-05   0.1209
    IWD   US Large Value            2000-05   0.1209
    IWO   US Small Growth           2000-07   0.0134
    IWN   US Small Value            2000-07   0.0134
    EFA   Intl Developed Equity     2001-08   0.2418
    EEM   Intl Emerging Equity      2003-04   0.0349

Alle acht ETFs haben über den gesamten Analysezeitraum 2008-2024 lückenlose
Daten (BWX ab Okt 2007). In-sample existiert also KEINE Datenlücke.

w_eq (Prior-Anker)
------------------
w_eq sind die von Idzorek (2004, S. 14) publizierten Marktkapitalisierungs-
gewichte dieses Universums. Sie gehen ins BLM ausschliesslich als neutraler
Prior-Anker ein (kein Stichproben-Schätzer), weshalb ihr exaktes Datum für
die Gültigkeit des Priors unerheblich ist. Quelle: eine einzige, zitierbare
Referenz; identisch zu core/data_loader.IDZOREK_W_EQ.
"""
import os
import sys

try:
    import yfinance as yf
except ImportError:
    sys.exit("Fehler: yfinance fehlt.  ->  pip install yfinance pandas")

import pandas as pd

# Konfiguration
START = "2007-12-01"   # ein Monat Vorlauf (pct_change verliert ersten Monat)
END   = "2024-12-31"

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")
os.makedirs(OUT_DIR, exist_ok=True)

# Idzorek (2004, S. 14) publizierte Gleichgewichtsgewichte (Summe = 1)
IDZOREK_W_EQ = {
    "AGG": 0.1934,   # US Bonds
    "BWX": 0.2613,   # Intl Bonds
    "IWF": 0.1209,   # US Large Growth
    "IWD": 0.1209,   # US Large Value
    "IWO": 0.0134,   # US Small Growth
    "IWN": 0.0134,   # US Small Value
    "EFA": 0.2418,   # Intl Developed Equity
    "EEM": 0.0349,   # Intl Emerging Equity
}
TICKERS = list(IDZOREK_W_EQ.keys())

LABELS = {
    "AGG": "US Bonds (AGG)",
    "BWX": "Intl Bonds (BWX)",
    "IWF": "US Lg Growth (IWF)",
    "IWD": "US Lg Value (IWD)",
    "IWO": "US Sm Growth (IWO)",
    "IWN": "US Sm Value (IWN)",
    "EFA": "Intl Dev (EFA)",
    "EEM": "Emg Mkts (EEM)",
}


def _to_month_end(idx):
    return pd.to_datetime(idx).to_period("M").to_timestamp("M")


def _extract_close(raw, tickers):
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            return pd.DataFrame({t: raw[t]["Close"] for t in tickers})
        except KeyError:
            return pd.DataFrame({t: raw["Close"][t] for t in tickers})
    col = "Close" if "Close" in raw.columns else raw.columns[0]
    return raw[[col]].rename(columns={col: tickers[0]})


# 1. ETF-Renditen
print("=" * 60)
print(f"1. Idzorek-ETF-Renditen ({len(TICKERS)} Ticker)")
print("=" * 60)
print(f"   Ticker:   {TICKERS}")
print(f"   Zeitraum: {START} bis {END}")

raw_etf = yf.download(TICKERS, start=START, end=END, interval="1mo",
                      auto_adjust=True, progress=True, group_by="ticker")
prices = _extract_close(raw_etf, TICKERS).dropna(how="all").sort_index()
returns = prices.pct_change().dropna(how="all").loc["2008-01-01":"2024-12-31"]
returns.index = _to_month_end(returns.index)
returns = returns[TICKERS]

assert len(returns) >= 200, f"Zu wenige Monate: {len(returns)}"
missing = [t for t in TICKERS if t not in returns.columns]
if missing:
    sys.exit(f"Fehlende Ticker: {missing}")
n_nan = returns.isna().sum()
if n_nan.any():
    print("   WARNUNG: fehlende Werte je Ticker:\n", n_nan[n_nan > 0])

train = returns.loc["2008-01-01":"2018-12-31"]
print(f"\n   IWF-IWD Korrelation (sehr hoch erwartet, 2008-2018): "
      f"{train['IWF'].corr(train['IWD']):+.3f}")
print(f"   IWF-AGG Korrelation (niedrig erwartet):              "
      f"{train['IWF'].corr(train['AGG']):+.3f}")
print(f"   Zeitraum: {returns.index[0].date()} bis {returns.index[-1].date()} "
      f"({len(returns)} Monate)")
returns.to_csv(os.path.join(OUT_DIR, "etf_returns_monthly.csv"),
               index_label="Date")
print("   Gespeichert: data/etf_returns_monthly.csv")


# 2. Risk-Free Rate (^IRX)
print("\n" + "=" * 60)
print("2. Risk-Free Rate (^IRX, 13-Week T-Bill)")
print("=" * 60)
raw_rf = yf.download("^IRX", start=START, end=END, interval="1mo",
                     auto_adjust=False, progress=False)["Close"]
if isinstance(raw_rf, pd.DataFrame):
    raw_rf = raw_rf.iloc[:, 0]
rf = (raw_rf / 100.0) / 12.0
rf = rf.reindex(returns.index, method="ffill")
rf.index = _to_month_end(rf.index)
rf.name = "rf"
rf.to_csv(os.path.join(OUT_DIR, "rf_monthly.csv"), header=True, index_label="Date")
print(f"   {rf.index[0].date()} bis {rf.index[-1].date()} ({len(rf)} Monate)")
print("   Gespeichert: data/rf_monthly.csv")


# 3. S&P-500-Excess-Returns (^GSPC - rf)
print("\n" + "=" * 60)
print("3. S&P-500 Excess Returns (^GSPC)")
print("=" * 60)
raw_spx = yf.download("^GSPC", start=START, end=END, interval="1mo",
                      auto_adjust=True, progress=False)["Close"]
if isinstance(raw_spx, pd.DataFrame):
    raw_spx = raw_spx.iloc[:, 0]
spx_ret = raw_spx.pct_change().dropna().loc["2008-01-01":"2024-12-31"]
spx_ret.index = _to_month_end(spx_ret.index)
spx_excess = (spx_ret - rf).dropna()
spx_excess.name = "spx_excess"
spx_excess.to_csv(os.path.join(OUT_DIR, "spx_returns_monthly.csv"),
                  header=True, index_label="Date")
print(f"   {spx_excess.index[0].date()} bis {spx_excess.index[-1].date()} "
      f"({len(spx_excess)} Monate)")
print("   Gespeichert: data/spx_returns_monthly.csv")


# 4. w_eq = Idzorek-Gleichgewichtsgewichte (fest, publiziert)
print("\n" + "=" * 60)
print("4. w_eq (Idzorek 2004, publizierte Gleichgewichtsgewichte)")
print("=" * 60)
w_eq = pd.Series(IDZOREK_W_EQ)
w_eq = (w_eq / w_eq.sum()).reindex(TICKERS)
source = ("Idzorek (2004), 'A Step-by-Step Guide to the Black-Litterman Model', S. 14 - "
          "publizierte Marktkapitalisierungsgewichte; statischer Prior-Anker")
eq_df = pd.DataFrame({"w_eq": w_eq.round(6), "source": source})
eq_df.index.name = "ticker"
eq_df.to_csv(os.path.join(OUT_DIR, "eq_weights_train_start.csv"))
for t in TICKERS:
    print(f"   {t} {LABELS[t]:22s}  {w_eq[t]*100:6.2f} %")
print(f"   Summe: {w_eq.sum():.6f}")
print("   Gespeichert: data/eq_weights_train_start.csv")

print("\n" + "=" * 60)
print("Fertig. Alle Dateien in:", OUT_DIR)
print("=" * 60)
