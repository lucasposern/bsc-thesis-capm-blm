# Thesis-Code - Kritik des CAPM und Erweiterung durch das Black-Litterman-Modell

**Lucas Posern | TU Dresden | Bachelorarbeit 2026**

Vier Jupyter-Notebooks erzeugen alle berechneten Zahlen, Tabellen und Grafiken der Arbeit. Datengrundlage ist das Acht-ETF-Universum des kanonischen Black-Litterman-Beispiels von Idzorek (2004): AGG, BWX, IWF, IWD, IWO, IWN, EFA, EEM. Marktproxy ist der S&P 500 (^GSPC).

**Zum Ansehen ist nichts zu installieren oder auszuführen.** Alle Ausgaben sind inline in den Notebooks gespeichert und rendern direkt hier auf GitHub - einfach das jeweilige `.ipynb` anklicken.

---

## Universum & w_eq

| ETF | Anlageklasse | w_eq (Idzorek) |
|-----|--------------------------|---------------:|
| AGG | US Bonds | 19,34 % |
| BWX | Intl Bonds | 26,13 % |
| IWF | US Large Growth | 12,09 % |
| IWD | US Large Value | 12,09 % |
| IWO | US Small Growth | 1,34 % |
| IWN | US Small Value | 1,34 % |
| EFA | Intl Developed Equity | 24,18 % |
| EEM | Intl Emerging Equity | 3,49 % |

`w_eq` sind die von Idzorek (2004, S. 14) publizierten Marktkapitalisierungsgewichte. Sie gehen ins Black-Litterman-Modell ausschließlich als neutraler Prior-Anker ein (kein Stichproben-Schätzer), weshalb ihr exaktes Datum für die Gültigkeit des Priors unerheblich ist - eine bewusste, aus einer einzigen Quelle belegbare Designentscheidung. Σ und Backtest beruhen auf den Daten 2008-2024 (Train 2008-2018, Test 2019-2024). Alle acht ETFs haben über diesen Zeitraum lückenlose Daten (BWX ab Okt 2007), es existiert also keine In-Sample-Datenlücke.

---

## Zuordnung zur Thesis

| Notebook | Abschnitt | Inhalt |
|----------|-----------|--------|
| `02_Efficient_Frontier.ipynb` | §2 | MV-Frontier, CML, MVP, Tangentialportfolio |
| `03_CAPM.ipynb` | §3.2 | Zeitreihen- und Querschnittsregression, Jensen's Alpha, MV-Sensitivität, Backtest CAPM vs. naives MV |
| `04_BLM.ipynb` | §4.6 | Szenario A/B: Erwartungsrenditen, Gewichte, Performance |
| `04_BLM.ipynb` | §4.7.1 | Sensitivität in τ und c |
| `04_BLM.ipynb` | §4.7.3 | Jarque-Bera-Tests, Fat Tails, VaR-Vergleich |
| `05_Modifikationen.ipynb` | §5.1 | Ledoit-Wolf vs. Sample-Kovarianz; Fixed-Ω: ρ*-Tabelle, Δc_eff, 2×2-Beispiel IWF/IWD |
| `05_Modifikationen.ipynb` | §5.2 | Student-t: ν-Schätzung (MoM/MLE), Dichte-Fit, Backtest |
| `05_Modifikationen.ipynb` | §5.3 | Risk Budgeting: MV, BLM, Risk Parity, RBMV, BLM-RBMV |

---

## Struktur

```
bsc-thesis-capm-blm/
├── 02_Efficient_Frontier.ipynb  # §2: MV-Frontier, CML (3-Asset-Beispiel, analytisch)
├── 03_CAPM.ipynb                # §3: OLS-Regressionen, SML, Jensen's Alpha, Michaud-Effekt, Backtest
├── 04_BLM.ipynb                 # §4: BLM Szenario A/B, Parametersensitivität, Fat Tails
├── 05_Modifikationen.ipynb      # §5: Ledoit-Wolf, Fixed-Ω, Student-t, Risk Budgeting (RBMV)
├── data/
│   ├── etf_returns_monthly.csv      # Monatsrenditen der 8 Idzorek-ETFs (2008-2024)
│   ├── rf_monthly.csv               # Risikofreier Zinssatz (^IRX)
│   ├── spx_returns_monthly.csv      # S&P-500-Überschussrendite (Marktproxy, ^GSPC)
│   └── eq_weights_train_start.csv   # Idzorek-Marktkapitalisierungsgewichte (Prior-Anker)
├── core/
│   ├── data_loader.py        # CSVs als ReturnPanel laden, Idzorek-Konstanten
│   ├── covariance.py         # Kovarianzschätzung (Sample, Ledoit-Wolf)
│   ├── black_litterman.py    # BLM-Kernberechnung
│   ├── views.py              # View-Matrizen P, Q, Ω (inkl. Idzorek-Kalibrierung)
│   ├── optimize.py           # MV-Optimierung, Long-Only-QP (cvxpy)
│   ├── portfolio_metrics.py  # Sharpe-Ratio, Volatilität, Drawdown
│   └── risk_budgeting.py     # Risk Parity, RBMV (solve_rb, solve_rbmv, solve_mv)
└── download_offline_data.py  # Lädt die Rohdaten neu nach data/ (optional, benötigt Internet/yfinance)
```

Alle Tabellen, Zahlen und Grafiken der Arbeit werden direkt in den Notebooks erzeugt und inline angezeigt.

---

## Lokal ausführen (optional)

Nur nötig, wenn die Notebooks selbst neu gerechnet werden sollen. Python 3.10 oder neuer:

```bash
git clone https://github.com/lucasposern/bsc-thesis-capm-blm.git
cd bsc-thesis-capm-blm
pip install -r requirements.txt
jupyter notebook 03_CAPM.ipynb   # dann: Kernel -> Restart & Run All
```

Jedes Notebook läuft eigenständig und in beliebiger Reihenfolge. Die nötigen Daten liegen als CSV in `data/` bei, eine Internetverbindung ist dafür nicht erforderlich.

`download_offline_data.py` lädt die Rohdaten über yfinance neu nach `data/`. Das ist für die Prüfung nicht nötig - alle Notebooks lesen ausschließlich die beiliegenden CSVs. Hinweis: yfinance/Yahoo ist keine stabile Archivquelle, frisch geladene Daten können leicht von den eingefrorenen CSVs abweichen, die die maßgebliche Datengrundlage der Arbeit bilden.
