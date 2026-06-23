"""
optimize.py
===========

Mean-Variance-Gewichte mit und ohne Leerverkaufsrestriktion.

Stellt eine *einheitliche* Schnittstelle für alle Notebooks bereit, damit
Long-only-Pendants konsistent (gleiche Zielfunktion, nur zusätzliche
Restriktion) berechnet werden - statt per Hand "clip auf 0 + renormieren".

Zielfunktion (Mean-Variance-Nutzen des repräsentativen Investors):

    max_w   mu^T w  -  (delta / 2) * w^T Sigma w

Unrestringiert (Closed-Form):
    w_raw = (delta * Sigma)^{-1} * mu        -> auf sum(w) = 1 normiert
    (delta kürzt sich bei der Normierung heraus -> identisch zum
     normierten Tangentialportfolio Sigma^{-1} mu / sum(...))

Long-only (echte QP via cvxpy):
    max_w   mu^T w - (delta/2) w^T Sigma w
    u.d.N.  sum_i w_i = 1,   w_i >= 0

Hintergrund: Ein Leerverkaufsverbot wirkt als implizite Regularisierung
(Jagannathan & Ma 2003, "Risk Reduction in Large Portfolios: Why Imposing
the Wrong Constraints Helps"). Es dämpft genau die Extremallokationen, die
das unrestringierte Markowitz-Problem bei verrauschten Schätzern erzeugt.

Konvention: mu und Sigma müssen *konsistent* skaliert sein (beide
annualisiert oder beide monatlich). delta wird passend zur Skalierung von
mu/Sigma übergeben (annualisiert, falls mu/Sigma annualisiert).

Autor: Lucas Posern, Bachelorarbeit "Kritik des CAPM und Erweiterung
       durch das Black-Litterman-Model".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp


# Kern: Mean-Variance-Gewichte (restringiert / unrestringiert)

def mv_weights(
    mu,
    Sigma,
    delta: float,
    long_only: bool = False,
    fully_invested: bool = True,
    solver: str = "CLARABEL",
):
    """Mean-Variance-optimale Portfoliogewichte.

    Löst  max_w  mu^T w - (delta/2) w^T Sigma w.

    Parameter
    ---------
    mu        : (n,) Erwartete (Excess-)Renditen. pd.Series oder ndarray.
    Sigma     : (n,n) Kovarianzmatrix, konsistent zu mu skaliert.
    delta     : Risiko-Aversion (zur Skalierung von mu/Sigma passend).
    long_only : False -> unrestringierte Closed-Form (kann Shorts enthalten).
                True  -> QP mit w >= 0.
    fully_invested : sum(w) = 1 erzwingen (Standard True). Bei der
                unrestringierten Lösung wird ohnehin auf 1 normiert.
    solver    : cvxpy-Solver für den Long-only-Fall ("CLARABEL", "SCS", ...).

    Rückgabe
    ---------
    Gewichte als pd.Series (falls mu eine Series ist) bzw. ndarray.
    Summe der Gewichte = 1.
    """
    is_series = isinstance(mu, pd.Series)
    index = mu.index if is_series else (
        Sigma.index if isinstance(Sigma, pd.DataFrame) else None
    )

    mu_v = np.asarray(mu, dtype=float).ravel()
    S = np.asarray(Sigma, dtype=float)
    n = mu_v.shape[0]

    if not long_only:
        # Unrestringierte Nutzenmaximierung: w = (delta*Sigma)^{-1} mu
        try:
            w = np.linalg.solve(delta * S, mu_v)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(delta * S, mu_v, rcond=None)[0]
        if fully_invested:
            s = w.sum()
            w = w / (s if abs(s) > 1e-12 else 1.0)
    else:
        w_var = cp.Variable(n)
        # psd_wrap: erlaubt auch nicht-perfekt-PSD Stichprobenmatrizen
        risk = 0.5 * delta * cp.quad_form(w_var, cp.psd_wrap(S))
        objective = cp.Maximize(mu_v @ w_var - risk)
        constraints = [w_var >= 0]
        if fully_invested:
            constraints.append(cp.sum(w_var) == 1)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver, verbose=False)
        if w_var.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Long-only QP fehlgeschlagen: Status = {prob.status}")
        w = np.maximum(w_var.value, 0.0)
        if fully_invested and w.sum() > 1e-12:
            w = w / w.sum()

    if is_series or index is not None:
        return pd.Series(w, index=index)
    return w


def mv_weights_pair(mu, Sigma, delta: float, **kwargs):
    """Bequemlichkeit: liefert (unrestringiert, long_only) als Tupel.

    Praktisch für Backtests, in denen beide Varianten verglichen werden.
    """
    w_unc = mv_weights(mu, Sigma, delta, long_only=False, **kwargs)
    w_lo = mv_weights(mu, Sigma, delta, long_only=True, **kwargs)
    return w_unc, w_lo


# Diagnostik: wie stark bindet die Restriktion?

def constraint_diagnostics(w_unconstrained, w_long_only) -> dict:
    """Kennzahlen dafür, wie stark das Leerverkaufsverbot bindet.

    Gibt ein dict zurück mit:
        n_short        : Anzahl Short-Positionen in der unrestr. Lösung
        sum_short      : Summe der (negativen) Short-Gewichte (unrestr.)
        gross_exposure : sum |w| der unrestringierten Lösung (1.0 = keine Shorts)
        turnover_to_lo : 0.5 * sum |w_unc - w_lo|  (L1-Abstand / 2)
        n_active_lo    : Anzahl Assets mit Gewicht > 0 im Long-only-Portfolio
    """
    w_u = np.asarray(w_unconstrained, dtype=float).ravel()
    w_l = np.asarray(w_long_only, dtype=float).ravel()
    return {
        "n_short":        int((w_u < -1e-6).sum()),
        "sum_short":      float(w_u[w_u < 0].sum()),
        "gross_exposure": float(np.abs(w_u).sum()),
        "turnover_to_lo": float(0.5 * np.abs(w_u - w_l).sum()),
        "n_active_lo":    int((w_l > 1e-6).sum()),
    }
