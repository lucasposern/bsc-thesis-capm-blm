# risk_budgeting.py
# Implementierung des Risk-Budgeting- und RBMV-Portfolio-Frameworks
# nach Freitas Paulo da Costa, Riva & Targino (2025)
#
# Kernformeln:
#   RC_i(v)  = v_i * (Sigma @ v)_i / sqrt(v^T Sigma v)         [Risikobeirtrag]
#   sigma(v) = sum_i RC_i(v)                                    [Euler-Zerlegung]
#
# RB-Problem  (Proposition 1):
#   min_{v>=0}  sqrt(v^T Sigma v)
#   s.t.        sum_i b_i log(v_i) >= 0
#
# RBMV-Problem (Definition 3):
#   min_{v>=0}  sqrt(v^T Sigma v)
#   s.t.        sum_i b_i log(v_i) >= 0          [lambda_v]
#               mu^T v >= mu_min * sum_i v_i      [lambda_mu]
#               sqrt(v^T Sigma v) <= sigma_max * sum_i v_i  [lambda_sigma]
#
# MV-Problem:
#   min_{v>=0}  sqrt(v^T Sigma v)
#   s.t.        sum_i v_i = v0
#               mu^T v >= mu_min * v0
#               sqrt(v^T Sigma v) <= sigma_max * v0

import numpy as np
import pandas as pd
import cvxpy as cp


# Hilfsfunktionen

def risk_contributions(v: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Berechnet den Risikobeitrag jedes Assets:
        RC_i(v) = v_i * (Sigma @ v)_i / sqrt(v^T Sigma v)

    Euler-Theorem garantiert: sum_i RC_i(v) = sigma(R(v))

    Parameter
    ----------
    v     : (d,) Dollar-Exposures
    Sigma : (d,d) Kovarianzmatrix der Returns
    """
    port_var = float(v @ Sigma @ v)
    port_std = np.sqrt(max(port_var, 1e-20))
    return v * (Sigma @ v) / port_std


def gini_index(weights: np.ndarray) -> float:
    """
    Gini-Konzentrationsmass für einen Gewichtsvektor.
    0 = gleichmässig verteilt, 1 = alles in einem Asset.
    """
    w = np.abs(weights)
    n = len(w)
    total = w.sum()
    if total < 1e-12:
        return 0.0
    return float(np.sum(np.abs(np.subtract.outer(w, w))) / (2 * n * total))


def _chol(Sigma: np.ndarray) -> np.ndarray:
    """Cholesky-Faktor von Sigma (macht DCP-konforme Volatilitätsformulierung möglich)."""
    # Kleine Regularisierung für numerische Stabilität
    jitter = 1e-8 * np.eye(Sigma.shape[0])
    return np.linalg.cholesky(Sigma + jitter)


def _sigma_minvar(Sigma: np.ndarray) -> float:
    """Volatilität des Minimum-Varianz-Portfolios (ohne Rendite-NB)."""
    L = _chol(Sigma)
    d = Sigma.shape[0]
    v = cp.Variable(d, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.norm(L.T @ v, 2)),
                      [cp.sum(v) == 1.0])
    prob.solve(solver=cp.SCS, verbose=False, eps=1e-6)
    return float(np.sqrt(v.value @ Sigma @ v.value))


# Optimierer

def solve_rb(
    b: np.ndarray,
    Sigma: np.ndarray,
    v0: float = 1.0,
    solver: str = "SCS",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Löst das Risk-Budgeting-Problem (Proposition 1 aus Targino et al. 2025):

        min_{v >= 0}  sqrt(v^T Sigma v)
        s.t.          sum_i b_i * log(v_i) >= 0

    Die Lösung v* ist proportional zum RB-Portfolio; sie wird auf
    sum(v) = v0 normiert zurückgegeben.

    Parameter
    ----------
    b      : (d,) Risikobudget, b_i > 0, sum(b) = 1
    Sigma  : (d,d) Kovarianzmatrix (annualisiert)
    v0     : Gesamtbudget (Standard: 1.0 -> Gewichte)
    """
    L = _chol(Sigma)
    d = len(b)
    v = cp.Variable(d, nonneg=True)

    objective   = cp.Minimize(cp.norm(L.T @ v, 2))
    constraints = [cp.sum(cp.multiply(b, cp.log(v))) >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=False, eps=eps)

    if v.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"RB-Solver Fehler: Status = {prob.status}")

    v_star = np.maximum(v.value, 0.0)
    return v0 * v_star / v_star.sum()


def solve_rbmv(
    b: np.ndarray,
    Sigma: np.ndarray,
    mu: np.ndarray,
    mu_min: float,
    sigma_max: float,
    v0: float = 1.0,
    solver: str = "SCS",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Löst das Risk-Budgeted Mean-Variance Portfolio-Problem (Definition 3):

        min_{v >= 0}  sqrt(v^T Sigma v)
        s.t.          sum_i b_i * log(v_i) >= 0          [lambda_v]
                      mu^T v >= mu_min * sum_i v_i        [lambda_mu]
                      sqrt(v^T Sigma v) <= sigma_max * sum_i v_i  [lambda_sigma]

    Interpoliert zwischen RB (mu_min klein / sigma_max gross) und MV.

    Parameter
    ----------
    b         : (d,) Risikobudget, b_i > 0, sum(b) = 1
    Sigma     : (d,d) Kovarianzmatrix (annualisiert)
    mu        : (d,) Erwartungsrenditen (annualisiert; z.B. mu_BL oder pi)
    mu_min    : Mindesterwartungsrendite des Portfolios (p.a.)
    sigma_max : Maximale erlaubte Portfoliovolatilität (p.a.)
    v0        : Gesamtbudget (Standard: 1.0 -> Gewichte)
    """
    L = _chol(Sigma)
    d = len(b)
    v = cp.Variable(d, nonneg=True)

    port_vol = cp.norm(L.T @ v, 2)
    total_v  = cp.sum(v)

    objective   = cp.Minimize(port_vol)
    constraints = [
        cp.sum(cp.multiply(b, cp.log(v))) >= 0,   # Risk-Budgeting-NB
        mu @ v >= mu_min * total_v,                # Mindestrendite
        port_vol <= sigma_max * total_v,           # Max-Volatilität
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=False, eps=eps)

    if v.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"RBMV-Solver Fehler: Status = {prob.status}")

    v_star = np.maximum(v.value, 0.0)
    return v0 * v_star / v_star.sum()


def solve_mv(
    Sigma: np.ndarray,
    mu: np.ndarray,
    mu_min: float,
    sigma_max: float,
    v0: float = 1.0,
    solver: str = "SCS",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Löst das klassische (long-only) Mean-Variance-Problem:

        min_{v >= 0}  sqrt(v^T Sigma v)
        s.t.          sum_i v_i = v0
                      mu^T v / v0 >= mu_min
                      sqrt(v^T Sigma v) / v0 <= sigma_max

    Parameter
    ----------
    Sigma     : (d,d) Kovarianzmatrix (annualisiert)
    mu        : (d,) Erwartungsrenditen (annualisiert)
    mu_min    : Mindesterwartungsrendite des Portfolios (p.a.)
    sigma_max : Maximale Portfoliovolatilität (p.a.)
    v0        : Gesamtbudget (Standard: 1.0 -> Gewichte)
    """
    L = _chol(Sigma)
    d = len(mu)
    v = cp.Variable(d, nonneg=True)

    port_vol = cp.norm(L.T @ v, 2)

    objective   = cp.Minimize(port_vol)
    constraints = [
        cp.sum(v) == v0,
        mu @ v >= mu_min * v0,
        port_vol <= sigma_max * v0,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=False, eps=eps)

    if v.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"MV-Solver Fehler: Status = {prob.status}")

    return np.maximum(v.value, 0.0)


def auto_params(
    Sigma: np.ndarray,
    mu: np.ndarray,
    sigma_buffer: float = 0.02,
    sigma_cap: float = 0.25,
    mu_discount: float = 0.80,
) -> tuple[float, float]:
    """
    Leitet sigma_max und mu_min automatisch aus den geschätzten
    Momenten ab, sodass das RBMV-Problem immer lösbar ist:

        sigma_max = min(sigma_MinVar + sigma_buffer, sigma_cap)
        mu_min    = mu_discount * mu_MV

    Gibt (mu_min, sigma_max) zurück.
    """
    sigma_mv = _sigma_minvar(Sigma)
    sigma_max = min(sigma_mv + sigma_buffer, sigma_cap)

    # MV-Portfolio zur Referenz: max Rendite unter Volatilitätsdeckel
    L = _chol(Sigma)
    d = len(mu)
    v = cp.Variable(d, nonneg=True)
    prob = cp.Problem(
        cp.Maximize(mu @ v),
        [cp.sum(v) == 1.0,
         cp.norm(L.T @ v, 2) <= sigma_max]
    )
    try:
        prob.solve(solver=cp.SCS, verbose=False)
        mu_mv = float(mu @ v.value) if v.value is not None else float(mu.mean())
    except Exception:
        mu_mv = float(mu.mean())

    mu_min = mu_discount * max(mu_mv, 0.0)
    return float(mu_min), float(sigma_max)


# Kompakt-API für den Backtest

def portfolio_weights(
    strategy: str,
    Sigma: np.ndarray,
    mu: np.ndarray,
    b: np.ndarray | None = None,
    mu_min: float | None = None,
    sigma_max: float | None = None,
    v0: float = 1.0,
) -> np.ndarray:
    """
    Einheitlicher Einstiegspunkt für alle Strategien.

    strategy : "rb"   -> reines Risk Budgeting
               "rbmv" -> RBMV (braucht mu_min, sigma_max)
               "mv"   -> Mean-Variance (braucht mu_min, sigma_max)
    b        : Risikobudget (Standard: gleiches Budget 1/d)
    """
    d = len(mu)
    if b is None:
        b = np.ones(d) / d

    if strategy == "rb":
        return solve_rb(b, Sigma, v0)
    elif strategy == "rbmv":
        if mu_min is None or sigma_max is None:
            mu_min, sigma_max = auto_params(Sigma, mu)
        return solve_rbmv(b, Sigma, mu, mu_min, sigma_max, v0)
    elif strategy == "mv":
        if mu_min is None or sigma_max is None:
            mu_min, sigma_max = auto_params(Sigma, mu)
        return solve_mv(Sigma, mu, mu_min, sigma_max, v0)
    else:
        raise ValueError(f"Unbekannte Strategie: '{strategy}'")
