"""
black_litterman.py
==================

Black-Litterman-Modell (He & Litterman, 1999).

Variablen (LaTeX-Notation):
    pi          (\\pi)                implizite Gleichgewichtsrenditen (Excess)
    Sigma       (\\Sigma)             Kovarianzmatrix der Asset-Renditen (annualisiert)
    w_eq                             Gleichgewichtsgewichte (\\sum w_eq = 1)
    delta       (\\delta)             Risiko-Aversion des repräsentativen Investors
    tau         (\\tau)               Skalierungsfaktor des Priors (~0.025 bis 0.05)
    P           (k \\times n)         Pick-Matrix der Views
    Q           (k,)                 erwartete Renditen der Views
    Omega       (\\Omega, k \\times k) Unsicherheitsmatrix der Views (diagonal)
    mu_BL       (\\mu_{BL})           Posterior-Erwartungswert
    M                                Posterior-Kovarianz des Mittelwerts
    Sigma_post  (\\Sigma_{post})      Posterior-Kovarianz der Renditen

Kernformeln (He-Litterman Master Formula):
    pi         = delta * Sigma * w_eq
    A          = P * tau*Sigma * P^T + Omega
    mu_BL      = pi + tau*Sigma * P^T * A^{-1} * (Q - P * pi)
    M          = tau*Sigma - tau*Sigma * P^T * A^{-1} * P * tau*Sigma
    Sigma_post = Sigma + M
    w_BL       = (delta * Sigma_post)^{-1} * mu_BL

Parameter-Imports:
    from covariance import sample_cov, get_estimator   # \\Sigma
    from views        import ViewSet                      # P, Q, \\Omega

Typischer Aufruf:
    from covariance    import sample_cov, get_estimator
    from views           import ViewSet
    from black_litterman import run_blm, implied_risk_aversion

    Sigma       = sample_cov(excess_returns)
    delta       = implied_risk_aversion(market_excess)
    views       = ViewSet(assets)
    views.add_absolute("AAPL", 0.10, confidence=0.5)
    P, Q, Omega = views.build(Sigma=Sigma, tau=0.05)
    result      = run_blm(Sigma, w_eq, P, Q, Omega, delta=delta, tau=0.05)

Autor: Lucas Posern, Bachelorarbeit "Kritik des CAPM und Erweiterung
       durch das Black-Litterman-Model".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Parameter-Module importieren
from covariance import sample_cov, get_estimator   # Kovarianzmatrix \\Sigma
from views import ViewSet                             # Views P, Q, \\Omega


# 1. Risiko-Aversion delta aus Marktdaten schätzen

def implied_risk_aversion(
    market_excess_returns: pd.Series,
    annualize: bool = True,
    frequency: str = "M",
) -> float:
    """
    Schätzt delta = E[r_M - r_f] / Var(r_M - r_f).

    Folgt aus dem CAPM-Optimum des repräsentativen Investors:
        delta = E[r_M] / Var(r_M)

    Parameter
    ---------
    market_excess_returns : monatliche/tägliche Excess Returns des Marktindex
    annualize             : auf Jahresbasis hochrechnen
    frequency             : "M", "D" oder "W"
    """
    er  = market_excess_returns.mean()
    var = market_excess_returns.var(ddof=1)
    factor = {"M": 12, "D": 252, "W": 52}[frequency] if annualize else 1
    return float((er * factor) / (var * factor))


# 2. Implizite Gleichgewichtsrenditen pi

def implied_returns(
    Sigma: pd.DataFrame,
    w_eq: pd.Series,
    delta: float,
) -> pd.Series:
    """
    pi = delta * Sigma * w_eq

    Leitet aus dem CAPM-Marktgleichgewicht implizierte Excess-Returns ab.
    Kehrt die Mean-Variance-Optimierung um: gegeben Sigma, w_eq und delta
    ergibt sich der erwartete Return, der diese Gewichte als optimal begründet.

    Parameter
    ---------
    Sigma : pd.DataFrame (n x n)  annualisierte Kovarianzmatrix
    w_eq  : pd.Series (n,)        Gleichgewichtsgewichte
    delta : float                 Risiko-Aversion (\\delta)
    """
    pi = delta * Sigma.values @ w_eq.values
    return pd.Series(pi, index=Sigma.index, name="pi")


# 3. BLM-Posterior: mu_BL, M, Sigma_post

def posterior(
    pi: pd.Series,
    Sigma: pd.DataFrame,
    delta: float,
    tau: float,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    add_M_to_Sigma: bool = True,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Berechnet den BLM-Posterior nach He-Litterman (1999).

    Formeln:
        A          = P * tau*Sigma * P^T + Omega
        mu_BL      = pi + tau*Sigma * P^T * A^{-1} * (Q - P * pi)
        M          = tau*Sigma - tau*Sigma * P^T * A^{-1} * P * tau*Sigma
        Sigma_post = Sigma + M    (Standard, He-Litterman)

    Anmerkung:
        Es wird nur die kleine (k x k)-Matrix A invertiert (Woodbury-/
        He-Litterman-Form), nicht die volle (n x n)-Praezisionsmatrix.
        Statt eines expliziten Inversen wird np.linalg.solve verwendet.

    Parameter
    ---------
    pi             : implizite Gleichgewichtsrenditen (aus implied_returns)
    Sigma          : annualisierte Kovarianzmatrix (n x n)
    delta          : Risiko-Aversion (\\delta)
    tau            : Prior-Skalierungsfaktor (\\tau)
    P              : Pick-Matrix der Views (k x n)
    Q              : erwartete View-Renditen (k,)
    Omega          : View-Unsicherheitsmatrix (k x k), positiv definit
    add_M_to_Sigma : Sigma_post = Sigma + M (True) oder Sigma (False)

    Rückgabe
    ---------
    (mu_BL, M, Sigma_post)
    """
    assets  = list(Sigma.index)
    S       = Sigma.values
    tau_S   = tau * S
    pi_v    = pi.values

    # A = P * tau*Sigma * P^T + Omega   (k x k)
    A   = P @ tau_S @ P.T + Omega
    PtS = P @ tau_S                      # (k x n);  tau_S symmetrisch => tau_S @ P.T = PtS.T

    # mu_BL = pi + tau*Sigma * P^T * A^{-1} * (Q - P * pi)
    mu_BL_v = pi_v + PtS.T @ np.linalg.solve(A, Q - P @ pi_v)
    mu_BL   = pd.Series(mu_BL_v, index=assets, name="mu_BL")

    # M = tau*Sigma - tau*Sigma * P^T * A^{-1} * P * tau*Sigma
    M_mat = tau_S - PtS.T @ np.linalg.solve(A, PtS)
    M     = pd.DataFrame(M_mat, index=assets, columns=assets)

    # Sigma_post = Sigma + M  (Standard, He-Litterman 1999)
    S_post_mat = S + M_mat if add_M_to_Sigma else S
    Sigma_post = pd.DataFrame(S_post_mat, index=assets, columns=assets)

    return mu_BL, M, Sigma_post


# 4. Optimale Gewichte w_BL

def optimal_weights(
    mu_BL: pd.Series,
    Sigma_post: pd.DataFrame,
    delta: float,
) -> pd.Series:
    """
    w_BL = (delta * Sigma_post)^{-1} * mu_BL

    Löst das unrestringierte Mean-Variance-Problem mit dem BLM-Posterior.
    Achtung: kann negative Gewichte (Leerverkäufe) enthalten.

    Parameter
    ---------
    mu_BL      : Posterior-Erwartungswert (aus posterior())
    Sigma_post : Posterior-Kovarianzmatrix (aus posterior())
    delta      : Risiko-Aversion (\\delta)
    """
    w = np.linalg.solve(delta * Sigma_post.values, mu_BL.values)
    return pd.Series(w, index=mu_BL.index, name="w_BL")


# 5. Ergebnis-Zusammenfassung

def summary(
    pi: pd.Series,
    mu_BL: pd.Series,
    w_eq: pd.Series,
    w_BL: pd.Series,
) -> pd.DataFrame:
    """
    Vergleichstabelle: implizite vs. Posterior-Renditen und Gewichte.

    Spalten: pi, mu_BL, delta_mu (= mu_BL - pi), w_eq, w_BL, delta_w (= w_BL - w_eq)
    """
    return pd.DataFrame({
        "pi (implied)":      pi,
        "mu_BL (posterior)": mu_BL,
        "delta_mu":          mu_BL - pi,
        "w_eq":              w_eq,
        "w_BL":              w_BL,
        "delta_w":           w_BL - w_eq,
    })


# 6. Vollständiger BLM-Durchlauf

def run_blm(
    Sigma: pd.DataFrame,
    w_eq: pd.Series,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    delta: float = 2.5,
    tau: float = 0.05,
    add_M_to_Sigma: bool = True,
) -> dict:
    """
    Vollständiger BLM-Durchlauf in einem Aufruf.

    Sigma kommt aus covariance (z.B. sample_cov, ledoit_wolf_cov).
    P, Q, Omega kommen aus views.ViewSet.build().

    Ablauf:
        Schritt 1: pi         = delta * Sigma * w_eq
        Schritt 2: mu_BL, M, Sigma_post = posterior(pi, Sigma, ...)
        Schritt 3: w_BL       = (delta * Sigma_post)^{-1} * mu_BL

    Parameter
    ---------
    Sigma          : annualisierte Kovarianzmatrix (aus covariance)
    w_eq           : Gleichgewichtsgewichte
    P, Q, Omega    : Views (aus views.ViewSet.build())
    delta          : Risiko-Aversion (\\delta)
    tau            : Prior-Skalierungsfaktor (\\tau)
    add_M_to_Sigma : Sigma_post = Sigma + M (True, Standard)

    Rückgabe
    ---------
    dict mit den Schlüsseln:
        "Sigma"      : annualisierte Kovarianzmatrix (Eingabe, unverändert)
        "pi"         : implizite Gleichgewichtsrenditen
        "mu_BL"      : Posterior-Erwartungswert
        "M"          : Posterior-Kovarianz des Mittelwerts
        "Sigma_post" : Posterior-Kovarianz der Renditen
        "w_eq"       : Gleichgewichtsgewichte (Eingabe, unverändert)
        "w_BL"       : BLM-optimale Gewichte
    """
    # Schritt 1: pi = delta * Sigma * w_eq
    pi = implied_returns(Sigma, w_eq, delta)

    # Schritt 2: Posterior (mu_BL, M, Sigma_post)
    mu_BL, M, Sigma_post = posterior(pi, Sigma, delta, tau, P, Q, Omega, add_M_to_Sigma)

    # Schritt 3: w_BL = (delta * Sigma_post)^{-1} * mu_BL
    w_BL = optimal_weights(mu_BL, Sigma_post, delta)

    return {
        "Sigma":      Sigma,
        "pi":         pi,
        "mu_BL":      mu_BL,
        "M":          M,
        "Sigma_post": Sigma_post,
        "w_eq":       w_eq,
        "w_BL":       w_BL,
    }


# 7. Notebook-Interface: Pi direkt übergeben, Omega intern berechnen

def black_litterman(
    Sigma: np.ndarray,
    Pi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    tau: float = 0.05,
    omega_method: str = "he_litterman",
    delta: float = 2.5,
    add_M_to_Sigma: bool = True,
) -> dict:
    """
    BLM-Funktion mit direkter Pi-Übergabe und interner Omega-Berechnung.

    Im Unterschied zu run_blm() nimmt diese Funktion Pi (implizite Renditen)
    direkt entgegen statt w_eq, und berechnet Omega aus omega_method.

    Parameter
    ---------
    Sigma        : Kovarianzmatrix als numpy-Array (n x n)
    Pi           : implizite Gleichgewichtsrenditen (n,)
    P            : Pick-Matrix der Views (k x n)
    Q            : erwartete View-Renditen (k,)
    tau          : Prior-Skalierungsfaktor
    omega_method : "he_litterman" => Omega = diag(P @ tau*Sigma @ P^T)
    delta        : Risiko-Aversion
    add_M_to_Sigma : Sigma_post = Sigma + M (True, Standard)

    Rückgabe
    ---------
    dict mit: Sigma, pi, mu_BL, M, Sigma_post, w_bl
    """
    S    = np.asarray(Sigma, dtype=float)
    pi_v = np.asarray(Pi,    dtype=float)
    tau_S = tau * S

    if omega_method == "he_litterman":
        Omega = np.diag(np.diag(P @ tau_S @ P.T))
    else:
        raise ValueError(
            f"Unbekannte omega_method: '{omega_method}'. "
            "Verfügbar: 'he_litterman'"
        )

    A   = P @ tau_S @ P.T + Omega
    PtS = P @ tau_S                      # (k x n);  tau_S symmetrisch => tau_S @ P.T = PtS.T

    mu_BL = pi_v + PtS.T @ np.linalg.solve(A, Q - P @ pi_v)
    M     = tau_S - PtS.T @ np.linalg.solve(A, PtS)
    S_post = S + M if add_M_to_Sigma else S

    w_bl = np.linalg.solve(delta * S_post, mu_BL)

    return {
        "Sigma":      S,
        "pi":         pi_v,
        "mu_BL":      mu_BL,
        "M":          M,
        "Sigma_post": S_post,
        "w_bl":       w_bl,
    }
