"""
views.py
========

Konstruktion von Investor Views für das Black-Litterman-Modell.

Ein View ist eine subjektive Aussage über erwartete Rendite(n). Im BLM-Modell
werden Views durch drei Grössen kodiert:

    P * mu = Q   mit Unsicherheit  epsilon ~ N(0, Omega)

- P     : (k x n) Pick-Matrix. Jede Zeile ist ein View.
          * absoluter View: eine 1 in Spalte i (Asset i) -> "E[r_i] = q"
          * relativer View: +w in einigen Spalten, -w in anderen
- Q     : (k x 1) erwartete Renditen aus den Views.
- Omega : (k x k) Diagonalmatrix mit den View-Varianzen (Vertrauensniveau).

Dieses Modul stellt eine kleine DSL bereit, um Views deklarativ zu bauen
und am Ende konsistent (P, Q, Omega) zurückzugeben.

Beispiel
--------
>>> v = ViewSet(asset_names=["A", "B", "C", "D"])
>>> v.add_absolute("A", 0.10, confidence=0.5)          # E[r_A] = 10 %
>>> v.add_relative(["B"], ["C"], 0.03, confidence=0.7) # B übertrifft C um 3 %
>>> P, Q, Omega = v.build(Sigma=Sigma, tau=0.05)

Autor: Lucas Posern
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd


# Datenstruktur für einen einzelnen View
@dataclass
class View:
    """Ein einzelner View. Wird in `ViewSet` aggregiert."""

    p_row: np.ndarray          # (n,) Vector
    q: float                   # erwartete Rendite des Views
    confidence: float = 0.5    # 0..1, mappt auf Omega (siehe ViewSet.build)
    label: str = ""            # Beschreibung für Reporting


# ViewSet: aggregiert Views, baut P, Q, Omega
class ViewSet:
    """Sammelt Views und baut daraus die Tripel (P, Q, Omega).

    Parameters
    ----------
    asset_names : list[str]
        Reihenfolge der Assets. Definiert Spaltenreihenfolge in P.
    """

    def __init__(self, asset_names: list[str]):
        self.asset_names = list(asset_names)
        self._index = {name: i for i, name in enumerate(self.asset_names)}
        self.views: list[View] = []

    @property
    def n_assets(self) -> int:
        return len(self.asset_names)

    @property
    def n_views(self) -> int:
        return len(self.views)

    # View hinzufügen
    def add_absolute(self, asset: str, expected_return: float, confidence: float = 0.5,
                     label: Optional[str] = None) -> "ViewSet":
        """Absoluter View: E[r_asset] = expected_return.

        `expected_return` ist als *Excess Return* zu verstehen (über r_f),
        konsistent mit der pi-Formulierung.
        """
        if asset not in self._index:
            raise KeyError(f"Asset '{asset}' nicht in {self.asset_names}")
        p = np.zeros(self.n_assets)
        p[self._index[asset]] = 1.0
        self.views.append(View(p_row=p, q=expected_return, confidence=confidence,
                               label=label or f"abs:{asset}={expected_return:.2%}"))
        return self

    def add_relative(self, outperform: list[str], underperform: list[str],
                     expected_diff: float, confidence: float = 0.5,
                     weights: Literal["equal", "value"] = "equal",
                     value_weights: Optional[pd.Series] = None,
                     label: Optional[str] = None) -> "ViewSet":
        """Relativer View: outperform-Korb übertrifft underperform-Korb um `expected_diff`.

        Standardmäßig werden beide Körbe gleichgewichtet aufgeteilt
        (Summe der positiven Einträge = +1, Summe der negativen = -1).
        Mit `weights="value"` und `value_weights` kann man auch Marktkapitalisierungs-
        gewichtete Körbe nutzen, wie es Black & Litterman in ihren Beispielen tun.
        """
        for asset in outperform + underperform:
            if asset not in self._index:
                raise KeyError(f"Asset '{asset}' nicht in {self.asset_names}")
        p = np.zeros(self.n_assets)

        if weights == "equal":
            for a in outperform:
                p[self._index[a]] = 1.0 / len(outperform)
            for a in underperform:
                p[self._index[a]] = -1.0 / len(underperform)
        elif weights == "value":
            if value_weights is None:
                raise ValueError("value_weights erforderlich wenn weights='value'")
            pos_sum = value_weights.reindex(outperform).sum()
            neg_sum = value_weights.reindex(underperform).sum()
            for a in outperform:
                p[self._index[a]] = value_weights[a] / pos_sum
            for a in underperform:
                p[self._index[a]] = -value_weights[a] / neg_sum
        else:
            raise ValueError(f"Unbekannte weights-Option: {weights}")

        self.views.append(View(
            p_row=p, q=expected_diff, confidence=confidence,
            label=label or f"rel:{'+'.join(outperform)}>{'+'.join(underperform)}",
        ))
        return self

    def add_custom(self, p_row: np.ndarray, q: float, confidence: float = 0.5,
                   label: str = "custom") -> "ViewSet":
        """Beliebige Linearkombination als View. Vorsichtig zu verwenden."""
        if len(p_row) != self.n_assets:
            raise ValueError(f"p_row Länge {len(p_row)} != n_assets {self.n_assets}")
        self.views.append(View(p_row=np.asarray(p_row, dtype=float), q=q,
                               confidence=confidence, label=label))
        return self

    # Build: P, Q, Omega
    def build(
        self,
        Sigma: pd.DataFrame,
        tau: float = 0.05,
        omega_method: Literal["he_litterman", "idzorek", "manual_diag", "proportional"] = "he_litterman",
        manual_omega_diag: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Baut das Tripel (P, Q, Omega).

        Parameters
        ----------
        Sigma : pd.DataFrame (n x n)
            Annualisierte Kovarianzmatrix. Wird für die Omega-Berechnung benoetigt.
        tau : float
            Skalierungsfaktor des Priors (0.025 - 0.05 üblich).
        omega_method : str
            "he_litterman" : Omega = diag(P * tau*Sigma * P^T)
                             Standard, nicht von User-Confidence abhängig.
            "idzorek"      : Omega(c_k) skaliert über Confidence c_k in (0,1].
                             c=1 => View ist sicher (kleines Omega), c->0 => ignoriert.
                             Konkret: omega_kk = (1 - c_k)/c_k * (P*tau*Sigma*P^T)_kk
                             (vereinfachte Idzorek-Kalibrierung in geschlossener Form;
                             nicht das iterative Tilt-Matching aus Idzorek 2007.)
            "proportional" : Omega = alpha * diag(P * Sigma * P^T), alpha = (1-c)/c
                             Variante ohne tau-Skalierung, verbreitet in Praxis.
            "manual_diag"  : Omega = diag(manual_omega_diag), Varianzen direkt angeben.

        Returns
        -------
        (P, Q, Omega) als np.ndarrays in der Reihenfolge der View-Hinzufügung.
        """
        if self.n_views == 0:
            raise ValueError("ViewSet ist leer - mindestens einen View hinzufügen.")

        P = np.vstack([v.p_row for v in self.views])      # (k, n)
        Q = np.array([v.q for v in self.views])           # (k,)
        c = np.array([v.confidence for v in self.views])  # (k,)

        Sig = Sigma.values if isinstance(Sigma, pd.DataFrame) else np.asarray(Sigma)

        if omega_method == "he_litterman":
            Omega = np.diag(np.diag(P @ (tau * Sig) @ P.T))
        elif omega_method == "idzorek":
            base = np.diag(P @ (tau * Sig) @ P.T)
            # 0 < c <= 1; bei c=1 sehr kleine Diagonale (sicher), c->0 sehr große (unsicher)
            c_safe = np.clip(c, 1e-6, 1.0)
            Omega = np.diag((1.0 - c_safe) / c_safe * base)
        elif omega_method == "proportional":
            base = np.diag(P @ Sig @ P.T)
            c_safe = np.clip(c, 1e-6, 1.0)
            Omega = np.diag((1.0 - c_safe) / c_safe * base)
        elif omega_method == "manual_diag":
            if manual_omega_diag is None or len(manual_omega_diag) != self.n_views:
                raise ValueError("manual_omega_diag muss Länge n_views haben")
            Omega = np.diag(manual_omega_diag)
        else:
            raise ValueError(f"Unbekannte omega_method: {omega_method}")

        return P, Q, Omega

    # Reporting
    def to_frame(self) -> pd.DataFrame:
        """Schöne Tabelle aller Views (für Druck im Skript / Notebook)."""
        rows = []
        for v in self.views:
            row = {"label": v.label, "q": v.q, "confidence": v.confidence}
            for i, a in enumerate(self.asset_names):
                if v.p_row[i] != 0:
                    row[a] = v.p_row[i]
            rows.append(row)
        return pd.DataFrame(rows).fillna(0.0)
