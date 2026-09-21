"""Mesures partagees, avec traitement CORRECT des ex aequo.

Pourquoi ce module. La fonction _auc ecrite a la main dans exp10/exp11/exp12
utilisait des rangs entiers (argsort) : quand toutes les valeurs sont egales
(signal constant, ex aequo parfaits), elle renvoyait 0.0 au lieu de 0.5.
L'erreur a ete attrapee par le controle V4 d'exp12 (fitness « sans bloc » =
0.000 la ou 0.5 etait attendu).

Principe : sur des ex aequo, l'AUC doit valoir 0.5 (aucune information). On
utilise les RANGS MOYENS (midranks) pour les ex aequo, convention standard.
"""
from __future__ import annotations

import numpy as np


def rangs_moyens(x: np.ndarray) -> np.ndarray:
    """Rangs 1..n avec la moyenne attribuee aux ex aequo."""
    n = x.size
    ordre = np.argsort(x, kind="mergesort")
    tri = x[ordre]
    rangs = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and tri[j + 1] == tri[i]:
            j += 1
        rangs[ordre[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return rangs


def auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) + 0.5 P(a = b). 0.5 si aucune separation (ex aequo partout)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.5
    tot = np.concatenate([a, b])
    r = rangs_moyens(tot)
    r_a = r[:a.size].sum()
    return float((r_a - a.size * (a.size + 1) / 2.0) / (a.size * b.size))
