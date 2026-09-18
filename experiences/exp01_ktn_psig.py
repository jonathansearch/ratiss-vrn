"""Experience 01 — KTN woven vs aligned, lus par P_sig.

Pont transdisciplinaire :
    KTN:Li (ferroelectrique, Travaux/ratiss_photoinduced/ktn_woven.py)
      -> nuage de points 3D
      -> P_sig (persistance H1, organes RATISS)
      -> lecture du neurone VRN (voir / aligne)

Question : P_sig distingue-t-il un tissage (brins entrelaces) d'un
alignement (domaines paralleles) ?

Verite terrain : le woven a beaucoup plus de structure de cycles que
l'aligned. Si P_sig les separe, la douane VRN a un premier capteur.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/vrn")

import numpy as np
from organes.psig import p_sig


def make_woven(n_points=900, n_strands=9, turn=2.0, noise=0.02, seed=42):
    """Repris de Travaux/ratiss_photoinduced/ktn_woven.py."""
    rng = np.random.default_rng(seed)
    pts = []
    per = n_points // n_strands
    for s in range(n_strands):
        t = np.linspace(0.0, np.pi * turn, per)
        phase = 2.0 * np.pi * s / n_strands
        e = rng.normal(0.0, noise, (len(t), 3))
        pts.append(np.column_stack([np.cos(t + phase) + e[:, 0],
                                    np.sin(t + phase) + e[:, 1],
                                    t * 3.0 + e[:, 2]]))
    return np.vstack(pts)


def make_aligned(n_points=900, n_strands=9, noise=0.02, seed=42):
    """Repris de Travaux/ratiss_photoinduced/ktn_woven.py."""
    rng = np.random.default_rng(seed)
    pts = []
    per = n_points // n_strands
    for s in range(n_strands):
        t = np.linspace(0.0, 5.0, per)
        e = rng.normal(0.0, noise, (len(t), 3))
        pts.append(np.column_stack([np.full_like(t, s * 2.0) + e[:, 0],
                                    t + e[:, 1],
                                    np.zeros(len(t)) + e[:, 2]]))
    return np.vstack(pts)


def subsample(pts, n=80, seed=0):
    rng = np.random.default_rng(seed)
    return pts[rng.choice(len(pts), min(n, len(pts)), replace=False)]


def mesure(nom, pts):
    r = p_sig(pts)
    print(f"{nom:22} p_sig={r['p_sig']:.4f}  robust_h1={r['robust_h1']:>3}  "
          f"n_barres={r['n_h1_bars']:>4}  max_vie={r['max_lifetime']:.4f}  "
          f"seuil={r['max_edge']:.4f}")
    return r


if __name__ == "__main__":
    print("=== Experience 01 : KTN woven vs aligned lus par P_sig ===\n")
    print("(sous-echantillon 80 points, plusieurs graines)\n")

    for seed in (0, 1, 2):
        woven = subsample(make_woven(seed=42), 80, seed=seed)
        aligned = subsample(make_aligned(seed=42), 80, seed=seed)
        print(f"-- graine de sous-echantillonnage {seed} --")
        mesure("  woven (tisse)", woven)
        mesure("  aligned (parallele)", aligned)
        print()