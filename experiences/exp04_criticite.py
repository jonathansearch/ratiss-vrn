"""Experience 04 — TEST DE LA CRITICITE.

PREDICTION ENONCEE AVANT LA MESURE.

Ma mesure exp03 dit : tout signal PERIODIQUE ferme la porte topologique
(P_sig = 0 : un cycle unique, aucune coexistence).
L'ordre JEPA dit : le periodique est PREVISIBLE, donc g_dyn devrait ouvrir.

Si on multiplie les deux lectures (g = g_dyn . g_topo), les deux ordres se
contredisent : le periodique est ferme par la topologie, l'aleatoire est
ferme par la dynamique.

PREDICTION : le neurone ne s'ouvre que dans un regime INTERMEDIAIRE.
y(p) doit avoir un MAXIMUM INTERIEUR, pas un maximum aux extremites.

Si c'est vrai, la douane VRN ne recompense ni l'ordre ni le chaos, mais
leur BORD — c'est la criticite.

PROTOCOLE : on interpole entre ordre pur (ATGC repete) et chaos pur
(base aleatoire), avec une probabilite p de desobeir au motif.
p=0 -> ordre pur ; p=1 -> chaos pur.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.neurone_vrn import neurone_vrn_v3


def chaine_interpolee(p: float, n: int = 240, seed: int = 12) -> str:
    """p = probabilite de desobeir au motif periodique ATGC."""
    rng = np.random.default_rng(seed)
    motif = "ATGC"
    return "".join(
        str(rng.choice(list("ATCG"))) if rng.random() < p else motif[i % 4]
        for i in range(n))


if __name__ == "__main__":
    print("=== Experience 04 : le neurone s'ouvre-t-il sur le bord du chaos ? ===\n")
    print("PREDICTION : y(p) a un maximum INTERIEUR, pas aux extremites.\n")
    print(f"{'p (desordre)':>13} {'s(P_sig)':>9} {'c(sem)':>8} {'g_dyn':>8} "
          f"{'g_topo':>8} {'g':>7} {'y':>8}")
    print("-" * 76)

    res = []
    for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        seq = chaine_interpolee(p)
        r = neurone_vrn_v3(seq)
        res.append((p, r))
        print(f"{p:>13.1f} {r['s_psig']:>9.4f} {r['c_semantique']:>8.4f} "
              f"{r['g_dyn']:>8.4f} {r['g_topo']:>8.4f} {r['g']:>7.4f} {r['y']:>8.4f}")

    ys = np.array([r["y"] for _, r in res])
    ps = np.array([p for p, _ in res])
    i_max = int(np.argmax(ys))
    print("\n--- VERDICT ---")
    print(f"maximum de y en p = {ps[i_max]:.1f}  (y={ys[i_max]:.4f})")
    if 0 < i_max < len(ys) - 1:
        print("PREDICTION CONFIRMEE : le maximum est INTERIEUR.")
        print("La douane VRN lit la criticite, pas la structure.")
    else:
        print("PREDICTION REFUTEE : le maximum est a une extremite.")
        print("La douane lit un regime monotone, pas la criticite.")

    # les deux gates se contredisent-ils ?
    g_dyn = np.array([r["g_dyn"] for _, r in res])
    g_topo = np.array([r["g_topo"] for _, r in res])
    corr = float(np.corrcoef(g_dyn, g_topo)[0, 1])
    print(f"\ncorrelation g_dyn vs g_topo = {corr:+.3f}")
    if corr < -0.3:
        print("ANTI-CORRELES : confirme que les deux lectures se contredisent.")
        print("Aucune des deux seules ne peut gouverner le neurone.")
    else:
        print("Non anti-correles : les deux lectures disent la meme chose.")