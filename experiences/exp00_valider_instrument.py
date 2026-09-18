"""Experience 00 — VALIDER L'INSTRUMENT.

Avant de croire une seule mesure issue du plongement de Takens sur un
profil d'hydrophobicite, on verifie que l'instrument (Takens + P_sig)
lit correctement des signaux dont on CONNAIT la reponse.

Signaux de verite connue :
  - ligne droite          -> aucun cycle          -> P_sig ~ 0
  - cercle                -> 1 cycle              -> P_sig bas (coexistence)
  - deux cercles lies     -> structure de cycles   -> P_sig plus haut
  - bruit                 -> beaucoup de petits    -> P_sig faible
  - sinus (boucle lisse)  -> 1 cycle propre        -> P_sig ~ 0

Si l'instrument ne retrouve pas ca, aucune conclusion sur les chaines
ATCG n'est valable. C'est la regle R7 appliquee a notre propre outil :
aucune valeur publiee sans reproduction en une commande.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/vrn")

import numpy as np
from organes.psig import p_sig_ripser as p_sig, takens_embed


def montre(nom, cloud, attendu=""):
    if len(cloud) < 4:
        print(f"{nom:28} TROP PEU DE POINTS")
        return
    r = p_sig(cloud)
    print(f"{nom:28} n={len(cloud):>4}  P_sig={r['p_sig']:.4f}  "
          f"robust_h1={r['robust_h1']:>4}  max_vie={r['max_lifetime']:.4f}   {attendu}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("=== Experience 00 : validation de l'instrument ===\n")

    # 1. ligne droite : aucun cycle
    x = np.linspace(0, 1, 60)
    montre("ligne droite", np.column_stack([x, x * 0.5, x * 0.2]), "attendu ~0")

    # 2. cercle parfait : 1 cycle
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    montre("cercle (1 cycle)", np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)]),
           "attendu bas (1 cycle)")

    # 3. sinusoide plongee par Takens : boucle lisse
    xs = np.sin(np.linspace(0, 6 * np.pi, 120))
    montre("sinus -> Takens", takens_embed(xs, 3, 3), "attendu bas (boucle)")

    # 4. bruit : pas de structure
    montre("bruit gaussien", rng.normal(0, 1, (60, 3)), "attendu faible")

    # 5. tore : 2 cycles -> le seul cas ou la coexistence doit s'allumer
    a, b = np.meshgrid(np.linspace(0, 2 * np.pi, 10, endpoint=False),
                       np.linspace(0, 2 * np.pi, 6, endpoint=False))
    R, r2 = 1.0, 0.4
    tore = np.column_stack([(R + r2 * np.cos(b.ravel())) * np.cos(a.ravel()),
                            (R + r2 * np.cos(b.ravel())) * np.sin(a.ravel()),
                            r2 * np.sin(b.ravel())])
    montre("tore (2 cycles)", tore, "attendu HAUT (coexistence)")

    # 6. signal aleatoire marche -> Takens
    mar = np.cumsum(rng.normal(0, 1, 120))
    montre("marche aleatoire -> Takens", takens_embed(mar, 3, 3), "attendu ?")