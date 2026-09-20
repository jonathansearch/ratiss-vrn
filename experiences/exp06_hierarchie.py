"""Experience 06 — HIERARCHIE vs PRODUIT, et le sort de c.

ORDRE DU CHEF :
  1. Hierarchique : g_dyn gouverne, s et g_topo MODULENT.
  2. Sans c : prouver que le langage apporte quelque chose, sinon le sortir
     du chemin de y (diagnostic seulement).
  3. Criteres scelles AVANT execution, graines tenues NEUVES.

------------------------------------------------------------------------
CRITERES SCELLES AVANT EXECUTION (graves ici, non modifies ensuite)
------------------------------------------------------------------------

Tache : detection du motif cache, identique au relais (classe 0 = aleatoire
pur, classe 1 = aleatoire + bloc "ATGC"x10 cache). Metrique AUC.

Graines NEUVES : 2000+i (jamais utilisees : 0-1199, 12-16, 21 sont prises).
N = 100 par classe, L = 160.

QUATRE CONFIGURATIONS, forme de modulation FIXEE A PRIORI :

  P   (produit, reference)  y = g_dyn . g_topo . (s . c)
  P-c (produit sans c)      y = g_dyn . g_topo . s
  H   (hierarchique)        y = g_dyn . m(s) . m(g_topo)
  H+c (hierarchique + c)    y = g_dyn . m(s) . m(g_topo) . m(c)

  avec m(x) = 0.5 + 0.5*x . Choix scelle : m va de 0.5 a 1, donc un
  modulateur peut au pire DIVISER le signal par deux, jamais l'annuler.
  C'est la definition operationnelle de "moduler" vs "multiplier".

Criteres scelles :
  C1 (hierarchie) : AUC(H) > AUC(P). Si vrai, la modulation bat le produit.
  C2 (c)          : |AUC(sans c) - AUC(avec c)| <= 0.02
                    -> le langage n'apporte rien, on le sort du chemin de y.
                    Si le gain depasse 0.02, c reste.
  C3 (organe)     : g_dyn seul reste le meilleur organe isole (AUC max).

Ces criteres sont ecrits avant toute mesure. Aucun ajustement apres.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.neurone_vrn import neurone_vrn_v3

N = 100
L = 160
MOTIF = "ATGC" * 10
GRAINE_BASE = 2000          # graines NEUVES


def auc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n1, n0 = int(labels.sum()), int((1 - labels).sum())
    ranks = np.argsort(np.argsort(scores)) + 1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def m(x: float) -> float:
    """Modulateur scelle : [0,1] -> [0.5,1]. Ne peut pas annuler."""
    return 0.5 + 0.5 * float(x)


def chaine(i, avec_motif, longueur=L):
    rng = np.random.default_rng(GRAINE_BASE + i)
    bases = [str(b) for b in rng.choice(list("ATCG"), size=longueur)]
    if avec_motif:
        pos = int(rng.integers(0, longueur - len(MOTIF) + 1))
        bases[pos:pos + len(MOTIF)] = list(MOTIF)
    return "".join(bases)


def main():
    print("=== exp06 : hierarchie vs produit, et le sort de c ===")
    print(f"graines NEUVES {GRAINE_BASE}+i, N={N}/classe, L={L}")
    print("criteres scelles : C1 AUC(H)>AUC(P) | C2 |delta_c|<=0.02 | "
          "C3 g_dyn meilleur organe\n")

    raw = {k: [] for k in
           ("g_dyn", "g_topo", "s", "c", "y_prod", "y_prod_noc",
            "y_hier", "y_hier_c")}
    labels = []

    for i in range(N):
        for lab in (0, 1):
            r = neurone_vrn_v3(chaine(i * 2 + lab, bool(lab)))
            gd, gt = r["g_dyn"], r["g_topo"]
            s, c = r["s_psig"], r["c_semantique"]
            raw["g_dyn"].append(gd)
            raw["g_topo"].append(gt)
            raw["s"].append(s)
            raw["c"].append(c)
            raw["y_prod"].append(gd * gt * (s * c))
            raw["y_prod_noc"].append(gd * gt * s)
            raw["y_hier"].append(gd * m(s) * m(gt))
            raw["y_hier_c"].append(gd * m(s) * m(gt) * m(c))
            labels.append(lab)

    labels = np.array(labels)
    A = {k: auc(v, labels) for k, v in raw.items()}

    print(f"{'mesure':14s} {'moy_c0':>8s} {'moy_c1':>8s} {'AUC':>8s}")
    print("-" * 42)
    for k in ("y_prod", "y_prod_noc", "y_hier", "y_hier_c",
              "g_dyn", "g_topo", "s", "c"):
        v = np.array(raw[k])
        print(f"{k:14s} {v[labels==0].mean():8.4f} {v[labels==1].mean():8.4f} "
              f"{A[k]:8.4f}")

    print("\n--- VERDICTS vs CRITERES SCELLES ---")
    c1 = A["y_hier"] > A["y_prod"]
    delta_c_prod = A["y_prod"] - A["y_prod_noc"]
    delta_c_hier = A["y_hier_c"] - A["y_hier"]
    c2 = abs(delta_c_hier) <= 0.02
    organes = {k: A[k] for k in ("g_dyn", "g_topo", "s", "c")}
    meilleur = max(organes, key=organes.get)
    c3 = meilleur == "g_dyn"

    print(f"C1 hierarchie   : AUC(H)={A['y_hier']:.4f} vs AUC(P)={A['y_prod']:.4f}"
          f"  -> {'CONFIRME' if c1 else 'INFIRME'}")
    print(f"C2 sort de c    : delta H+c vs H = {delta_c_hier:+.4f} ; "
          f"delta produit = {delta_c_prod:+.4f}"
          f"  -> {'c INUTILE (a sortir)' if c2 else 'c APPORTE'}")
    print(f"C3 meilleur organe isole : {meilleur} (AUC {organes[meilleur]:.4f})"
          f"  -> {'CONFIRME' if c3 else 'INFIRME'}")

    out = {
        "graine_base": GRAINE_BASE, "N_par_classe": N, "longueur": L,
        "criteres_scelles": {
            "C1": "AUC(y_hier) > AUC(y_prod)",
            "C2": "|AUC(y_hier_c) - AUC(y_hier)| <= 0.02 -> c inutile",
            "C3": "g_dyn est le meilleur organe isole",
        },
        "modulateur": "m(x)=0.5+0.5x (borne [0.5,1], ne peut pas annuler)",
        "AUC": {k: round(v, 4) for k, v in A.items()},
        "verdicts": {"C1_hierarchie_bat_produit": bool(c1),
                     "C2_c_inutile": bool(c2),
                     "C3_g_dyn_meilleur": bool(c3)},
        "deltas_c": {"produit": round(delta_c_prod, 4),
                     "hierarchique": round(delta_c_hier, 4)},
    }
    p = Path("experiences/exp06_hierarchie.json")
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nOK -> {p}")


if __name__ == "__main__":
    main()