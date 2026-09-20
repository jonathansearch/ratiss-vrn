#!/usr/bin/env python3
"""exp05 — TACHE A REPONSE CONNUE : detection d'un motif cache (relais Arena).

Question : le neurone v3 retrouve-t-il un ordre cache dont ON connait la reponse ?
  - Classe 0 : chaine aleatoire pure (160 bases).
  - Classe 1 : aleatoire + UN bloc periodique cache ("ATGC"x10 = 40 bases,
    position aleatoire). Verite terrain = labels (connus d'avance).

Mesures : y, s, c, g_dyn, g_topo, fusion. Metrique : AUC (sans seuil).
Critere SCELLE avant execution : AUC(y) >= 0.70 -> le neurone trouve l'ordre
cache ; AUC ~ 0.50 (+/-0.10) -> pas de detection (documente, pas juge).
Prediction enoncee avant mesure : separation faible (AUC 0.60-0.75),
portee surtout par g_dyn (predictibilite).

Graines 1000+i : held-out (jamais vues par les ateliers 00-04).
Rejouable : python3 experiences/exp05_tache_motif.py  (depuis la racine)
"""
import json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from organes.neurone_vrn import neurone_vrn_v3

N = 100          # chaines par classe
L = 160          # longueur
MOTIF = "ATGC" * 10  # 40 bases periodiques cachees


def auc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n1, n0 = int(labels.sum()), int((1 - labels).sum())
    ranks = np.argsort(np.argsort(scores)) + 1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def chaine(i, avec_motif):
    rng = np.random.default_rng(1000 + i)
    bases = [str(b) for b in rng.choice(list("ATCG"), size=L)]
    if avec_motif:
        pos = int(rng.integers(0, L - len(MOTIF) + 1))
        bases[pos:pos + len(MOTIF)] = list(MOTIF)
    return "".join(bases)


def main():
    print("=== exp05 : motif cache — critere scelle AUC(y) >= 0.70 ===\n")
    # NOTE : ce script a d'abord mesure y = g.(s*c) (produit). Depuis exp06,
    # neurone_vrn_v3 calcule y en HIERARCHIQUE (g_dyn x m_s x m_topo) et ne
    # renvoie plus "fusion". On lit donc m_s et m_topo, et c reste un
    # diagnostic. Voir JOURNAL, atelier 06.
    rec = {k: [] for k in ("y", "s", "c", "g_dyn", "g_topo", "m_s", "m_topo",
                           "y_prod_ref")}
    labels = []
    for i in range(N):
        for lab in (0, 1):
            r = neurone_vrn_v3(chaine(i * 2 + lab, bool(lab)))
            rec["y"].append(r["y"])
            rec["s"].append(r["s_psig"])
            rec["c"].append(r["c_semantique"])
            rec["g_dyn"].append(r["g_dyn"])
            rec["g_topo"].append(r["g_topo"])
            rec["m_s"].append(r["m_s"])
            rec["m_topo"].append(r["m_topo"])
            rec["y_prod_ref"].append(r["g_dyn"] * r["g_topo"]
                                     * (r["s_psig"] * r["c_semantique"]))
            rec["instrument"] = r["instrument"]
            labels.append(lab)
    labels = np.array(labels)
    out = {"N_par_classe": N, "longueur": L, "motif": MOTIF,
           "operateur": "hierarchique y=g_dyn*m_s*m_topo (exp06)",
           "instrument": rec.pop("instrument", "n/a"),
           "critere_scelle": "AUC(y) >= 0.70", "resultats": {}}
    print(f"{'mesure':8s} {'moy_c0':>8s} {'moy_c1':>8s} {'AUC':>7s}")
    for k, v in rec.items():
        v = np.array(v)
        a = auc(v, labels)
        out["resultats"][k] = {"moy_classe0": round(float(v[labels == 0].mean()), 4),
                               "moy_classe1": round(float(v[labels == 1].mean()), 4),
                               "AUC": round(a, 4)}
        print(f"{k:8s} {v[labels==0].mean():8.4f} {v[labels==1].mean():8.4f} {a:7.4f}")
    verdict = ("DETECTION — le neurone trouve l'ordre cache"
               if out["resultats"]["y"]["AUC"] >= 0.70
               else "pas de detection au seuil scelle — documente")
    out["verdict"] = verdict
    print(f"\n-> {verdict}")
    with open("experiences/exp05_tache_motif.json", "w") as f:
        json.dump(out, f, indent=2)
    print("OK -> experiences/exp05_tache_motif.json")


if __name__ == "__main__":
    main()
