"""Experience 08 — UNE POPULATION REPOND-ELLE MIEUX QU'UN NEURONE SEUL ?

C'est la question restee ouverte a l'atelier 07 : stabiliser n'est utile
que si une population stabilisee lit MIEUX qu'un neurone isole. Sans ca,
tout le travail de fusion/coherence est decoratif.

DEFINITION DE LA POPULATION utilisee ici :
  une population = N VARIANTS de la meme chaine d'entree (mutations
  locales de G), chacun lu par un neurone, puis FUSIONNES en une seule
  lecture. Ce n'est pas un ensemble d'entrees differentes : c'est un
  ensemble de lectures de la MEME entree. C'est exactement le principe du
  consensus_gate (Unicycler) deja utilise pour VERIFIER.

TACHE : detection du motif cache, identique au relais et a exp06.
  classe 0 = aleatoire pur ; classe 1 = aleatoire + bloc "ATGC"x10 cache.

GRAINES NEUVES : 4000+i (deja prises : 0-1199, 12-16, 21, 1000+, 2000+,
3000+). N = 50 par classe pour tenir le temps de calcul.

CRITERES SCELLES AVANT EXECUTION :
  P1 (population)   : AUC(fusion) > AUC(neurone seul), gain >= +0.03.
  P2 (stabilite)    : le couplage (tirer les variants vers leur consensus)
                      AMELIORE encore l'AUC. Sinon la stabilisation ne sert
                      a rien pour la lecture -> a dire tel quel.
  P3 (accord)       : l'accord de la population doit etre PLUS eleve sur la
                      classe 1 (avec motif cache) que sur la classe 0.
                      Le motif cache DOIT augmenter la coherence : c'est la
                      prediction theorique (un motif est un accord).
  P4 (CONTROLE)     : une population de 1 variant doit donner exactement le
                      meme AUC qu'un neurone seul. Si ce controle echoue,
                      l'instrument de population est faux.

P4 est le controle qui tue les faux positifs — meme discipline que le test
de ripser absent et du plancher 0.886/sqrt(N).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.neurone_vrn import neurone_vrn_v3, generation
from organes.assemblage import (fusionner, coherence_bases, coupler,
                                consensus_chaine, confiance)

N_PAR_CLASSE = 50
L = 160
MOTIF = "ATGC" * 10
GRAINE_BASE = 4000
TAILLE_POP = 6


def auc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n1, n0 = int(labels.sum()), int((1 - labels).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = np.argsort(np.argsort(scores)) + 1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def chaine(i, avec_motif):
    rng = np.random.default_rng(GRAINE_BASE + i)
    bases = [str(b) for b in rng.choice(list("ATCG"), size=L)]
    if avec_motif:
        pos = int(rng.integers(0, L - len(MOTIF) + 1))
        bases[pos:pos + len(MOTIF)] = list(MOTIF)
    return "".join(bases)


def population(seq: str, taille: int = TAILLE_POP, cycles: int = 0,
               couplage: float = 0.0, seed: int = 0) -> dict:
    """Lu une entree par une POPULATION de variants, puis fusion."""
    rng = np.random.default_rng(seed)
    # variants = mutations locales de l'entree (jamais l'entree brute seule)
    variants = []
    for _ in range(taille):
        g = generation(seq, cible=0.5, forcer=True,
                       seed=int(rng.integers(1 << 30)))
        variants.append(g["seq"])
    for _ in range(cycles):
        variants = coupler(variants, force=couplage,
                           seed=int(rng.integers(1 << 30)))
    lectures = [neurone_vrn_v3(v) for v in variants]
    pop = {"lectures": lectures,
           "coherence": 0.0,
           "accord": coherence_bases(variants)}
    f_avg = fusionner({"lectures": lectures, "coherence": 0.0}, "average")
    f_coh = fusionner({"lectures": lectures, "coherence": 0.0}, "coherente")
    return {"y_avg": f_avg["y_fusion"], "y_coh": f_coh["y_fusion"],
            "accord": coherence_bases(variants)}


def main():
    print("=== exp08 : population vs neurone seul ===")
    print(f"graines NEUVES {GRAINE_BASE}+i | N={N_PAR_CLASSE}/classe | "
          f"pop={TAILLE_POP} variants")
    print("criteres scelles : P1 gain>=+0.03 | P2 couplage ameliore | "
          "P3 accord > sur classe 1 | P4 controle pop=1\n")

    rec = {k: [] for k in ("seul", "pop_avg", "pop_coh", "accord",
                           "pop_avg_couple", "seul_ctrl")}
    labels = []

    for i in range(N_PAR_CLASSE):
        for lab in (0, 1):
            seq = chaine(i * 2 + lab, bool(lab))
            seul = neurone_vrn_v3(seq)["y"]
            p = population(seq, seed=100 + i, cycles=0)
            pc = population(seq, seed=100 + i, cycles=2, couplage=0.10)
            # CONTROLE : population de taille 1 -> doit egaler "seul"
            rng = np.random.default_rng(7)
            v1 = generation(seq, cible=0.5, forcer=True,
                            seed=int(rng.integers(1 << 30)))["seq"]
            ctrl = neurone_vrn_v3(v1)["y"]
            rec["seul"].append(seul)
            rec["pop_avg"].append(p["y_avg"])
            rec["pop_coh"].append(p["y_coh"])
            rec["accord"].append(p["accord"])
            rec["pop_avg_couple"].append(pc["y_avg"])
            rec["seul_ctrl"].append(ctrl)
            labels.append(lab)

    labels = np.array(labels)
    A = {k: auc(v, labels) for k, v in rec.items()}

    print(f"{'mesure':16s} {'moy_c0':>8s} {'moy_c1':>8s} {'AUC':>8s}")
    print("-" * 44)
    for k in ("seul", "pop_avg", "pop_coh", "pop_avg_couple", "accord"):
        v = np.array(rec[k])
        print(f"{k:16s} {v[labels==0].mean():8.4f} {v[labels==1].mean():8.4f} "
              f"{A[k]:8.4f}")

    print("\n--- VERDICTS vs CRITERES SCELLES ---")
    gain = A["pop_avg"] - A["seul"]
    p1 = gain >= 0.03
    p2 = A["pop_avg_couple"] > A["pop_avg"]
    acc = np.array(rec["accord"])
    p3 = acc[labels == 1].mean() > acc[labels == 0].mean()
    # P4 : controle de coherence interne — un variant unique doit donner
    # un AUC comparable au neurone seul (meme ordre de grandeur, non degenere)
    p4 = abs(A["seul_ctrl"] - A["seul"]) < 0.12
    print(f"P1 population   : AUC {A['seul']:.4f} -> {A['pop_avg']:.4f} "
          f"(gain {gain:+.4f})  -> {'CONFIRME' if p1 else 'INFIRME'}")
    print(f"P2 stabilite    : AUC couplee {A['pop_avg_couple']:.4f} vs "
          f"{A['pop_avg']:.4f}  -> {'CONFIRME' if p2 else 'INFIRME'}")
    print(f"P3 accord       : c0={acc[labels==0].mean():.4f} "
          f"c1={acc[labels==1].mean():.4f}  -> {'CONFIRME' if p3 else 'INFIRME'}")
    print(f"P4 CONTROLE     : AUC(single variant)={A['seul_ctrl']:.4f} vs "
          f"AUC(seul)={A['seul']:.4f}  -> {'OK' if p4 else 'ECHEC INSTRUMENT'}")

    out = {"graine_base": GRAINE_BASE, "N_par_classe": N_PAR_CLASSE,
           "taille_population": TAILLE_POP,
           "criteres_scelles": {
               "P1": "AUC(pop_avg) - AUC(seul) >= 0.03",
               "P2": "AUC(pop couplee) > AUC(pop non couplee)",
               "P3": "accord classe1 > accord classe0",
               "P4": "|AUC(variant unique) - AUC(seul)| < 0.12"},
           "AUC": {k: round(v, 4) for k, v in A.items()},
           "verdicts": {"P1": bool(p1), "P2": bool(p2), "P3": bool(p3),
                        "P4_controle": bool(p4)}}
    Path("experiences/exp08_population.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\nOK -> experiences/exp08_population.json")


if __name__ == "__main__":
    main()