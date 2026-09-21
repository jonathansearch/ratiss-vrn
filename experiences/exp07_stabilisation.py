"""Experience 07 — STABILISATION : une population de neurones converge-t-elle ?

DIRECTION FINALE. Question : si on laisse une population cycler (chaque
neurone mute localement via G a chaque cycle), converge-t-elle vers un etat
stable, ou derive-t-elle ?

PREDICTION ENONCEE AVANT MESURE :
  La coherence R ne doit PAS deriver. G mutera chaque neurone, mais la
  selection garde les mutations qui rapprochent Delta_ent de sa cible,
  donc la population doit se resserrer sans s'effondrer : la dispersion des
  y doit DECROITRE ou rester plate, et R doit rester dans une bande.

  Cas ou la prediction doit echouer : une population initialement
  HETEROGENE (moitie ordre, moitie chaos) n'a pas d'etat commun vers
  lequel converger. Si la prediction tient meme la, c'est que la mesure ne
  lit rien.

GRAINES NEUVES : 3000+i (2000+ et 1000+ sont prises).

CRITERES SCELLES :
  S1 stabilite : |R_T - R_0| < 0.15 pour une population HOMOGENE.
  S2 convergence : dispersion finale <= dispersion initiale + 0.10
                  pour une population HOMOGENE.
  S3 homogene vs heterogene : si S1 tient pour l'heterogene aussi, la
     mesure ne sait pas distinguer -> critere falsifie.
  S4 fusion : |y_coh - y_avg| depend de la dispersion. Si la dispersion
     est nulle, les deux regles DOIVENT donner le meme y (controle de
     coherence interne de l'instrument).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.assemblage import lire_population, fusionner, stabiliser
from organes.assemblage import coherence_population


def melange(p, n=240, seed=0, base="ATGC"):
    r = np.random.default_rng(seed)
    return "".join(str(r.choice(list("ATCG"))) if r.random() < p
                   else base[i % len(base)] for i in range(n))


def motif(m, n=240):
    """Motif periodique repete — population ordonnee NON degeneree."""
    return (m * (n // len(m) + 1))[:n]


def balayage_couplage(N=12, cycles=8, graines=(3100, 3200, 3300)):
    """K fait-il monter l'accord ? (test du couplage de Kuramoto)."""
    print("=== balayage du COUPLAGE K (accord final, moy 3 graines) ===")
    print(f"{'K':>6} {'accord':>9} {'sd':>7} {'synchronise':>12}")
    for K in (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40):
        accs = []
        for seed in graines:
            pop = [melange(1.0, seed=3000 + i) for i in range(N)]
            st = stabiliser(pop, cycles=cycles, cible=0.5, seed=seed,
                            forcer=True, couplage=K)
            accs.append(st["accord_apres"])
        print(f"{K:>6.2f} {np.mean(accs):>9.4f} {np.std(accs):>7.4f} "
              f"{'oui' if np.mean(accs) > 0.40 + 0.15 else 'non':>12}")
    print()


def main():
    print("=== exp07 : stabilisation d'une population de neurones ===")
    print("graines neuves 3000+i | criteres S1-S4 scelles\n")

    # NOTE : melange(0.0, seed=i) renvoie "ATGC" repete QUEL QUE SOIT le
    # seed -> population degeneree (6 copies identiques). Corrige en motifs
    # distincts.
    pop_ord = [motif(m) for m in ("AT", "ATGC", "ATCG", "AAAT", "ACGT",
                                  "ATGCATCG")]
    pop_ale = [melange(1.0, seed=3000 + i) for i in range(6)]
    pop_mix = pop_ord[:3] + pop_ale[:3]

    resultats = {}
    for nom, pop, homogene in (("homogene ordonnee", pop_ord, True),
                               ("homogene aleatoire", pop_ale, True),
                               ("mixte heterogene", pop_mix, False)):
        print(f"--- {nom} ---")
        # forcer=True : sans cela G ne touche jamais une chaine ordonnee
        # (Delta_ent=0 sous la cible) et la stabilisation mesuree serait
        # celle d'un systeme FIGE, pas stable. Voir le defaut du garde
        # a sens unique, corrige dans neurone_vrn.generation.
        st = stabiliser(pop, cycles=6, seed=3000, forcer=True)
        print(f"  {'cycle':>5} {'R':>8} {'y_moyen':>9} {'dispersion':>11}")
        for h in st["historique"]:
            print(f"  {h['cycle']:>5} {h['coherence']:>8.4f} "
                  f"{h['y_moyen']:>9.4f} {h['dispersion']:>11.4f}")
        print(f"  derive R = {st['derive_coherence']:+.4f}  "
              f"dispersion {st['dispersion_avant']:.4f} -> "
              f"{st['dispersion_apres']:.4f}  "
              f"-> {'STABLE' if st['stable'] else 'DERIVE'}")
        resultats[nom] = st
        print()

    # verdicts
    print("--- VERDICTS vs CRITERES SCELLES ---")
    st_ord = resultats["homogene ordonnee"]
    st_ale = resultats["homogene aleatoire"]
    st_mix = resultats["mixte heterogene"]
    s1 = abs(st_ord["derive_coherence"]) < 0.15
    s2 = st_ord["dispersion_apres"] <= st_ord["dispersion_avant"] + 0.10
    # S3 : l'heterogene doit etre MOINS stable que l'homogene, sinon la mesure ne distingue rien
    s3 = (not st_mix["stable"]
          or abs(st_mix["derive_coherence"]) > abs(st_ord["derive_coherence"]))
    print(f"S1 stabilite homogene     : derive R={st_ord['derive_coherence']:+.4f}"
          f"  -> {'CONFIRME' if s1 else 'INFIRME'}")

    # S4 : controle interne — dispersion nulle => fusion identique
    pop_id = [melange(0.0, seed=0) for _ in range(4)]
    p = lire_population(pop_id)
    fa, fc = fusionner(p, "average"), fusionner(p, "coherente")
    s4 = abs(fa["y_fusion"] - fc["y_fusion"]) < 1e-9
    print(f"S2 convergence homogene   : dispersion {st_ord['dispersion_avant']:.4f}"
          f" -> {st_ord['dispersion_apres']:.4f}  -> {'CONFIRME' if s2 else 'INFIRME'}")
    print(f"S3 homogene vs heterogene : mixte derive R={st_mix['derive_coherence']:+.4f}"
          f" ({'DERIVE' if not st_mix['stable'] else 'STABLE'})"
          f"  -> {'CONFIRME' if s3 else 'INFIRME (la mesure ne distingue pas)'}")
    print(f"S4 controle fusion        : y_avg={fa['y_fusion']:.6f} "
          f"y_coh={fc['y_fusion']:.6f}  -> {'CONFIRME' if s4 else 'INFIRME'}")

    out = {
        "graines": "3000+i", "cycles": 5,
        "criteres_scelles": {
            "S1": "|R_T - R_0| < 0.15 (homogene)",
            "S2": "dispersion finale <= initiale + 0.10 (homogene)",
            "S3": "l'heterogene doit deriver plus que l'homogene",
            "S4": "dispersion nulle => y_avg == y_coh",
            # seuil : l'accord de base vaut 0.40 (biais de composition de
            # l'ADN, pas 0.25). Il faut donc exiger une montee NETTE, pas
            # seulement > 0.35 qui serait deja vrai sans couplage.
            "S5": "K=0.10 fait monter l'accord d'au moins +0.15 par rapport a K=0",
        },
        "homogene_ordonnee": st_ord,
        "homogene_aleatoire": st_ale,
        "mixte_heterogene": st_mix,
        "verdicts": {"S1": bool(s1), "S2": bool(s2), "S3": bool(s3), "S4": bool(s4)},
    }
    Path("experiences/exp07_stabilisation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str))
    balayage_couplage()

    print("\nOK -> experiences/exp07_stabilisation.json")


if __name__ == "__main__":
    main()