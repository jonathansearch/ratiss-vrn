"""Experience 13 — LA RECOMBINAISON AJOUTE-T-ELLE QUELQUE CHOSE ?

Suite de la direction 3 (la piste vivante). exp11 montrait que la
recombinaison homologue RESISTE la ou la mutation s'effondre — mais les
parents partaient deja quasi optimaux (0.92), donc peu de marge pour monter.

Ici : partir d'une population DEGRADEE, et voir si la recombinaison RECUPERE.

--------------------------------------------------------------------------
v1 — CONTROLE QUI CHANGE LA QUESTION.
--------------------------------------------------------------------------
Le premier montage traitait « noop » (clones d'elites, aucune operation)
comme un controle de non-derive, avec un seuil |delta| <= 0.03. Il a echoue :
noop montait de +0.046.

Ce n'etait PAS une derive de mesure. C'est la SELECTION ELLE-MEME : garder
la moitie haute puis cloner fait monter la moyenne mecaniquement, sans
aucune operation. Mon critere etait mal concu — il condamnait un effet reel
en le prenant pour un artefact.

Consequence plus grave : noop n'est pas un controle nul, c'est LE TEMOIN
PRINCIPAL. Dans le smoke test, noop (0.945) BATTait la recombinaison
(0.922).

v2 (ci-dessous) corrige ca : noop devient la BASELINE, et le critere central
n'est plus « la recombinaison monte » (la selection suffit a faire monter)
mais « la recombinaison fait MIEUX QUE LA SELECTION SEULE ».

--------------------------------------------------------------------------
MONTAGE v2. Les trois bras reels partagent EXACTEMENT le meme etat de depart
(mu0 = 0.05, pres de la frontiere mu* = 0.063 mesuree en exp12). Quatre bras,
meme selection (moitie haute + elitisme) :

  noop          : clones d'elites         -> BASELINE : valeur de la
                                            selection seule
  mutation      : elite + mutation mu     -> variation aleatoire
  recomb_hom    : deux elites, crossover  -> l'hypothese
  recomb_nonhom : blocs a des positions differentes (crossover a travers)

THEORIE (falsifiable). Les degats sont disperses : chez un parent ici, chez
l'autre la. Un crossover peut ASSEMBLER deux segments non abimes -> enfant
plus sain que ses parents. Prediction : recomb_hom > noop. Si au contraire
noop >= recomb_hom, la recombinaison ne fait que du bruit et la montee vient
de la selection.

CRITERES SCELLES AVANT EXECUTION :
  T0 (mesure) : parent INTACT >= 0.80, et etat initial degrade < intact-0.10.
  T1 (baseline) : noop final > noop initial. Sanity : la selection doit
       faire monter, sinon rien n'est interpretable.
  T2 (LE CRITERE CENTRAL) : recomb_hom final - noop final >= +0.05. La
       recombinaison doit battre la SELECTION SEULE.
  T3 : recomb_hom final - mutation final >= +0.10.
  T4 (specificite) : recomb_hom final - nonhom final >= +0.15.
  T5 (dynamique) : recomb_hom >= noop sur >= 60% des generations.
  T6 (CONTROLE) : mutation final <= noop final + 0.05. Si la mutation fait
       aussi bien que la selection seule, « battre la mutation » ne dit rien.

GRAINES NEUVES : 15000+ (utilisees : ... 14000-14020, 99000 (smoke test)).
N = 10 populations par bras, pop = 8, 6 generations.

PERFORMANCE : ~0.9 s par evaluation, parallellise sur 4 processus ; identite
serie/parallele VERIFIEE avant usage. Duree attendue ~8 min.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.enroulement import signal
from organes.mesure import auc
from organes.psig import takens_embed, p_sig_ripser

L = 600
POS0, POS1 = 234, 324
POSE0, POSE1 = 60, 150          # bloc du controle non homologue
FENETRE = 48
MU = 0.05                       # taux de mutation applique pendant l'evolution
MU_INIT = 0.12                  # degradation INITIALE : au-dela de mu*=0.063 (exp12)
                                # v2a : mu_init=0.05 donnait degrade=0.882 vs intact
                                # 0.967 -> ecart 0.085 < 0.10, T0 a stoppe. Corrige.
GENERATIONS = 6
TAILLE_POP = 8
N_POP = 10
GRAINE_BASE = 15000
PHASES = ("ATG", "TGC", "CAT", "GTA")
N_PROC = 4
BRAS = ("noop", "mutation", "recomb_hom", "recomb_nonhom")


def _profil(X, w=FENETRE):
    out = []
    for i in range(0, len(X) - w):
        seg = X[i:i + w]
        if seg.std() < 1e-12:
            out.append(0.0)
            continue
        c = np.unique(np.round(takens_embed((seg - seg.mean()) / seg.std(), 3, 3), 10),
                      axis=0)
        out.append(p_sig_ripser(c)["p_sig"] if len(c) >= 6 else 0.0)
    return np.array(out)


@lru_cache(maxsize=16384)
def fitness(seq: str) -> float:
    """Separation AUC entre la region du bloc (234-324) et le reste."""
    X = signal(seq)
    if X.size < 80:
        return 0.5
    y = _profil(X)
    if y.size == 0:
        return 0.5
    centres = np.arange(y.size) + FENETRE // 2
    dedans = (centres >= POS0 / 3.0) & (centres < POS1 / 3.0)
    if dedans.sum() < 3 or (~dedans).sum() < 3:
        return 0.5
    return auc(y[dedans], y[~dedans])


def _init_pop(rng, pos=(POS0, POS1)) -> list[str]:
    """Population de depart DEGRADEE : fond ordonne + bloc desordre + mu0."""
    pop = []
    for j in range(TAILLE_POP):
        ph = PHASES[j % len(PHASES)]
        s = list((ph * (L // 3 + 1))[:L])
        s[pos[0]:pos[1]] = list(rng.choice(list("ATCG"), size=pos[1] - pos[0]))
        for i in range(len(s)):
            if rng.random() < MU_INIT:
                s[i] = str(rng.choice(list("ATCG")))
        pop.append("".join(s))
    return pop


def muter(seq: str, rng, mu=MU) -> str:
    s = list(seq)
    for i in range(len(s)):
        if rng.random() < mu:
            s[i] = str(rng.choice(list("ATCG")))
    return "".join(s)


def recombiner(a: str, b: str, rng) -> str:
    bp = int(rng.integers(1, len(a)))
    return a[:bp] + b[bp:]


def main():
    print("=== exp13 v2 : la recombinaison bat-elle la SELECTION SEULE ? ===")
    print(f"graines NEUVES {GRAINE_BASE}+ | N={N_POP} | pop={TAILLE_POP} | "
          f"{GENERATIONS} generations | mu_init={MU_INIT} mu_evol={MU}\n")

    # --- T0 : controles de mesure -----------------------------------------
    r0 = np.random.default_rng(GRAINE_BASE - 10)
    s0 = list((PHASES[0] * (L // 3 + 1))[:L])
    s0[POS0:POS1] = list(r0.choice(list("ATCG"), size=POS1 - POS0))
    fit_intact = fitness("".join(s0))
    fit_degrade = float(np.mean([fitness(s) for s in _init_pop(
        np.random.default_rng(GRAINE_BASE - 11))]))
    print(f"T0 — parent INTACT : {fit_intact:.3f} (attendu >= 0.80)")
    print(f"T0 — etat DEGRADE : {fit_degrade:.3f} (attendu < intact - 0.10)")
    if fit_intact < 0.80 or fit_degrade >= fit_intact - 0.10:
        print("\nSTOP : l'instrument ne lit pas la structure. Experience annulee.")
        return

    # --- les 3 bras reels partagent EXACTEMENT le meme depart --------------
    populations = {}
    for k in range(N_POP):
        commun = _init_pop(np.random.default_rng(GRAINE_BASE + k))
        populations[(k, "noop")] = [(s, None) for s in commun]
        populations[(k, "mutation")] = [(s, None) for s in commun]
        populations[(k, "recomb_hom")] = [(s, None) for s in commun]
        populations[(k, "recomb_nonhom")] = [
            (s, None) for s in _init_pop(np.random.default_rng(GRAINE_BASE + k),
                                         pos=(POSE0, POSE1))]

    rngs = {k: np.random.default_rng(GRAINE_BASE + 5000 + k) for k in range(N_POP)}
    courbe = {b: [] for b in BRAS}

    for g in range(GENERATIONS + 1):
        a_evaluer, cibles = [], []
        for (k, b), pop in populations.items():
            for i, (sq, f) in enumerate(pop):
                if f is None:
                    a_evaluer.append(sq)
                    cibles.append((k, b, i))
        if a_evaluer:
            with Pool(N_PROC) as pool:
                valeurs = pool.map(fitness, a_evaluer)
            for (k, b, i), v in zip(cibles, valeurs):
                populations[(k, b)][i] = (populations[(k, b)][i][0], v)

        for b in BRAS:
            courbe[b].append(float(np.mean([populations[(k, b)][i][1]
                                            for k in range(N_POP)
                                            for i in range(TAILLE_POP)])))
        etiq = "depart      " if g == 0 else f"generation {g}"
        print(f"  {etiq:12s} : " + " ".join(f"{b}={courbe[b][-1]:.3f}" for b in BRAS))
        if g == GENERATIONS:
            break

        for (k, b), pop in list(populations.items()):
            rng = rngs[k]
            elites = [sq for sq, _ in sorted(pop, key=lambda t: -t[1])][:TAILLE_POP // 2]
            enfants = [(elites[0], None)]          # elitisme
            while len(enfants) < TAILLE_POP:
                if b == "noop":
                    enfants.append((elites[int(rng.integers(0, len(elites)))], None))
                elif b == "mutation":
                    p = elites[int(rng.integers(0, len(elites)))]
                    enfants.append((muter(p, rng), None))
                else:
                    a = elites[int(rng.integers(0, len(elites)))]
                    c = elites[int(rng.integers(0, len(elites)))]
                    enfants.append((recombiner(a, c, rng), None))
            populations[(k, b)] = enfants

    init = {b: courbe[b][0] for b in BRAS}
    fin = {b: courbe[b][-1] for b in BRAS}

    t0 = fit_intact >= 0.80 and fit_degrade < fit_intact - 0.10
    t1 = fin["noop"] > init["noop"]
    t2 = (fin["recomb_hom"] - fin["noop"]) >= 0.05
    t3 = (fin["recomb_hom"] - fin["mutation"]) >= 0.10
    t4 = (fin["recomb_hom"] - fin["recomb_nonhom"]) >= 0.15
    avance = sum(1 for g in range(GENERATIONS + 1)
                 if courbe["recomb_hom"][g] >= courbe["noop"][g])
    t5 = avance >= int(0.6 * (GENERATIONS + 1))
    t6 = fin["mutation"] <= fin["noop"] + 0.05

    print("\n" + f"{'bras':14s} " + " ".join(f"g{g}" for g in range(GENERATIONS + 1)))
    print("-" * 66)
    for b in BRAS:
        print(f"{b:14s} " + " ".join(f"{v:.3f}" for v in courbe[b]))

    print("\n--- VERDICTS vs CRITERES SCELLES ---")
    print(f"T0 controles de mesure : {'OK' if t0 else 'MESURE CASSEE'}")
    print(f"T1 la selection seule fait monter noop ({fin['noop'] - init['noop']:+.3f}) : "
          f"{'CONFIRME' if t1 else 'INFIRME'}")
    print(f"T2 CRITERE CENTRAL : recomb_hom bat la SELECTION SEULE "
          f"({fin['recomb_hom'] - fin['noop']:+.3f}) : "
          f"{'CONFIRME' if t2 else 'INFIRME'}")
    print(f"T3 bat la mutation ({fin['recomb_hom'] - fin['mutation']:+.3f}) : "
          f"{'CONFIRME' if t3 else 'INFIRME'}")
    print(f"T4 specificite homologie ({fin['recomb_hom'] - fin['recomb_nonhom']:+.3f}) : "
          f"{'CONFIRME' if t4 else 'INFIRME'}")
    print(f"T5 avance sur >=60% des generations ({avance}/{GENERATIONS + 1}) : "
          f"{'CONFIRME' if t5 else 'INFIRME'}")
    print(f"T6 CONTROLE : mutation <= noop (+/-0.05) "
          f"({fin['mutation'] - fin['noop']:+.3f}) : "
          f"{'OK' if t6 else 'LA MUTATION VAUT LA SELECTION SEULE'}")

    out = {"version": "v2 (noop = baseline de selection ; v1 jetee, cf. docstring)",
           "graine_base": GRAINE_BASE, "N_pop": N_POP, "taille_pop": TAILLE_POP,
           "generations": GENERATIONS, "mu_init": MU_INIT, "mu_evol": MU, "fenetre": FENETRE,
           "criteres_scelles": {
               "T0": "intact >= 0.80 et degrade < intact - 0.10",
               "T1": "noop fin > noop init",
               "T2": "recomb_hom fin - noop fin >= +0.05",
               "T3": "recomb_hom fin - mutation fin >= +0.10",
               "T4": "recomb_hom fin - nonhom fin >= +0.15",
               "T5": "recomb_hom >= noop sur >= 60% des generations",
               "T6": "mutation fin <= noop fin + 0.05"},
           "controles_mesure": {"intact": round(fit_intact, 4),
                                "degrade_mu0": round(fit_degrade, 4)},
           "courbes": {b: [round(v, 4) for v in courbe[b]] for b in BRAS},
           "verdicts": {"T0_controle": bool(t0), "T1": bool(t1), "T2_central": bool(t2),
                        "T3": bool(t3), "T4": bool(t4), "T5": bool(t5),
                        "T6_controle": bool(t6)}}
    Path("experiences/exp13_reparation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\nOK -> experiences/exp13_reparation.json")


if __name__ == "__main__":
    main()