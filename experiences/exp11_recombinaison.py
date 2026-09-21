"""Experience 11 — RECOMBINAISON : reparer l'echec de population d'exp08.

CONTEXTE. exp08 : la population NE BAT PAS le neurone seul (0.9744 vs
0.9848). Diagnostic pose alors : les variants etaient des MUTANTS FORCES de
la MEME entree. Forcer eloigne chaque variant de sa cible, donc chacun lit
moins bien l'origine, et la fusion moyenne aussi le signal. Le probleme
n'etait pas la population — c'etait l'OPERATION appliquee aux variants.

HYPOTHESE (direction 3) : remplacer la mutation forcee par une VRAIE
operation genomique — recombinaison homologue entre chaines DISTINCTES.
Si la structure portee est ce sur quoi les parents s'accordent, alors la
recombinaison la PRESERVE, alors que la mutation la detruit.

MONTAGE. Toutes les chaines font L=600 bases. La structure porteuse est un
bloc de 90 bases a une position FIXE (234..324) ; le fond est une chaine
ORDONNEE (periode 3).

  v1 (graines 11000+, ECHEC) : j'avais mis le DESORDRE dans le fond et
  l'ORDRE dans le bloc, c'est-a-dire la direction D- de l'instrument — celle
  qui est INFIRMEE (exp10, U2). Le controle R5 l'a attrape : bloc intact ->
  fitness 0.313 au lieu de >= 0.80. La mesure ne lisait rien. v1 jetee.

  v2 (graines 12000+, ci-dessous) : on utilise la direction qui MARCHE
  (D+ : desordre plante dans l'ordre). Le bloc est un DESORDRE plante dans
  un fond ordonne — le thermometre le lit a AUC ~0.95 (exp10, U1).

  Deux parents distincts portent le fond ordonne SOUS DEUX PHASES
  DIFFERENTES (ATG vs TGC) : tous deux sont « ordonnes », mais ils
  different partout. C'est exactement le cas ou une recombinaison doit
  PRESERVER la qualite (l'ordre) en CREANT de la diversite.

Trois operations :
  A MUTATION FORCEE      : un parent, chaque base tiree avec prob mu.
                           La mutation ne regarde pas OU elle frappe : elle
                           detruit le fond ordonne ET le bloc.
  B RECOMBINAISON HOMOLOGUE : deux parents qui portent le bloc AU MEME
                           ENDROIT. Un crossover a un point de coupe tire.
                           Le fond reste ordonne (les deux phases le sont),
                           le bloc est preserve.
  C CONTROLE NON HOMOLOGUE : deux parents ou le bloc est a des positions
                           DIFFERENTES. Le crossover coupe presque toujours
                           a travers l'un des deux blocs.

MESURE. Fitness = capacite du neurone a LIRE le bloc plante : separation AUC
(direction D+ de l'instrument, validée exp10/U1). Une chaine dont le fond a
perdu son ordre, ou dont le bloc a disparu, tombe vers AUC 0.5.

CRITERES SCELLES AVANT EXECUTION :
  R1 : la recombinaison homologue PRESERVE la fitness mieux que la mutation
       forcee — moyenne(recomb) - moyenne(mutation) >= +0.10 d'AUC.
  R2 : le CONTROLE non homologue ne preserve PAS — sa fitness est nettement
       sous la recombinaison homologue (< -0.10 d'ecart).
  R3 : ce n'est pas un artefact de « ne rien changer ». On mesure la
       DIVERSITE des descendants (fraction de positions differentes du
       parent 1). La recombinaison doit produire une diversite >= 0.30 tout
       en preservant.
  R4 : sur 3 GENERATIONS avec selection, la lignee recombinaison garde une
       fitness que la lignee mutation ne garde pas.
  R5 : CONTROLE de la mesure — sur une chaine portant le bloc intact dans un
       fond ordonne, la fitness doit valoir >= 0.80.

GRAINES NEUVES : 12000+ (utilisees : 0-1199, 12-16, 21, 1000+, 2000+, 3000+,
4000+, 5000+, 8000+, 9000-9039, 11000-11600). N = 20 populations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.enroulement import signal
from organes.psig import takens_embed, p_sig_ripser

L = 600
POS0, POS1 = 234, 324          # bloc porteur, position FIXE
FENETRE = 48
MU = 0.05
N_POP = 20
GENERATIONS = 3
GRAINE_BASE = 12000
PHASES = ("ATG", "TGC", "CAT")   # trois lectures ordonnees, phase differente


def bloc_aleatoire(n: int, rng) -> list[str]:
    return list(rng.choice(list("ATCG"), size=n))


def fond_ordonne(n: int, phase: str) -> list[str]:
    return list((phase * (n // 3 + 1))[:n])


def _auc(a, b):
    if a.size == 0 or b.size == 0:
        return 0.5
    tot = np.concatenate([a, b])
    r = np.argsort(np.argsort(tot)) + 1.0
    return float((r[:a.size].sum() - a.size * (a.size + 1) / 2.0) / (a.size * b.size))


def profil(X, w=FENETRE):
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


def fitness(seq: str) -> float:
    """Separation AUC : le bloc desordonne se lit-il contre le fond ordonne ?"""
    X = signal(seq)
    if X.size < 80:
        return 0.5
    y = profil(X)
    if y.size == 0:
        return 0.5
    centres = np.arange(y.size) + FENETRE // 2
    dedans = (centres >= POS0 / 3.0) & (centres < POS1 / 3.0)
    if dedans.sum() < 3 or (~dedans).sum() < 3:
        return 0.5
    return _auc(y[dedans], y[~dedans])


def parent(rng, phase: str, pos=(POS0, POS1), bloc=None) -> str:
    """Fond ordonne (phase donnee) + bloc desordonne a `pos`."""
    s = fond_ordonne(L, phase)
    s[pos[0]:pos[1]] = bloc if bloc is not None else bloc_aleatoire(pos[1] - pos[0], rng)
    return "".join(s)


def muter(seq: str, rng, mu=MU) -> str:
    s = list(seq)
    for i in range(len(s)):
        if rng.random() < mu:
            s[i] = str(rng.choice(list("ATCG")))
    return "".join(s)


def recombiner(a: str, b: str, rng) -> str:
    bp = int(rng.integers(1, len(a)))
    return a[:bp] + b[bp:]


def diversite(a: str, b: str) -> float:
    n = min(len(a), len(b))
    return sum(x != y for x, y in zip(a[:n], b[:n])) / n


def main():
    print("=== exp11 : recombinaison homologue vs mutation forcee (v2) ===")
    print(f"graines NEUVES {GRAINE_BASE}+ | N={N_POP} populations | "
          f"bloc [{POS0},{POS1}] | mu={MU}\n")

    r0 = np.random.default_rng(GRAINE_BASE - 1)
    temoin = parent(r0, PHASES[0])
    fit_temoin = fitness(temoin)
    print(f"CONTROLE R5 — bloc desordonne dans fond ordonne : fitness = "
          f"{fit_temoin:.3f} (attendu >= 0.80)\n")

    fit = {"mutation": [], "recomb_hom": [], "recomb_nonhom": [],
           "identite": [], "parent1": []}
    div = {"mutation": [], "recomb_hom": [], "recomb_nonhom": [], "identite": []}

    for k in range(N_POP):
        rng = np.random.default_rng(GRAINE_BASE + k)
        ph1, ph2 = PHASES[k % 3], PHASES[(k + 1) % 3]
        p1 = parent(rng, ph1)                 # bloc desordre @ POS0..POS1
        p2_hom = parent(rng, ph2)             # bloc desordre @ MEME position, phase differente
        p2_nonhom = parent(rng, ph2, pos=(60, 150))   # bloc AILLEURS

        fit["parent1"].append(fitness(p1))

        mutants = [muter(p1, rng) for _ in range(6)]
        fit["mutation"].append(float(np.mean([fitness(m) for m in mutants])))
        div["mutation"].append(float(np.mean([diversite(p1, m) for m in mutants])))

        recs = [recombiner(p1, p2_hom, rng) for _ in range(6)]
        fit["recomb_hom"].append(float(np.mean([fitness(r) for r in recs])))
        div["recomb_hom"].append(float(np.mean([diversite(p1, r) for r in recs])))

        recn = [recombiner(p1, p2_nonhom, rng) for _ in range(6)]
        fit["recomb_nonhom"].append(float(np.mean([fitness(r) for r in recn])))
        div["recomb_nonhom"].append(float(np.mean([diversite(p1, r) for r in recn])))

        fit["identite"].append(fitness(p1))
        div["identite"].append(0.0)

    print(f"{'operation':16s} {'fitness moy':>12s} {'diversite moy':>14s}")
    print("-" * 46)
    for op in ("parent1", "identite", "recomb_hom", "recomb_nonhom", "mutation"):
        d = np.mean(div[op]) if op in div else float("nan")
        print(f"{op:16s} {np.mean(fit[op]):>12.4f} {d:>14.4f}")

    print("\n--- R4 : 3 generations, selection = moitie haute ---")
    gen_m, gen_r = [], []
    vals_m, vals_r = [], []
    for k in range(N_POP):
        rng = np.random.default_rng(GRAINE_BASE + 500 + k)
        pop_m = [parent(rng, PHASES[j % 3]) for j in range(8)]
        pop_r = [parent(rng, PHASES[j % 3]) for j in range(8)]
        vals_m.append(float(np.mean([fitness(s) for s in pop_m])))
        vals_r.append(float(np.mean([fitness(s) for s in pop_r])))
        for g in range(GENERATIONS):
            fm = np.array([fitness(s) for s in pop_m])
            fr = np.array([fitness(s) for s in pop_r])
            keep_m = [pop_m[i] for i in np.argsort(fm)[-4:]]
            keep_r = [pop_r[i] for i in np.argsort(fr)[-4:]]
            pop_m = [muter(keep_m[int(rng.integers(0, 4))], rng) for _ in range(8)]
            pop_r = [recombiner(keep_r[int(rng.integers(0, 4))],
                                keep_r[int(rng.integers(0, 4))], rng)
                     for _ in range(8)]
            vals_m.append(float(np.mean([fitness(s) for s in pop_m])))
            vals_r.append(float(np.mean([fitness(s) for s in pop_r])))
    gen_m = [float(np.mean(vals_m[i::GENERATIONS + 1])) for i in range(GENERATIONS + 1)]
    gen_r = [float(np.mean(vals_r[i::GENERATIONS + 1])) for i in range(GENERATIONS + 1)]
    print(f"  mutation  : {' '.join(f'{v:.3f}' for v in gen_m)}")
    print(f"  recombine : {' '.join(f'{v:.3f}' for v in gen_r)}")

    print("\n--- VERDICTS vs CRITERES SCELLES ---")
    fm_, fr_, fn_ = (np.mean(fit["mutation"]), np.mean(fit["recomb_hom"]),
                     np.mean(fit["recomb_nonhom"]))
    r1 = (fr_ - fm_) >= 0.10
    r2 = (fr_ - fn_) >= 0.10
    r3 = np.mean(div["recomb_hom"]) >= 0.30
    r4 = gen_r[-1] > gen_m[-1] + 0.10
    r5 = fit_temoin >= 0.80
    print(f"R5 controle mesure (bloc intact {fit_temoin:.3f}) : "
          f"{'OK' if r5 else 'MESURE CASSEE'}")
    print(f"R1 recombinaison preserve vs mutation ({fr_ - fm_:+.3f}) : "
          f"{'CONFIRME' if r1 else 'INFIRME'}")
    print(f"R2 le controle non homologue ne preserve pas "
          f"({fr_ - fn_:+.3f}) : {'CONFIRME' if r2 else 'INFIRME'}")
    print(f"R3 diversite preservee ({np.mean(div['recomb_hom']):.3f}) : "
          f"{'CONFIRME' if r3 else 'INFIRME'}")
    print(f"R4 3 generations ({gen_r[-1]:.3f} vs {gen_m[-1]:.3f}) : "
          f"{'CONFIRME' if r4 else 'INFIRME'}")

    out = {"graine_base": GRAINE_BASE, "N_pop": N_POP, "mu": MU,
           "bloc": [POS0, POS1], "fenetre": FENETRE,
           "version": "v2 (direction D+ ; la v1 sur graines 11000+ a ete jetee par R5)",
           "criteres_scelles": {
               "R1": "recomb_hom - mutation >= +0.10 AUC",
               "R2": "recomb_hom - recomb_nonhom >= +0.10 AUC",
               "R3": "diversite recomb >= 0.30",
               "R4": "gen3 recomb > gen3 mutation + 0.10",
               "R5": "controle: bloc intact dans fond ordonne >= 0.80"},
           "fitness_moyenne": {k: round(float(np.mean(v)), 4) for k, v in fit.items()},
           "diversite_moyenne": {k: round(float(np.mean(v)), 4) for k, v in div.items()},
           "generations": {"mutation": [round(v, 4) for v in gen_m],
                           "recomb_hom": [round(v, 4) for v in gen_r]},
           "controle_mesure": round(fit_temoin, 4),
           "verdicts": {"R1": bool(r1), "R2": bool(r2), "R3": bool(r3),
                        "R4": bool(r4), "R5_controle": bool(r5)}}
    Path("experiences/exp11_recombinaison.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\nOK -> experiences/exp11_recombinaison.json")


if __name__ == "__main__":
    main()