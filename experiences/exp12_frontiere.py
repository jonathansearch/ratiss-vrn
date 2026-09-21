"""Experience 12 — GENERATEUR DE CONTRAINTES : la frontiere de validite.

DIRECTION 2. « La zone ou la performance se degrade definit une frontiere
de validite. Ce que le reseau rejette est aussi informatif que ce qu'il
accepte. »

On transforme ca en mesure : jusqu'a QUELLE intensite de degradation le
neurone lit-il encore la structure ? Le seuil de rejet est LU sur la courbe,
pas choisi arbitrairement.

MONTAGE. Chaine = fond ordonne + bloc desordre plante (direction D+ de
l'instrument, AUC ~0.95, exp10/U1). On degrade la chaine par mutation
aleatoire a taux mu = 0.00 a 0.30, et on mesure la fitness (separation AUC)
a chaque taux. La courbe fitness(mu) EST le generateur de contraintes.
Frontiere mu* = le taux ou la fitness passe sous 0.80.

CRITERES SCELLES AVANT EXECUTION :
  V1 (monotonie) : la fitness decroit avec mu — non croissante a 80% des
       points (la degradation doit degrader).
  V2 (frontiere nette) : la courbe passe SOUS 0.80 — il existe un mu* ou le
       neurone cesse de lire. Si la fitness tient partout, il n'y a pas de
       frontiere a genere.
  V3 (reproductible) : mu* a un ecart-type <= 0.03 sur les graines. Une
       frontiere qui bouge d'une graine a l'autre n'est pas un seuil, c'est
       du bruit.
  V4 (CONTROLE) : sur une chaine SANS bloc plante (fond ordonne seul), la
       fitness(mu) reste proche de 0.5 et ne descend pas sous 0.35 — sinon la
       « frontiere » mesure juste la disparition du signal en general, pas la
       perte de la structure.
  V5 (non trivial) : mu* n'est ni 0 (rejet de tout) ni >= 0.30 (rejet de
       rien).

GRAINES NEUVES : 13000+ (utilisees : 0-1199, 12-16, 21, 1000+, 2000+, 3000+,
4000+, 5000+, 8000+, 9000-9039, 11000-11600, 12000-12000+500). N = 15.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.enroulement import signal
from organes.psig import takens_embed, p_sig_ripser
from organes.mesure import auc as _auc_ok

L = 600
POS0, POS1 = 234, 324
FENETRE = 48
TAUX = (0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.30)
SEUIL_REJET = 0.80
GRAINE_BASE = 13000
N = 15
PHASES = ("ATG", "TGC")


def _auc(a, b):
    """Delègue a organes.mesure.auc : les ex aequo valent 0.5, pas 0.

    La version locale precedente (rangs entiers) renvoyait 0.0 sur un signal
    constant, au lieu de 0.5 « aucune information ». Le controle V4 d'exp12
    l'a attrape : fitness « sans bloc » = 0.000 la ou 0.5 est attendu.
    """
    return _auc_ok(a, b)


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


def construire(rng, avec_bloc: bool, mu: float) -> str:
    ph = PHASES[int(rng.integers(0, len(PHASES)))]
    s = list((ph * (L // 3 + 1))[:L])
    if avec_bloc:
        s[POS0:POS1] = list(rng.choice(list("ATCG"), size=POS1 - POS0))
    if mu > 0:
        for i in range(len(s)):
            if rng.random() < mu:
                s[i] = str(rng.choice(list("ATCG")))
    return "".join(s)


def main():
    print("=== exp12 : generateur de contraintes — la frontiere de validite ===")
    print(f"graines NEUVES {GRAINE_BASE}+ | N={N} | seuil de rejet = {SEUIL_REJET}\n")

    courbe, courbe_ctrl = {}, {}
    for mu in TAUX:
        f, fc = [], []
        for k in range(N):
            rng = np.random.default_rng(GRAINE_BASE + k)
            f.append(fitness(construire(rng, True, mu)))
            rng2 = np.random.default_rng(GRAINE_BASE + 700 + k)
            fc.append(fitness(construire(rng2, False, mu)))
        courbe[mu] = float(np.mean(f))
        courbe_ctrl[mu] = float(np.mean(fc))

    print(f"{'mu':>6s} {'fitness (bloc)':>15s} {'fitness (sans bloc)':>20s}")
    print("-" * 44)
    for mu in TAUX:
        print(f"{mu:>6.2f} {courbe[mu]:>15.4f} {courbe_ctrl[mu]:>20.4f}")

    # frontiere mu* = interpolation lineaire du croisement avec SEUIL_REJET
    def franchissement(vals):
        for i in range(len(TAUX) - 1):
            a, b = vals[i], vals[i + 1]
            if a >= SEUIL_REJET > b:
                t = (a - SEUIL_REJET) / (a - b + 1e-12)
                return TAUX[i] + t * (TAUX[i + 1] - TAUX[i])
        return float("nan")

    mus = []
    for k in range(8):
        vals = []
        for mu in TAUX:
            rng = np.random.default_rng(GRAINE_BASE + 2000 + k)
            vals.append(fitness(construire(rng, True, mu)))
        m = franchissement(vals)
        if np.isfinite(m):
            mus.append(m)
    mu_star = float(np.mean(mus)) if mus else float("nan")
    mu_sd = float(np.std(mus)) if mus else float("nan")

    decroissante = sum(1 for i in range(len(TAUX) - 1)
                       if courbe[TAUX[i + 1]] <= courbe[TAUX[i]] + 0.02)
    v1 = decroissante >= int(0.8 * (len(TAUX) - 1))
    v2 = min(courbe.values()) < SEUIL_REJET
    v3 = np.isfinite(mu_sd) and mu_sd <= 0.03
    v4 = (max(courbe_ctrl.values()) <= 0.65) and (min(courbe_ctrl.values()) >= 0.35)
    v5 = np.isfinite(mu_star) and 0.0 < mu_star < 0.30

    print("\n--- VERDICTS vs CRITERES SCELLES ---")
    print(f"V1 degradation monotone ({decroissante}/{len(TAUX)-1} segments) : "
          f"{'CONFIRME' if v1 else 'INFIRME'}")
    print(f"V2 frontiere existe (min {min(courbe.values()):.3f} < "
          f"{SEUIL_REJET}) : {'CONFIRME' if v2 else 'INFIRME'}")
    print(f"V3 reproductible (mu* = {mu_star:.3f} +/- {mu_sd:.3f}) : "
          f"{'CONFIRME' if v3 else 'INFIRME'}")
    print(f"V4 CONTROLE sans bloc (plage "
          f"[{min(courbe_ctrl.values()):.3f},{max(courbe_ctrl.values()):.3f}]) : "
          f"{'OK' if v4 else 'LA FRONTIERE EST UN ARTEFACT'}")
    print(f"V5 frontiere non triviale : {'CONFIRME' if v5 else 'INFIRME'}")

    out = {"graine_base": GRAINE_BASE, "N": N, "seuil_rejet": SEUIL_REJET,
           "criteres_scelles": {
               "V1": "fitness non croissante sur >=80% des segments",
               "V2": "min(fitness) < 0.80",
               "V3": "ecart-type de mu* <= 0.03",
               "V4": "sans bloc : fitness dans [0.35,0.65] partout",
               "V5": "0 < mu* < 0.30"},
           "courbe_fitness": {f"mu={mu:.2f}": round(v, 4) for mu, v in courbe.items()},
           "courbe_sans_bloc": {f"mu={mu:.2f}": round(v, 4) for mu, v in courbe_ctrl.items()},
           "frontiere_mu_star": round(mu_star, 4),
           "frontiere_mu_sd": round(mu_sd, 4),
           "verdicts": {"V1": bool(v1), "V2": bool(v2), "V3": bool(v3),
                        "V4_controle": bool(v4), "V5": bool(v5)}}
    Path("experiences/exp12_frontiere.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\nOK -> experiences/exp12_frontiere.json")


if __name__ == "__main__":
    main()