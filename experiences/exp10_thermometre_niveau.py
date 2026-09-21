"""Experience 10 — LE THERMOMETRE, CORRIGE : le niveau, pas le pic.

POURQUOI CETTE EXPERIENCE EXISTE.

exp09 testait le thermometre avec un critere de LOCALISATION PAR PIC :
« l'argmax du P_sig local doit tomber dans la zone plantee ». Ce critere a
ete INFIRME (33% contre 70% exige, D- meme SOUS la chance). Ce verdict est
consigne et ne bouge pas.

MAIS en diagnostiquant l'echec (graines 5000+), une structure DIFFERENTE et
plus forte est apparue : la zone plantee ne produit pas forcement LE pic,
mais elle produit un DEPLACEMENT DE NIVEAU. Mesure sur 4 chaines :
    D+  dedans 0.211 / 0.362 / 0.206   dehors 0.090 / 0.056 / 0.045
    D-  dedans 0.195                    dehors 0.326

Ce constat est EXPLORATOIRE : il a ete trouve sur les graines 5000+. Le
tester sur les memes graines serait du bricolage. On le scelle donc et on le
teste sur des GRAINES NEUVES 8000+.

REFORMULATION (le thermometre lit un NIVEAU, pas un sommet) :
  Le P_sig local est-il en moyenne plus eleve DANS la zone plantee que
  DEHORS ? Mesure : AUC entre les fenetres dedans et les fenetres dehors.
  AUC = 0.5 : aucune separation. AUC = 1.0 : separation parfaite.

GRAINES NEUVES : 8000+. Utilisees : 0-1199, 12-16, 21, 1000+, 2000+, 3000+,
4000+, 5000+, 9001-9002. N = 40 chaines par condition.

CRITERES SCELLES AVANT EXECUTION :
  U1 : AUC(dedans > dehors) >= 0.75 pour D+, a chaque fenetre.
  U2 : AUC(dedans < dehors) >= 0.75 pour D-, a chaque fenetre.
  U3 : U1 et U2 tiennent aux TROIS fenetres w = 36, 48, 64 (robustesse).
  U4 : CONTROLE — sur chaine SANS zone plantee, l'AUC contre une zone
       fictive doit rester dans [0.35, 0.65]. Sinon le thermometre lit du
       bruit structurel et tout s'effondre.
  U5 : le P_sig local bat le temoin naif (variance locale) : AUC(P_sig) >
       AUC(variance) pour D+.
  U6 : l'effet TIENT sur des chaines plus LONGUES (L=900) — un thermometre
       qui ne marche qu'a une longueur donnee n'est pas un thermometre.

Note : U6 est un test de generalisation a une condition NON vue pendant
l'exploration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.enroulement import signal
from organes.psig import takens_embed, p_sig_ripser

N = 40
TAILLE_BLOC = 90
FENETRES = (36, 48, 64)
GRAINE_BASE = 8000
GRAINE_CONTROLE = 9000
MOTIF_ORDRE = "ATG"


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(une valeur de a depasse une valeur de b). Statistique de rang."""
    if a.size == 0 or b.size == 0:
        return 0.5
    tot = np.concatenate([a, b])
    rangs = np.argsort(np.argsort(tot)) + 1.0
    r_a = rangs[:a.size].sum()
    return float((r_a - a.size * (a.size + 1) / 2.0) / (a.size * b.size))


def chaine(D_plus: bool, seed: int, L: int = 600) -> tuple[str, int]:
    r = np.random.default_rng(seed)
    pos = int(r.integers(60, L - TAILLE_BLOC - 60))
    if D_plus:
        fond = list((MOTIF_ORDRE * (L // 3 + 1))[:L])
        bloc = list(r.choice(list("ATCG"), size=TAILLE_BLOC))
    else:
        fond = list(r.choice(list("ATCG"), size=L))
        bloc = list((MOTIF_ORDRE * (TAILLE_BLOC // 3 + 1))[:TAILLE_BLOC])
    s = fond[:]
    s[pos:pos + TAILLE_BLOC] = bloc
    return "".join(s), pos


def profil(X: np.ndarray, w: int, mesure: str) -> np.ndarray:
    out = []
    for i in range(0, len(X) - w):
        seg = X[i:i + w]
        if seg.std() < 1e-12:
            out.append(0.0)
            continue
        if mesure == "variance":
            out.append(float(seg.std()))
            continue
        c = np.unique(np.round(takens_embed((seg - seg.mean()) / seg.std(), 3, 3), 10),
                      axis=0)
        out.append(p_sig_ripser(c)["p_sig"] if len(c) >= 6 else 0.0)
    return np.array(out)


def separer(X: np.ndarray, w: int, a: float, b: float, mesure: str) -> float:
    """AUC entre les fenetres centrees DANS [a,b] et celles DEHORS."""
    y = profil(X, w, mesure)
    if y.size == 0:
        return 0.5
    centres = np.arange(y.size) + w // 2
    dedans = (centres >= a) & (centres < b)
    if dedans.sum() < 3 or (~dedans).sum() < 3:
        return 0.5
    return _auc(y[dedans], y[~dedans])


def main():
    print("=== exp10 : thermometre corrige — lire le NIVEAU, pas le pic ===")
    print(f"graines NEUVES {GRAINE_BASE}+ | N={N}/condition | scelle sur l'exploration 5000+\n")

    res = {}
    for L in (600, 900):
        for D_plus, nom in ((True, "D+"), (False, "D-")):
            for w in FENETRES:
                for mesure in ("psig", "variance"):
                    aucs = []
                    for k in range(N):
                        seq, pos = chaine(D_plus, GRAINE_BASE + k * 2 + int(D_plus), L)
                        X = signal(seq)
                        if X.size < 80:
                            continue
                        a, b = pos / 3.0, (pos + TAILLE_BLOC) / 3.0
                        aucs.append(separer(X, w, a, b, mesure))
                    res[(L, nom, w, mesure)] = float(np.mean(aucs))

    print(f"{'L':5s} {'cond':5s} {'mesure':9s} " + " ".join(f"w={w:<6}" for w in FENETRES))
    print("-" * 50)
    for L in (600, 900):
        for nom in ("D+", "D-"):
            for mesure in ("psig", "variance"):
                print(f"{L:<5d} {nom:5s} {mesure:9s} " +
                      " ".join(f"{res[(L, nom, w, mesure)]:6.3f} " for w in FENETRES))

    # controle sans zone
    ctrl = {w: [] for w in FENETRES}
    for k in range(N):
        r = np.random.default_rng(GRAINE_CONTROLE + k)
        seq = "".join(r.choice(list("ATCG"), size=600))
        X = signal(seq)
        if X.size < 80:
            continue
        f0 = 0.3 * X.size
        f1 = f0 + TAILLE_BLOC / 3.0
        for w in FENETRES:
            ctrl[w].append(separer(X, w, f0, f1, "psig"))
    ctrl_moy = {w: float(np.mean(v)) for w, v in ctrl.items()}

    print("\n--- VERDICTS vs CRITERES SCELLES ---")
    u1 = all(res[(600, "D+", w, "psig")] >= 0.75 for w in FENETRES)
    u2 = all((1 - res[(600, "D-", w, "psig")]) >= 0.75 for w in FENETRES)
    u3 = u1 and u2
    u4 = all(0.35 <= ctrl_moy[w] <= 0.65 for w in FENETRES)
    u5 = all(res[(600, "D+", w, "psig")] > res[(600, "D+", w, "variance")]
             for w in FENETRES)
    u6 = all(res[(900, "D+", w, "psig")] >= 0.70 for w in FENETRES)

    print(f"U1 separation D+ (baseline, 600) : "
          f"{[round(res[(600,'D+',w,'psig')],3) for w in FENETRES]} -> "
          f"{'CONFIRME' if u1 else 'INFIRME'}")
    print(f"U2 separation D- (inversee)      : "
          f"{[round(1-res[(600,'D-',w,'psig')],3) for w in FENETRES]} -> "
          f"{'CONFIRME' if u2 else 'INFIRME'}")
    print(f"U3 robustesse 3 fenetres         : {'CONFIRME' if u3 else 'INFIRME'}")
    print(f"U4 CONTROLE sans zone (attendu 0.5) : "
          f"{[round(ctrl_moy[w],3) for w in FENETRES]} -> "
          f"{'OK' if u4 else 'LIT DU BRUIT'}")
    print(f"U5 bat la variance comme temoin  : "
          f"{'CONFIRME' if u5 else 'INFIRME'}")
    print(f"U6 generalise a L=900            : "
          f"{[round(res[(900,'D+',w,'psig')],3) for w in FENETRES]} -> "
          f"{'CONFIRME' if u6 else 'INFIRME'}")

    out = {"graine_base": GRAINE_BASE, "N": N, "fenetres": list(FENETRES),
           "criteres_scelles": {
               "U1": "AUC(dedans>dehors) >= 0.75 en D+ (L=600)",
               "U2": "AUC(dedans<dehors) >= 0.75 en D- (L=600)",
               "U3": "U1 et U2 aux 3 fenetres",
               "U4": "controle sans zone dans [0.35,0.65]",
               "U5": "P_sig bat la variance",
               "U6": "tient a L=900 (condition non vue)"},
           "auc": {f"L{L}|{n}|w{w}|{m}": round(v, 4)
                   for (L, n, w, m), v in res.items()},
           "controle_sans_zone": {f"w{w}": round(v, 4) for w, v in ctrl_moy.items()},
           "verdicts": {"U1": bool(u1), "U2": bool(u2), "U3": bool(u3),
                        "U4_controle": bool(u4), "U5": bool(u5), "U6": bool(u6)}}
    Path("experiences/exp10_thermometre_niveau.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\nOK -> experiences/exp10_thermometre_niveau.json")


if __name__ == "__main__":
    main()