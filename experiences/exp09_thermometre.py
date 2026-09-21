"""Experience 09 — LE THERMOMETRE VRN : localiser les zones de desordre.

CHANGEMENT DE CAP. Jusqu'ici on demandait au neurone de CLASSER (motif
cache) ou de PREDIRE. Ici on l'utilise comme INSTRUMENT DE MESURE : un
thermometre de sante informationnelle le long d'une chaine. On ne lui
demande pas « quelle est la classe ? » mais « OU la structure change-t-elle ?».

TACHE (verite-terrain connue) :
  On plante une zone de structure differente dans une chaine.
    D+  zone DESORDONNEE plantee dans un fond ORDONNE
    D-  zone ORDONNEE plantee dans un fond DESORDONNE
  On balaie la chaine par une fenetre glissante, on lit le P_sig local,
  et on demande au detecteur de LOCALISER la zone plantee.

  Prediction : en D+, la zone plantee doit avoir un P_sig local PLUS ELEVE
  (le desordre cree des cycles topologiques) ; en D-, PLUS BAS.

GRAINES NEUVES : 5000+ (prises : 0-1199, 12-16, 21, 1000+, 2000+, 3000+,
4000+, sondes 9001-9002). N = 30 chaines par condition.

CRITERES SCELLES AVANT EXECUTION :
  T1 (localisation D+) : argmax(P_sig local) tombe dans la zone plantee
                         pour >= 70% des chaines.
  T2 (localisation D-) : argmin(P_sig local) tombe dans la zone plantee
                         pour >= 70% des chaines.
  T3 (robustesse)      : T1 et T2 tiennent pour LES TROIS fenetres
                         w = 36, 48, 64, avec moins de 20 points d'ecart.
                         Un detecteur qui ne marche qu'a une fenetre est
                         ACCORDE, pas robuste.
  T4 (CONTROLE ANTI-HALLUCINATION) : sur une chaine SANS zone plantee, le
                         taux de « succes » mesure contre une zone tiree au
                         hasard doit rester proche du niveau de chance
                         (couverture), soit =< chance + 15 points.
                         Si le detecteur « trouve » des zones la ou il n'y
                         en a pas, il hallucine et tout le reste tombe.
  T5 (comparaison)     : P_sig local doit battre un temoin naif (variance
                         locale du signal). Sinon, dire que le temoin
                         suffit.

Le seuil de chance = fraction du signal couverte par la zone plantee. Il est
calcule, pas suppose.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.enroulement import signal
from organes.psig import takens_embed, p_sig_ripser

N_PAR_CONDITION = 30
L = 600                      # bases
TAILLE_BLOC = 90             # bases plantees
FENETRES = (36, 48, 64)
GRAINE_BASE = 5000
MOTIF_ORDRE = "ATG"


def chaine(D_plus: bool, seed: int) -> tuple[str, tuple[int, int]]:
    """Renvoie (chaine, (debut, fin) de la zone plantee EN BASES)."""
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
    return "".join(s), (pos, pos + TAILLE_BLOC)


def profil(X: np.ndarray, w: int, mesure: str) -> np.ndarray:
    """Profil local le long du signal. mesure = 'psig' ou 'variance'."""
    out = []
    for i in range(0, len(X) - w):
        seg = X[i:i + w]
        if seg.std() < 1e-12:
            out.append(0.0)
            continue
        if mesure == "variance":
            out.append(float(seg.std()))
            continue
        segn = (seg - seg.mean()) / seg.std()
        c = np.unique(np.round(takens_embed(segn, 3, 3), 10), axis=0)
        out.append(p_sig_ripser(c)["p_sig"] if len(c) >= 6 else 0.0)
    return np.array(out)


def zone_aa(zone_bases: tuple[int, int], len_signal: int) -> tuple[float, float]:
    """Convertit la zone (bases) en indices de signal (aa), echelle 1/3."""
    d, f = zone_bases
    return d / 3.0, f / 3.0


def localise(y: np.ndarray, zone, len_signal: int, sens: str) -> bool:
    """Le pic (D+) ou le creux (D-) tombe-t-il dans la zone ?"""
    if y.size == 0:
        return False
    i = int(np.argmax(y)) if sens == "max" else int(np.argmin(y))
    d, f = zone
    return d <= i <= f


def main():
    print("=== exp09 : thermometre VRN — localiser le desordre structural ===")
    print(f"graines NEUVES {GRAINE_BASE}+ | N={N_PAR_CONDITION}/condition | "
          f"L={L} bases | bloc={TAILLE_BLOC} bases")
    print("criteres scelles : T1/T2 >=70% | T3 robuste aux 3 fenetres | "
          "T4 controle == chance | T5 bat la variance\n")

    # couverture moyenne = niveau de chance
    couv = []
    res = {("D+", "psig"): {}, ("D-", "psig"): {},
           ("D+", "variance"): {}, ("D-", "variance"): {}}

    for k in range(N_PAR_CONDITION):
        for D_plus, nom in ((True, "D+"), (False, "D-")):
            seq, zone_b = chaine(D_plus, GRAINE_BASE + k * 2 + int(D_plus))
            X = signal(seq)
            if X.size < 80:
                continue
            z = zone_aa(zone_b, X.size)
            couv.append((z[1] - z[0]) / X.size)
            sens = "max" if D_plus else "min"
            for w in FENETRES:
                for mesure in ("psig", "variance"):
                    y = profil(X, w, mesure)
                    hit = localise(y, z, X.size, sens)
                    res[(nom, mesure)].setdefault(w, []).append(hit)

    chance = float(np.mean(couv))
    print(f"niveau de chance (couverture de la zone) = {chance:.3f}\n")

    print(f"{'condition':10s} {'mesure':9s} " +
          " ".join(f"w={w:<6}" for w in FENETRES))
    print("-" * 48)
    taux = {}
    for (nom, mesure), d in res.items():
        ligne = []
        for w in FENETRES:
            t = float(np.mean(d.get(w, [0]))) if d.get(w) else 0.0
            taux[(nom, mesure, w)] = t
            ligne.append(f"{t:6.3f} ")
        print(f"{nom:10s} {mesure:9s} " + " ".join(ligne))

    print("\n--- VERDICTS vs CRITERES SCELLES ---")
    t1 = all(taux[("D+", "psig", w)] >= 0.70 for w in FENETRES)
    t2 = all(taux[("D-", "psig", w)] >= 0.70 for w in FENETRES)
    ecart = max(abs(taux[("D+", "psig", w)] - taux[("D-", "psig", w2)])
                for w in FENETRES for w2 in FENETRES)
    t3 = ecart <= 0.20
    p_ok = all(taux[("D+", "psig", w)] >= 0.70 and taux[("D-", "psig", w)] >= 0.70
               for w in FENETRES)
    # T4 : sur chaine SANS zone, le detecteur ne doit pas trouver mieux que le hasard
    faux = []
    for k in range(N_PAR_CONDITION):
        r = np.random.default_rng(7000 + k)
        seq = "".join(r.choice(list("ATCG"), size=L))
        X = signal(seq)
        if X.size < 80:
            continue
        # zone FICTIVE de meme taille relative
        f0 = 0.3 * X.size
        f1 = f0 + (TAILLE_BLOC / 3.0)
        w = 48
        y = profil(X, w, "psig")
        faux.append(localise(y, (f0, f1), X.size, "max"))
    t_hall = float(np.mean(faux)) if faux else 1.0
    t4 = t_hall <= chance + 0.15
    best_psig = max(taux[("D+", "psig", w)] for w in FENETRES)
    best_var = max(max(taux[("D+", "variance", w)], taux[("D-", "variance", w)])
                   for w in FENETRES)
    t5 = best_psig > best_var
    print(f"T1 localisation D+ (max dans la zone) : {t1} -> "
          f"{'CONFIRME' if t1 else 'INFIRME'}")
    print(f"T2 localisation D- (min dans la zone) : {t2} -> "
          f"{'CONFIRME' if t2 else 'INFIRME'}")
    print(f"T3 robustesse fenetres (ecart {ecart:.3f}) : {t3} -> "
          f"{'CONFIRME' if t3 else 'INFIRME'}")
    print(f"T4 CONTROLE anti-hallucination : taux sur chaine SANS zone = "
          f"{t_hall:.3f} vs chance {chance:.3f} -> "
          f"{'OK' if t4 else 'HALLUCINE'}")
    print(f"T5 battre le temoin naif : P_sig {best_psig:.3f} vs "
          f"variance {best_var:.3f} -> {'CONFIRME' if t5 else 'INFIRME'}")

    out = {"graine_base": GRAINE_BASE, "N_par_condition": N_PAR_CONDITION,
           "fenetres": list(FENETRES), "chance": round(chance, 4),
           "criteres_scelles": {
               "T1": "argmax P_sig dans la zone (D+) >= 70%",
               "T2": "argmin P_sig dans la zone (D-) >= 70%",
               "T3": "ecart entre fenetres <= 20 points",
               "T4": "sur chaine sans zone, taux <= chance+15 points",
               "T5": "P_sig bat la variance locale comme temoin"},
           "taux": {f"{k[0]}|{k[1]}|w{k[2]}": round(v, 4)
                    for k, v in taux.items()},
           "controle_hallucination": round(t_hall, 4),
           "verdicts": {"T1": bool(t1), "T2": bool(t2), "T3": bool(t3),
                        "T4_controle": bool(t4), "T5": bool(t5)}}
    Path("experiences/exp09_thermometre.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\nOK -> experiences/exp09_thermometre.json")


if __name__ == "__main__":
    main()