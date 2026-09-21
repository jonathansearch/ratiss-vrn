"""Assemblage — fusion et stabilisation d'une population de neurones VRN.

DIRECTION FINALE. Jusqu'ici on lisait UN neurone. Ici on lit une
POPULATION, et on pose les deux questions finales :

  1. FUSION — comment agreger N lectures en une seule ?
  2. STABILISATION — la population converge-t-elle quand on la laisse
     cycler, ou derive-t-elle ?

FONDEMENT (suite directe de l'enroulement) :
  Un neurone est un oscillateur (sa trajectoire de Takens tourne, cf.
  enroulement.py). N neurones = N oscillateurs couples. La theorie de
  Kuramoto dit qu'ils se synchronisent au-dela d'un couplage critique, et
  le PARAMETRE D'ORDRE R = |moyenne(e^{i.theta})| mesure cette coherence :
  R=0 phases independantes, R=1 parfaitement verrouillees.

  C'est exactement la generalisation du PLV a deux neurones (exp05) a N.

DEUX REGLES DE FUSION, testees l'une contre l'autre :

  F-average : moyenne simple des y. Suppose que chaque neurone est un
              temoin aussi fiable qu'un autre.
  F-coherent: ponderation par la CONFIANCE. Un neurone est confiant quand
              son y est tranche (loin de l'incertitude 0.5, ici : soit
              g_dyn franc). Les neurones confiants pesent plus.

LA MESURE DE STABILITE :
  On fait cycler la population : a chaque cycle chaque neurone subit une
  mutation LOCALE dirigee (G, organes/neurone_vrn.py::generation). On
  mesure, cycle apres cycle, la coherence R et l'ecart-type de y entre
  neurones. STABLE = R ne derive pas et l'ecart-type se reduit. INSTABLE =
  R s'effondre ou l'ecart-type explose.

STATUT : exploration. Aucun de ces choix n'est fige (manifeste VRN).
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.neurone_vrn import neurone_vrn_v3, generation
from organes.enroulement import construire_script, _phase, plv
from organes.psig import takens_embed
from organes.atcg import hydrophobic_signal


def _trajectoire(seq: str) -> np.ndarray:
    """Trajectoire de Takens du neurone, normalisee."""
    x = hydrophobic_signal(seq)
    if x.size < 6:
        return np.zeros((0, 3))
    x = (x - x.mean()) / (x.std() + 1e-12)
    return takens_embed(x, dim=3, delay=3)


def coherence_population(trajectoires: list[np.ndarray]) -> float:
    """Parametre d'ordre R : coherence de phase de la population.

    Generalise le PLV a 2 (exp05) a N neurones. R = |moyenne(e^{i.theta})|,
    moyenne sur le temps puis sur les neurones, en prenant soin d'aligner
    les longueurs (troncature a la plus courte).
    """
    if len(trajectoires) < 2:
        return 0.0
    tailles = [len(t) for t in trajectoires if len(t) >= 4]
    if len(tailles) < 2:
        return 0.0
    n = min(tailles)
    phases = []
    for t in trajectoires:
        if len(t) < 4:
            continue
        phases.append(_phase(t)[:n])
    if len(phases) < 2:
        return 0.0
    P = np.vstack(phases)                       # (n_neurones, n_temps)
    z = np.exp(1j * P)
    R_par_temps = np.abs(z.mean(axis=0))        # coherence a chaque instant
    return float(np.mean(R_par_temps))


def matrice_plv(trajectoires: list[np.ndarray]) -> np.ndarray:
    """PLV deux a deux — la structure fine du verrouillage."""
    n = len(trajectoires)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            p = plv(trajectoires[i], trajectoires[j])
            M[i, j] = M[j, i] = p
    return M


def confiance(lecture: dict) -> float:
    """Confiante = g_dyn franc (loin de 0.5).

    Un neurone dont g_dyn vaut exactement 0.5 ne sait pas s'il voit une
    dynamique. Un neurone a g_dyn 1.0 ou 0.0 sait. On mesure la distance
    au doute : 2|g_dyn - 0.5| dans [0,1].
    """
    return float(2.0 * abs(lecture["g_dyn"] - 0.5))


def lire_population(sequences: list[str]) -> dict:
    """Lit une population : lectures, trajectoires, coherence, PLV."""
    lectures = [neurone_vrn_v3(s) for s in sequences]
    trajs = [_trajectoire(s) for s in sequences]
    return {
        "lectures": lectures,
        "trajectoires": trajs,
        "coherence": coherence_population(trajs),
        "plv": matrice_plv(trajs),
    }


def fusionner(pop: dict, regle: str = "coherente") -> dict:
    """Agrege les y d'une population en un seul y."""
    lec = pop["lectures"]
    ys = np.array([r["y"] for r in lec], float)
    if regle == "average":
        w = np.ones_like(ys)
    elif regle == "coherente":
        w = np.array([confiance(r) for r in lec], float)
        if w.sum() < 1e-9:
            w = np.ones_like(ys)
    else:
        raise ValueError(f"regle inconnue : {regle}")
    w = w / w.sum()
    return {
        "y_fusion": float(np.sum(w * ys)),
        "poids": w.tolist(),
        "coherence": pop["coherence"],
        "regle": regle,
        "n": len(ys),
        "dispersion": float(np.std(ys)),
    }


def coherence_bases(sequences: list[str]) -> float:
    """Coherence d'ACCORD : accord moyen des chaines au consensus.

    POURQUOI CE N'EST PAS `coherence_population`. Diagnostic exp07 :

        forcer 0%  des positions au consensus -> R = 0.368
        forcer 50% des positions au consensus -> R = 0.388
        forcer 90% des positions au consensus -> R = 0.315
        forcer 100% (tous identiques)         -> R = 1.000

    R (phase de Takens) NE MONTE PAS quand les chaines se rapprochent : il
    ne lit pas la similitude des chaines. La phase est invariante aux
    mutations locales, donc le couplage (qui agit sur les bases) et R (qui
    lit la phase) mesuraient deux choses differentes. D'ou le resultat nul
    de `coupler` : ratio 0.96-1.04 sur le plancher, quelle que soit la
    force K.

    Ici on mesure ce que le couplage fait vraiment : la fraction de
    positions ou chaque chaine s'accorde avec le consensus, moyennee.
    1.0 = population parfaitement unanime, ~0.25 = desaccord total (ADN).
    L'accord par pur hasard vaut 1/4 (+ biais de composition).
    """
    if len(sequences) < 2:
        return 0.0
    cons = consensus_chaine(sequences)
    n = min(len(cons), min(len(s) for s in sequences))
    if n == 0:
        return 0.0
    accords = [sum(a == b for a, b in zip(s[:n], cons[:n])) / n
               for s in sequences]
    return float(np.mean(accords))


def consensus_chaine(sequences: list[str]) -> str:
    """Consensus base par base (vote majoritaire) — repris de Unicycler.

    La "voix" de la population, au sens d'un assemblage consensus : a
    chaque position, la base la plus representee. C'est le point vers
    lequel un neurone couple sera tire.
    """
    if not sequences:
        return ""
    L = min(len(s) for s in sequences)
    out = []
    for i in range(L):
        col = [s[i] for s in sequences]
        vals, counts = np.unique(col, return_counts=True)
        out.append(str(vals[int(np.argmax(counts))]))
    return "".join(out)


def coupler(sequences: list[str], force: float = 0.05,
            seed: int = 0) -> list[str]:
    """COUPLAGE de Kuramoto, en version ADN.

    Sans ce terme, la population mesuree en exp07 restait EXACTEMENT au
    plancher des phases independantes, R = sqrt(pi)/(2 sqrt(N)) = 0.886/sqrt(N)
    (verifie : N=32 -> mes 0.159 vs theorie 0.157). Autrement dit les
    neurones ne se parlaient pas : le "R stable" d'exp07 etait une
    trivialite, pas une synchronisation.

    Ici chaque chaine est tiree vers le consensus de la population, d'une
    fraction `force` de ses positions. `force=0` -> pas de couplage
    (controle) ; `force>0` -> les neurones doivent se rapprocher et R doit
    PASSER AU-DESSUS du plancher.

    C'est l'analogue discret du terme de Kuramoto : dtheta_i/dt =
    omega_i + (K/N) sum_j sin(theta_j - theta_i).
    """
    if force <= 0 or len(sequences) < 2:
        return list(sequences)
    rng = np.random.default_rng(seed)
    cons = consensus_chaine(sequences)
    out = []
    for s in sequences:
        v = list(s)
        n = min(len(v), len(cons))
        for i in range(n):
            if v[i] != cons[i] and rng.random() < force:
                v[i] = cons[i]
        out.append("".join(v))
    return out


def stabiliser(sequences: list[str], cycles: int = 5, cible: float = 0.5,
               seed: int = 0, forcer: bool = False,
               couplage: float = 0.0) -> dict:
    """Fait cycler la population et regarde si elle se stabilise.

    A chaque cycle, chaque neurone subit une mutation LOCALE dirigee (G),
    PUIS un couplage vers le consensus (`couplage` > 0). On suit la
    coherence R et la dispersion des y.

    Rappel mesure : sans couplage, R reste au plancher 0.886/sqrt(N) — ce
    n'est PAS de la stabilisation, c'est l'absence d'interaction. Le
    couplage est ce qui doit faire monter R au-dessus du plancher.
    """
    courant = list(sequences)
    hist = []
    rng = np.random.default_rng(seed)
    for c in range(cycles + 1):
        pop = lire_population(courant)
        ys = np.array([r["y"] for r in pop["lectures"]], float)
        hist.append({"cycle": c, "coherence": pop["coherence"],
                     "accord_bases": coherence_bases(courant),
                     "y_moyen": float(ys.mean()), "dispersion": float(ys.std())})
        if c == cycles:
            break
        courant = [generation(s, cible=cible, seed=int(rng.integers(1 << 30)),
                              forcer=forcer)["seq"]
                   for s in courant]
        courant = coupler(courant, force=couplage,
                          seed=int(rng.integers(1 << 30)))
    R0, RT = hist[0]["coherence"], hist[-1]["coherence"]
    A0, AT = hist[0]["accord_bases"], hist[-1]["accord_bases"]
    d0, dT = hist[0]["dispersion"], hist[-1]["dispersion"]
    # plancher theorique des phases independantes
    plancher = float(np.sqrt(np.pi) / (2 * np.sqrt(len(sequences))))
    return {
        "historique": hist,
        "derive_coherence": RT - R0,
        "accord_avant": A0,
        "accord_apres": AT,
        "derive_accord": AT - A0,
        "dispersion_avant": d0,
        "dispersion_apres": dT,
        "plancher_hasard": plancher,
        "au_dessus_du_plancher": bool(RT > plancher * 1.15),
        "synchronise": bool(AT > 0.25 + 0.10),
        "stable": bool(dT <= d0 + 0.10 or AT - A0 > 0.05),
    }


if __name__ == "__main__":
    print("=== Assemblage : coherence d'une population de neurones ===\n")
    rng = np.random.default_rng(7)

    def melange(p, n=240, seed=0):
        r = np.random.default_rng(seed)
        return "".join(str(r.choice(list("ATCG"))) if r.random() < p else "ATGC"[i % 4]
                       for i in range(n))

    # population homogene ordonnee
    pop_ordre = [melange(0.0, seed=i) for i in range(6)]
    # population homogene aleatoire
    pop_alea = [melange(1.0, seed=i) for i in range(6)]
    # population MIXTE (moitie ordre, moitie chaos)
    pop_mixte = pop_ordre[:3] + pop_alea[:3]

    for nom, pop in (("homogene ordonnee", pop_ordre),
                     ("homogene aleatoire", pop_alea),
                     ("mixte (ordre + chaos)", pop_mixte)):
        r = lire_population(pop)
        f_avg = fusionner(r, "average")
        f_coh = fusionner(r, "coherente")
        ys = [round(x["y"], 4) for x in r["lectures"]]
        print(f"{nom:24s} R={r['coherence']:.4f}  dispersion={f_avg['dispersion']:.4f}")
        print(f"{'':24s} y_avg={f_avg['y_fusion']:.4f}  y_coh={f_coh['y_fusion']:.4f}")
        print(f"{'':24s} y individuels={ys}\n")