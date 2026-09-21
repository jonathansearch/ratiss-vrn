"""Neurone VRN v3 — les trois briques branchees.

    VOIR      -> dynamique.jepa(seq)        g_dyn  (test de predictibilite)
    VERIFIER  -> consensus_gate(seq)        g_topo (robustesse topologique)
    NOMMER    -> semantique.concept(seq)    c      (signature semantique)

    s  = P_sig (persistance H1, topologie)
    G  = entropy_delta (plasticite) — voir plus bas
    y  = g . (s * c)^alpha      (fusion, pas somme)
    g  = g_dyn * g_topo

CHANGEMENTS ASSUMES vs la v2 :

  (a) `c` n'est plus la cohesion hydrophobe, c'est la SIGNATURE SEMANTIQUE
      (organe repris de ratis_net/topo_tokenizer.py). L'ordre du chef :
      "c devient la signature semantique extraite de la chaine ATCG".

  (b) `g` n'est plus un seul gate, il est le PRODUIT de deux lectures :
      robustesse topologique (VERIFIER) x predictibilite dynamique (VOIR).

  (c) L'operateur ⊕ (somme) est remplace par un PRODUIT (s*c)^alpha.
      "Une forte topologie sans concept pertinent = hallucination
      structurelle. Un concept fort sans support topologique = delire
      semantique." Le produit annule les deux cas.

  (d) G n'est plus decoratif. Il porte la PLASTICITE : Delta_ent mesure le
      desalignement, et G applique une mutation LOCALE dirigee quand la
      coherence est insuffisante. Voir `generation()`.

  (e) La coherence hydrophobe de la v2 n'est PAS jetee : elle devient
      `c_physique`, disponible comme diagnostic, hors de y.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.atcg import (
    ROLE_VRN, fold_coherence, translate, chain_to_cloud, consensus_gate,
)
from organes.psig import p_sig_ripser
from organes.semantique import concept
from organes.dynamique import jepa


def entropy_delta(seq: str) -> float:
    """Delta_ent = desalignement : entropie(bases) vs entropie(proteine).

    Mesure l'ecart entre ce que porte la chaine brute et ce que porte sa
    lecture traduite. Un codage qui ne change rien a la lecture est aligne.
    """
    def H(s):
        if not s:
            return 0.0
        _, c = np.unique(list(s), return_counts=True)
        p = c / c.sum()
        return float(-np.sum(p * np.log(p)))
    h_base = H(seq)
    h_prot = H(translate(seq, jusquau_stop=False))
    ref = max(h_base, h_prot, 1e-9)
    return float(np.clip(abs(h_base - h_prot) / np.log(4), 0.0, 1.0))


def generation(seq: str, cible: float = 0.5, essais: int = 12,
               seed: int = 0, forcer: bool = False) -> dict:
    """G — plasticite dirigee par la coherence. G MODIFIE la chaine.

    Ordre du chef : "G ne compte pas. G modifie la structure de la chaine
    en reponse a l'ecart mesure par la Douane VR. C'est la boucle de
    retroaction biologique."

    Phi D = |Delta_ent - cible| : la distance a la cible de desalignement.
    G cherche par mutation LOCALE (un codon a la fois) la variante qui
    reduit Phi D. G renvoie la chaine MUTEE, pas un score.

    --- DEFAUT CORRIGE : LE GARDE A SENS UNIQUE ---

    La v1 faisait :

        if d_ent < cible:
            return "aligne — pas de mutation necessaire"

    C'etait un test a SENS UNIQUE. Il confondait "sous la cible" avec
    "aligne". Consequence mesuree : une chaine ordonnee (Delta_ent = 0.0)
    etait un POINT FIXE ABSORBANT — G ne la touchait jamais
    (`mutations=0`). Une population ordonnee devenait donc un systeme FIGE,
    et la stabilisation mesuree etait celle d'un systeme MORT, pas d'un
    systeme stable. Bug trouve en exp07 en comparant les cycles : valeurs
    identiques au chiffre pres.

    Le bon critere est la DISTANCE a la cible, pas le signe de l'ecart.
    `Forcer=True` autorise G a s'eloigner de la cible — necessaire pour
    explorer depuis un etat degenere (ex. ordre parfait), sous peine de ne
    jamais pouvoir en sortir.
    """
    d_ent = entropy_delta(seq)
    phi = abs(d_ent - cible)
    if phi < 1e-9 and not forcer:
        return {"seq": seq, "delta_ent": d_ent, "n_mutations": 0,
                "phi": phi, "verdict": "sur la cible — pas de mutation"}

    rng = np.random.default_rng(seed)
    bases = list(seq)
    meilleure, meilleur_phi = seq, phi
    n_mut = 0
    for _ in range(essais):
        v = bases.copy()
        i = int(rng.integers(0, len(v)))
        v[i] = str(rng.choice(list("ATCG")))
        cand = "".join(v)
        cand_phi = abs(entropy_delta(cand) - cible)
        if cand_phi < meilleur_phi or (forcer and cand_phi <= meilleur_phi + 1e-12):
            meilleure, meilleur_phi = cand, cand_phi
            bases = v
            n_mut += 1
    return {"seq": meilleure, "delta_ent": entropy_delta(meilleure),
            "n_mutations": n_mut, "phi": meilleur_phi,
            "verdict": (f"Phi D {phi:.3f} -> {meilleur_phi:.3f} "
                        f"({n_mut} mutations locales)")}


def neurone_vrn_v3(seq: str, alpha: float = 1.0, plastique: bool = False) -> dict:
    """y = g_dyn . m(s) . m(g_topo),  m(x) = 0.5 + 0.5x.

    OPERATEUR DECIDE PAR MESURE (exp06, criteres scelles) :

      produit  y = g_dyn . g_topo . (s . c)   AUC 0.8277
      hierar.  y = g_dyn . m(s) . m(g_topo)   AUC 0.9743   <- RETENU

    g_dyn GOUVERNE (facteur). s et g_topo MODULENT : m va de 0.5 a 1, donc
    un modulateur peut au pire diviser le signal par deux, jamais l'annuler.
    C'est la difference avec le produit, ou un organe a zero tue tout.

    `c` EST SORTI DU CHEMIN DE y (exp06 C2). Mesure : gain de c en
    hierarchique = +0.0003 (negligeable), et dans le produit il DETERIORE
    l'AUC de 0.027. c seul vaut 0.4061 — sous 0.5, il discrimine a
    l'envers. c reste calcule et renvoye, en DIAGNOSTIC seulement (voir la
    cle "c_semantique"), jamais dans y. Le langage n'a pas prouve qu'il
    apportait quelque chose ; il est mis hors du chemin tant que ce n'est
    pas le cas.
    """
    seq_eff = seq
    gen = {"seq": seq, "delta_ent": entropy_delta(seq), "n_mutations": 0,
           "verdict": "G inactif"}
    if plastique:
        gen = generation(seq)
        seq_eff = gen["seq"]

    cloud = chain_to_cloud(seq_eff)
    # s_defini distingue "non mesurable" de "mesure a zero". Une chaine trop
    # courte pour plonger n'a PAS un P_sig nul : elle n'a pas de P_sig.
    # Meme discipline que le correctif ripser : pas de zero qui se fait
    # passer pour une mesure.
    s_defini = len(cloud) >= 6
    if s_defini:
        s = p_sig_ripser(cloud)
    else:
        s = {"p_sig": 0.0, "robust_h1": 0, "instrument": "non mesure (nuage trop petit)"}
    d = jepa(seq_eff)
    topo = consensus_gate(seq_eff)
    sem = concept(seq_eff)

    g_dyn = d["g_topo"]
    g_topo = topo["g"]
    # modulateurs : bornes [0.5, 1], un organe ne peut pas annuler le signal
    m_s = 0.5 + 0.5 * float(max(s["p_sig"], 0.0))
    m_topo = 0.5 + 0.5 * float(max(g_topo, 0.0))
    y = g_dyn * m_s * m_topo

    roles = {}
    for b in seq_eff:
        roles[ROLE_VRN[b]] = roles.get(ROLE_VRN[b], 0) + 1
    return {
        "seq": seq_eff,
        "roles": roles,
        "s_psig": s["p_sig"],
        "s_defini": s_defini,
        "instrument": s.get("instrument", "n/a"),
        "c_semantique": sem["c"],              # DIAGNOSTIC, hors de y
        "c_physique": fold_coherence(translate(seq_eff, jusquau_stop=False)),
        "g_dyn": g_dyn,
        "g_topo": g_topo,
        "m_s": m_s,
        "m_topo": m_topo,
        "y": y,
        "delta_ent": gen["delta_ent"],
        "delta_ent_avant": entropy_delta(seq),
        "n_mutations": gen["n_mutations"],
        "verdict_plastique": gen["verdict"],
        "verdict_dyn": d["verdict_topo"],
        "verdict_topo": topo["verdict"],
    }


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    print("=== Neurone VRN v3 — operateur hierarchique (exp06) ===\n")

    def montre(nom, seq):
        r = neurone_vrn_v3(seq)
        print(f"{nom}")
        print(f"   y={r['y']:.4f} = g_dyn {r['g_dyn']:.3f} x m_s {r['m_s']:.3f} "
              f"x m_topo {r['m_topo']:.3f}")
        print(f"   s={r['s_psig']:.4f} (defini={r['s_defini']}, "
              f"{r['instrument']})  |  c(diagnostic)={r['c_semantique']:.4f}")
        print(f"   -> VOIR : {r['verdict_dyn']}")
        print(f"   -> VERIFIER : {r['verdict_topo']}\n")

    montre("ORDRE PUR — ATGC repete", "ATGC" * 40)
    montre("ALEATOIRE", "".join(rng.choice(list("ATCG")) for _ in range(160)))

    print("=== G : plasticite dirige le desalignement ===\n")
    desaligne = "ATG" + "AAA" * 10 + "".join(rng.choice(list("ATCG")) for _ in range(90))
    r = neurone_vrn_v3(desaligne, plastique=True)
    print(f"   Delta_ent avant={r['delta_ent_avant']:.4f}  apres={r['delta_ent']:.4f}")
    print(f"   mutations={r['n_mutations']}  -> {r['verdict_plastique']}")
