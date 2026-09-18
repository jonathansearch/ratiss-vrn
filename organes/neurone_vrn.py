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
               seed: int = 0) -> dict:
    """G — plasticite dirigee par la coherence. G MODIFIE la chaine.

    Ordre du chef : "G ne compte pas. G modifie la structure de la chaine
    en reponse a l'ecart mesure par la Douane VR. C'est la boucle de
    retroaction biologique."

    Ici : on mesure Delta_ent (desalignement). Si il est eleve, on cherche
    par mutation LOCALE (un codon a la fois) la variante qui rapproche la
    coherence de sa cible. On garde la meilleure. G renvoie la chaine
    MUTEE, pas un score.
    """
    d_ent = entropy_delta(seq)
    if d_ent < cible:
        return {"seq": seq, "delta_ent": d_ent, "n_mutations": 0,
                "verdict": "aligne — pas de mutation necessaire"}

    rng = np.random.default_rng(seed)
    bases = list(seq)
    meilleure, meilleur_ecart = seq, abs(d_ent - cible)
    n_mut = 0
    for _ in range(essais):
        v = bases.copy()
        i = int(rng.integers(0, len(v)))
        v[i] = str(rng.choice(list("ATCG")))
        cand = "".join(v)
        ecart = abs(entropy_delta(cand) - cible)
        if ecart < meilleur_ecart:
            meilleure, meilleur_ecart = cand, ecart
            bases = v
            n_mut += 1
    return {"seq": meilleure, "delta_ent": entropy_delta(meilleure),
            "n_mutations": n_mut,
            "verdict": f"desaligne (Delta_ent={d_ent:.3f}) — {n_mut} mutations locales"}


def neurone_vrn_v3(seq: str, alpha: float = 1.0, plastique: bool = False) -> dict:
    """y = g . (s * c)^alpha,  g = g_dyn * g_topo."""
    seq_eff = seq
    gen = {"seq": seq, "delta_ent": entropy_delta(seq), "n_mutations": 0,
           "verdict": "G inactif"}
    if plastique:
        gen = generation(seq)
        seq_eff = gen["seq"]

    cloud = chain_to_cloud(seq_eff)
    s = p_sig_ripser(cloud) if len(cloud) >= 6 else {"p_sig": 0.0, "robust_h1": 0}
    d = jepa(seq_eff)
    topo = consensus_gate(seq_eff)
    sem = concept(seq_eff)

    g_dyn = d["g_topo"]
    g_topo = topo["g"]
    g = g_dyn * g_topo
    fusion = float((max(s["p_sig"], 0.0) * max(sem["c"], 0.0)) ** alpha)
    y = g * fusion

    roles = {}
    for b in seq_eff:
        roles[ROLE_VRN[b]] = roles.get(ROLE_VRN[b], 0) + 1
    return {
        "seq": seq_eff,
        "roles": roles,
        "s_psig": s["p_sig"],
        "c_semantique": sem["c"],
        "c_physique": fold_coherence(translate(seq_eff, jusquau_stop=False)),
        "g_dyn": g_dyn,
        "g_topo": g_topo,
        "g": g,
        "fusion": fusion,
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
    print("=== Neurone VRN v3 — trois briques branchees ===\n")

    def montre(nom, seq):
        r = neurone_vrn_v3(seq)
        print(f"{nom}")
        print(f"   s(P_sig)={r['s_psig']:.4f}  c(sem)={r['c_semantique']:.4f}  "
              f"fusion={r['fusion']:.4f}")
        print(f"   g_dyn={r['g_dyn']:.4f}  g_topo={r['g_topo']:.4f}  g={r['g']:.4f}  "
              f"y={r['y']:.4f}")
        print(f"   -> VOIR : {r['verdict_dyn']}")
        print(f"   -> VERIFIER : {r['verdict_topo']}")
        print(f"   -> NOMMER : c={r['c_semantique']:.4f}\n")

    montre("ORDRE PUR — ATGC repete", "ATGC" * 40)
    montre("ALEATOIRE", "".join(rng.choice(list("ATCG")) for _ in range(160)))

    print("=== G : plasticite dirige le desalignement ===\n")
    desaligne = "ATG" + "AAA" * 10 + "".join(rng.choice(list("ATCG")) for _ in range(90))
    r = neurone_vrn_v3(desaligne, plastique=True)
    print(f"   Delta_ent avant={r['delta_ent_avant']:.4f}  apres={r['delta_ent']:.4f}")
    print(f"   mutations={r['n_mutations']}  -> {r['verdict_plastique']}")