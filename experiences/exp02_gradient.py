"""Experience 02 — la metrique a-t-elle un GRADIENT ?

Question : P_sig (via Takens) augmente-t-il quand la chaine gagne
de la structure ? Si oui, la douane VRN a une echelle de lecture.
Si non, elle ne lit qu'un binaire (structure / pas structure).

Protocole : on construit des chaines avec un nombre croissant d'acides
amines distincts (1, 2, 4, 8, 20). Le profil d'hydrophobicite passe du
plat au riche. On lit P_sig et le gate a chaque niveau.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/vrn")

import numpy as np
from organes.atcg import CODON_TABLE, chain_to_cloud, fold_coherence, translate, consensus_gate
from organes.psig import p_sig_ripser as p_sig

NON_STOP = [aa for aa in CODON_TABLE if CODON_TABLE[aa] != "STOP"]
ACIDES = sorted(set(CODON_TABLE[c] for c in NON_STOP if CODON_TABLE[c] != "STOP"))
CODONS_DE = {aa: [c for c in CODON_TABLE if CODON_TABLE[c] == aa] for aa in ACIDES}


def chaine_de_diversite(n_distinct: int, n_codons: int = 60, seed: int = 3) -> str:
    """Chaine aleatoire tiree dans un alphabet de `n_distinct` acides amines."""
    rng = np.random.default_rng(seed)
    pool = ACIDES[:n_distinct] if n_distinct <= len(ACIDES) else ACIDES
    codons = [c for aa in pool for c in CODONS_DE[aa]]
    return "ATG" + "".join(rng.choice(codons) for _ in range(n_codons))


if __name__ == "__main__":
    print("=== Experience 02 : gradient de P_sig selon la structure ===\n")
    print(f"{'acides distincts':>17} {'P_sig':>8} {'repli':>8} {'g':>8} {'y':>8}  verdict")
    print("-" * 78)

    for n in (1, 2, 4, 8, 20):
        seq = chaine_de_diversite(n)
        cloud = chain_to_cloud(seq)
        s = p_sig(cloud)["p_sig"] if len(cloud) >= 6 else 0.0
        c = fold_coherence(translate(seq))
        gate = consensus_gate(seq, n_variants=10)
        y = gate["g"] * (s + c)
        print(f"{n:>17} {s:>8.4f} {c:>8.4f} {gate['g']:>8.4f} {y:>8.4f}  {gate['verdict'][:34]}")