"""Experience 03 — la metrique lit-elle la PERIODICITE ?

Hypothese issue de l'exp02 : P_sig ne suit pas la diversite de l'alphabet,
il culmine a 2 acides. Or 2 acides font un signal qui ALTERNE — donc
periodique de periode 2. Un signal periodique trace une boucle fermee
dans l'espace des phases de Takens.

On teste directement : motifs periodiques de periode croissante,
contre un signal aleatoire de meme longueur.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/vrn")

import numpy as np
from organes.atcg import CODON_TABLE, chain_to_cloud, consensus_gate
from organes.psig import p_sig_ripser as p_sig

NON_STOP = [c for c in CODON_TABLE if CODON_TABLE[c] != "STOP"]
AA_HYDRO = {"Val": 4.2, "Leu": 3.8, "Ile": 4.5, "Glu": -3.5, "Lys": -3.9, "Gly": -0.4}
# deux acides aux extremes opposes : coeur vs surface
HYDRO = "GTT"   # Val  +4.2
PHILE = "GAT"   # Asp  -3.5


def chaine_periodique(periode: int, n_codons: int = 60) -> str:
    """Motif de `periode` codons, alterne fort / faible, repete."""
    motif = "".join("GTT" if i % 2 == 0 else "GAT" for i in range(periode))
    n = n_codons // periode + 1
    return "ATG" + (motif * n)[:n_codons * 3]


def chaine_aleatoire(n_codons: int = 60, seed: int = 5) -> str:
    rng = np.random.default_rng(seed)
    return "ATG" + "".join(rng.choice([HYDRO, PHILE]) for _ in range(n_codons))


if __name__ == "__main__":
    print("=== Experience 03 : P_sig lit-il la periodicite ? ===\n")
    print(f"{'periode (codons)':>16} {'P_sig':>8} {'robust_h1':>10} {'g':>8}  verdict")
    print("-" * 78)

    for p in (1, 2, 3, 4, 5, 6, 10, 15, 30, 60):
        seq = chaine_periodique(p)
        cloud = chain_to_cloud(seq)
        r = p_sig(cloud) if len(cloud) >= 6 else {"p_sig": 0.0, "robust_h1": 0}
        gate = consensus_gate(seq, n_variants=10)
        print(f"{p:>16} {r['p_sig']:>8.4f} {r['robust_h1']:>10} {gate['g']:>8.4f}  {gate['verdict'][:30]}")

    print("\n--- reference : signal aleatoire (aucune periode) ---")
    seq = chaine_aleatoire()
    cloud = chain_to_cloud(seq)
    r = p_sig(cloud)
    gate = consensus_gate(seq, n_variants=10)
    print(f"{'aleatoire':>16} {r['p_sig']:>8.4f} {r['robust_h1']:>10} {gate['g']:>8.4f}  {gate['verdict'][:30]}")