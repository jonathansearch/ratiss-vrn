"""Semantique — NOMMER. Porte de ratis_net/topo_tokenizer.py.

ORGANE REPRIS : `_word_to_cloud` + `topo_signature`, de
Ratiss-experimental-IA-/ratis_net/topo_tokenizer.py.

Principe de RATISS (et c'est le coeur de la brique semantique) : un mot
n'est pas identifie par un hash, il est identifie par sa SIGNATURE
TOPOLOGIQUE — les cycles H1 persistants de son nuage de points. Deux
donnees topologiquement equivalentes produisent le meme token.

Adaptation assumee : le backend de persistance passe de
`persistence_optimizer.compute_persistence` a `ripser` — l'organe que
`ratiss-neuro/topology.py` appelle deja. Meme mesure, backend different.

Pour le neurone VRN, la chaine ATCG joue le role du "mot" : on demande
a la chaine de NOMMER ce qu'elle est. La reponse est un vecteur de
signature. On en tire un scalaire `c` = CONCENTRATION de la signature :
  - signature concentree sur peu de composantes -> la chaine nomme
    quelque chose de precis -> c eleve.
  - signature etalee -> aucun concept identifiable -> c faible.
"""
from __future__ import annotations

import hashlib
import math
import sys
sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.psig import h1_diagram_ripser


def _word_to_cloud(word: str, n_points: int = 40, seed: int = 42) -> np.ndarray:
    """Repris de topo_tokenizer.py::_word_to_cloud.

    Chaque caractere unique devient un anneau ; rayon, centre et deformation
    dependent du code du caractere. La topologie (nombre de cycles, leurs
    persistances, leurs intersections) varie donc d'un mot a l'autre.
    """
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") + seed)
    coords = []
    unique_chars = list(dict.fromkeys(word))
    n_rings = max(1, len(unique_chars))
    pts_per_ring = max(6, n_points // n_rings)
    for ci, ch in enumerate(unique_chars):
        code = ord(ch)
        radius = 0.6 + 0.8 * (code % 17) / 17.0
        cx = 1.5 * ci + 0.5 * (code % 7)
        cy = 0.4 * ((code % 13) - 6)
        for k in range(pts_per_ring):
            theta = 2 * math.pi * k / pts_per_ring
            r_eff = radius * (1.0 + 0.2 * math.sin(theta * (1 + code % 4)))
            x = cx + r_eff * math.cos(theta)
            y = cy + r_eff * math.sin(theta)
            z = 0.4 * math.sin(theta * (1 + code % 3))
            coords.append([x, y, z])
    coords = np.array(coords)
    coords += rng.normal(0, 0.03, coords.shape)
    return coords


def topo_signature(word: str, dim: int = 8) -> np.ndarray:
    """Signature topologique d'un mot = vecteur de dimension `dim`.

    Repris de topo_tokenizer.py::topo_signature.
    [b0, b1, densite_cycles, pers_max, pers_mean, pers_median, pers_std, skew]
    normalise.
    """
    cloud = _word_to_cloud(word)
    dgm = h1_diagram_ripser(cloud)
    pers = (dgm[:, 1] - dgm[:, 0]) if len(dgm) else np.array([])
    pers = pers[np.isfinite(pers)]
    pers = pers[pers > 0]
    n_cycles = pers.size
    n_pts = max(len(cloud), 2)

    if n_cycles:
        p_max = float(pers.max())
        p_mean = float(pers.mean())
        p_med = float(np.median(pers))
        p_std = float(pers.std())
        p_skew = float(((pers - p_mean) ** 3).mean() / (p_std ** 3)) if p_std > 1e-9 else 0.0
    else:
        p_max = p_mean = p_med = p_std = p_skew = 0.0

    feat = [
        1.0 / 10.0,                                       # b0 (une composante)
        float(n_cycles) / 10.0,                           # cycles
        math.log1p(n_cycles) / math.log1p(n_pts),         # densite
        min(p_max, 1.0),
        min(p_mean * 10.0, 1.0),
        min(p_med * 10.0, 1.0),
        min(p_std * 10.0, 1.0),
        min(max(p_skew, -1.0), 1.0),
    ]
    sig = np.array(feat, dtype=float)
    if len(sig) < dim:
        sig = np.pad(sig, (0, dim - len(sig)))
    elif len(sig) > dim:
        sig = sig[:dim]
    n = np.linalg.norm(sig)
    return sig / n if n > 1e-9 else sig


def _proteine_en_lettres(seq: str) -> str:
    """La chaine -> son "mot" : la proteine, sur un alphabet de 20 lettres.

    Correction d'une limite reelle : appliquer _word_to_cloud directement a
    la sequence ATCG ne voit que 4 caracteres uniques — l'ordre des codons
    ne change alors presque rien (c=0.0770 vs 0.0791 selon la chaine).
    En passant par la proteine, l'alphabet passe a 20 lettres et l'ordre
    devient porteur d'information.
    """
    from organes.atcg import translate
    prot = translate(seq, jusquau_stop=False)
    aas = sorted(set(prot))
    code = {aa: chr(65 + i) for i, aa in enumerate(aas)}
    return "".join(code[aa] for aa in prot)


def concept(seq: str) -> dict:
    """c = dominance du cycle principal dans la signature semantique.

    Correction d'une v1 fausse : je mesurais l'entropie de Shannon du
    vecteur de signature 8D, or elle vaut presque toujours ~1.9 quelle que
    soit la chaine -> c reste colle a ~0.05 et ne discrimine rien.

    Ici : c = p_max / sum(p). Si un seul cycle porte l'essentiel de la
    signature, la chaine NOMME une structure identifiable -> c eleve.
    Si la signature est etalee sur 8 composantes -> aucun concept -> c bas.
    """
    mot = _proteine_en_lettres(seq)
    if len(mot) < 2 or len(set(mot)) < 2:
        return {"c": 0.0, "signature": [], "domination": 0.0, "mot": mot}
    sig = topo_signature(mot)
    p = np.abs(sig)
    total = p.sum()
    if total < 1e-12:
        return {"c": 0.0, "signature": sig.tolist(), "domination": 0.0, "mot": mot}
    p = p / total
    dom = float(p.max())
    # 1/dim = signature parfaitement etalee -> c = 0 ; 1 = signature concentree -> c = 1
    c = float(np.clip((dom - 1.0 / len(p)) / (1.0 - 1.0 / len(p)), 0.0, 1.0))
    return {"c": c, "signature": sig.tolist(), "domination": dom, "mot": mot}


if __name__ == "__main__":
    print("=== Semantique : la chaine nomme-t-elle quelque chose ? ===\n")
    for mot in ["ATCG", "ATGCATGC", "ATGAAACCCGGGTTT", "AAAA", "bonjour"]:
        r = concept(mot)
        print(f"  {mot:20} c={r['c']:.4f}  dominance={r.get('domination',0):.4f}")