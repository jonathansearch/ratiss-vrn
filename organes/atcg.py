"""Neurone ATCG — chaine genomique qui se replie, avec douane VRN.

ORGANES REPRIS (rien reecrit de zero) :

  - CODON_TABLE, HYDROPHOBICITY, protein_folding_coherence
      -> ratiss-bio/bio114/cascade.py
      (traduction codon -> acide amine, repliement par hydrophobicite)

  - all_paths + scoring + culling a 95% du meilleur
      -> Unicycler/unicycler/path_finding.py
      (resolution d'ambiguite : enumerer les chemins, scorer, eliminer)

  - multiplicite entree/sortie d'un repeat
      -> Flye/flye/trestle/graph_resolver.py

IDEE :

  Le neurone VRN n'est pas un noeud scalaire. C'est une CHAINE ATCG.
  La chaine se replie (coeur hydrophobe / surface hydrophile). Sa forme
  repliee est lue par P_sig. Et quand la chaine est ambigue — un repeat,
  deux lectures possibles — la douane VRN fait ce que fait un assembleur
  de genome : elle enumere les chemins, les score, et si aucun ne gagne
  nettement, ELLE SE TAIT.

  C'est la regle d'or : si la forme s'effondre, le neurone se tait.

LES 4 BASES (double lecture) :

    Base | Genetique      | Role VRN              | Homologie
    -----|----------------|-----------------------|-----------
    A    | Adenine        | Ancrage (H0)          | stabilite
    T    | Thymine        | Transmission (H1/H2)  | connectivite
    C    | Cytosine       | Coherence (contrainte) | garde-fou
    G    | Guanine        | Generation (plasticite)| adaptation

LE CODON (ce qui manquait) :

  3 bases -> 64 combinaisons -> 20 acides amines + stop.
  C'est la REDONDANCE qui donne a l'ADN sa tolerance aux erreurs :
  plusieurs codons donnent le meme acide amine. Un code a 4 symboles
  n'a pas cette propriete ; un code a 64 codons si.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.psig import p_sig_ripser as p_sig, takens_embed

# --------------------------------------------------------------------------
# Table des codons — reprise de ratiss-bio/bio114/cascade.py
# --------------------------------------------------------------------------

CODON_TABLE = {
    "ATG": "Met", "TTT": "Phe", "TTC": "Phe", "TTA": "Leu", "TTG": "Leu",
    "CTT": "Leu", "CTC": "Leu", "CTA": "Leu", "CTG": "Leu",
    "ATT": "Ile", "ATC": "Ile", "ATA": "Ile", "GTT": "Val", "GTC": "Val",
    "GTA": "Val", "GTG": "Val", "TCT": "Ser", "TCC": "Ser", "TCA": "Ser",
    "TCG": "Ser", "CCT": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
    "ACT": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
    "GCT": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
    "TAT": "Tyr", "TAC": "Tyr", "CAT": "His", "CAC": "His",
    "CAA": "Gln", "CAG": "Gln", "AAT": "Asn", "AAC": "Asn",
    "AAA": "Lys", "AAG": "Lys", "GAT": "Asp", "GAC": "Asp",
    "GAA": "Glu", "GAG": "Glu", "TGT": "Cys", "TGC": "Cys",
    "TGG": "Trp", "CGT": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg",
    "AGT": "Ser", "AGC": "Ser", "AGA": "Arg", "AGG": "Arg",
    "GGT": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
    "TAA": "STOP", "TAG": "STOP", "TGA": "STOP",
}

HYDROPHOBICITY = {
    "Ile": 4.5, "Val": 4.2, "Leu": 3.8, "Phe": 2.8, "Cys": 2.5,
    "Met": 1.9, "Ala": 1.8, "Gly": -0.4, "Thr": -0.7, "Ser": -0.8,
    "Trp": -0.9, "Tyr": -1.3, "Pro": -1.6, "His": -3.2, "Gln": -3.5,
    "Asn": -3.5, "Glu": -3.5, "Asp": -3.5, "Lys": -3.9, "Arg": -4.5,
    "STOP": 0.0,
}

BASES = ("A", "T", "C", "G")
ROLE_VRN = {"A": "ancrage", "T": "transmission", "C": "coherence", "G": "generation"}


def translate(seq: str) -> list[str]:
    """Codon -> acide amine. Repris de cascade.py::translate_protein."""
    out = []
    for i in range(0, len(seq) - 2, 3):
        aa = CODON_TABLE.get(seq[i:i + 3], "Gly")
        if aa == "STOP":
            break
        out.append(aa)
    return out


def fold_coherence(protein: list[str]) -> float:
    """Coherence de repliement. Repris de cascade.py::protein_folding_coherence.

    Un coeur hydrophobe + une surface hydrophile = repliement stable.
    Mesure l'alternance locale de l'hydrophobicite.
    """
    if len(protein) < 3:
        return 0.0
    arr = np.array([HYDROPHOBICITY.get(aa, 0.0) for aa in protein])
    return float(np.clip(np.mean(np.abs(np.diff(arr))) / 4.0, 0.0, 1.0))


def hydrophobic_signal(seq: str) -> np.ndarray:
    """Le signal 1D que porte la chaine : le profil d'hydrophobicite.

    Un genome porte un signal (comme un EEG porte une trace). Ici le signal
    est le profil d'hydrophobicite des acides amines le long de la chaine.
    C'est ce signal que l'on plonge dans l'espace des phases.
    """
    prot = translate(seq)
    if not prot:
        return np.zeros(0)
    return np.array([HYDROPHOBICITY.get(aa, 0.0) for aa in prot], dtype=np.float64)


def chain_to_cloud(seq: str, dim: int = 3, delay: int = 3) -> np.ndarray:
    """La chaine -> nuage de points par PLONGEMENT DE TAKENS.

    Remplace la v1 (helice + hydrophobicite, une construction ad hoc) :
    la v1 faisait mesurer a P_sig une forme que J'AVAIS fabriquee, pas une
    forme portee par la chaine.

    Ici on applique l'organe de ratiss-neuro/topology.py : le signal
    d'hydrophobicite est plonge dans l'espace des phases (Takens), et
    c'est la GEOMETRIE DU SIGNAL que P_sig va lire.
    """
    x = hydrophobic_signal(seq)
    if x.size < 8:
        return np.zeros((0, dim))
    x = (x - x.mean()) / (x.std() + 1e-12)
    cloud = takens_embed(x, dim=dim, delay=delay)
    if len(cloud) == 0:
        return cloud
    # Dedoublonnage : dans un complexe de Vietoris-Rips, des points confondus
    # ne creent aucun cycle — ils gonflent le comptage et faussent la mesure.
    # Un signal discret (20 niveaux d'hydrophobicite) produit des dizaines de
    # copies du meme point. On ne garde que les positions distinctes : c'est
    # mathematiquement equivalent et ca supprime l'artefact.
    return np.unique(np.round(cloud, 10), axis=0)


def chain_to_cloud_plie(seq: str) -> np.ndarray:
    """v1 conservee pour comparaison : repliement 3D en helice hydrophobe."""
    prot = translate(seq)
    if len(prot) < 4:
        return np.zeros((0, 3))
    h = np.array([HYDROPHOBICITY.get(aa, 0.0) for aa in prot])
    hn = (h - h.min()) / (np.ptp(h) + 1e-12)
    n = len(hn)
    theta = np.linspace(0, 4 * np.pi, n)
    radius = 0.35 + 0.65 * hn
    z = (hn - 0.5) * 2.0
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])


# --------------------------------------------------------------------------
# DOUANE VRN : resolution d'ambiguite a la maniere d'un assembleur
#   logique reprise de Unicycler/path_finding.py::get_best_paths_for_seq
# --------------------------------------------------------------------------

def _kmer_graph(seq: str, k: int = 3) -> dict:
    """Graphe de de Bruijn : k-mer -> k-mers suivants."""
    g: dict[str, list[str]] = {}
    for i in range(len(seq) - k):
        a, b = seq[i:i + k], seq[i + 1:i + 1 + k]
        g.setdefault(a, []).append(b)
    return g


def branching_kmers(seq: str, k: int = 3) -> dict:
    """k-mers avec plusieurs successeurs = zones ambigues (repeats).

    Repris de la logique de Flye/trestle : un noeud a multiplicite > 1
    est une zone a resoudre.
    """
    g = _kmer_graph(seq, k)
    return {n: sorted(set(s)) for n, s in g.items() if len(set(s)) > 1}


def enumerate_paths(seq: str, start: str, max_paths: int = 64, k: int = 3) -> list[str]:
    """Chemins possibles depuis un k-mer ambigu.

    Repris de Unicycler/path_finding.py::all_paths (version bornee).
    """
    g = _kmer_graph(seq, k)
    paths = [[start]]
    out = []
    while paths and len(out) < max_paths:
        new = []
        for p in paths:
            nxt = g.get(p[-1])
            if not nxt:
                if len(p) > 1:
                    out.append("".join(x[0] for x in p) + p[-1][1:])
                continue
            for nx in set(nxt):
                if len(p) > 12:
                    out.append("".join(x[0] for x in p) + p[-1][1:])
                    continue
                new.append(p + [nx])
        paths = new
    return out


def consensus_gate(seq: str, n_variants: int = 24, mut: float = 0.06,
                   seed: int = 7, k: int = 3) -> dict:
    """La douane VRN v2 : CONSENSUS de topologie.

    ORGANE REPRIS : le principe de consensus d'un assembleur de genome
    (Unicycler : les lectures se recouvrent, on vote ; Flye : multiplicite
    entree/sortie d'un repeat). Un assembleur ne resout pas une zone par
    branchement — il la resout par ACCORD entre lectures.

    Ici : on perturbe la chaine (lectures alternatives), on replie chaque
    variante, on mesure P_sig de chaque repliement, et on demande :
    la signature topologique SURVIT-ELLE aux perturbations ?

      - Les variantes s'accordent  -> la forme tient -> g ouvre.
      - Les variantes divergent    -> la forme s'effondre -> g ferme.

    C'est la regle d'or rendue mesurable : si la forme s'effondre, le
    neurone se tait. Et c'est le consensus d'assemblage, applique a la
    coherence d'un neurone.

    v1 (abandonnee) : le gate par branchement de k-mers etait INVERSE —
    il penalisait les chaines aleatoires et recompensait les repetitions
    parfaites. Une repetition n'est pas une ambiguite : c'est un CYCLE H1.
    Le branchement n'est pas la bonne moitie de l'assembleur. Le consensus
    l'est.
    """
    rng = np.random.default_rng(seed)
    bases = np.array(list(seq))

    sigs = []
    for _ in range(n_variants):
        v = bases.copy()
        n_mut = max(1, int(mut * len(v)))
        idx = rng.choice(len(v), size=n_mut, replace=False)
        for i in idx:
            v[i] = rng.choice(list("ATCG"))
        cloud = chain_to_cloud("".join(v))
        if len(cloud) >= 6:
            sigs.append(p_sig(cloud)["p_sig"])
    if len(sigs) < 4:
        return {"g": 0.0, "n_variants": len(sigs), "dispersion": None,
                "verdict": "chaine trop courte pour un repliement lisible"}

    sigs = np.array(sigs)
    mu, sd = float(sigs.mean()), float(sigs.std())
    # g = stabilite : accord entre variantes. sd grand -> desaccord -> silence.
    g = float(np.clip(1.0 - sd / (abs(mu) + 1e-9), 0.0, 1.0))
    verdict = ("les variantes s'accordent — la forme tient, la porte s'ouvre"
               if g > 0.5 else
               "les variantes divergent — la forme s'effondre, le neurone se tait")
    return {"g": g, "n_variants": len(sigs), "mu_psig": mu, "sd_psig": sd,
            "dispersion": sd / (abs(mu) + 1e-9), "verdict": verdict}


# --------------------------------------------------------------------------
# LE NEURONE VRN
# --------------------------------------------------------------------------

def neurone_vrn(seq: str, k: int = 3, plie: bool = False) -> dict:
    """y = g_VR . (s (+) c)

    s : lecture topologique de la chaine repliee (P_sig)
    c : lecture semantique (acide amines -> roles VRN)
    g : douane (resolution d'ambiguite)
    """
    cloud = chain_to_cloud_plie(seq) if plie else chain_to_cloud(seq)
    prot = translate(seq)

    s = p_sig(cloud) if len(cloud) >= 4 else {"p_sig": 0.0, "robust_h1": 0, "n_h1_bars": 0}
    c = fold_coherence(prot)
    gate = consensus_gate(seq, k=k)

    roles = {}
    for i, base in enumerate(seq):
        roles[ROLE_VRN[base]] = roles.get(ROLE_VRN[base], 0) + 1

    return {
        "n_bases": len(seq),
        "n_codons": len(prot),
        "roles": roles,
        "s_psig": s["p_sig"],
        "s_robust_h1": s.get("robust_h1", 0),
        "c_repliement": c,
        "g_douane": gate["g"],
        "verdict_douane": gate["verdict"],
        "mu_psig_variantes": gate.get("mu_psig"),
        "sd_psig_variantes": gate.get("sd_psig"),
        "y": gate["g"] * (s["p_sig"] + c),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    print("=== Neurone ATCG — chaine genomique + douane VRN ===\n")

    def montre(nom, seq, plie=False):
        r = neurone_vrn(seq, plie=plie)
        tag = " [repliement v1]" if plie else " [Takens]"
        print(f"{nom}{tag}")
        print(f"   bases={r['n_bases']} codons={r['n_codons']} roles={r['roles']}")
        print(f"   s(P_sig)={r['s_psig']:.4f}  c(repli)={r['c_repliement']:.4f}  "
              f"g(douane)={r['g_douane']:.4f}")
        if r['mu_psig_variantes'] is not None:
            print(f"   variantes : mu={r['mu_psig_variantes']:.4f} "
                  f"sd={r['sd_psig_variantes']:.4f}")
        print(f"   y={r['y']:.4f}")
        print(f"   -> {r['verdict_douane']}\n")

    # chaines de travail : 60 codons, sans STOP precoce (on evite TAA/TAG/TGA)
    def chaine(n_codons=60, seed=0):
        r = np.random.default_rng(seed)
        aas = [aa for aa in CODON_TABLE if CODON_TABLE[aa] != "STOP"]
        return "ATG" + "".join(r.choice(aas) for _ in range(n_codons))

    print("--- A. Chaine aleatoire, 60 codons ---")
    montre("CHAINE A", chaine(60, 1))

    print("--- B. Coeur hydrophobe construit (GTT/GTG = Val) puis surface (GAT) ---")
    montre("CHAINE B", "ATG" + "GTT" * 20 + "GAT" * 20 + "GAA" * 18)

    print("--- C. Profil plat : un seul acide amine (Gly, hydrophile) ---")
    montre("CHAINE C", "ATG" + "GGT" * 59)

    print("--- D. Repeat pur (ATGCATGC...) ---")
    montre("CHAINE D", ("ATGC" * 16))

    print("--- E. Comparaison des deux representations sur la meme chaine ---")
    montre("CHAINE B", "ATG" + "GTT" * 20 + "GAT" * 20 + "GAA" * 18, plie=True)