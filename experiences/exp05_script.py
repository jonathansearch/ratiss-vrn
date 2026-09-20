"""Experience 05 — CORRELATION INTRA-SCRIPT.

CONSIGNE : evoluer les chaines de repliement de chaque neurone afin de
surveiller exactement d'ou l'un va a l'autre, determiner les taux
d'enroulement, et le moment ou ca casse.

Un SCRIPT = une suite de neurones. Ici trois scripts, du plus ordonne au
plus chaotique, et un script mixte qui force une transition.

Ce que l'on regarde :
  - le taux d'enroulement PROPRE de chaque neurone (sa signature)
  - le taux d'enroulement LOCAL le long du script (fenetre glissante)
  - la ou le local quitte la loi de son neurone -> CASSURE
  - le PLV entre neurones voisins -> verrouillage ou decrochage

PREDICTION (enoncee avant mesure) :
  Le taux d'enroulement doit DECROITRE quand le desordre augmente : un
  signal ordonneur tourne beaucoup autour de son centre, un signal
  chaotique tourne peu et de facon instable. L'instabilite locale du taux
  doit donc AUGMENTER avec le desordre. Une cassure doit apparaitre a la
  frontiere ordre -> chaos, pas a l'interieur d'un regime homogene.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.enroulement import analyse_script, taux_enroulement, plv
from organes.psig import takens_embed


def neurone_motif(motif: str, n: int = 120) -> str:
    return (motif * (n // len(motif) + 1))[:n]


def neurone_melange(p: float, entropie: float = 0.5, n: int = 120, seed: int = 0) -> str:
    """Chaine avec un biais controll par entropie (probabilite d'utiliser
    une base non-uniforme) et un taux de desordre p (probabilite de casser
    le motif local)."""
    rng = np.random.default_rng(seed)
    return "".join(str(rng.choice(list("ATCG"))) for _ in range(n))


if __name__ == "__main__":
    print("=== Experience 05 : correlation intra-script ===\n")
    rng = np.random.default_rng(21)

    # --- SCRIPT 1 : homogene ordonne (motifs de plus en plus longs) ---
    print("--- SCRIPT 1 : trois neurones ordonnes (periodes 2, 4, 8) ---")
    s1 = [neurone_motif("AT"), neurone_motif("ATGC"), neurone_motif("ATGCATCG")]
    r1 = analyse_script(s1, fenetre=24)
    print(f"   enroulement propre : {[round(x,4) for x in r1['refs']]}")
    print(f"   seuil de cassure   : {r1['seuil']:.5f}")
    for f in r1["frontieres"]:
        print(f"   frontiere {f['i']}->{f['i']+1} @pos {f['position']:4d} : "
              f"avant={f['avant']:+.4f} apres={f['apres']:+.4f} ecart={f['ecart']:.4f} "
              f"PLV={f['plv']:.3f}  {'CASSURE' if f['cassure'] else 'continue'} ({f['raison']})")

    # --- SCRIPT 2 : homogene chaotique ---
    print("\n--- SCRIPT 2 : trois neurones aleatoires ---")
    s2 = [neurone_melange(1.0, seed=i) for i in range(3)]
    r2 = analyse_script(s2, fenetre=24)
    print(f"   enroulement propre : {[round(x,4) for x in r2['refs']]}")
    print(f"   seuil de cassure   : {r2['seuil']:.5f}")
    for f in r2["frontieres"]:
        print(f"   frontiere {f['i']}->{f['i']+1} @pos {f['position']:4d} : "
              f"avant={f['avant']:+.4f} apres={f['apres']:+.4f} ecart={f['ecart']:.4f} "
              f"PLV={f['plv']:.3f}  {'CASSURE' if f['cassure'] else 'continue'} ({f['raison']})")

    # --- SCRIPT 3 : MIXTE ordre -> chaos (la transition doit casser) ---
    print("\n--- SCRIPT 3 : ordre pur -> chaos pur (transition forcee) ---")
    s3 = [neurone_motif("ATGC", 120),
          neurone_motif("ATGC", 120),
          neurone_melange(1.0, seed=7)]
    r3 = analyse_script(s3, fenetre=24)
    print(f"   enroulement propre : {[round(x,4) for x in r3['refs']]}")
    print(f"   seuil de cassure   : {r3['seuil']:.5f}")
    for f in r3["frontieres"]:
        print(f"   frontiere {f['i']}->{f['i']+1} @pos {f['position']:4d} : "
              f"avant={f['avant']:+.4f} apres={f['apres']:+.4f} ecart={f['ecart']:.4f} "
              f"PLV={f['plv']:.3f}  {'CASSURE' if f['cassure'] else 'continue'} ({f['raison']})")

    # --- etude : taux d'enroulement selon desordre ---
    print("\n--- Taux d'enroulement global selon le desordre ---")
    print(f"{'p desordre':>11} {'tours':>9} {'taux':>9} {'pas actifs':>11}")
    print("-" * 44)
    for p in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        seq = "".join(
            str(rng.choice(list("ATCG"))) if rng.random() < p else "ATGC"[i % 4]
            for i in range(240))
        from organes.atcg import hydrophobic_signal
        cloud = takens_embed((lambda z: (z - z.mean()) / (z.std() + 1e-12))(hydrophobic_signal(seq)), 3, 3)
        r = taux_enroulement(cloud)
        print(f"{p:>11.1f} {r['tours']:>9.4f} {r['taux']:>9.5f} {r['pas_actifs']:>11d}")