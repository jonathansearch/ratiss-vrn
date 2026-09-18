# Journal d'atelier VRN

---

## 2026-09-18 — Expérience 01 : KTN woven vs aligned lus par P_sig

**Pont transdisciplinaire testé :**
`Travaux/ratiss_photoinduced/ktn_woven.py` (ferroélectrique KTN:Li)
→ nuage de points 3D
→ `organes/psig.py` (persistance H1, repris de ratiss-neuro + persistence_optimizer)
→ lecture VRN (aligne / se tait).

**Protocole :** 80 points sous-échantillonnés, 3 graines, woven vs aligned.

| Graine | p_sig woven | p_sig aligned | robust_h1 woven | robust_h1 aligned |
|---|---|---|---|---|
| 0 | 0.1956 | 0.1471 | 1178 | 744 |
| 1 | 0.1874 | 0.1226 | 1148 | 539 |
| 2 | 0.2559 | 0.1554 | 1537 | 796 |

**Résultat : le woven est au-dessus de l'aligned 3 fois sur 3.**
Écart p_sig : ×1.33, ×1.53, ×1.65.
Écart robust_h1 : ×1.58, ×2.13, ×1.93.

**Lecture :** P_sig distingue un tissage (brins entrelacés) d'un alignement
(domaines parallèles). La douane VRN a un premier capteur qui marche sur
une vérité terrain externe à l'IA — un matériau ferroélectrique.

**Réserve honnête (pas un verdict) :** `n_barres ≈ 2100` pour 80 points.
Le seuil de filtration (~7.8) connecte presque tout, donc la mesure est
noyée sous des cycles de taille minuscule. La séparation tient, mais elle
est portée par une métrique encore bruitée. Le seuil mérite un travail.

**Ce que ça change :** la corrélation transdisciplinaire n'est plus une
intuition. Un algorithme né sur un cristal lit structurellement un autre
domaine. C'est le principe VRN en action — coupler, pas séparer.

---

## 2026-09-18 — P_sig : découverte de formule

En portant le comptage par Kruskal (`topo_plasticity.py`) pour compter les
cycles H1, échec net : sur un cercle de 40 points, 741 « cycles » annoncés
pour 1 seul trou réel. Le comptage d'arêtes hors-arbre (E − V + 1) n'est
pas un comptage de cycles indépendants.

**Leçon :** P_sig ne se mesure pas en comptant des arêtes, mais en
**réduisant une matrice de bordure**. Coût : le prix de la vérité.

Et dans le code existant de `ratiss-neuro/topology.py` :

```
score = sum(pers) / (n * max(pers)) - 1/n
```

Un cycle unique → score = 0. Le cycle solitaire ne fait pas une signature.
Il faut de la **coexistence**. Jonathan l'a dit à l'oral ; sa formule
l'encodait déjà.

---

## Inventaire des organes disponibles (43 dépôts, 1255 fichiers py)

Algos identifiés comme réutilisables pour la VRN :

| Domaine | Source | Algo |
|---|---|---|
| Topologie | ratiss-neuro/topology.py | P_sig, Takens, H0 sous-niveau |
| Topologie | Ratiss-experimental-IA-/persistence_optimizer.py | réduction de bordure numpy |
| Plasticité | ratiss-neuro/topo_plasticity.py | H1 → masque synapses |
| Neurone | ratiss-experimental-IA-/lct_neuron.py | ΔW = η·φ·P_sig·C |
| Quasiment quantique | ratiss-neuro/topo_qubit.py | qubit topologique logiciel |
| Hamiltonien | RATISS-LABS-GTT/gtt/quantum/lanczos.py | Lanczos + Jacobi, stdlib seule |
| Décohérence | RATISS-LABS-GTT/gtt/quantum/decoherence.py | T1/T2, Von Neumann |
| **KTN** | Travaux/ratiss_photoinduced/ktn_woven.py | tissage vs aligné (vérité terrain) |
| **KTN** | Travaux/ratiss_photoinduced/ssh_model.py | SSH, winding, états de bord |
| **Génomique** | ratiss-bio/bio114/mutagenesis.py | mutagenèse dirigée, Xi_IDP |
| **Génomique** | ratiss-bio/bio114/cascade.py | codon→protéine→organe, repliement |
| **Auto-amélioration** | RATISS-ODV-AEON/orchestrator/auto_improve.py | trajectoire→leçons→harnais |
| Réplique | ratiss-neuro/tryperposition.py | collapse dirigé |

**Fait notable :** `bio114/cascade.py` contient déjà un algorithme de
repliement protéique (CODON_TABLE, HYDROPHOBICITY, protein_folding_coherence)
et une **cascade multi-échelle** où la cohérence d'un étage module l'étage
supérieur. C'est un substrat candidat pour la VRN — sans torch.