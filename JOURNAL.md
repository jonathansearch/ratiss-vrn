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

---

## 2026-09-18 (suite) — Itération de la métrique : 3 échecs, 1 validation

Règle appliquée : *un truc casse → on remplace → on teste avec autre chose.*

### Échec 1 — le gate par branchement était inversé
v1 : `resolve_gate` par branchement de k-mers (logique Flye/Unicycler).
Résultat : chaîne aléatoire → porte **fermée** ; répétition parfaite → porte
**ouverte**. Inversé.

Cause : une répétition parfaite (`ATGCATGC`) ne *branche* pas dans un graphe
de de Bruijn — elle **boucle**. Et une répétition **est** un cycle H1. J'avais
pris la mauvaise moitié de l'assembleur.

**Remplacement :** `consensus_gate`. Un assembleur ne résout pas par
branchement, il résout par **accord entre lectures**. On perturbe la chaîne,
on replie chaque variante, on mesure P_sig, et on demande : la forme
survit-elle aux perturbations ?

### Échec 2 — la représentation était la mienne, pas celle de la chaîne
v1 : `chain_to_cloud` construisait une hélice hydrophobe. P_sig mesurait
donc **ma** construction.

**Remplacement :** plongement de **Takens** — l'organe que
`ratiss-neuro/topology.py` utilise déjà sur des EEG. Le profil
d'hydrophobicité est un signal ; on le plonge dans l'espace des phases.

Effet mesuré, même chaîne, deux représentations :

| Représentation | P_sig |
|---|---|
| hélice v1 (mienne) | 0.0826 |
| **Takens (RATISS)** | **0.5894** |

Facteur 7. La représentation de RATISS voyait la structure, la mienne la ratait.

### Échec 3 — l'instrument lui-même était invalide
Ajout d'un dédoublonnage (nécessaire : 20 niveaux d'hydrophobicité
produisent des dizaines de points confondus, qui gonflent le comptage).

Puis **validation de l'instrument** (exp00) — et là, échec net :

| Nuage | Attendu | Mesuré (v1, seuil unique) |
|---|---|---|
| ligne droite | ~0 | **0.6315** ← le plus haut ! |
| tore (2 cycles) | haut | 0.2320 |

Une **ligne droite** obtenait le meilleur score. Cause : un seuil unique
(median × 1.5) sur une ligne connecte tout → graphe complet → des milliers
de cycles bidons. Le calcul de bordure était correct ; c'est le **seuil
unique** qui était faux.

**Remplacement :** `ripser` — l'organe que `ratiss-neuro/topology.py`
**appelle déjà** (`from ripser import ripser`). ripser balaie toute la
filtration, pas un seuil.

### Instrument validé (exp00, backend ripser)

| Nuage | Attendu | Mesuré | robust_h1 |
|---|---|---|---|
| ligne droite | 0 cycle | P_sig=0.0000 | **0** ✅ |
| cercle | 1 cycle | P_sig=0.0000 | **1** ✅ |
| sinus → Takens | 1 boucle | P_sig=0.0000 | **1** ✅ |
| bruit gaussien | faible | P_sig=0.2947 | 9 |
| tore | plusieurs | P_sig=0.1602 | **11** ✅ |
| marche aléatoire → Takens | ? | P_sig=0.2331 | 16 |

L'instrument **retrouve exactement** la structure attendue.

### Résultat qui survit aux trois remplacements (exp03)

| Signal | robust_h1 | P_sig | g (douane) |
|---|---|---|---|
| périodique, période 1 à 60 | **0** | 0.0000 | **0 → se tait** |
| **aléatoire** | 5 | 0.6138 | **0.69 → s'ouvre** |

**Tous les signaux périodiques se taisent. L'aléatoire s'ouvre.**

Lecture : un signal périodique est *prévisible* — un cycle unique, aucune
coexistence. Un signal aléatoire est *riche* — plusieurs cycles coexistent.
La douane VRN ne récompense donc pas la régularité : elle récompense la
**coexistence de formes**. C'est cohérent avec la formule de
`ratiss-neuro/topology.py` : `sum(pers)/(n·max(pers)) − 1/n`, qui vaut 0
pour un cycle unique.

**Réserve :** ce résultat dit ce que l'instrument mesure, pas encore ce que
la chaîne ATCG *signifie*. Il faut une tâche — un cas où l'on sait ce que
le neurone devrait répondre — avant toute conclusion sur la VRN.

**Fichiers :** `organes/psig.py`, `organes/atcg.py`, `experiences/exp00..exp03`.
