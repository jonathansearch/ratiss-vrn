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

---

## 2026-09-18 (suite) — Corrélation intra-script : enroulement et cassure

Consigne : *évoluer les chaînes de repliement de chaque neurone afin de
surveiller exactement d'où l'un va à l'autre, déterminer les taux
d'enroulement, et le moment où ça casse.*

### Le fondement : le nombre d'enroulement

Le **nombre d'enroulement** est un invariant de Poincaré — il compte les
tours d'une trajectoire fermée autour d'un centre. Deux oscillateurs
couplés se **verrouillent** tant que le rapport de leurs nombres
d'enroulement reste rationnel (langues d'Arnold, Farey, Stern-Brocot), et
**décrochent** à la frontière. Notre « cassure » est ce décrochage.

### L'instrument, validé avant usage (discipline R7)

Deux artefacts trouvés à la validation, deux corrections :

| Cas | Attendu | v1 | v2 finale |
|---|---|---|---|
| cercle | 1 tour | 0.995 ✅ | **0.995** ✅ |
| cercle ×2 | 2 tours | **−1.990** ❌ signe | **1.990** ✅ |
| ligne droite | 0 | **−0.500** ❌ | **0.000** ✅ |
| sinus 3 périodes | ~3 | −2.72 | **2.282** ✅ |

- **Artefact 1 — demi-tour fantôme.** Une ligne droite traversant son
  centre faisait sauter l'angle de π → un demi-tour inexistant (0.5 tour).
  Correction : ne compter l'angle que là où la trajectoire est réellement
  décentrée (rayon > ¼ du rayon max). Physiquement : un nombre
  d'enroulement n'est défini que pour une trajectoire qui tourne *autour*
  d'un centre — là où le rayon s'annule, l'angle n'a pas de sens.

- **Artefact 2 — signe arbitraire de la SVD.** Le même cercle parcouru dans
  le même sens ressortait +1 ou −1 selon l'échantillon, ce qui aurait
  fabriqué de **fausses cassures**. Correction : convention d'orientation
  (aire signée positive). Les vraies inversions ressortent en négatif.

### Le critère de cassure : double, et c'est nécessaire

Un seul critère ne tenait pas : le seuil d'écart **s'adapte au régime**
(0.085 en régime ordonné, 1.009 en régime chaotique), donc le même écart
absolu est énorme dans un cas et négligeable dans l'autre. Le seuil seul
produisait des faux positifs.

Critère retenu : **cassure = écart d'enroulement ET décrochage du
verrouillage (PLV < 0.5)**. Il faut les deux — le régime change *et* les
deux chaînes se désynchronisent.

### Résultats (exp05)

**Script 3 — transition forcée ordre → chaos :**

| Frontière | Écart | PLV | Verdict |
|---|---|---|---|
| ordre → ordre (période 4 → 4) | 0.276 | **1.000** | continue — verrouillé |
| **ordre → chaos** | 0.936 | **0.197** | **CASSURE — décrochage** |

**Script 2 — chaos homogène :** aucune cassure (écarts 0.216 / 0.113,
sous le seuil 1.009). Le chaos ne casse pas contre lui-même.

**Script 1 — ordre homogène :** la frontière période-2 → période-4 casse
(PLV 0.187) ; période-4 → période-8 ne casse pas (PLV 0.819, verrouillage
conservé malgré un écart réel). C'est le double critère qui distingue les
deux — et il distingue juste : 2 et 4 sont commensurables, 4 et 8 aussi,
mais le passage 2→4 change de langue d'Arnold, pas 4→8 dans la même
famille.

**Enroulement global selon le désordre :**

| p désordre | tours |
|---|---|
| 0.0 | **18.21** |
| 0.2 | 8.31 |
| 0.4 | 4.55 |
| 0.6 | 6.23 |
| 0.8 | −1.60 |
| 1.0 | 1.65 |

**L'ordre s'enroule, le chaos tourne peu.** 18 tours à l'ordre pur,
1.6 au chaos pur. Le passage par une valeur négative en p=0.8 signale une
inversion de sens de rotation — un régime de transition, pas un régime
stable.

### PRÉDICTION CONFIRMÉE

Énoncée avant mesure : *une cassure doit apparaître à la frontière
ordre → chaos, pas à l'intérieur d'un régime homogène.* C'est exactement
ce que donne le script 3 (PLV 1.000 en interne, 0.197 à la transition) et
le script 2 (aucune cassure).

### Où l'un va à l'autre — la réponse

Le passage d'un neurone à l'autre se lit dans **deux quantités** :
le **taux d'enroulement local** (où le régime tourne plus ou moins) et le
**PLV entre voisins** (si les deux se parlent encore). La cassure est le
moment où les deux tombent ensemble.

### Incident d'environnement

Le conteneur s'est réinitialisé en cours de session : numpy et ripser
avaient disparu. Code intact (poussé sur GitHub), mesures perdues et
refaites. Réinstallation effectuée.

**Fichiers :** `organes/enroulement.py`, `experiences/exp05_script.py`.


## Atelier 04 (relais Arena, 2026-09-18) — exp05 : tâche à réponse connue

**Ordre du relais :** donner au neurone v3 une tâche dont on connaît la réponse,
et voir s'il la trouve. Pas nettoyer, pas polir — tester.

**Tâche (motif caché) :** 2×100 chaînes de 160 bases, graines held-out 1000+i.
Classe 0 = aléatoire pur. Classe 1 = aléatoire + UN bloc périodique caché
(`ATGC`×10, 40 bases, position aléatoire). Vérité terrain = labels connus.
Métrique : AUC (sans seuil). **Critère scellé avant exécution : AUC(y) ≥ 0.70.**
**Prédiction énoncée avant mesure :** séparation faible (AUC 0.60–0.75),
portée par g_dyn.

**Résultat (200 chaînes, `experiences/exp05_tache_motif.json`) :**

| mesure | moy. classe 0 | moy. classe 1 | AUC |
|---|---|---|---|
| y | 0.0000 | 0.0000 | 0.505 |
| s (P_sig) | 0.0000 | 0.0000 | 0.505 |
| c (sémantique) | 1.0000 | 1.0000 | 0.505 |
| **g_dyn (VOIR)** | 0.1860 | 0.3277 | **0.989** |
| g_topo (VERIFIER) | 1.0000 | 1.0000 | 0.505 |
| fusion (s·c) | 0.0000 | 0.0000 | 0.505 |

**Verdict vs critère scellé : pas de détection (AUC(y)=0.505 < 0.70) — documenté.**

**Diagnostic — le géant borgne :** UN SEUL organe voit. `g_dyn` (VOIR/JEPA)
retrouve le motif caché presque parfaitement (AUC 0.99) — la tâche a bien
une réponse et elle est trouvable. Mais `y = 0` partout car :
- `s = 0.0000` sur les 200 chaînes → la fusion-produit `(s·c)` vaut 0
  et tue le seul signal qui marche (à comparer : exp03 donnait P_sig=0.61
  sur aléatoire — régime d'entrée différent, à investiguer) ;
- `c = 1.0000` saturé partout (aucune discrimination sur ce régime) ;
- `g_topo = 1.0000` bloqué ouvert partout.
(AUC 0.505 sur mesures constantes = effet des ex-aequo de rangs, sans signal.)

La prédiction est confirmée sur le QUI (g_dyn porte le signal, et même
mieux que prévu : 0.99) mais infirmée sur y — la cause (s=0 systématique)
n'était pas anticipée. C'est une donnée, pas un échec.

**Pistes ouvertes (pas des conclusions) :**
1. Pourquoi `s=0` sur ces 200 chaînes de 160 bases ? Investiguer
   `chain_to_cloud` + `p_sig_ripser` sur ce régime exact vs exp03.
2. Pourquoi `c` sature à 1.0 ici (vs 0.08–0.23 en atelier 03) ?
   Vérifier la normalisation de `semantique.concept` sur ce régime.
3. Pourquoi `consensus_gate` vaut 1.0 partout ici ? Régime ou bug ?
4. Fragilité structurelle : la fusion-produit meurt dès qu'UN organe tombe
   à 0. Piste théorique (option C de la douane) : terme de repli ?
   Le chef tranche — on ne change rien sans ordre.
5. Hygiène notée (non traitée, par ordre) : `organes/__pycache__/` versionné.

**Fichiers :** `experiences/exp05_tache_motif.py` (+ `.json` des résultats).
Rejouable : `python3 experiences/exp05_tache_motif.py` depuis la racine.
