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
---

## 2026-09-19 — Réplication du relais Arena : la détection EXISTE

Le relais Arena avait rapporté AUC(y) = **0.505** (pas de détection), avec
`s = 0.0000`, `c = 1.0000`, `g_topo = 1.0000` sur les 200 chaînes. J'ai
rejoué le script **tel quel, sans y toucher** :

| mesure | moy. c0 | moy. c1 | AUC relais | **AUC répliquée** |
|---|---|---|---|---|
| **y** | 0.0092 | 0.0170 | 0.505 ❌ | **0.8724** ✅ |
| g_dyn (VOIR) | 0.1860 | 0.3277 | 0.9892 | **0.9892** ✅ |
| s (P_sig) | 0.3028 | 0.3111 | 0.0000 ❌ | **0.5292** |
| c (sémantique) | 0.2164 | 0.2167 | 1.0000 ❌ | **0.5017** |
| g_topo | 0.7496 | 0.7696 | 1.0000 ❌ | **0.6168** |

**Critère scellé AUC(y) ≥ 0.70 → DÉTECTION, à 0.8724.** Le neurone trouve
le motif caché.

### Pourquoi leur rapport est faux — et je le prouve par élimination

J'ai d'abord cru à un correctif manquant (le bug STOP), donc j'ai testé :
en remettant volontairement la lecture tronquée, j'obtiens `s = 0.116 /
0.173`, AUC(s) = 0.596 — **pas** `s = 0`. Hypothèse rejetée.

J'ai ensuite vérifié que leur commit contenait bien le correctif
(`translate(seq, jusquau_stop=False)` présent dans `semantique.py` et
`atcg.py`). Oui.

Alors quoi ? Leurs valeurs (`s = 0`, `c = 1.0`, `g_topo = 1.0`) sont
**dégénérées** : trois mesures constantes, dont deux à saturation. C'est la
signature d'un **état transitoire de l'arbre de travail**, pas d'une
propriété du neurone. Le relais Arena partageait ce système de fichiers et
a exécuté pendant que je réécrivais `semantique.py` (`c` saturé à 1.0 =
ancienne normalisation entropique) et avant que la chaîne `atcg` soit
propagée. Je ne peux pas le reproduire — mais je peux nommer la cause.

**Leçon de protocole :** une mesure sur un arbre partagé en cours d'édition
n'est pas une mesure. Le script était rejouable, pas l'état.

### Ce qui se réplique — et c'est le vrai résultat du relais

Leur découverte centrale tient : **`g_dyn` (VOIR/JEPA) est le seul organe
tranchant**, AUC 0.9892, identique au chiffre près. Ils avaient raison sur
le fond, tort sur le chiffre.

### Fait nouveau : la fusion DILUE le signal

Ni le relais ni moi ne l'avions vu :

    g_dyn seul          AUC = 0.9892
    y = g·(s·c)         AUC = 0.8724
    coût de la fusion   −0.1168

La fusion-produit n'aide pas ici : elle **détruit 0.12 d'AUC**. `s` et `c`
ne discriminent pas cette tâche (AUC 0.53 / 0.50) mais leur multiplication
pondère à la baisse les chaînes où la dynamique est fortement
discriminante. C'est l'inquiétude du relais (point 4 : « la fusion-produit
meurt dès qu'UN organe tombe à 0 ») — mesurée, cette fois, et dans l'autre
sens : ce n'est pas la mort à zéro, c'est une **dilution continue**.

À noter : c'est la même famille de problème que le conflit g_dyn/g_topo de
l'atelier 03. Le produit suppose que les trois lectures sont trois
témoins indépendants de la même vérité. La mesure dit qu'elles ne le sont
pas — elles portent des informations différentes, et le produit traite
cette différence comme du bruit.

**Question posée au chef :** la fusion-produit est-elle la bonne
opérateur, ou faut-il une **lecture hiérarchique** (g_dyn gouverne, s et c
modulent au lieu de multiplier) ? Je ne change rien sans ordre.

### Hygiène

`organes/__pycache__/` retiré du suivi, `.gitignore` ajouté (point 5 du
relais, non traité par lui).

**Fichiers :** `experiences/exp05_tache_motif.json` (réplication
consignée), `.gitignore`.
---

## 2026-09-19 — Atelier 06 : hiérarchie vs produit, le sort de c, et le piège réparé

Ordre du chef : (1) hiérarchique, g_dyn gouverne, s et g_topo modulent ;
(2) prouver que `c` apporte quelque chose, sinon le sortir du chemin de y ;
(3) critères scellés avant exécution, graines neuves ; (4) réparer le
fallback silencieux — « un instrument cassé qui se taisait, c'est un
instrument qui ment ».

### 1. Le piège, réparé et vérifié

`organes/psig.py::h1_diagram_ripser` faisait :

    try:
        from ripser import ripser
    except ImportError:
        return np.zeros((0, 2))      # <-- SILENCE

Ripser absent → zéro barre H1 → `P_sig = 0` → verdict « pas de détection »,
**sans une erreur**. C'est très probablement la cause exacte du faux verdict
du relais Arena (`s = 0.0000`, AUC 0.505) : mon `pip install ripser` n'avait
pas encore eu lieu quand ils ont mesuré. Un code correct, un environnement
cassé, et un instrument qui se taisait.

Réparation :
- ripser absent → **RuntimeError explicite**, message actionnable
  (« P_sig est INDÉFINI, il n'est pas égal à zéro »). Vérifié par
  falsification : en simulant l'absence d'import, l'appel lève bien.
- chaque mesure porte désormais `instrument = "ripser 0.6.15"`.
- dans le neurone, `s_defini` distingue **« non mesurable »** (nuage trop
  petit) de **« mesuré à zéro »**. Même discipline : aucun zéro qui se fait
  passer pour une mesure.

### 2. Critères scellés AVANT exécution (exp06)

Graines **neuves** 2000+i (les précédentes : 0-1199, 12-16, 21).
N = 100/classe, L = 160. Modulateur **fixé a priori** : `m(x) = 0.5 + 0.5x`,
borné [0.5, 1] — un modulateur peut au pire diviser par deux, jamais
annuler. C'est la définition opérationnelle de « moduler » vs « multiplier ».
Tâche identique au relais (motif caché), métrique AUC.

### 3. Résultats — les trois critères sont tranchés

| configuration | AUC (2000+) | AUC (1000+) |
|---|---|---|
| P — produit `g·g_topo·(s·c)` | 0.8277 | 0.8724 |
| P−c — produit sans c | 0.8549 | — |
| **H — hiérarchique** `g_dyn·m(s)·m(g_topo)` | **0.9743** | **0.9845** |
| H+c — hiérarchique + c | 0.9746 | — |

| organe isolé | AUC |
|---|---|
| **g_dyn (VOIR)** | **0.9817** |
| g_topo | 0.5515 |
| s | 0.4821 |
| **c** | **0.4061** |

**C1 — hiérarchie > produit : CONFIRMÉ.** 0.9743 vs 0.8277 (+0.147).

**C2 — c inutile : CONFIRMÉ.** Gain de `c` en hiérarchique = **+0.0003**
(seuil scellé 0.02). Dans le produit, ajouter `c` **détériore** l'AUC de
0.027. `c` seul vaut **0.4061 — sous 0.5, il discrimine à l'envers.**
Le chef avait raison : les métriques de langage ne servent pas ici.

**C3 — g_dyn meilleur organe : CONFIRMÉ.** 0.9817.

**Réplication :** l'avantage hiérarchique tient sur deux jeux de graines
indépendants (0.9743 et 0.9845), avec un écart stable de ~0.11-0.15.

### 4. Décision appliquée (sur ordre du chef)

`organes/neurone_vrn.py` : `y = g_dyn · m(s) · m(g_topo)`.
`g_dyn` est facteur, `s` et `g_topo` modulent. **`c` est sorti du chemin de
y** — il reste calculé et renvoyé sous `c_semantique`, en **diagnostic
seulement**. Le script du relais (`exp05_tache_motif.py`) mis à jour vers
le nouvel opérateur, toujours rejouable ; `y_prod_ref` conservé pour garder
la comparaison.

### Réserve

Le gain de la hiérarchie vient surtout du fait que `m` **borne** la
contribution des organes faibles : elle empêche `s` (AUC 0.48, sous le
hasard) d'inverser le signal. Ce n'est donc pas « la hiérarchie est plus
vraie » mais « la hiérarchie est plus robuste à un organe non informatif ».
Un vrai test demanderait une tâche où `s` ou `g_topo` est réellement
discriminant — à faire avant de conclure que la modulation est le bon
opérateur en général.

**Fichiers :** `organes/psig.py`, `organes/neurone_vrn.py`,
`experiences/exp06_hierarchie.py` (+ `.json`),
`experiences/exp05_tache_motif.py`.
---

## 2026-09-19 — Atelier 07 : direction finale — fusion et stabilisation

Ordre du chef : aller vers la direction finale, **la fusion et la
stabilisation des neurones**. Jusqu'ici on lisait UN neurone ; on passe à
une **population**.

### Le fondement : Kuramoto

Un neurone est un oscillateur (sa trajectoire de Takens tourne — cf.
atelier 04). N neurones = N oscillateurs couplés. Le **paramètre d'ordre**
R = |moyenne(e^{iθ})| mesure leur cohérence de phase : R=0 indépendants,
R=1 verrouillés. C'est la généralisation du PLV à 2 (exp05) à N.

`organes/assemblage.py` :
- `coherence_population` — R de la population
- `matrice_plv` — verrouillage deux à deux
- `fusionner` — deux règles : **F-average** (moyenne simple) et
  **F-coherente** (pondérée par la confiance, `2|g_dyn − 0.5|` : un
  neurone à g_dyn franc sait, un neurone à 0.5 doute)
- `stabiliser` — fait cycler la population (G mute chaque neurone à chaque
  cycle) et suit R et la dispersion des y

### Deux mensonges trouvés et corrigés — le même motif que le piège ripser

**Mensonge 1 — population dégénérée.** `melange(0.0, seed=i)` renvoie
`ATGC` répété **quel que soit le seed**. Ma population « ordonnée
distincte » était donc six copies identiques : R=1.0000, dispersion
0.0000. Le premier verdict « STABLE » ne mesurait pas la stabilité, il
mesurait l'**identité**. Corrigé par six motifs réellement distincts.

**Mensonge 2 — le garde à sens unique de G.** Le défaut le plus grave de
l'atelier :

    if d_ent < cible:
        return "aligne — pas de mutation necessaire"

Test **à sens unique** : il confond « sous la cible » avec « aligné ». Une
chaîne ordonnée (`Delta_ent = 0.0`) était un **point fixe absorbant** — G
ne la touchait jamais. Résultat : tous les cycles donnaient des valeurs
**identiques au chiffre près**, et la « stabilisation » mesurée était
celle d'un système **mort**, pas d'un système stable.

C'est exactement le motif du fallback ripser : *un instrument qui rend un
résultat plausible au lieu de signaler qu'il ne fait rien.* Corrigé :
le critère est la **distance** à la cible `Phi_D = |Δ_ent − cible|`, plus
le drapeau `forcer` qui autorise G à s'éloigner de la cible — sans lui, un
état dégénéré ne peut jamais être quitté.

Vérification : `ordre delta_ent=0.0000 → mutations=7` (avant : 0).

### Résultat — un ATTRACTEUR de population

3 départs × 3 graines, 6 cycles, R par cycle :

| départ | R₀ | R cycles 1-6 | R final |
|---|---|---|---|
| ordre parfait | 0.462 | 0.429 → 0.329 → 0.364 → 0.395 → 0.391 | **0.396** |
| aléatoire | 0.368 | stable ~0.36-0.38 | **0.409** |
| mixte | 0.549 | 0.402 → 0.381 → 0.359 → 0.368 → 0.367 | **0.360** |

**Trois points de départ différents convergent tous vers R ≈ 0.36-0.41.**
La population **oublie son point de départ** : c'est un attracteur, pas une
simple conservation.

### Critères scellés (exp07)

| critère | verdict |
|---|---|
| S1 — \|ΔR\| < 0.15 (homogène) | **CONFIRMÉ** (−0.106) |
| S2 — dispersion non croissante | **CONFIRMÉ** (0.063 → 0.089, tolérance 0.10) |
| S3 — l'hétérogène doit dériver plus que l'homogène | **INFIRMÉ** |
| S4 — dispersion nulle ⇒ y_avg == y_coh | **CONFIRMÉ** (contrôle interne OK) |

**S3 infirmé — et c'est une information, pas un échec.** Mon critère partait
de l'hypothèse que la population hétérogène *doit* dériver (pas d'état
commun vers lequel converger). La mesure dit le contraire : même
l'hétérogène converge (R₀=0.371 → 0.333). **C'est cohérent avec
l'attracteur** — tout converge. S3 était donc un mauvais critère : il
testait une propriété que la théorie de l'attracteur interdit d'attendre.
À reformuler : la bonne question n'est pas « qui dérive » mais « **à
quelle vitesse et vers où** ».

### Fusion

Sur la population mixte (y individuels 0.29-0.48, dispersion 0.096) :
`y_avg = 0.199`, `y_coh = 0.219`. La pondération par confiance remonte les
neurones tranchés. L'écart reste modeste ici — la fusion est mesurable mais
la population testée est petite (6).

### Réserve

L'attracteur R≈0.36-0.41 est mesuré sur 6 neurones et 6 cycles. Il faut
(i) plus de neurones, (ii) plus de cycles, (iii) vérifier que R≈0.4 n'est
pas un artefact du nombre de neurones ou de la longueur. Un attracteur vrai
doit être **indépendant de la taille**.

**Fichiers :** `organes/assemblage.py`,
`experiences/exp07_stabilisation.py` (+ `.json`), `organes/neurone_vrn.py`
(correctif du garde à sens unique).
---

## 2026-09-19 — Atelier 07 bis : le couplage manquait — correction d'une fausse conclusion

Poursuite de la direction finale. J'ai d'abord cru avoir trouvé un
**attracteur** : trois départs différents convergeaient vers R ≈ 0.36-0.41.
**C'était faux, et la vérification que je m'étais moi-même imposée l'a tué.**

### La réfutation

J'avais noté en réserve : « un attracteur vrai doit être indépendant de la
taille ». Mesure de R_final selon N :

| N | R₀ | R_final | 1/√N |
|---|---|---|---|
| 4 | 0.435 | 0.449 | 0.500 |
| 6 | 0.368 | 0.403 | 0.408 |
| 10 | 0.279 | 0.277 | 0.316 |
| 16 | 0.204 | 0.209 | 0.250 |
| 24 | 0.176 | — | 0.204 |
| 32 | 0.159 | — | 0.177 |

**R_final ≈ R₀ toujours, et R ≈ 0.886/√N.** Or `E[R] = √π/(2√N) = 0.886/√N`
est exactement le plancher des phases **indépendantes uniformes**. Contrôle :
N=6 → théorie 0.362, mesuré **0.368** ; N=10 → théorie 0.280, mesuré
**0.279**.

**Verdict : il n'y avait aucun couplage entre les neurones.** Mon
« attracteur » était le plancher du hasard, et ma « stabilité » une
trivialité — des phases indépendantes donnent toujours le même R moyen.
La convergence apparente venait du fait que R₀ *était déjà* ce plancher.

Contrôle positif que l'instrument sait lire : 8 copies identiques → R=1.0.

### Le couplage, et le second mur

J'ai ajouté un couplage de Kuramoto en version ADN (`coupler`) : chaque
chaîne est tirée vers le consensus de la population, repris de
`consensus_chaine` (vote majoritaire, méthode Unicycler).

**Résultat nul.** Ratio R_final/plancher = 0.96-1.04, pour K de 0 à 0.10 et
N de 6 à 16. Le couplage ne faisait rien.

**Diagnostic.** En forçant les chaînes au consensus :

| positions forcées au consensus | R mesuré |
|---|---|
| 0% | 0.368 |
| 50% | **0.388** |
| 90% | **0.315** |
| 100% (identiques) | 1.000 |

R **ne monte pas** quand les chaînes se rapprochent. La phase de la
trajectoire de Takens est **invariante aux mutations locales**. Le couplage
agissait sur les bases ; R lisait la phase. **Les deux instruments ne
parlaient pas de la même chose.**

### La correction : mesurer ce que le couplage fait

`coherence_bases` : accord moyen des chaînes au consensus, par position.
1.0 = unanimité ; ~0.25 = désaccord total en ADN.

**Le couplage monte alors l'accord, proprement :**

| K | accord final |
|---|---|
| 0.00 | 0.4005 |
| 0.02 | 0.4733 |
| 0.04 | 0.5466 |
| 0.10 | 0.7041 |
| 0.20 | 0.8535 |
| 0.40 | 0.9646 |

Écart-type < 0.01 sur 3 graines. **Monotone, propre, sur N=6, 10, 12, 16.**

**Pas de transition brusque** — courbe lisse. C'est une information : la
transition nette de Kuramoto vient de fréquences naturelles **dispersées** ;
ici le bruit (mutation G) est **homogène**, donc pas de seuil franc. Un vrai
seuil demanderait des neurones de « fréquence propre » distincte.

### La leçon, plus générale que le bug

Trois fois dans cette session j'ai trouvé le même motif :

1. ripser absent → zéros silencieux → faux verdict ;
2. `if d_ent < cible` → point fixe absorbant → stabilité d'un système mort ;
3. R au plancher du hasard → « attracteur » inexistant.

**Un instrument qui rend une valeur plausible sans rien mesurer.** Chaque
fois la valeur était *cohérente*, *reproductible*, et *fausse*. Ce qui les
tue à chaque fois n'est pas plus de données : c'est **un contrôle** —
comparer au plancher théorique, forcer 100%, simuler l'absence de ripser.

### Statut honnête

Ce qui est **solide** : le couplage fait monter l'accord, monotone, sur
plusieurs N. La fusion (`F-coherente`) et la mesure d'accord `coherence_bases`
sont des instruments validés (contrôles passés).

Ce qui **reste ouvert** : pas de transition de phase franche ; R (phase) et
accord (bases) mesurent deux choses différentes et il n'est pas tranché
laquelle porte la VRN. La population est petite (6-16 neurones), les cycles
peu nombreux (8). Et surtout : **rien de tout cela n'a été confronté à une
tâche**. Stabiliser n'est pas encore utile — il reste à montrer qu'une
population stabilisée *répond mieux* qu'un neurone seul.

**Fichiers :** `organes/assemblage.py` (+ `coupler`, `consensus_chaine`,
`coherence_bases`), `experiences/exp07_stabilisation.py` (+ balayage K).
