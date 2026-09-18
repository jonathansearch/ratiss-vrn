# SPEC-V0 — Neurone VRN (version exploratoire 0)

> Statut : **brouillon ouvert**. Tout paramètre noté `ouvert` est une piste,
> pas une décision. Rien n'est figé, rien n'est conclu.

## 1. Définitions

- `x` : entrée sensorielle (ex. pixels).
- `s = JEPA(x)` : embedding latent, `s ∈ R^d` (ex. d=192, LeWM).
- `P_sig(s)` : signature topologique (persistance H1, ripser), `∈ R+`.
- `g = Γ(P_sig(s))` : interrupteur topologique, `Γ` = fonction `ouverte`.
- `c = concept(s)` : attachement sémantique (RATIS-Net), `ouvert`.
- `y` : sortie du neurone VRN.

## 2. Fonction du neurone (v0)

```
VOIR     :  s = JEPA(x)
VÉRIFIER :  g = Γ(P_sig(s))        # Γ ouverte : seuil ? sigmoïde ? autre ?
NOMMER   :  y = g · (s ⊕ c)        # ⊕ ouverte : concaténation ? somme ? autre ?
```

Règle d'or héritée : **si la forme s'effondre, le neurone se tait** (g → 0).

## 3. Paramètres ouverts

| Paramètre | Rôle | Statut |
|---|---|---|
| `Γ` (interrupteur) | convertit P_sig en gate | ouvert |
| `θ` (seuil éventuel) | niveau de cohérence requis | ouvert |
| `⊕` (fusion) | marie signal et concept | ouvert |
| `concept(.)` | source du sens (grammaire ? faits ? LCT ?) | ouvert |
| `d, n` (dimensions) | tailles latent / cognitif | ouvert |

## 4. Prototype minimal proposé (à explorer, pas à valider)

1. Prendre 1 embedding LeWM réel (checkpoint `quentinll/lewm-pusht`).
2. Mesurer son P_sig (plugin `TopologicalInvariantPlugin`, T ≥ 16).
3. Attacher 1 concept RATIS-Net.
4. Faire tourner VOIR → VÉRIFIER → NOMMER sur cet exemple unique.
5. Observer et documenter ce qui sort — sans conclure.

## 5. Journal d'itération

| Date | Version | Observation |
|---|---|---|
| 2026-09-17 | v0 | Dépôt créé, spec posée. Atelier ouvert. |

---
*RATISS Labs — VRN. Le réseau ne parle pas. Il pense. Et maintenant, il nomme.* 💬
