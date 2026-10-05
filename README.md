# 🧠⚡ RATISS-VRN — Neural Virtual Reality

> [!NOTE]
> ## 🌱 Open exploration — nothing is frozen
> This repository is a workshop, not a verdict. Every idea is a track to explore,
> every measurement an observation to document. **No conclusion — neither positive
> nor negative — is sealed here.** Permanent iteration.

**VRN** unifies three types of neurons into a new type:
the neuron that **sees**, **verifies** and **names** — the atom of a neural
virtual reality, an inner world where AI thinks what it sees.

## The three origin neurons

| Neuron | What it does | RATISS organ |
|---|---|---|
| 👁️ **JEPA** | SEES — predicts the next state (pixels → latent) | [ratiss-lewm-integration](https://github.com/jonathansearch/ratiss-lewm-integration) |
| 🕸️ **Topological** | VERIFIES — measures the coherence of the shape (P_sig) | [ratiss-neuro](https://github.com/jonathansearch/ratiss-neuro) · [ratiss-Skynet](https://github.com/jonathansearch/ratiss-Skynet) |
| 💬 **Semantic** | NAMES — attaches meaning, the concept | [Ratiss-experimental-IA-](https://github.com/jonathansearch/Ratiss-experimental-IA-) (RATIS-Net) |

## The VRN neuron — operation in 3 steps (v0)

1. **SEE** — `s = JEPA(pixels)`: the raw perceived signal.
2. **VERIFY** — `g = switch(P_sig(s))`: is the shape coherent?
   If not, the neuron stays silent (golden rule inherited from Skynet/Fusion).
3. **NAME** — `y = verified signal + concept`: the output carries the meaning and its proof.

Starting equation (v0, to explore):

```
y = g · (s ⊕ concept(s))    where g = topological switch
```

## Exploration tracks (open)

- [ ] What shape for `g`: hard threshold, sigmoid, other?
- [ ] How does `concept(s)` dialogue with RATIS-Net (grammar, facts, LCT)?
- [ ] First minimal coupling: 1 LeWM embedding + 1 P_sig + 1 concept → observe.
- [ ] VRN as a space: what becomes a network of connected VRN neurons?
- [ ] Bridge with [RATISS-FUSION v2](https://github.com/jonathansearch/Ratiss-Fusion-stark-):
      the VRN neuron as a graft of the complete being.

See [`SPEC-V0.md`](SPEC-V0.md) for the formal v0 specification.

## Workshop principles

- **Permanent iteration** — every version is a draft of the next one.
- **Transdisciplinarity** — topology, semantics, prediction: couple, don't separate.
- **Demonstration by functioning** — we couple, we observe, we document.

---
**RATISS Labs** — *JEPA sees. Topology verifies. Semantics names. VRN unifies.* 🌌

Set in motion by **Jonathan Evina** (night of Sept. 17, 2026), written with the assistance of an AI agent.
