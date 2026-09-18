"""P_sig — signature topologique RATISS, numpy pur.

Organes repris des depots RATISS (rien n'est reecrit de zero) :

  - compute_persistence_cpu  : repris de
      Ratiss-experimental-IA-/ratis_net/persistence_optimizer.py
      (H0 + H1 par filtration Rips et reduction de bordure, numpy vectorise)
  - persistence_score        : repris de ratiss-neuro/ratiss_neuro/topology.py
      (score de coexistence de cycles)
  - robust_h1                : repris du rapport
      ratiss-lewm-integration/reports/psig_sensitivity_report.md
      (barres H1 depassant 15% de la duree de vie maximale)

DECOUVERTE DE L'ATELIER (2026-09-18) :

  Le portage du comptage par Kruskal (topo_plasticity.py) a echoue : il
  compte les aretes hors-arbre (E - V + 1), pas les cycles H1 independants.
  Sur un cercle de 40 points il annonce 741 "cycles" pour 1 seul trou reel.

  => Un NaN d'information. Il faut la REDUCTION DE BORDURE pour obtenir les
  vrais couples (naissance, mort). C'est ce que fait persistence_optimizer.

  Enseignement : P_sig ne se mesure pas en comptant des aretes. Il se mesure
  en reduisant une matrice de bordure. Le cout est le prix de la verite.

POURQUOI PLUSIEURS CYCLES SONT NECESSAIRES :

    score = sum(pers) / (n * max(pers)) - 1/n

  Un cycle unique -> score = 0. Le tore (2 cycles) -> 0.5. Il faut de la
  coexistence. P_sig n'est pas "y a-t-il un trou" mais "combien de trous
  tiennent ensemble".
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# H0 + H1 : filtration Rips + reduction de bordure
#   repris de persistence_optimizer.py::compute_persistence_cpu
# --------------------------------------------------------------------------

def compute_persistence_cpu(points: np.ndarray, max_edge: float) -> dict:
    """Diagrammes de persistance {0: [[b,d],...], 1: [[b,d],...]}.

    Rips : aretes sous max_edge, triangles sous max_edge.
    H0 par union-find, H1 par reduction de la matrice de bordure.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        return {0: [[0.0, float("inf")] for _ in range(n)], 1: []}

    diff = points[:, None, :] - points[None, :, :]
    D = np.linalg.norm(diff, axis=2)

    iu, ju = np.triu_indices(n, k=1)
    dists = D[iu, ju]
    keep = dists <= max_edge
    ei, ej, ed = iu[keep], ju[keep], dists[keep]
    order = np.argsort(ed)
    ei, ej, ed = ei[order], ej[order], ed[order]
    edges = list(zip(ed.tolist(), ei.tolist(), ej.tolist()))

    diagrams = {0: [], 1: []}

    parent = list(range(n))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    components = n
    for d, i, j in edges:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
            components -= 1
            diagrams[0].append([0.0, d])
    for _ in range(components):
        diagrams[0].append([0.0, float("inf")])

    # --- H1 : triangles puis reduction de bordure ---
    edge_set = {(int(a), int(b)): d for d, a, b in edges}
    edge_index = {(int(a), int(b)): k for k, (_, a, b) in enumerate(edges)}
    A = np.zeros((n, n), dtype=bool)
    A[ei, ej] = True
    A[ej, ei] = True
    np.fill_diagonal(A, False)

    triangles = []
    for d_ab, a, b in edges:
        a, b = int(a), int(b)
        for c in np.where(A[a] & A[b])[0]:
            c = int(c)
            if c > b:
                d_ac = edge_set[(a, c)]
                d_bc = edge_set[(min(b, c), max(b, c))]
                triangles.append((max(d_ab, d_ac, d_bc), a, b, c))
    triangles.sort(key=lambda t: t[0])

    edge_order = [(d, i, j) for d, i, j in edges]
    low_marker: dict[int, set[int]] = {}
    pairs = []
    for d_tri, a, b, c in triangles:
        bnd = [edge_index[(min(a, b), max(a, b))],
               edge_index[(min(a, c), max(a, c))],
               edge_index[(min(b, c), max(b, c))]]
        reduced = set(bnd)
        while True:
            present = sorted(reduced)
            if not present:
                break
            low = present[0]
            if low in low_marker:
                reduced = reduced.symmetric_difference(low_marker[low])
            else:
                low_marker[low] = reduced
                birth = edge_order[low][0]
                if d_tri > birth + 1e-9:
                    pairs.append((birth, d_tri))
                break

    for birth, death in pairs:
        diagrams[1].append([float(birth), float(death)])
    killed: set[int] = set()
    for reduced in low_marker.values():
        killed |= reduced
    for k, (d, i, j) in enumerate(edge_order):
        if k not in killed:
            diagrams[1].append([float(d), float("inf")])
    return diagrams


# --------------------------------------------------------------------------
# les mesures de decision
# --------------------------------------------------------------------------

def _finite_lifetimes(dgm_h1) -> np.ndarray:
    if len(dgm_h1) == 0:
        return np.array([])
    d = np.asarray(dgm_h1, dtype=np.float64)
    pers = d[:, 1] - d[:, 0]
    pers = pers[np.isfinite(pers)]
    return pers[pers > 0]


def persistence_score(lifetimes: np.ndarray) -> float:
    """Coexistence de cycles. Repris de ratiss-neuro/topology.py.

        sum(pers) / (n * max(pers)) - 1/n

    1 cycle -> 0.0 ; n cycles egaux -> 1 - 1/n.
    """
    pers = np.asarray(lifetimes, dtype=np.float64)
    pers = pers[np.isfinite(pers)]
    pers = pers[pers > 0]
    if pers.size == 0:
        return 0.0
    n = pers.size
    m = pers.max()
    return float(np.clip(pers.sum() / (n * m) - 1.0 / n, 0.0, 1.0))


def robust_h1(dgm_h1, ratio: float = 0.15) -> int:
    """Barres H1 depassant `ratio` de la duree de vie maximale.

    Repris de psig_sensitivity_report.md : le comptage brut H1 est sensible
    au bruit, la decision doit utiliser la persistance des barres.
    """
    pers = _finite_lifetimes(dgm_h1)
    if pers.size == 0:
        return 0
    return int(np.sum(pers >= ratio * pers.max()))


# --------------------------------------------------------------------------
# P_sig complet
# --------------------------------------------------------------------------

def p_sig(cloud: np.ndarray, max_edge: float | None = None) -> dict:
    cloud = np.asarray(cloud, dtype=np.float64)
    n = len(cloud)
    if max_edge is None:
        D = np.linalg.norm(cloud[:, None, :] - cloud[None, :, :], axis=2)
        off = D[~np.eye(n, dtype=bool)]
        max_edge = float(np.median(off)) * 1.5

    dgms = compute_persistence_cpu(cloud, max_edge)
    pers = _finite_lifetimes(dgms[1])
    return {
        "p_sig": persistence_score(pers),
        "robust_h1": robust_h1(dgms[1]),
        "n_h1_bars": int(pers.size),
        "max_lifetime": float(pers.max()) if pers.size else 0.0,
        "sum_lifetime": float(pers.sum()) if pers.size else 0.0,
        "max_edge": max_edge,
        "dgms_h1": dgms[1],
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    print("=== P_sig sur nuages de controle (reduction de bordure) ===\n")

    def montre(nom, cloud, note=""):
        r = p_sig(cloud)
        print(f"{nom:26} coexistence={r['p_sig']:.4f}  robust_h1={r['robust_h1']:>3}  "
              f"n_barres={r['n_h1_bars']:>3}  max_vie={r['max_lifetime']:.3f}  {note}")

    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    montre("cercle (1 trou)", np.column_stack([np.cos(t), np.sin(t)]), "<- 1 cycle = 0")

    a, b = np.meshgrid(np.linspace(0, 2 * np.pi, 12, endpoint=False),
                       np.linspace(0, 2 * np.pi, 6, endpoint=False))
    R, r2 = 1.0, 0.35
    tore = np.column_stack([(R + r2 * np.cos(b.ravel())) * np.cos(a.ravel()),
                            (R + r2 * np.cos(b.ravel())) * np.sin(a.ravel()),
                            r2 * np.sin(b.ravel())])
    montre("tore (2 trous)", tore, "<- 2 cycles doit s'allumer")

    montre("bruit gaussien", rng.normal(0, 1, (60, 3)), "<- bruit = beaucoup de petits")

    montre("deux cercles disjoints", np.vstack([
        np.column_stack([np.cos(t), np.sin(t)]) * 0.5 + np.array([-2, 0]),
        np.column_stack([np.cos(t), np.sin(t)]) * 0.5 + np.array([2, 0])]), "<- 2 trous")
