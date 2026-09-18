"""Dynamique — VOIR. Test de predictibilite JEPA, sans reseau entraine.

ORDRE : "Pas de reseau entraine. Juste une verification : la prochaine
base/propriete est-elle deductible des N precedentes ?"

DEUX NIVEAUX, et c'est volontaire. Un test unique n'est pas falsifiable :
n'importe quel signal a une autocorrelation non nulle, donc un test unique
dirait toujours "predictible". Il faut donc deux lectures qui se
CONTREDISENT :

  Niveau 1 — g_topo : predictibilite TOPOLOGIQUE (VRAI JEPA).
      On plonge la sequence par Takens, on fait un pas de dynamique
      (x_{t+1} = F(x_t) par plus proche voisin), et on mesure la
      persistance H1 du nuage (etat, etat_predit). Si la prediction
      respecte la forme, P_sig reste eleve ; si elle detruit la forme,
      P_sig s'effondre.

  Niveau 2 — g_lin : predictibilite LINEAIRE (baseline volontairement
      faible, autocorrelation d'ordre 1 sur la sequence ATCG). Elle ne
      regarde pas la forme, juste la memoire statistique.

POURQUOI LES DEUX. Le resultat de exp03 dit : tout signal PERIODIQUE
ferme la porte topologique (P_sig = 0, un cycle unique, pas de
coexistence). Si on n'utilisait que g_topo, le test JEPA "le periodique
est previsible donc j'ouvre" serait contredit. En gardant les deux,
on voit que :

    g_topo et g_lin sont ANTI-CORRELES.

Un signal tres previsible lineairement (periodique) a une topologie
pauvre ; un signal riche topologiquement (aleatoire) est imprevisible
lineairement. Le neurone ne peut donc s'ouvrir que dans un regime
INTERMEDIAIRE — c'est la prediction a tester (exp04).

Consequence theorique : la douane VRN ne recompense ni l'ordre ni le
chaos, mais leur BORD. C'est la criticite, pas la structure.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/vrn")

import numpy as np

from organes.psig import h1_diagram_ripser, takens_embed


def _pers_score(cloud: np.ndarray) -> float:
    """P_sig normalise d'un nuage (0 si moins de 2 cycles)."""
    dgm = h1_diagram_ripser(cloud)
    pers = (dgm[:, 1] - dgm[:, 0]) if len(dgm) else np.array([])
    pers = pers[np.isfinite(pers)]
    pers = pers[pers > 0]
    if pers.size == 0:
        return 0.0
    n = pers.size
    return float(pers.sum() / (n * pers.max()) - 1.0 / n)


def predictibilite_topologique(seq: str, dim: int = 3, delay: int = 3,
                               k: int = 4) -> dict:
    """VRAI JEPA : erreur de prediction d'un pas dans l'espace des phases.

    Correction d'une v1 fausse : la v1 comparait P_sig(reel) et
    P_sig(predit), et trouvait 0.23 PARTOUT (periodique comme aleatoire).
    Elle mesurait donc la stabilite de la forme, pas la predictibilite.

    Ici on mesure l'erreur de prediction elle-meme :
      - voisins locaux dans l'espace des phases (pas de reseau)
      - vitesse moyenne des voisins -> etat predit
      - erreur = ||predit - reel|| / ||pas moyen||
      - g_topo = 1 - erreur
    Deterministe -> erreur ~ 0 -> g_topo ~ 1.
    Bruit pur    -> erreur ~ pas moyen -> g_topo ~ 0.
    """
    x = np.array([{"A": 0.0, "T": 1.0, "C": 2.0, "G": 3.0}[b] for b in seq],
                 dtype=np.float64)
    if x.size < 20:
        return {"g_topo": 0.0, "erreur": None, "psig_reel": 0.0,
                "verdict": "sequence trop courte"}

    cloud = takens_embed(x, dim=dim, delay=delay)
    if len(cloud) < k + 3:
        return {"g_topo": 0.0, "erreur": None, "psig_reel": 0.0,
                "verdict": "nuage trop petit"}

    errs, scales = [], []
    for i in range(len(cloud) - 1):
        p = cloud[i]
        d = np.linalg.norm(cloud[:-1] - p, axis=1)
        d[i] = np.inf
        idx = np.argsort(d)[:k]
        vel = np.mean(cloud[idx + 1] - cloud[idx], axis=0)
        pred = p + vel
        errs.append(np.linalg.norm(cloud[i + 1] - pred))
        scales.append(np.linalg.norm(cloud[i + 1] - p))

    err = float(np.mean(errs))
    scale = float(np.mean(scales)) + 1e-12
    g = float(np.clip(1.0 - err / scale, 0.0, 1.0))
    return {"g_topo": g, "erreur": err / scale, "psig_reel": _pers_score(cloud),
            "verdict": "dynamique deterministe (le neurone voit)" if g > 0.5
                       else "dynamique non deterministe (le neurone se tait)"}


def predictibilite_lineaire(seq: str, max_lag: int = 8) -> dict:
    """Baseline : memoire statistique, autocorrelation sur plusieurs lags.

    Correction : un periodique de periode 4 (ATGC) a une autocorrelation
    NULLE au lag 1. Mesurer un seul lag ratait donc exactement les
    signaux les plus previsibles. On balaie les lags et on garde le max.
    """
    x = np.array([{"A": 0.0, "T": 1.0, "C": 2.0, "G": 3.0}[b] for b in seq],
                 dtype=np.float64)
    if x.size < max_lag + 3 or x.std() < 1e-12:
        return {"g_lin": 0.0, "autocorr": 0.0, "lag": 0}
    best, best_lag = 0.0, 0
    for lag in range(1, max_lag + 1):
        a, b = x[:-lag], x[lag:]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        r = abs(float(np.corrcoef(a, b)[0, 1]))
        if r > best:
            best, best_lag = r, lag
    return {"g_lin": best, "autocorr": best, "lag": best_lag}


def jepa(seq: str) -> dict:
    """Les deux lectures reunies."""
    t = predictibilite_topologique(seq)
    l = predictibilite_lineaire(seq)
    return {
        "g_topo": t["g_topo"],
        "g_lin": l["g_lin"],
        "psig_reel": t["psig_reel"],
        "erreur": t.get("erreur"),
        "verdict_topo": t["verdict"],
    }


if __name__ == "__main__":
    print("=== JEPA : la chaine est-elle dynamiquement vivante ? ===\n")
    rng = np.random.default_rng(11)

    def montre(nom, seq):
        r = jepa(seq)
        print(f"{nom:34} g_topo={r['g_topo']:.4f}  g_lin={r['g_lin']:.4f}  "
              f"P_sig={r['psig_reel']:.4f}")
        print(f"{'':34} -> {r['verdict_topo']}")

    montre("periodique ATGC (ordre pur)", "ATGC" * 40)
    montre("aleatoire", "".join(rng.choice(list("ATCG")) for _ in range(160)))
    montre("motif + bruit", "".join(
        rng.choice(list("ATCG")) if rng.random() < 0.35 else "ATGC"[i % 4]
        for i in range(160)))