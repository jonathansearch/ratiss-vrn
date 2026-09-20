"""Enroulement — correlation intra-script entre chaines repliees.

CONSIGNE : evoluer les chaines de repliement de chaque neurone afin de
surveiller exactement d'ou l'un va a l'autre, determiner les taux
d'enroulement, et le moment ou ca casse.

FONDEMENT PHYSIQUE (correlation transdisciplinaire) :
  Le NOMBRE D'ENROULEMENT est un invariant de Poincare. Pour une
  trajectoire fermee autour d'un centre, il compte les tours. Deux
  oscillateurs couples se VERROUILLENT tant que le rapport de leurs
  nombres d'enroulement reste rationnel (langues d'Arnold, arbre de
  Stern-Brocot, suites de Farey) et DECROCHENT a la frontiere.

  Une chaine ATCG repliee trace une trajectoire dans l'espace des phases
  (Takens). Chaque neurone a donc un nombre d'enroulement. Un SCRIPT est
  une suite de neurones. "Ou l'un va a l'autre" = le passage d'un
  enroulement au suivant. "Le moment ou ca casse" = le decrochage.

DEUX MESURES, validees avant usage (discipline R7) :

  taux_enroulement(cloud) : tours de la trajectoire autour de son centre,
      par point. Calcule par projection PCA sur le plan principal, puis
      angle cumule deroule.

  cassure(script) : on concatene les signaux des neurones, on plonge,
      on mesure le taux d'enroulement LOCAL par fenetre glissante, et on
      compare au taux propre a chaque neurone. Une cassure est un point
      ou l'enroulement local quitte la loi de son propre neurone.

  plv(a, b) : phase-locking value entre deux trajectoires — la mesure
      standard du verrouillage entre deux oscillateurs.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/vrn")

import numpy as np

BASES = {"A": 0.0, "T": 1.0, "C": 2.0, "G": 3.0}


def signal(seq: str, complet: bool = True) -> np.ndarray:
    """Signal 1D portee par la chaine : profil d'hydrophobicite."""
    from organes.atcg import hydrophobic_signal
    return hydrophobic_signal(seq, complet=complet)


def signal_bases(seq: str) -> np.ndarray:
    """Signal brut des bases (0..3) — pour les mesures de phase pures."""
    return np.array([BASES[b] for b in seq], dtype=np.float64)


def _plan_principal(cloud: np.ndarray) -> np.ndarray:
    """Projette le nuage sur son plan principal (2D), en convention d'orientation.

    La SVD a un signe arbitraire sur chaque axe propre. Sans convention, un
    meme cercle parcouru dans le meme sens peut ressortir +1 ou -1 tour
    selon l'echantillon — ce qui fabriquerait de fausses cassures dans un
    script. On oriente donc le plan pour que l'aire signee de la
    trajectoire soit positive : le sens de parcours devient une constante,
    et seules les VRAIES inversions ressortent en negatif.
    """
    c = cloud - cloud.mean(axis=0)
    if len(c) < 3:
        return np.zeros((len(c), 2))
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    p = c @ vt[:2].T
    # aire signee (formule du lacet) : si negative, on retourne le 2e axe
    x, y = p[:, 0], p[:, 1]
    aire = 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    if aire < 0:
        p[:, 1] = -p[:, 1]
    return p


def rotation_series(cloud: np.ndarray) -> np.ndarray:
    """Serie des increments d'angle de la trajectoire (radians par pas).

    Projection PCA -> angle -> deroulement -> differences.
    """
    if len(cloud) < 4:
        return np.zeros(0)
    p = _plan_principal(cloud)
    ang = np.arctan2(p[:, 1], p[:, 0])
    return np.diff(np.unwrap(ang))


def taux_enroulement(cloud: np.ndarray, frac_rayon: float = 0.25) -> dict:
    """Tours autour du centre, par point.

    Tours = angle deroule / 2pi, en ne comptant QUE les pas ou la
    trajectoire est reellement decentree (rayon > frac_rayon x rayon max).

    POURQUOI CE MASQUE. Validation de l'instrument (v1) :

        cercle (1 tour)   ->  0.995  OK
        cercle x2         -> -1.990  signe inverse
        ligne droite      -> -0.500  doit valoir 0

    La ligne droite a -0.5 : quand elle traverse son centre, l'angle saute
    de pi, ce qui fabrique un demi-tour inexistant. Masquer les pas ou le
    rayon est faible supprime l'artefact, et le filtre de signe de la v1
    (qui renvoyait -2 pour un cercle double) disparait aussi.

    Physiquement : un nombre d'enroulement n'est defini que pour une
    trajectoire qui tourne AUTOUR d'un centre — la ou le rayon s'annule,
    l'angle n'a pas de sens.
    """
    if len(cloud) < 4:
        return {"tours": 0.0, "taux": 0.0, "main": 0.0, "n_points": len(cloud)}
    p = _plan_principal(cloud)
    r = np.linalg.norm(p, axis=1)
    ang = np.unwrap(np.arctan2(p[:, 1], p[:, 0]))
    d = np.diff(ang)
    r_mid = 0.5 * (r[:-1] + r[1:])
    seuil = frac_rayon * (r.max() + 1e-12)
    d = np.where(r_mid >= seuil, d, 0.0)
    total = float(d.sum())
    n_pt = len(cloud)
    actifs = int((r_mid >= seuil).sum())
    return {"tours": total / (2 * np.pi), "taux": total / (2 * np.pi) / n_pt,
            "main": float(d[d != 0].mean()) if actifs else 0.0,
            "n_points": n_pt, "pas_actifs": actifs}


# ----------------------------------------------------------------------------
# Phase-locking : verrouillage entre deux chaines
# ----------------------------------------------------------------------------

def _phase(cloud: np.ndarray) -> np.ndarray:
    """Phase instantanee de la trajectoire (angle dans le plan principal)."""
    if len(cloud) < 2:
        return np.zeros(0)
    p = _plan_principal(cloud)
    return np.unwrap(np.arctan2(p[:, 1], p[:, 0]))


def plv(a: np.ndarray, b: np.ndarray) -> float:
    """Phase-locking value entre deux trajectoires, sur leur longueur commune.

    PLV = |moyenne(exp(i(phi_a - phi_b)))| dans [0,1].
    1 = parfaitement verrouillees, 0 = phases independantes.
    """
    pa, pb = _phase(a), _phase(b)
    n = min(len(pa), len(pb))
    if n < 4:
        return 0.0
    d = pa[:n] - pb[:n]
    return float(abs(np.mean(np.exp(1j * d))))


# ----------------------------------------------------------------------------
# SCRIPT : une suite de neurones, et la ou ca casse
# ----------------------------------------------------------------------------

def construire_script(neurones: list[str], overlap: int = 0) -> dict:
    """Concatene les signaux des neurones, sans normaliser entre eux.

    Chaque neurone est normalise individuellement (z-score) PUIS concatene :
    un script est une succession d'etats normalises, pas une bouillie
    d'echelles differentes. Le marquage des frontieres permet ensuite de
    savoir OU un enroulement passe au suivant.
    """
    blocs, bornes = [], [0]
    for seq in neurones:
        x = signal(seq)
        if x.size < 4:
            x = np.zeros(4)
        x = (x - x.mean()) / (x.std() + 1e-12)
        blocs.append(x)
        bornes.append(bornes[-1] + x.size)
    return {"X": np.concatenate(blocs), "bornes": bornes, "neurones": neurones}


def enroulement_local(X: np.ndarray, fenetre: int = 24, pas: int = 1,
                      dim: int = 3, delay: int = 3) -> np.ndarray:
    """Taux d'enroulement local, par fenetre glissante le long du script."""
    from organes.psig import takens_embed
    out, centres = [], []
    for i in range(0, len(X) - fenetre, pas):
        w = X[i:i + fenetre]
        cloud = takens_embed(w, dim=dim, delay=delay)
        if len(cloud) < 4:
            continue
        d = rotation_series(cloud)
        out.append(float(d.mean()) if d.size else 0.0)
        centres.append(i + fenetre // 2)
    return np.array(centres), np.array(out)


def analyse_script(neurones: list[str], fenetre: int = 24) -> dict:
    """Ou l'un va a l'autre, et le moment ou ca casse.

    Pour chaque neurone : son taux d'enroulement PROPRE (reference).
    Pour chaque frontiere : le taux local de part et d'autre, l'ecart, et
    le PLV entre les deux neurones voisins.
    Cassure = la frontiere ou l'ecart depasse la dispersion interne des
    neurones — c'est-a-dire la ou la loi d'enroulement change vraiment.
    """
    from organes.psig import takens_embed

    script = construire_script(neurones)
    X, bornes = script["X"], script["bornes"]

    refs = []
    for i, seq in enumerate(neurones):
        a, b = bornes[i], bornes[i + 1]
        cloud = takens_embed(X[a:b], dim=3, delay=3)
        refs.append(taux_enroulement(cloud)["taux"] if len(cloud) >= 4 else 0.0)

    # dispersion interne : combien le taux local d'un neurone varie autour de sa reference
    centres, locaux = enroulement_local(X, fenetre=fenetre)
    disps = []
    for i in range(len(neurones)):
        a, b = bornes[i] + fenetre // 2, bornes[i + 1] - fenetre // 2
        m = (centres >= a) & (centres < b)
        if m.sum() > 2:
            disps.append(float(np.std(locaux[m])))
    seuil = float(np.mean(disps)) * 3.0 if disps else 0.0

    frontieres = []
    for i in range(len(neurones) - 1):
        b = bornes[i + 1]
        la = (centres >= b - 2 * fenetre) & (centres < b)
        lb = (centres >= b) & (centres < b + 2 * fenetre)
        ca = float(np.mean(locaux[la])) if la.sum() else refs[i]
        cb = float(np.mean(locaux[lb])) if lb.sum() else refs[i + 1]
        ecart = abs(cb - ca)
        ca_cloud = takens_embed(X[bornes[i]:b], dim=3, delay=3)
        cb_cloud = takens_embed(X[b:bornes[i + 2]], dim=3, delay=3)
        p = plv(ca_cloud, cb_cloud)
        # Deux criteres independants : un ECART d'enroulement, et un
        # DECROCHAGE de verrouillage (PLV). Une cassure vraie demande les
        # deux : le regime change ET les deux chaines se desynchronisent.
        # Le seul ecart ne suffit pas — il depend d'un seuil qui s'adapte
        # au regime (voir JOURNAL).
        cassure = bool(ecart > seuil and p < 0.5)
        frontieres.append({
            "i": i, "position": b, "avant": ca, "apres": cb, "ecart": ecart,
            "plv": p, "cassure": cassure,
            "raison": ("ecart + decrochage" if cassure else
                       "ecart seul (verrouillage conserve)" if ecart > seuil else
                       "regime homogene"),
        })

    return {"refs": refs, "seuil": seuil, "frontieres": frontieres,
            "centres": centres, "locaux": locaux, "bornes": bornes, "X": X}


if __name__ == "__main__":
    print("=== Enroulement : validation de l'instrument ===\n")
    from organes.psig import takens_embed

    # verite connue : cercle -> 1 tour
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    cercle = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    r = taux_enroulement(cercle)
    print(f"cercle (1 tour attendu)      tours={r['tours']:.3f}  taux={r['taux']:.4f}")
    # deux tours
    t2 = np.linspace(0, 4 * np.pi, 200, endpoint=False)
    c2 = np.column_stack([np.cos(t2), np.sin(t2), np.zeros_like(t2)])
    r2 = taux_enroulement(c2)
    print(f"cercle x2 (2 tours attendus) tours={r2['tours']:.3f}  taux={r2['taux']:.4f}")
    # ligne droite -> 0
    x = np.linspace(0, 1, 200)
    rl = taux_enroulement(np.column_stack([x, x * .5, x * .2]))
    print(f"ligne droite (0 attendu)     tours={rl['tours']:.3f}  taux={rl['taux']:.4f}")
    # sinus -> Takens -> boucle
    xs = np.sin(np.linspace(0, 2 * np.pi * 3, 200))
    rs = taux_enroulement(takens_embed(xs, 3, 3))
    print(f"sinus 3 periodes -> Takens   tours={rs['tours']:.3f}  taux={rs['taux']:.4f}")