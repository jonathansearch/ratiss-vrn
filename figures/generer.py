"""Generation des figures du depot — VRN / RATISS Labs.

Produit les figures utilisees dans le README, a partir des resultats
consignes (experiences/*.json) et de mesures reproductibles. Style sobre,
fond clair, une idee par figure.

Lancement : python -m figures.generer
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/vrn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "figures" / "img"
SORTIE.mkdir(parents=True, exist_ok=True)

BLEU = "#1f4e79"
ORANGE = "#c55a11"
GRIS = "#7f7f7f"
VERT = "#548235"
ROUGE = "#a02020"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140,
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _json(nom):
    return json.loads((RACINE / "experiences" / nom).read_text())


def fig_criticite():
    """y(p) a un maximum INTERIEUR — la criticite (exp04)."""
    from organes.neurone_vrn import neurone_vrn_v3

    def chaine_interpolee(p, n=240, seed=12):
        rng = np.random.default_rng(seed)
        return "".join(str(rng.choice(list("ATCG"))) if rng.random() < p
                       else "ATGC"[i % 4] for i in range(n))

    ps = np.linspace(0, 1, 11)
    ys, gd, gt = [], [], []
    for p in ps:
        r = neurone_vrn_v3(chaine_interpolee(float(p)))
        ys.append(r["y"]); gd.append(r["g_dyn"]); gt.append(r["g_topo"])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ps, ys, "-o", color=BLEU, lw=2, label="y (douane VRN)")
    ax.plot(ps, gd, "--", color=ORANGE, lw=1.4, label="g_dyn (VOIR)")
    ax.plot(ps, gt, ":", color=GRIS, lw=1.4, label="g_topo (VERIFIER)")
    imax = int(np.argmax(ys))
    ax.axvline(ps[imax], color=ROUGE, lw=1, ls="--", alpha=0.6)
    ax.annotate(f"max interieur\np={ps[imax]:.1f}", (ps[imax], ys[imax]),
                xytext=(ps[imax] + 0.12, ys[imax] + 0.02), color=ROUGE,
                fontsize=9, arrowprops=dict(arrowstyle="->", color=ROUGE))
    ax.set_xlabel("p — probabilite de desordre (0 = ordre pur, 1 = chaos)")
    ax.set_ylabel("lecture")
    ax.set_title("Criticite : le neurone s'ouvre sur le BORD, pas aux extremes")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(SORTIE / "criticite.png")
    plt.close(fig)


def fig_enroulement():
    """Nombre d'enroulement : cas stables ET le defaut d'echantillonnage.

    Le cercle et la droite sont invariants a la densite d'echantillonnage.
    Le sinus, lui, NE L'EST PAS : la valeur depend du nombre de points
    (200 -> 2.28, 400 -> 0.90, 800 -> 0.43). Un nombre d'enroulement doit
    etre invariant : c'est un defaut PREEXISTANT de l'instrument, et la
    figure doit le montrer, pas le masquer en choisissant 200 points.
    """
    from organes.enroulement import taux_enroulement
    from organes.psig import takens_embed

    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    c1 = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    t2 = np.linspace(0, 4 * np.pi, 200, endpoint=False)
    c2 = np.column_stack([np.cos(t2), np.sin(t2), np.zeros_like(t2)])
    x = np.linspace(0, 1, 200)
    ligne = np.column_stack([x, x * .5, x * .2])
    noms = ["cercle\n(1 tour)", "cercle x2\n(2 tours)", "ligne\ndroite"]
    attendus = [1.0, 2.0, 0.0]
    mesures = [abs(taux_enroulement(c)["tours"]) for c in (c1, c2, ligne)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    xi = np.arange(len(noms))
    ax.bar(xi - 0.2, attendus, 0.4, color=GRIS, label="attendu")
    ax.bar(xi + 0.2, mesures, 0.4, color=BLEU, label="mesure")
    for i, m in zip(xi, mesures):
        ax.text(i + 0.2, m + 0.04, f"{m:.2f}", ha="center", fontsize=9,
                color=BLEU)
    ax.set_xticks(xi); ax.set_xticklabels(noms)
    ax.set_ylabel("nombre d'enroulement"); ax.set_ylim(0, 2.4)
    ax.set_title("Cas valides : invariant a l'echantillonnage")
    ax.legend(frameon=False)

    ns = [50, 100, 200, 400, 800, 1600]
    vals = [abs(taux_enroulement(takens_embed(
        np.sin(np.linspace(0, 2 * np.pi * 3, n)), 3, 3))["tours"])
        for n in ns]
    ax2.plot(ns, vals, "-o", color=ROUGE, lw=2)
    ax2.axhline(3.0, color=GRIS, ls=":", lw=1)
    ax2.text(50, 3.05, "attendu ~3", fontsize=8, color=GRIS)
    ax2.set_xscale("log"); ax2.set_xlabel("nombre de points")
    ax2.set_ylabel("enroulement mesure")
    ax2.set_title("DEFAUT : le sinus depend de l'echantillonnage")
    fig.tight_layout()
    fig.savefig(SORTIE / "enroulement.png")
    plt.close(fig)


def fig_hierarchie():
    """Hierarchie vs produit vs organes isoles (exp06)."""
    d = _json("exp06_hierarchie.json")["AUC"]
    ordre = [("g_dyn", "g_dyn (VOIR)"), ("y_hier", "hierarchique H"),
             ("y_hier_c", "H + c"), ("s", "s (P_sig)"),
             ("y_prod_noc", "produit sans c"), ("y_prod", "produit P"),
             ("g_topo", "g_topo"), ("c", "c (langage)")]
    noms = [n for _, n in ordre]
    vals = [d[k] for k, _ in ordre]
    couleurs = [BLEU if k.startswith("y_hier") or k == "g_dyn"
                else (ROUGE if k == "c" else GRIS) for k, _ in ordre]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    y = np.arange(len(noms))[::-1]
    ax.barh(y, vals, color=couleurs)
    ax.axvline(0.5, color="k", lw=1, ls=":")
    ax.text(0.505, len(noms) - 0.4, "hasard (0.5)", fontsize=8, color="k")
    for yi, v in zip(y, vals):
        ax.text(v + 0.005, yi, f"{v:.4f}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(noms)
    ax.set_xlim(0, 1.08); ax.set_xlabel("AUC")
    ax.set_title("Organe tranchant : g_dyn gouverne, c sous le hasard")
    fig.tight_layout()
    fig.savefig(SORTIE / "hierarchie.png")
    plt.close(fig)


def fig_couplage():
    """Accord de population selon le couplage K (exp07)."""
    K = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]
    acc = [0.4005, 0.4376, 0.4733, 0.5466, 0.6049, 0.6594,
           0.7041, 0.7902, 0.8535, 0.9287, 0.9646]
    sd = [0.0023, 0.0047, 0.0051, 0.0028, 0.0035, 0.0070,
          0.0074, 0.0084, 0.0054, 0.0073, 0.0023]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(K, acc, yerr=sd, fmt="-o", color=BLEU, lw=2, capsize=3)
    ax.axhline(0.4005, color=GRIS, ls=":", lw=1)
    ax.text(0.25, 0.408, "sans couplage (K=0)", fontsize=8, color=GRIS)
    ax.set_xlabel("K — force de couplage")
    ax.set_ylabel("accord de la population")
    ax.set_ylim(0.35, 1.0)
    ax.set_title("Couplage : la population se synchronise (crossover, pas seuil)")
    fig.tight_layout()
    fig.savefig(SORTIE / "couplage.png")
    plt.close(fig)


def fig_coherence_taille():
    """R tombe en 0.886/sqrt(N) : le plancher du hasard (exp07 bis)."""
    Ns = np.array([4, 6, 8, 10, 16, 24, 32])
    R = np.array([0.4350, 0.3679, 0.3173, 0.2791, 0.2038, 0.1755, 0.1591])
    n = np.linspace(3, 34, 100)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(n, np.sqrt(np.pi) / (2 * np.sqrt(n)), "-", color=ROUGE, lw=1.6,
            label=r"theorie hasard $\sqrt{\pi}/(2\sqrt{N})$")
    ax.plot(Ns, R, "o", color=BLEU, ms=7,
            label="R mesure (population non couplee)")
    ax.set_xlabel("N — nombre de neurones")
    ax.set_ylabel("coherence R")
    ax.set_title("R mesure le plancher du hasard, pas un attracteur")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SORTIE / "coherence_taille.png")
    plt.close(fig)


def fig_population():
    """Population vs neurone seul (exp08) — resultat negatif assume."""
    d = _json("exp08_population.json")["AUC"]
    noms = ["neurone\nseul", "pop\nfusion", "pop\ncouplee"]
    vals = [d["seul"], d["pop_avg"], d["pop_avg_couple"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(noms, vals, color=[BLEU, ORANGE, VERT])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0.95, 0.995); ax.set_ylabel("AUC")
    ax.set_title("La population ne bat PAS le neurone seul\n"
                 "(critere scelle P1 : gain >= +0.03 — infirme)")
    fig.tight_layout()
    fig.savefig(SORTIE / "population.png")
    plt.close(fig)


def fig_tache_motif():
    """AUC par organe sur la tache motif cache (exp05/exp06)."""
    d5 = _json("exp05_tache_motif.json")["resultats"]
    d6 = _json("exp06_hierarchie.json")["AUC"]
    noms = ["g_dyn", "hierarchique", "produit", "g_topo", "s", "c"]
    vals = [d6["g_dyn"], d6["y_hier"], d6["y_prod"],
            d5["g_topo"]["AUC"], d5["s"]["AUC"], d5["c"]["AUC"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    couleurs = [BLEU, VERT, ORANGE, GRIS, GRIS, ROUGE]
    x = np.arange(len(noms))
    ax.bar(x, vals, color=couleurs)
    ax.axhline(0.5, color="k", lw=1, ls=":")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(noms)
    ax.set_ylim(0, 1.06); ax.set_ylabel("AUC (motif cache)")
    ax.set_title("Qui voit le motif cache : g_dyn, puis la hierarchie")
    fig.tight_layout()
    fig.savefig(SORTIE / "tache_motif.png")
    plt.close(fig)


def main():
    print("generation des figures ->", SORTIE)
    for f in (fig_criticite, fig_enroulement, fig_hierarchie, fig_couplage,
              fig_coherence_taille, fig_population, fig_tache_motif):
        f()
        print("  ok", f.__name__)
    print("termine.")


if __name__ == "__main__":
    main()