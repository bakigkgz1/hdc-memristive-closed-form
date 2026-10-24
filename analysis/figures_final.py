"""Consolidated figures for the submitted version (6 main figures + 1 supplementary).

  figA_prediction  (Fig. 2): (a) v1 parity, (b) v2 parity, (c) v1 error distribution
  figB_validity    (Fig. 3): (a,b) MAE vs noise scale, (c,d) MAE by precision and periphery
  figC_training    (Fig. 6): (a-c) analog loss by rule and device, (d) margin-to-noise law
Fig. 1 (schematic), Fig. 4 (clip mechanism), Fig. 5 (CrossSim) and Supplementary Fig. S1
(end-to-end encoding) are produced by figures.py / figures_late.py unchanged.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figures as F
import figures_late as L
import margin_law as ML
from paths import WORK

MARGIN = ["largemargin", "dependable", "noisecal"]


def _parity(ax, x, y, title, fig):
    e = y - x
    hb = ax.hexbin(x, y, gridsize=60, bins="log", cmap=F.BLUES, mincnt=1, linewidths=0)
    ax.plot([0, 100], [0, 100], color=F.INK2, lw=0.8, ls="--")
    ax.set_xlim(10, 100)
    ax.set_ylim(10, 100)
    ax.set_title(title)
    ax.set_xlabel("Simulated accuracy, 8 chips (%)")
    ax.text(13, 96, f"MAE {e.abs().mean():.2f} pp\n>2 pp: {(e.abs() > 2).mean() * 100:.2f} %",
            va="top", color=F.INK, fontsize=7)
    return hb


def fig_prediction():
    q = pd.read_pickle(WORK / "val_core_quad.pkl")          # exact v1 (Gauss-Hermite)
    d1 = pd.DataFrame({"sim": q.sim / 100, "an": q.quad / 100, "method": q.m})
    d2 = L.v1v2("core")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    _parity(axs[0], d1.sim * 100, d1.an * 100, "(a) Closed form v1 (exact)", fig)
    hb = _parity(axs[1], d2.sim, d2.v2, "(b) Semi-analytic v2", fig)
    axs[0].set_ylabel("Predicted accuracy (%)")
    axs[1].set_yticklabels([])
    e = (d1.an - d1.sim) * 100
    mm = d1.method.isin(MARGIN)
    bins = np.linspace(-1.5, 1.5, 61)
    ax = axs[2]
    ax.hist(np.clip(e[mm], -1.5, 1.5), bins=bins, color=F.C1, alpha=0.9, histtype="stepfilled", lw=0,
            label=f"margin-trained ({e[mm].abs().mean():.2f} pp)")
    ax.hist(np.clip(e[~mm], -1.5, 1.5), bins=bins, color=F.C2, histtype="step", lw=1.3,
            label=f"no margin ({e[~mm].abs().mean():.2f} pp)")
    ax.set_yscale("log")
    ax.set_ylim(top=3e5)
    ax.set_xlabel("v1 error (pp, clipped to ±1.5)")
    ax.set_ylabel("count")
    ax.set_title("(c) v1 error by training rule")
    ax.legend(loc="upper center", fontsize=6.5)
    fig.tight_layout()
    F.save(fig, "figA_prediction")


def fig_validity():
    st = pd.read_pickle(WORK / "val_stress_analytic.pkl").dropna(subset=["sdiff", "shat"])
    pr = pd.read_pickle(WORK / "val_prec_analytic.pkl").dropna(subset=["sdiff", "shat"])
    pv = pd.read_pickle(WORK / "val_paired_all.pkl")                 # from paired_table.py (pp)
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0))
    lines = [("an", F.C1, "o", "v1 closed form (no chip)"),
             ("sdiff", F.C2, "s", r"$\sigma_\Delta$ (1 calibration chip)"),
             ("shat", F.C3, "^", r"$\hat\sigma$ (1 calibration chip)")]
    for ax, cen, title in [(axs[0, 0], True, "(a) Centred queries"), (axs[0, 1], False, "(b) Uncentred queries")]:
        s = st[st.center == cen]
        g = s.groupby("noise")
        ns = np.array(sorted(s.noise.unique()))
        for col, c, mk, lab in lines:
            mae = g.apply(lambda q: (q[col] - q.sim).abs().mean() * 100).reindex(ns)
            ax.plot(ns, mae, color=c, marker=mk, ms=4.5, label=lab)
        s2 = pv[(pv.stage == "stress") & (pv.center == cen)]          # same paired points as v1, sigma
        mae2 = s2.groupby("noise").apply(lambda q: (q.v2 - q.sim).abs().mean()).reindex(ns)
        ax.plot(ns, mae2, color=F.INK, marker="D", ms=4, ls="--", label="v2 semi-analytic (no chip)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([f"×{v:g}" for v in ns])
        ax.set_xlabel("Device-noise scale")
        ax.set_title(title)
    axs[0, 1].set_ylim(axs[0, 0].get_ylim()[0], None)
    axs[0, 0].set_ylabel("Mean absolute error (pp)")
    axs[0, 0].legend(loc="upper left", fontsize=6.5)
    order = ["fp", "4bit", "2bit", "binary"]
    w = 0.26
    ymax = 0
    for ax, per, title in [(axs[1, 0], "ideal", "(c) Ideal periphery"),
                           (axs[1, 1], "realistic", "(d) 9-bit ADC, 7-bit DAC, output noise")]:
        s = pr[pr.periphery == per]
        x = np.arange(len(order))
        for j, (col, c, _, lab) in enumerate(lines):
            v = [((s[s.precision == p][col] - s[s.precision == p].sim).abs().mean() * 100) for p in order]
            ymax = max(ymax, max(v))
            ax.bar(x + (j - 1) * w, v, width=w - 0.03, color=c, label=lab)
            for xi, vi in zip(x + (j - 1) * w, v):
                ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", va="bottom", fontsize=5.3, color=F.INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(["FP32", "4 bit", "2 bit", "binary"])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    for ax in axs[1]:
        ax.set_ylim(0, ymax * 1.15)
    axs[1, 0].set_ylabel("Mean absolute error (pp)")
    fig.tight_layout()
    F.save(fig, "figB_validity")


def fig_training():
    d = F.load_core_ref()
    rng = np.random.default_rng(1)
    fig = plt.figure(figsize=(7.0, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.45, wspace=0.08)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for i, (ax, dev) in enumerate(zip(axs, ["pcm", "reram_cmo", "reram_wan"])):
        s = d[d.device == dev]
        for k, m in enumerate(F.METH_ORDER):
            v = s[s.method == m].loss.to_numpy()
            ds = s[s.method == m].dataset.to_numpy()
            groups = [v[ds == c] for c in np.unique(ds)]
            boots = [np.concatenate([groups[j][rng.integers(len(groups[j]), size=len(groups[j]))]
                                     for j in rng.integers(len(groups), size=len(groups))]).mean()
                     for _ in range(1000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            margin = m in MARGIN
            c = F.C1 if margin else F.C2
            ax.plot([lo, hi], [k, k], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(v.mean(), k, "o" if margin else "s", color=c, ms=5,
                    markeredgecolor=F.SURF, markeredgewidth=0.8)
        ax.set_yticks(np.arange(len(F.METH_ORDER)))
        ax.set_yticklabels([F.METH_LABEL[m] for m in F.METH_ORDER] if i == 0 else [])
        ax.set_title(f"({'abc'[i]}) {F.DEV_LABEL[dev]}")
        ax.set_xlabel("Analog loss at 1 day (pp)")
        ax.set_xlim(left=0)
        ax.invert_yaxis()
    ax = fig.add_subplot(gs[1, :])
    m = ML.load()
    edges = np.quantile(m.x, np.linspace(0, 1, 9))
    m["bin"] = pd.cut(m.x, edges, include_lowest=True)
    pos = m.loss > 1e-3
    ax.scatter(m.x[pos], m.loss[pos], s=1.5, color=F.GRID, rasterized=True, zorder=1)
    ax.text(0.99, 0.97, f"{(~pos).sum():,} records with loss ≤ 0.001 pp not shown",
            transform=ax.transAxes, ha="right", va="top", fontsize=6, color=F.INK2)
    for meth in ["single", "perceptron", "onlinehd", "largemargin", "dependable", "noisecal"]:
        s = m[m.method == meth].groupby("bin", observed=True).agg(x=("x", "median"), l=("loss", "median"))
        s = s[s.l > 1e-3]
        ax.plot(s.x, s.l, "-o", ms=3, lw=1.2, color=ML.COL[meth], label=ML.NAMES[meth], zorder=3)
    rr = ["perceptron", "onlinehd", "largemargin", "dependable", "noisecal"]
    sub = m[m.method.isin(rr)].groupby("bin", observed=True).agg(x=("x", "median"), l=("loss", "median")).iloc[1:7]
    bb = np.polyfit(np.log(sub.x), np.log(sub.l), 1)[::-1]
    xx = np.geomspace(4, 21, 50)
    ax.plot(xx, np.exp(bb[0]) * xx ** bb[1], "--", color=F.INK, lw=0.8, zorder=2,
            label=f"retraining rules: ∝ x$^{{{bb[1]:.1f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 140)
    ax.set_ylim(1e-3, 30)
    ax.set_xlabel("Median digital margin / σ$_Δ$")
    ax.set_ylabel("Analog loss at 1 day (pp)")
    ax.set_title("(d) Margin-to-noise law, all devices")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
    F.save(fig, "figC_training")


if __name__ == "__main__":
    fig_prediction()
    fig_validity()
    fig_training()
    print("done")
