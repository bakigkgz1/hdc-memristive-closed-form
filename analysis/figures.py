"""Manuscript figures from the finished stages (core, stress, precision, encoders, e2e).
Figures that need cell 13 (analytic v2) or cell 14 (CrossSim) are made by figures_late.py.
Palette: validated categorical slots 1-3 (blue, orange, aqua; all-pairs CVD-safe) plus
markers / direct labels as secondary encoding; single-series magnitude in one blue ramp.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
A2 = ROOT / "analysis2"
from paths import R, WORK
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DEV_LABEL = {"pcm": "PCM", "reram_wan": "ReRAM (NeuRRAM)", "reram_cmo": "ReRAM (CMO/HfOx)"}
METH_ORDER = ["single", "perceptron", "onlinehd", "largemargin", "dependable", "noisecal"]
METH_LABEL = {"single": "Single-pass", "perceptron": "Perceptron", "onlinehd": "OnlineHD",
              "largemargin": "Large-margin", "dependable": "DependableHD-style",
              "noisecal": "Noise-calibrated"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 300,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 2 parity plot
def fig_parity():
    d = pd.read_pickle(WORK / "val_core_analytic.pkl")
    x, y = d.sim * 100, d.an * 100
    e = y - x
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    hb = ax.hexbin(x, y, gridsize=70, bins="log", cmap=BLUES, mincnt=1, linewidths=0)
    ax.plot([0, 100], [0, 100], color=INK2, lw=0.8, ls="--")
    ax.set_xlim(10, 100)
    ax.set_ylim(10, 100)
    ax.set_xlabel("Simulated analog accuracy, mean of 8 chips (%)")
    ax.set_ylabel("Closed-form prediction (%)")
    ax.set_title("(a) Prediction vs. simulation, 98,280 points")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("points per bin (log)", color=INK2)
    cb.outline.set_visible(False)
    ax.text(12, 94, f"MAE {e.abs().mean():.2f} pp\nr = {np.corrcoef(x, y)[0, 1]:.3f}",
            va="top", color=INK)
    ax = axs[1]
    mm = d.method.isin(["largemargin", "dependable", "noisecal"])
    bins = np.linspace(-1.5, 1.5, 61)
    ax.hist(np.clip(e[mm], -1.5, 1.5), bins=bins, color=C1, alpha=0.9, label=f"margin-trained (MAE {e[mm].abs().mean():.2f} pp)",
            histtype="stepfilled", lw=0)
    ax.hist(np.clip(e[~mm], -1.5, 1.5), bins=bins, color=C2, histtype="step", lw=1.4,
            label=f"no margin (MAE {e[~mm].abs().mean():.2f} pp)")
    ax.set_yscale("log")
    ax.set_xlabel("Prediction error (pp, clipped to ±1.5)")
    ax.set_ylabel("count")
    ax.set_title("(b) Error distribution")
    ax.set_ylim(top=2e5)
    ax.legend(loc="upper center", ncol=1)
    fig.tight_layout()
    save(fig, "fig2_parity")


# ------------------------------------------------------------------ Fig. 3 predictors vs noise
def fig_predictors_noise():
    st = pd.read_pickle(WORK / "val_stress_analytic.pkl").dropna(subset=["sdiff", "shat"])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, cen, title in [(axs[0], True, "(a) Centred queries"), (axs[1], False, "(b) Uncentred queries")]:
        s = st[st.center == cen]
        g = s.groupby("noise")
        ns = np.array(sorted(s.noise.unique()))
        for col, c, mk, lab in [("an", C1, "o", "Closed form (no chip)"),
                                ("sdiff", C2, "s", r"$\sigma_\Delta$ (1 calibration chip)"),
                                ("shat", C3, "^", r"$\hat\sigma$ (1 calibration chip)")]:
            mae = g.apply(lambda q: (q[col] - q.sim).abs().mean() * 100).reindex(ns)
            ax.plot(ns, mae, color=c, marker=mk, ms=4.5, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([f"×{v:g}" for v in ns])
        ax.set_xlabel("Device-noise scale")
        ax.set_title(title)
    axs[0].set_ylabel("Mean absolute error (pp)")
    axs[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_predictors_noise")


# ------------------------------------------------------------------ Fig. 4 analog loss by method
def load_core_ref():
    rows = []
    for line in open(R("core")):
        r = json.loads(line)
        if r.get("e2e"):
            continue
        i = r["times"].index(86400.0)
        rows.append(dict(dataset=r["dataset"], encoder=r["encoder"], D=r["dim"], seed=r["seed"],
                         method=r["method"], precision=r["precision"], device=r["device"],
                         dig=100 * r["digital_acc"], acc=100 * r["acc_mean"][i],
                         chip_sd=100 * r["acc_std"][i]))
    d = pd.DataFrame(rows)
    d["loss"] = d.dig - d.acc
    return d


def fig_method_loss():
    d = load_core_ref()
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
    for ax, dev in zip(axs, ["pcm", "reram_cmo", "reram_wan"]):
        s = d[d.device == dev]
        ys = np.arange(len(METH_ORDER))
        for k, m in enumerate(METH_ORDER):
            v = s[s.method == m].loss.to_numpy()
            ds = s[s.method == m].dataset.to_numpy()
            groups = [v[ds == c] for c in np.unique(ds)]
            boots = [np.concatenate([groups[i][rng.integers(len(groups[i]), size=len(groups[i]))]
                                     for i in rng.integers(len(groups), size=len(groups))]).mean()
                     for _ in range(1000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            margin = m in ("largemargin", "dependable", "noisecal")
            c = C1 if margin else C2
            ax.plot([lo, hi], [k, k], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(v.mean(), k, "o" if margin else "s", color=c, ms=5.5,
                    markeredgecolor=SURF, markeredgewidth=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([METH_LABEL[m] for m in METH_ORDER])
        ax.set_title(DEV_LABEL[dev])
        ax.set_xlabel("Analog loss at 1 day (pp)")
        ax.set_xlim(left=0)
    axs[0].invert_yaxis()
    fig.text(0.99, 0.01, "mean ± 95 % cluster-bootstrap CI (datasets); circles: margin-trained",
             ha="right", color=INK2, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig4_method_loss")


# ------------------------------------------------------------------ Fig. 5 ADC mechanism
def fig_clip_mechanism():
    rows = []
    K = ["job_key", "method", "precision", "device", "noise_scale", "periphery"]
    S = {}
    for line in open(R("core")):
        r = json.loads(line)
        if not r.get("e2e"):
            S[tuple(r[k] for k in K)] = r
    for line in open(R("core_analytic")):
        a = json.loads(line)
        s = S[tuple(a[k] for k in K)]
        for i, t in enumerate(a["times"]):
            rows.append(dict(method=a["method"], device=a["device"], s=a["in_scale"],
                             e=100 * (a["pred_acc_analytic_t"][i] - s["acc_mean"][i])))
    d = pd.DataFrame(rows)
    d["margin"] = d.method.isin(["largemargin", "dependable", "noisecal"])
    edges = np.quantile(d.s, np.linspace(0, 1, 11))
    d["bin"] = pd.cut(d.s, edges, include_lowest=True, labels=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for dev, c, mk in [("reram_wan", C2, "s"), ("pcm", C1, "o"), ("reram_cmo", C3, "^")]:
        g = d[(d.device == dev) & ~d.margin].groupby("bin")
        x = g.s.median()
        y = g.e.apply(lambda q: (q.abs() > 2).mean() * 100)
        ax.plot(x, y, color=c, marker=mk, ms=4, label=DEV_LABEL[dev])
    ax.set_xscale("log")
    ticks = [0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Calibrated read scale s (log)")
    ax.set_ylabel("Points with |error| > 2 pp (%)")
    ax.set_title("Closed form v1, models without margin training")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig5_clip_mechanism")


# ------------------------------------------------------------------ Fig. 6 precision x periphery
def fig_precision_periphery():
    pr = pd.read_pickle(WORK / "val_prec_analytic.pkl").dropna(subset=["sdiff", "shat"])
    order = ["fp", "4bit", "2bit", "binary"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    w = 0.26
    for ax, per, title in [(axs[0], "ideal", "(a) Ideal periphery"),
                           (axs[1], "realistic", "(b) 9-bit ADC, 7-bit DAC, output noise")]:
        s = pr[pr.periphery == per]
        x = np.arange(len(order))
        for j, (col, c, lab) in enumerate([("an", C1, "Closed form"), ("sdiff", C2, r"$\sigma_\Delta$"),
                                           ("shat", C3, r"$\hat\sigma$")]):
            v = [((s[s.precision == p][col] - s[s.precision == p].sim).abs().mean() * 100) for p in order]
            ax.bar(x + (j - 1) * w, v, width=w - 0.03, color=c, label=lab)
            for xi, vi in zip(x + (j - 1) * w, v):
                ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", va="bottom", fontsize=5.5, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(["FP32", "4 bit", "2 bit", "binary"])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Mean absolute error (pp)")
    axs[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig6_precision_periphery")


# ------------------------------------------------------------------ Fig. 7 end-to-end projection
def fig_e2e():
    def load(p):
        rows = []
        for line in open(p):
            r = json.loads(line)
            i = r["times"].index(86400.0)
            rows.append(dict(dataset=r["dataset"], encoder=r["encoder"], D=r["dim"], seed=r["seed"],
                             method=r["method"], device=r["device"], dig=100 * r["digital_acc"],
                             acc=100 * r["acc_mean"][i]))
        return pd.DataFrame(rows)
    g = load(R("e2e"))
    g = g[(g.encoder == "rp") & (g.method == "largemargin")]
    b = load(R("e2e_bipolar"))
    k = ["dataset", "D", "seed", "device"]
    j = g.set_index(k)[["acc", "dig"]].join(b.set_index(k)[["acc", "dig"]], lsuffix="_g", rsuffix="_b",
                                             how="inner").reset_index()
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6), sharex=True, sharey=True)
    for ax, dev in zip(axs, ["pcm", "reram_cmo", "reram_wan"]):
        s = j[j.device == dev]
        ax.plot([0, 100], [0, 100], color=INK2, lw=0.8, ls="--")
        ax.scatter(s.acc_g, s.acc_b, s=12, color=C1, edgecolor=SURF, linewidth=0.4)
        ax.set_title(DEV_LABEL[dev])
        ax.set_xlabel("Gaussian projection (%)")
        ax.text(0.96, 0.04, f"mean Δ = {(s.acc_b - s.acc_g).mean():+.1f} pp\n"
                            f"bipolar better in {((s.acc_b - s.acc_g) > 0.05).sum()} of {len(s)}",
                transform=ax.transAxes, va="bottom", ha="right", color=INK)
    axs[0].set_ylabel("Bipolar (±1) projection (%)")
    lo = min(j.acc_g.min(), j.acc_b.min()) - 2
    axs[0].set_xlim(lo, 100)
    axs[0].set_ylim(lo, 100)
    fig.suptitle("End-to-end analog HDC (encoder and prototypes on crossbars), accuracy at 1 day",
                 fontsize=8.5, color=INK)
    fig.tight_layout()
    save(fig, "fig7_e2e_projection")


if __name__ == "__main__":
    for f in [fig_parity, fig_predictors_noise, fig_method_loss, fig_clip_mechanism,
              fig_precision_periphery, fig_e2e]:
        f()
        print("ok", f.__name__)
