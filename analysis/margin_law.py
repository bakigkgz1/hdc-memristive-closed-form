"""Margin-to-noise law: analog loss at 1 day vs the median digital cosine margin divided by
the decision-variable noise sigma_Delta (calibration chip, reproduced core records).

Outputs: tables/margin_law.csv (binned medians per rule, fit), figures/fig10_margin_law.png/pdf.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figures as F
from paths import HERE, R

K = ["job_key", "method", "precision", "device", "noise_scale", "periphery"]
NAMES = {"single": "Single-pass", "perceptron": "Perceptron", "onlinehd": "OnlineHD",
         "largemargin": "Large-margin", "dependable": "DependableHD-style",
         "noisecal": "Noise-calibrated"}
COL = {"single": F.INK2, "perceptron": F.C2, "onlinehd": "#b8860b", "largemargin": F.C1,
       "dependable": F.C3, "noisecal": "#7b4fd6"}


def load():
    P = {tuple(r[k] for k in K): r for r in map(json.loads, open(R("core_pred"))) if r["reproduced"]}
    rows = []
    for r in map(json.loads, open(R("core"))):
        p = P.get(tuple(r[k] for k in K))
        if r.get("e2e") or not p:
            continue
        rows.append(dict(dataset=r["dataset"], method=r["method"], precision=r["precision"],
                         device=r["device"], D=r["dim"],
                         loss=100 * (r["digital_acc"] - r["acc_mean"][1]),
                         x=r["margin_quantiles"][2] / p["sigma_diff_t"][1]))
    return pd.DataFrame(rows)


def main():
    d = load()
    g = d[(d.x > 1) & (d.loss > 1e-3)]
    X = np.c_[np.ones(len(g)), np.log(g.x)]
    y = np.log(g.loss)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    M = pd.get_dummies(g.method).astype(float).drop(columns="perceptron")
    X2 = np.c_[X, M.values]
    b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    factor = dict(zip(M.columns, np.exp(b2[2:])))
    edges = np.quantile(d.x, np.linspace(0, 1, 9))
    d["bin"] = pd.cut(d.x, edges, include_lowest=True)
    t = d.pivot_table(index="bin", columns="method", values="loss", aggfunc="median", observed=True)
    t["x_median"] = d.groupby("bin", observed=True).x.median()
    t.to_csv(HERE / "tables" / "margin_law.csv")
    rho = d[["x", "loss"]].corr(method="spearman").iloc[0, 1]
    med = d.groupby("method").x.median()
    print(f"n={len(d)} spearman={rho:.3f} power-law slope={b[1]:.2f} R2={r2:.2f} (n={len(g)})")
    print("loss factor vs perceptron at equal ratio:", {k: round(v, 2) for k, v in factor.items()})
    print("median ratio per rule:", med.round(1).to_dict())
    print(t.round(3).to_string())
    # power law through the binned medians of the five retraining rules (bins 2-7, x ~ 4-21)
    rr = ["perceptron", "onlinehd", "largemargin", "dependable", "noisecal"]
    sub = d[d.method.isin(rr)].groupby("bin", observed=True).agg(x=("x", "median"), l=("loss", "median")).iloc[1:7]
    bb = np.polyfit(np.log(sub.x), np.log(sub.l), 1)[::-1]
    spread = (t[rr].iloc[1:7].max(1) / t[rr].iloc[1:7].min(1))
    print(f"binned power law slope {bb[1]:.2f}; max/min across retraining rules per bin: {spread.round(2).tolist()}")

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.scatter(d.x, d.loss.clip(1e-3), s=2, color=F.GRID, rasterized=True, zorder=1)
    for m in ["single", "perceptron", "onlinehd", "largemargin", "dependable", "noisecal"]:
        s = d[d.method == m].groupby("bin", observed=True).agg(x=("x", "median"), l=("loss", "median"))
        ax.plot(s.x, s.l.clip(1e-3), "-o", ms=3, lw=1.2, color=COL[m], label=NAMES[m], zorder=3)
    xx = np.geomspace(3, 40, 50)
    ax.plot(xx, np.exp(bb[0]) * xx ** bb[1], "--", color=F.INK, lw=0.8, zorder=2,
            label=f"retraining rules: ∝ x$^{{{bb[1]:.1f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 140)
    ax.set_ylim(1e-3, 30)
    ax.set_xlabel("Median digital margin / σ$_Δ$")
    ax.set_ylabel("Analog loss at 1 day (pp)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    F.save(fig, "fig10_margin_law")


if __name__ == "__main__":
    main()
