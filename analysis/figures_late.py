"""Figures / tables that need cell 13 (closed form v2) and cell 14 (CrossSim).
Run after placing the outputs in analysis2/:
  analysis2/<stage>_analytic2.jsonl  (stage in core, stress, precision, encoders)
  analysis2/crosssim.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import figures as F

plt = F.plt
from paths import R
OUT = F.ROOT / "paper" / "tables"
K = ["job_key", "method", "precision", "device", "noise_scale", "periphery"]
SIM = {s: R(s) for s in ("core", "stress", "precision", "encoders")}


def v1v2(stage):
    S = {tuple(r[k] for k in K): r for r in map(json.loads, open(SIM[stage])) if not r.get("e2e")}
    a1 = {tuple(r[k] for k in K): r for r in map(json.loads, open(R(f"{stage}_analytic")))}
    rows = []
    for r2 in map(json.loads, open(R(f"{stage}_analytic2"))):
        k = tuple(r2[x] for x in K)
        s, r1 = S[k], a1[k]
        for i, t in enumerate(r2["times"]):
            rows.append(dict(stage=stage, dataset=s["dataset"], encoder=s["encoder"], D=s["dim"],
                             center=s["center"], method=r2["method"], precision=r2["precision"],
                             device=r2["device"], noise=r2["noise_scale"], periphery=r2["periphery"],
                             t=t, sim=100 * s["acc_mean"][i], dig=100 * s["digital_acc"],
                             v1=100 * r1["pred_acc_analytic_t"][i], v2=100 * r2["pred_acc_analytic2_t"][i],
                             clip=r2["clip_frac_t"][i], s=r2["in_scale"]))
    return pd.DataFrame(rows)


def validity_table():
    d = pd.concat([v1v2(s) for s in SIM if R(f"{s}_analytic2").exists()])
    d["e1"], d["e2"] = (d.v1 - d.sim).abs(), (d.v2 - d.sim).abs()
    d["sub"] = np.where(d.stage == "stress",
                        np.where(d.center, "centred", "uncentred") + " x" + d.noise.map("{:g}".format),
                        np.where(d.stage == "precision", d.periphery, "all"))
    t = d.groupby(["stage", "sub"]).agg(n=("e1", "size"), MAE_v1=("e1", "mean"), MAE_v2=("e2", "mean"),
                                        p95_v1=("e1", lambda x: x.quantile(.95)),
                                        p95_v2=("e2", lambda x: x.quantile(.95)),
                                        gt2_v1=("e1", lambda x: (x > 2).mean() * 100),
                                        gt2_v2=("e2", lambda x: (x > 2).mean() * 100))
    t.to_csv(OUT / "closed_form_v1_v2.csv")
    print(t.round(3).to_string())
    return d


def fig_v1_v2(d):
    c = d[d.stage == "core"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True, sharey=True)
    for ax, col, title in [(axs[0], "v1", "(a) Gaussian closed form (v1)"),
                           (axs[1], "v2", "(b) Periphery-aware closed form (v2)")]:
        hb = ax.hexbin(c.sim, c[col], gridsize=70, bins="log", cmap=F.BLUES, mincnt=1, linewidths=0)
        ax.plot([0, 100], [0, 100], color=F.INK2, lw=0.8, ls="--")
        e = (c[col] - c.sim).abs()
        ax.text(0.04, 0.96, f"MAE {e.mean():.2f} pp\nmax {e.max():.1f} pp\n>2 pp: {(e > 2).mean() * 100:.2f} %",
                transform=ax.transAxes, va="top", color=F.INK)
        ax.set_title(title)
        ax.set_xlabel("Simulated analog accuracy (%)")
    axs[0].set_ylabel("Predicted accuracy (%)")
    axs[0].set_xlim(0, 100)
    axs[0].set_ylim(0, 100)
    fig.tight_layout()
    F.save(fig, "fig8_v1_v2")


def crosssim():
    p = R("crosssim")
    if not p.exists():
        print("no crosssim.jsonl yet")
        return
    rows = []
    for r in map(json.loads, open(p)):
        for i, t in enumerate(r["times_days"]):
            rows.append(dict(dataset=r["dataset"], encoder=r["encoder"], D=r["dim"], seed=r["seed"],
                             method=r["method"], precision=r["precision"], device=r["xs_device"],
                             periphery=r["periphery"], t=t, dig=100 * r["digital_acc"],
                             sim=100 * r["acc_mean"][i], pred=100 * r["pred_acc_t"][i],
                             reproduced=r["reproduced"]))
    d = pd.DataFrame(rows)
    d["err"] = (d.pred - d.sim).abs()
    d["loss"] = d.dig - d.sim
    print("reproduced", d.reproduced.mean())
    t1 = d.groupby(["device", "periphery"]).agg(n=("err", "size"), MAE=("err", "mean"),
                                                 p95=("err", lambda x: x.quantile(.95)),
                                                 loss=("loss", "mean"))
    t2 = d.pivot_table(index="method", columns="device", values="loss", aggfunc="mean")
    t1.to_csv(OUT / "crosssim_closed_form.csv")
    t2.to_csv(OUT / "crosssim_loss_method_device.csv")
    print(t1.round(3).to_string(), "\n", t2.round(3).to_string())
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axs[0]
    hb = ax.hexbin(d.sim, d.pred, gridsize=60, bins="log", cmap=F.BLUES, mincnt=1, linewidths=0)
    ax.plot([0, 100], [0, 100], color=F.INK2, lw=0.8, ls="--")
    lo = min(d.sim.min(), d.pred.min()) - 2
    ax.set_xlim(lo, 100)
    ax.set_ylim(lo, 100)
    ax.text(0.04, 0.96, f"MAE {d.err.mean():.2f} pp\nmax {d.err.max():.1f} pp\nr > 0.9999",
            transform=ax.transAxes, va="top", color=F.INK)
    ax.set_xlabel("CrossSim accuracy, mean of 8 chips (%)")
    ax.set_ylabel("Closed form from CrossSim cell statistics (%)")
    ax.set_title("(a) Closed form in a second simulator")
    ax = axs[1]
    devs = ["pcm_joshi", "rram_milo", "rram_wan", "sonos"]
    lab = {"pcm_joshi": "PCM (Joshi)", "rram_wan": "ReRAM (Wan)", "rram_milo": "ReRAM (Milo)",
           "sonos": "SONOS (5 days)"}
    r = d[(d.periphery == "realistic") & (d.t == d.groupby("device").t.transform("max"))]
    rng = np.random.default_rng(3)
    meths = [("perceptron", F.C2, "s"), ("largemargin", F.C1, "o"), ("dependable", F.C3, "^")]
    for k, (m, c, mk) in enumerate(meths):
        for i, dv in enumerate(devs):
            q = r[(r.method == m) & (r.device == dv)]
            v, ds = q.loss.to_numpy(), q.dataset.to_numpy()
            g = [v[ds == u] for u in np.unique(ds)]
            bs = [np.concatenate([g[j][rng.integers(len(g[j]), size=len(g[j]))]
                                  for j in rng.integers(len(g), size=len(g))]).mean() for _ in range(1000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            y = i + (k - 1) * 0.25
            ax.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(v.mean(), y, mk, color=c, ms=6, markeredgecolor="white", markeredgewidth=0.8,
                    label=F.METH_LABEL[m] if i == 0 else None)
    ax.set_yticks(range(len(devs)))
    ax.set_yticklabels([lab[x] for x in devs])
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(left=-0.002)
    ax.set_xlabel("Analog loss (pp, symlog), 9-bit ADC")
    ax.set_title("(b) Margin training in CrossSim")
    ax.legend(loc="lower right")
    fig.tight_layout()
    F.save(fig, "fig9_crosssim")


if __name__ == "__main__":
    if R("core_analytic2").exists():
        fig_v1_v2(validity_table())
    crosssim()
