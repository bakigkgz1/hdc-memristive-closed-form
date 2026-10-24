"""Analysis of the revision runs: 64-chip reference (cell 22) and OnlineHD step size (cell 23)."""
import json
import numpy as np, pandas as pd
from paths import HERE
A = HERE.parent / "analysis2"
rng = np.random.default_rng(3)

def boot(diff, cl, n=4000):
    gs = [diff[cl == c] for c in np.unique(cl)]
    m = [np.concatenate([gs[i][rng.integers(len(gs[i]), size=len(gs[i]))] for i in rng.integers(len(gs), size=len(gs))]).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])

c = pd.DataFrame(map(json.loads, open(A / "v3_chips64.jsonl")))
c["chips"] = c.acc_chips.apply(np.array)
c["sim64"] = c.chips.apply(np.mean) * 100
c["sim8"] = c.chips.apply(lambda a: a[:8].mean()) * 100
c["sd64"] = c.chips.apply(lambda a: a.std(ddof=1)) * 100
c["se64"] = c.sd64 / 8
for k in ["v1_quad", "v2", "pred_sdiff", "pred_shat", "pred_pc_affine", "digital_acc"]:
    c[k] = c[k] * 100
c["sur1_sd"] = c.sur_v1.apply(lambda a: np.std(a, ddof=1)) * 100
c["sur2_sd"] = c.sur_v2.apply(lambda a: np.std(a, ddof=1)) * 100
c["sur2_mean"] = c.sur_v2.apply(np.mean) * 100
c["p5_64"] = c.chips.apply(lambda a: np.percentile(a, 5)) * 100
c["p5_sur2"] = c.sur_v2.apply(lambda a: np.percentile(a, 5)) * 100
c["loss64"] = c.digital_acc - c.sim64
M = c.method == "largemargin"
print("n points", len(c), " median se64", round(c.se64.median(), 4), " floor64", round(np.sqrt(2/np.pi)*c.se64.mean(), 4))
rows = []
for name, g in [("all", c), ("large-margin", c[M]), ("no margin", c[~M])]:
    for k, lab in [("v1_quad", "v1"), ("v2", "v2"), ("pred_sdiff", "sigma_Delta"), ("pred_shat", "sigma_hat"), ("pred_pc_affine", "per-class affine")]:
        e = g[k] - g.sim64
        rows.append(dict(subset=name, predictor=lab, n=len(g), mae=e.abs().mean(), bias=e.mean(),
                         p95=e.abs().quantile(.95), mae_vs8=(g[k]-g.sim8).abs().mean()))
t = pd.DataFrame(rows); print(t.round(3).to_string())
t.to_csv(HERE/"tables"/"chips64_predictors.csv", index=False)
# paired: v2 vs per-class affine, v1 vs per-class
for a, b in [("v2", "pred_pc_affine"), ("v1_quad", "pred_pc_affine"), ("v2", "pred_sdiff")]:
    d = ((c[a]-c.sim64).abs() - (c[b]-c.sim64).abs()).to_numpy()
    lo, hi = boot(d, c.dataset.to_numpy())
    cm = c.assign(d=d).groupby("dataset").d.mean()
    print(f"{a} - {b}: {d.mean():+.3f} CI [{lo:+.3f},{hi:+.3f}] datasets same sign {(np.sign(cm)==np.sign(d.mean())).sum()}/7")
# bias significance of v2 vs 64 chips: mean e and its se
e2 = c.v2 - c.sim64
print("v2 bias", round(e2.mean(),4), "+-", round(e2.std()/np.sqrt(len(e2)),4), " v1 bias", round((c.v1_quad-c.sim64).mean(),4))
# chip spread
s = c[c.sd64 > 0]
r1 = np.corrcoef(np.log(s.sd64), np.log(s.sur1_sd.clip(1e-4)))[0,1]; r2 = np.corrcoef(np.log(s.sd64), np.log(s.sur2_sd.clip(1e-4)))[0,1]
print("chip sd: median sim", round(c.sd64.median(),3), "sur1", round(c.sur1_sd.median(),3), "sur2", round(c.sur2_sd.median(),3))
print("log-corr sd sim vs sur1", round(r1,3), "sur2", round(r2,3))
print("ratio sur2/sim median", round((s.sur2_sd/s.sd64).median(),3), "IQR", (s.sur2_sd/s.sd64).quantile([.25,.75]).round(3).tolist())
print("p5 error sur2 vs sim64 MAE", round((c.p5_sur2-c.p5_64).abs().mean(),3), "bias", round((c.p5_sur2-c.p5_64).mean(),3))
print(c.groupby("device")[["sd64","sur1_sd","sur2_sd"]].median().round(3))
# read scale subsampling
d_s = (c.in_scale_full / c.in_scale - 1).abs()
print("in_scale rel diff: max", d_s.max(), "n differing", (d_s > 1e-9).sum())
c.to_pickle(A / "val_chips64.pkl")

o = pd.DataFrame(map(json.loads, open(A / "v3_onlinehd.jsonl")))
o["loss"] = (o.digital_acc - o.acc_mean) * 100
o["dig"] = o.digital_acc * 100; o["ana"] = o.acc_mean * 100
o["x"] = o.margin_median / o.sdiff
K = ["dataset", "encoder", "dim", "seed", "precision", "device"]
p = o.pivot_table(index=K, columns="method", values=["loss", "dig", "ana", "x"])
print(o.groupby("method")[["dig","ana","loss","x"]].mean().round(3))
print("lr_used dist", o[o.method=="onlinehd_tuned"].drop_duplicates(K[:4]).lr_used.value_counts().to_dict())
cl = p.index.get_level_values("dataset").to_numpy()
for meas in ["loss", "ana", "dig"]:
    for a, b in [("onlinehd_tuned", "perceptron"), ("onlinehd", "perceptron"), ("onlinehd_tuned", "onlinehd")]:
        d = (p[(meas, a)] - p[(meas, b)]).to_numpy()
        lo, hi = boot(d, cl)
        cm = pd.Series(d, index=cl).groupby(level=0).mean()
        print(f"{meas}: {a} - {b}: {d.mean():+.3f} CI [{lo:+.3f},{hi:+.3f}] same sign {(np.sign(cm)==np.sign(d.mean())).sum()}/7")
o.to_pickle(A / "val_onlinehd.pkl")

# ---- cell 24: further baselines on the 64-chip points
b = pd.DataFrame(map(json.loads, open(A / "v3_baselines.jsonl")))
K2 = ["dataset", "encoder", "dim", "method", "precision", "device"]
m = c.merge(b[K2 + ["digital_acc", "pred_sdiff8", "pred_pc_affine8", "pred_device_sigma"]].rename(columns={"digital_acc": "dig_b"}), on=K2)
m = m[np.abs(m.dig_b * 100 - m.digital_acc) < 1e-6].copy()
print("baselines merged, reproduced models:", len(m))
for k in ["pred_sdiff8", "pred_pc_affine8", "pred_device_sigma"]:
    m[k] = m[k] * 100
rows = []
for name, g in [("all", m), ("large-margin", m[m.method == "largemargin"]), ("no margin", m[m.method != "largemargin"])]:
    for k, lab in [("v1_quad", "v1"), ("v2", "v2"), ("pred_device_sigma", "device-sigma Gaussian (no chip)"),
                   ("pred_sdiff", "sigma_Delta, 1 chip"), ("pred_sdiff8", "sigma_Delta, 8 chips"),
                   ("pred_pc_affine", "per-class affine, 1 chip"), ("pred_pc_affine8", "per-class affine, 8 chips")]:
        e = g[k] - g.sim64
        rows.append(dict(subset=name, predictor=lab, n=len(g), mae=e.abs().mean(), bias=e.mean(), p95=e.abs().quantile(.95)))
tb = pd.DataFrame(rows); print(tb.round(3).to_string())
tb.to_csv(HERE / "tables" / "chips64_baselines.csv", index=False)
for a_, b_ in [("v2", "pred_pc_affine8"), ("v2", "pred_sdiff8"), ("v2", "pred_device_sigma"), ("v1_quad", "pred_device_sigma"), ("v1_quad", "pred_pc_affine8")]:
    d = ((m[a_] - m.sim64).abs() - (m[b_] - m.sim64).abs()).to_numpy()
    lo, hi = boot(d, m.dataset.to_numpy())
    cm = m.assign(d=d).groupby("dataset").d.mean()
    print(f"{a_} - {b_}: {d.mean():+.3f} CI [{lo:+.3f},{hi:+.3f}] same sign {(np.sign(cm)==np.sign(d.mean())).sum()}/7")
