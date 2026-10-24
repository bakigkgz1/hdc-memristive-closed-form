"""Numbers quoted in the text that no other script prints (second pre-submission review):
64-chip v2 bias CI, ID-level v2 vs sigma_Delta, OnlineHD median margin-to-noise ratios,
bootstrap CI of the margin-law exponent, t-interval sensitivity for Table 2, reproducibility
rates per stage and the reproduced vs non-reproduced v1 error. Output: tables/extra_numbers.csv"""
import numpy as np, pandas as pd
from scipy import stats as S
import stats as ST
import margin_law as ML
from paths import HERE, WORK, R
A = HERE.parent / "analysis2"
rng = np.random.default_rng(2026)
out = []
def add(k, v): out.append(dict(item=k, value=v)); print(f"{k}: {v}")

# 1. 64-chip check: signed error of v2, 95 % cluster-bootstrap CI (datasets, then points)
c = pd.read_pickle(A / "val_chips64.pkl")
sc = 100 if c.sim64.max() <= 1 else 1
e = (c.v2 * (100 if c.v2.max() <= 1 else 1) - c.sim64 * sc).to_numpy()
lo, hi = ST.cluster_boot_ci(e, c.dataset.to_numpy())
add("chips64 v2 bias (pp) mean, CI", f"{e.mean():+.4f} [{lo:+.3f}, {hi:+.3f}], SE {e.std(ddof=1)/np.sqrt(len(e)):.4f}")

# 2. ID-level: |v2 err| - |sigma_Delta err| on paired points
p = pd.read_pickle(WORK / "val_paired_all.pkl")
g = p[p.stage == "encoders"]
d = ((g.v2 - g.sim).abs() - (g.sdiff - g.sim).abs()).to_numpy()
lo, hi = ST.cluster_boot_ci(d, g.dataset.to_numpy())
add("ID-level v2 - sigma_Delta (pp)", f"{d.mean():+.3f} [{lo:+.2f}, {hi:+.2f}], n={len(d)}, "
    f"datasets same sign {ST.cluster_test(d, g.dataset.to_numpy())['datasets_same_sign']}/7")

# 3. OnlineHD step-size subset: median margin-to-noise ratio per rule
o = pd.read_pickle(A / "val_onlinehd.pkl")
add("OnlineHD subset median x", o.groupby("method").x.median().round(2).to_dict())

# 4. margin law: exponent of the binned power law, bootstrap over datasets
m = ML.load()
edges = np.quantile(m.x, np.linspace(0, 1, 9))
rr = ["perceptron", "onlinehd", "largemargin", "dependable", "noisecal"]
def slope(dd):
    dd = dd.assign(bin=pd.cut(dd.x, edges, include_lowest=True))
    s = dd[dd.method.isin(rr)].groupby("bin", observed=True).agg(x=("x", "median"), l=("loss", "median")).iloc[1:7]
    s = s[s.l > 0]
    return np.polyfit(np.log(s.x), np.log(s.l), 1)[0]
ds = m.dataset.unique()
bs = [slope(pd.concat([m[m.dataset == k] for k in rng.choice(ds, len(ds))])) for _ in range(1000)]
add("margin-law exponent, CI (dataset bootstrap)", f"{slope(m):.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]")
add("core median x per rule", m.groupby("method").x.median().round(1).to_dict())

# 5. Table 2: t-interval over the seven dataset means (6 df) as a sensitivity check
core = ST.load(R("core")); core = core[~core.e2e]
K = ["dataset", "encoder", "D", "seed", "device", "precision"]
for a, b, v in [("largemargin", "perceptron", "acc"), ("dependable", "perceptron", "acc"),
                ("noisecal", "perceptron", "acc"), ("largemargin", "perceptron", "loss"),
                ("dependable", "perceptron", "loss"), ("noisecal", "perceptron", "loss"),
                ("onlinehd", "perceptron", "loss"), ("noisecal", "largemargin", "acc")]:
    pa = core[core.method == a].set_index(K)[v]; pb = core[core.method == b].set_index(K)[v]
    j = (pa - pb).dropna()
    cm = j.groupby(level="dataset").mean()
    h = S.t.ppf(0.975, len(cm) - 1) * cm.std(ddof=1) / np.sqrt(len(cm))
    add(f"Table 2 t-interval {a}-{b} {v}", f"{cm.mean():+.2f} [{cm.mean()-h:+.2f}, {cm.mean()+h:+.2f}]")

# 6. reproducibility of calibration-chip records and v1 error on reproduced vs other points
for st, f in [("core", "val_core_analytic"), ("stress", "val_stress_analytic"),
              ("precision", "val_prec_analytic"), ("ID-level", "val_enc_analytic")]:
    a = pd.read_pickle(WORK / f"{f}.pkl"); r = a.sdiff.notna(); err = (a.an - a.sim).abs() * 100
    extra = ""
    if st == "stress":
        extra = f", centred {a[a.center].sdiff.notna().mean()*100:.1f} %, uncentred {a[~a.center].sdiff.notna().mean()*100:.1f} %"
    add(f"reproduced share {st}", f"{r.mean()*100:.1f} %{extra}; v1 MAE reproduced {err[r].mean():.3f}, other {err[~r].mean() if (~r).any() else float('nan'):.3f}")
    if st == "core":
        add("core reproduced share per rule", (r.groupby(a.method).mean() * 100).round(1).to_dict())
        add("core v1 MAE other/reproduced per rule", err.groupby([a.method, r]).mean().round(3).to_dict())
pd.DataFrame(out).to_csv(HERE / "tables" / "extra_numbers.csv", index=False)
