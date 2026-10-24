"""Merged validity table (former Tables 4 and 5): v1, v2, MC floor, bias, sigma_Delta, true loss.
Core v1 is the exact (quadrature) value. Output: tables/validity_merged.csv"""
import numpy as np
import pandas as pd

import noise_floor as NF
from paths import HERE, WORK

MARGIN = ["largemargin", "dependable", "noisecal"]
d = pd.concat([NF.load(s) for s in NF.NCH])
d["loss"] = np.nan
q = pd.read_pickle(WORK / "val_core_quad.pkl")
c = d.stage == "core"
A = {"core": "val_core_analytic", "stress": "val_stress_analytic", "precision": "val_prec_analytic",
     "encoders": "val_enc_analytic"}
sd = []
for st, f in A.items():
    a = pd.read_pickle(WORK / f"{f}.pkl")
    a["stage"] = st
    a["sim"], a["sdiff"], a["dig"] = a.sim * 100, a.sdiff * 100, a.dig * 100
    sd.append(a)
sd = pd.concat(sd)


def sub(df):
    return np.where(df.stage == "stress",
                    np.where(df.center, "centred x", "uncentred x") + df.noise.map("{:g}".format),
                    np.where(df.stage == "precision", df.periphery, "all"))


d["sub"], sd["sub"] = sub(d), sub(sd)
rows = []
groups = [("core", "all"), ("core", "margin-trained"), ("core", "no margin")] + \
         [("stress", f"{c} x{n}") for c in ("centred", "uncentred") for n in ("0.5", "1", "2", "4", "8")] + \
         [("precision", "ideal"), ("precision", "realistic"), ("encoders", "all")]
for st, s in groups:
    if s in ("margin-trained", "no margin"):
        f = lambda df: (df.stage == st) & (df.method.isin(MARGIN) == (s == "margin-trained"))
    else:
        f = lambda df: (df.stage == st) & (df["sub"] == s)
    g, h = d[f(d)], sd[f(sd)].dropna(subset=["sdiff"])
    e1, e2 = g.v1 - g.sim, g.v2 - g.sim
    if st == "core":  # exact v1 (quadrature) for the core stage
        qq = q if s == "all" else q[q.m.isin(MARGIN) == (s == "margin-trained")]
        e1 = qq.quad - qq.sim
    rows.append(dict(stage=st, subset=s, n=len(g), mae_v1=e1.abs().mean(), mae_v2=e2.abs().mean(),
                     bias_v2=e2.mean(), floor=np.sqrt(2 / np.pi) * g.se.mean(),
                     gt2_v1=(e1.abs() > 2).mean() * 100, gt2_v2=(e2.abs() > 2).mean() * 100,
                     p95_v1=e1.abs().quantile(.95), p95_v2=e2.abs().quantile(.95),
                     n_sdiff=len(h), mae_sdiff=(h.sdiff - h.sim).abs().mean(),
                     true_loss=(g.dig - g.sim).mean()))
t = pd.DataFrame(rows)
t.to_csv(HERE / "tables" / "validity_merged.csv", index=False)
print(t.round(3).to_string())
