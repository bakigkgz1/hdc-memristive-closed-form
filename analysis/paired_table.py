"""Table 1 companion: v1, v2 and sigma_Delta on identical (paired) points, i.e. the points with a
reproduced calibration chip. Output: tables/validity_paired.csv"""
import numpy as np, pandas as pd
import figures_late as L
from paths import HERE, WORK
MARGIN = ["largemargin", "dependable", "noisecal"]
F = {"core": "val_core_analytic", "stress": "val_stress_analytic", "precision": "val_prec_analytic",
     "encoders": "val_enc_analytic"}
K = ["dataset", "encoder", "D", "center", "method", "precision", "device", "noise", "periphery", "t"]
rows, out = [], []
for st, f in F.items():
    a = pd.read_pickle(WORK / f"{f}.pkl")
    a["sim_r"] = (a.sim * 100).round(4)
    v = L.v1v2(st)
    v["sim_r"] = v.sim.round(4)
    v = v.drop_duplicates(K + ["sim_r"])
    m = a.merge(v[K + ["sim_r", "v2"]], on=K + ["sim_r"], how="left")
    assert len(m) == len(a), (st, len(m), len(a))
    m["stage"] = st
    out.append(m)
d = pd.concat(out)
d = d[d.sdiff.notna()].copy()
for c in ["sim", "an", "sdiff", "shat"]:
    d[c] = d[c] * 100
sub = np.where(d.stage == "stress", np.where(d.center, "centred x", "uncentred x") + d.noise.map("{:g}".format),
               np.where(d.stage == "precision", d.periphery, "all"))
d["sub"] = sub
groups = [("core", "all"), ("core", "margin-trained"), ("core", "no margin")] + \
         [("stress", f"{c} x{n}") for c in ("centred", "uncentred") for n in ("0.5", "1", "2", "4", "8")] + \
         [("precision", "ideal"), ("precision", "realistic"), ("encoders", "all")]
for st, s in groups:
    if s in ("margin-trained", "no margin"):
        g = d[(d.stage == st) & (d.method.isin(MARGIN) == (s == "margin-trained"))]
    else:
        g = d[(d.stage == st) & (d["sub"] == s)]
    rows.append(dict(stage=st, subset=s, n_paired=len(g), v2_missing=int(g.v2.isna().sum()),
                     mae_v1=(g.an - g.sim).abs().mean(), mae_v2=(g.v2 - g.sim).abs().mean(),
                     mae_sdiff=(g.sdiff - g.sim).abs().mean(), mae_shat=(g.shat - g.sim).abs().mean()))
t = pd.DataFrame(rows)
t.to_csv(HERE / "tables" / "validity_paired.csv", index=False)
print(t.round(3).to_string())
d.to_pickle(WORK / "val_paired_all.pkl")
