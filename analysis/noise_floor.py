"""Monte-Carlo floor of the reference and signed bias of every predictor (reviewer item 1).

The reference is the mean over n simulated chips, so even a perfect predictor shows an
absolute error of E|N(0, se)| = sqrt(2/pi) se with se = chip sd / sqrt(n) (unbiased sd).
For each subset we report MAE, signed mean error (bias), the expected MAE of a perfect
predictor (MC floor) and the model RMS error left after removing the MC variance,
sqrt(max(mean(e^2) - mean(se^2), 0)).
Output: tables/noise_floor.csv
"""
import json

import numpy as np
import pandas as pd

from figures_late import K, SIM
from paths import HERE, R, WORK

NCH = {"core": 8, "encoders": 8, "stress": 6, "precision": 6}


def load(stage):
    S = {tuple(r[k] for k in K): r for r in map(json.loads, open(SIM[stage])) if not r.get("e2e")}
    a1 = {tuple(r[k] for k in K): r for r in map(json.loads, open(R(f"{stage}_analytic")))}
    rows = []
    n = NCH[stage]
    for r2 in map(json.loads, open(R(f"{stage}_analytic2"))):
        k = tuple(r2[x] for x in K)
        s, r1 = S[k], a1[k]
        for i, t in enumerate(r2["times"]):
            rows.append(dict(stage=stage, method=r2["method"], center=s["center"],
                             noise=r2["noise_scale"], periphery=r2["periphery"], device=r2["device"],
                             sim=100 * s["acc_mean"][i], dig=100 * s["digital_acc"],
                             se=100 * s["acc_std"][i] * np.sqrt(n / (n - 1)) / np.sqrt(n),
                             v1=100 * r1["pred_acc_analytic_t"][i],
                             v2=100 * r2["pred_acc_analytic2_t"][i]))
    return pd.DataFrame(rows)


def summarise(g, cols=("v1", "v2")):
    out = {"n": len(g), "floor": np.sqrt(2 / np.pi) * g.se.mean(),
           "se_rms": np.sqrt((g.se ** 2).mean())}
    for c in cols:
        e = g[c] - g.sim
        out[f"mae_{c}"] = e.abs().mean()
        out[f"bias_{c}"] = e.mean()
        out[f"excess_rms_{c}"] = np.sqrt(max((e ** 2).mean() - (g.se ** 2).mean(), 0.0))
    return pd.Series(out)


def main():
    d = pd.concat([load(s) for s in SIM])
    MARGIN = ["largemargin", "dependable", "noisecal"]
    d["sub"] = np.where(d.stage == "stress",
                        np.where(d.center, "centred x", "uncentred x") + d.noise.map("{:g}".format),
                        np.where(d.stage == "precision", d.periphery, "all"))
    rows = []
    for (st, sub), g in d.groupby(["stage", "sub"]):
        rows.append({"stage": st, "subset": sub, **summarise(g)})
    c = d[d.stage == "core"]
    for name, g in [("margin-trained", c[c.method.isin(MARGIN)]),
                    ("no margin", c[~c.method.isin(MARGIN)])]:
        rows.append({"stage": "core", "subset": name, **summarise(g)})
    q = pd.read_pickle(WORK / "val_core_quad.pkl")       # exact v1 (quadrature), core
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "tables" / "noise_floor.csv", index=False)
    print(t.round(3).to_string())
    print("core quad v1 MAE", round((q.quad - q.sim).abs().mean() * (100 if q.sim.max() <= 1 else 1), 3))
    return d


if __name__ == "__main__":
    main()
