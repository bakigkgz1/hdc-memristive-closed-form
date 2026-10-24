"""Iso-accuracy dimension saving of margin training (core, t = 1 day, realistic periphery).
For each (dataset, encoder, device, precision): target = mean analog accuracy of perceptron
retraining at D = 8192 (over seeds). D_needed(method) = smallest D of the grid whose
seed-mean analog accuracy reaches the target (inf if none). Saving = D_needed(perceptron) /
D_needed(method)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from paths import R
ROOT = Path(__file__).resolve().parents[1]
rows = []
for l in open(R("core")):
    r = json.loads(l)
    if r.get("e2e"): continue
    i = r["times"].index(86400.0)
    rows.append(dict(dataset=r["dataset"], encoder=r["encoder"], D=r["dim"], seed=r["seed"],
                     method=r["method"], precision=r["precision"], device=r["device"], acc=100 * r["acc_mean"][i]))
d = pd.DataFrame(rows).groupby(["dataset", "encoder", "device", "precision", "method", "D"]).acc.mean().reset_index()
out = []
for k, g in d.groupby(["dataset", "encoder", "device", "precision"]):
    tgt = g[(g.method == "perceptron") & (g.D == 8192)].acc.iloc[0]
    need = {}
    for m, gm in g.groupby("method"):
        ok = gm[gm.acc >= tgt - 1e-9].D
        need[m] = ok.min() if len(ok) else np.inf
    for m in ["largemargin", "dependable", "noisecal"]:
        out.append(dict(zip(["dataset", "encoder", "device", "precision"], k), method=m,
                        D_perc=need["perceptron"], D_m=need[m], saving=need["perceptron"] / need[m]))
o = pd.DataFrame(out)
o.to_csv(ROOT / "paper/tables/iso_dimension.csv", index=False)
for m, g in o.groupby("method"):
    s = g.saving
    print(m, "n", len(s), "median %.1fx" % s.median(), "IQR", s.quantile([.25, .75]).tolist(),
          "min", s.min(), "max", s.max(), "share>=2x %.0f%%" % ((s >= 2).mean() * 100), "share>1x %.0f%%" % ((s > 1).mean() * 100))
print(o[o.method == "largemargin"].groupby("precision").saving.median())
