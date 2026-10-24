"""Joins simulated accuracies with the calibration-chip and closed-form (v1) predictions
and writes the per-point tables used by stats_predictors.py and figures.py."""
import json
import numpy as np
import pandas as pd
from paths import R, WORK

K = ["job_key", "method", "precision", "device", "noise_scale", "periphery"]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def build(stage):
    S = {tuple(r[k] for k in K): r for r in load(R(stage)) if not r.get("e2e")}
    P = {tuple(r[k] for k in K): r for r in load(R(f"{stage}_pred"))}
    rows = []
    for a in load(R(f"{stage}_analytic")):
        k = tuple(a[x] for x in K)
        s, p = S[k], P.get(k)
        ok = p is not None and p["reproduced"]
        for i, t in enumerate(a["times"]):
            rows.append(dict(dataset=s["dataset"], encoder=s["encoder"], D=s["dim"], center=s["center"],
                             method=a["method"], precision=a["precision"], device=a["device"],
                             noise=a["noise_scale"], periphery=a["periphery"], t=t,
                             sim=s["acc_mean"][i], chipsd=s["acc_std"][i], dig=s["digital_acc"],
                             an=a["pred_acc_analytic_t"][i], in_scale=a["in_scale"],
                             sdiff=p["pred_acc_diff_t"][i] if ok else np.nan,
                             shat=p["pred_acc_t"][i] if ok else np.nan, reproduced=a["reproduced"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for stage, name in [("core", "core"), ("stress", "stress"), ("precision", "prec"), ("encoders", "enc")]:
        d = build(stage)
        d.to_pickle(WORK / f"val_{name}_analytic.pkl")
        e = (d.an - d.sim).abs() * 100
        print(f"{stage:10s} points {len(d):6d}  closed-form MAE {e.mean():.3f} pp")
