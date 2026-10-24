"""v2 component ablation and run-time comparison (cell 16 output).

Inputs: <stage>_ablation (results/v2_<stage>_ablation.jsonl) paired with the simulated
chip means of the same records. Outputs: tables/v2_ablation.csv (MAE / >2 pp share per
variant, stage and subset), tables/v2_ablation_device.csv, tables/runtime.csv.
"""
import json

import numpy as np
import pandas as pd

from paths import HERE, R

OUT = HERE / "tables"
K = ["job_key", "method", "precision", "device", "noise_scale", "periphery"]
VAR = ["full", "no_dac", "no_gain", "no_clip"]


def load(stage):
    S = {tuple(r[k] for k in K): r for r in map(json.loads, open(R(stage))) if not r.get("e2e")}
    rows, timing = [], []
    for a in map(json.loads, open(R(f"{stage}_ablation"))):
        s = S[tuple(a[k] for k in K)]
        for i, t in enumerate(a["times"]):
            rows.append(dict(stage=stage, dataset=s["dataset"], D=s["dim"], center=s["center"],
                             method=a["method"], precision=a["precision"], device=a["device"],
                             noise=a["noise_scale"], t=t, sim=100 * s["acc_mean"][i],
                             **{v: 100 * a[f"pred_{v}_t"][i] for v in VAR}))
        if "t_mc8_s" in a:
            timing.append(dict(stage=stage, dataset=s["dataset"], D=s["dim"], device=a["device"],
                               n_test=a["n_test"], n_times=len(a["times"]),
                               t_mc8=a["t_mc8_s"], t_closed=a["t_closed_s"]))
    return pd.DataFrame(rows), pd.DataFrame(timing)


def main():
    ds, ts = zip(*[load(s) for s in ("core", "stress", "encoders") if R(f"{s}_ablation").exists()])
    d, tm = pd.concat(ds), pd.concat(ts)
    d["sub"] = np.where(d.stage == "stress",
                        np.where(d.center, "centred", "uncentred") + " x" + d.noise.map("{:g}".format),
                        "all")
    for v in VAR:
        d[f"e_{v}"] = (d[v] - d.sim).abs()
    agg = {f"MAE_{v}": (f"e_{v}", "mean") for v in VAR}
    agg.update({f"gt2_{v}": (f"e_{v}", lambda x: (x > 2).mean() * 100) for v in VAR})
    agg["n"] = ("e_full", "size")
    t = d.groupby(["stage", "sub"]).agg(**agg)
    t.to_csv(OUT / "v2_ablation.csv")
    print(t.round(3).to_string())
    td = d.groupby(["stage", "device"]).agg(**{f"MAE_{v}": (f"e_{v}", "mean") for v in VAR})
    td.to_csv(OUT / "v2_ablation_device.csv")
    print(td.round(3).to_string())
    tm["speedup"] = tm.t_mc8 / tm.t_closed
    rt = tm.groupby(["D"]).agg(n=("speedup", "size"), t_mc8_med=("t_mc8", "median"),
                               t_v2_med=("t_closed", "median"),
                               speedup_med=("speedup", "median"),
                               speedup_min=("speedup", "min"), speedup_max=("speedup", "max"))
    rt.to_csv(OUT / "runtime.csv")
    print(rt.round(2).to_string())
    print("overall median speed-up", tm.speedup.median().round(2),
          "| total MC8", tm.t_mc8.sum().round(0), "s vs closed", tm.t_closed.sum().round(0), "s")


if __name__ == "__main__":
    main()
