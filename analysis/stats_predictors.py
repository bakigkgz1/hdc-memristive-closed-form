"""Accuracy-predictor comparison: closed form (no chip) vs sigma_Delta and sigma_hat
(one calibration chip each). Absolute prediction error |pred - mean over 8 chips| in pp,
paired per point; Wilcoxon signed-rank + Holm, cluster bootstrap (datasets) CI.
Inputs: analysis2/val_*_analytic.pkl written by the validation scripts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from stats import cluster_test

ROOT = Path(__file__).resolve().parents[1]
from paths import WORK as A2
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)


def boot(diff, cl, n=4000):
    gs = [diff[cl == c] for c in np.unique(cl)]
    k = len(gs)
    m = np.empty(n)
    for b in range(n):
        m[b] = np.concatenate([gs[i][RNG.integers(len(gs[i]), size=len(gs[i]))]
                               for i in RNG.integers(k, size=k)]).mean()
    return np.percentile(m, [2.5, 97.5])


def compare(d, stage, subset):
    d = d.dropna(subset=["sdiff", "shat"])
    ea = (d.an - d.sim).abs() * 100
    rows = []
    for name, col in [("sigma_Delta", "sdiff"), ("sigma_hat", "shat")]:
        eo = (d[col] - d.sim).abs() * 100
        diff = (ea - eo).to_numpy()
        lo, hi = boot(diff, d.dataset.to_numpy())
        rows.append(dict(stage=stage, subset=subset, n=len(d), MAE_closed=ea.mean(),
                         MAE_other=eo.mean(), other=name, mean_diff=diff.mean(), ci_lo=lo, ci_hi=hi,
                         frac_closed_better=(diff < 0).mean(),
                         p=wilcoxon(diff, zero_method="pratt").pvalue,
                         **cluster_test(diff, d.dataset.to_numpy())))
    return rows


def main():
    core = pd.read_pickle(A2 / "val_core_analytic.pkl")
    st = pd.read_pickle(A2 / "val_stress_analytic.pkl")
    pr = pd.read_pickle(A2 / "val_prec_analytic.pkl")
    en = pd.read_pickle(A2 / "val_enc_analytic.pkl")
    rows = []
    rows += compare(core, "core", "all (centred, x1)")
    for ns in sorted(st.noise.unique()):
        rows += compare(st[st.center & (st.noise == ns)], "stress", f"centred x{ns:g}")
        rows += compare(st[~st.center & (st.noise == ns)], "stress", f"uncentred x{ns:g}")
    for per in ["ideal", "realistic"]:
        rows += compare(pr[pr.periphery == per], "precision", per)
    rows += compare(en, "encoders", "ID-level")
    r = pd.DataFrame(rows)
    r["p_holm"] = np.nan
    order = r.p.sort_values().index
    run = 0.0
    for k, idx in enumerate(order):
        run = max(run, min(1.0, (len(r) - k) * r.at[idx, "p"]))
        r.at[idx, "p_holm"] = run
    r.to_csv(OUT / "predictor_comparison.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.drop(columns=["p"]).to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    # validity map of the closed form (MAE, p95, share > 2 pp), all points incl. no-chip-pred ones
    vm = []
    def add(d, stage, subset):
        e = (d.an - d.sim).abs() * 100
        vm.append(dict(stage=stage, subset=subset, n=len(d), MAE=e.mean(), p95=e.quantile(.95),
                       share_gt2pp=(e > 2).mean() * 100, true_loss=((d.dig - d.sim) * 100).mean()))
    add(core, "core", "all")
    mm = core.method.isin(["largemargin", "dependable", "noisecal"])
    add(core[mm], "core", "margin-trained")
    add(core[~mm], "core", "no margin")
    for ns in sorted(st.noise.unique()):
        add(st[st.center & (st.noise == ns)], "stress", f"centred x{ns:g}")
        add(st[~st.center & (st.noise == ns)], "stress", f"uncentred x{ns:g}")
    for per in ["ideal", "realistic"]:
        add(pr[pr.periphery == per], "precision", per)
    add(en, "encoders", "ID-level")
    vm = pd.DataFrame(vm)
    vm.to_csv(OUT / "closed_form_validity.csv", index=False)
    print("\n", vm.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
