"""Statistics for the manuscript (all paired, non-parametric).

Unit of analysis: one (dataset, encoder, D, seed, device, precision[, noise]) configuration,
evaluated at the reference read time t = 1 day; its value is the mean over 8 programmed
chips. Comparisons are paired on identical configurations.
  * Wilcoxon signed-rank test (two-sided, zero-differences split, 'pratt' handling),
  * Holm-Bonferroni correction within each family of comparisons,
  * 95 % CI of the mean paired difference by a cluster bootstrap that resamples whole
    datasets (7 clusters) and, within them, configurations -> honest about the fact that
    configurations from one dataset are not independent,
  * matched-pairs rank-biserial correlation as effect size,
  * win / tie / loss counts (tie: |diff| < 0.05 pp).
Outputs CSV tables to paper/tables/ and prints a summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
A2 = ROOT / "analysis2"
from paths import R
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
REF_T = 86400.0
RNG = np.random.default_rng(2026)
N_BOOT = 4000


def load(path) -> pd.DataFrame:
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if REF_T not in r["times"]:
            continue
        i = r["times"].index(REF_T)
        rows.append(dict(dataset=r["dataset"], encoder=r["encoder"], D=r["dim"], seed=r["seed"],
                         center=r["center"], e2e=r.get("e2e", False), method=r["method"],
                         precision=r["precision"], device=r["device"], noise=r["noise_scale"],
                         periphery=r["periphery"], dig=100 * r["digital_acc"],
                         acc=100 * r["acc_mean"][i], chip_sd=100 * r["acc_std"][i]))
    d = pd.DataFrame(rows)
    d["loss"] = d.dig - d.acc
    return d


def cluster_boot_ci(diff: np.ndarray, cluster: np.ndarray, n=N_BOOT):
    groups = [diff[cluster == c] for c in np.unique(cluster)]
    k = len(groups)
    means = np.empty(n)
    for b in range(n):
        pick = RNG.integers(k, size=k)
        vals = np.concatenate([g[RNG.integers(len(g), size=len(g))] for g in (groups[p] for p in pick)])
        means[b] = vals.mean()
    return np.percentile(means, [2.5, 97.5])


def paired(df, key_cols, factor, a, b, value, family, label=None):
    """value(a) - value(b) paired on key_cols."""
    pa = df[df[factor] == a].set_index(key_cols)[value]
    pb = df[df[factor] == b].set_index(key_cols)[value]
    j = pa.to_frame("a").join(pb.to_frame("b"), how="inner")
    diff = (j.a - j.b).to_numpy()
    clusters = j.index.get_level_values("dataset").to_numpy()
    nz = diff[np.abs(diff) > 1e-12]
    if len(nz) > 0:
        stat = wilcoxon(diff, zero_method="pratt")
        p = stat.pvalue
        r = np.sign(nz) * pd.Series(np.abs(nz)).rank().to_numpy()
        rbc = r.sum() / pd.Series(np.abs(nz)).rank().sum()
    else:
        p, rbc = 1.0, 0.0
    lo, hi = cluster_boot_ci(diff, clusters)
    return dict(family=family, comparison=label or f"{a} - {b}", value=value, n=len(diff),
                mean=diff.mean(), median=np.median(diff), ci_lo=lo, ci_hi=hi,
                win=int((diff > 0.05).sum()), tie=int((np.abs(diff) <= 0.05).sum()),
                loss=int((diff < -0.05).sum()), rank_biserial=rbc, p=p,
                **cluster_test(diff, clusters))


def cluster_test(diff, clusters):
    """Dataset-level inference: the dataset is the independent unit. Exact two-sided
    sign-flip permutation test on the per-dataset mean differences (2^k flips) and the
    number of datasets whose mean difference has the sign of the overall mean."""
    import itertools
    cm = np.array([diff[clusters == c].mean() for c in np.unique(clusters)])
    k = len(cm)
    obs = abs(cm.mean())
    flips = np.array(list(itertools.product([-1, 1], repeat=k)))
    p = float((np.abs((flips * cm).mean(1)) >= obs - 1e-12).mean())
    same = int((np.sign(cm) == np.sign(cm.mean())).sum())
    return dict(k_datasets=k, datasets_same_sign=same, p_cluster=p)


def holm(res: pd.DataFrame) -> pd.DataFrame:
    res = res.copy()
    res["p_holm"] = np.nan
    for fam, g in res.groupby("family"):
        order = g.p.sort_values().index
        m = len(order)
        running = 0.0
        for k, idx in enumerate(order):
            running = max(running, min(1.0, (m - k) * res.at[idx, "p"]))
            res.at[idx, "p_holm"] = running
    return res


def main():
    res = []
    # ---------------- core: training methods (analog accuracy and analog loss) -------------
    core = load(R("core"))
    core = core[~core.e2e]
    K = ["dataset", "encoder", "D", "seed", "device", "precision"]
    for m in ["single", "onlinehd", "largemargin", "dependable", "noisecal"]:
        res.append(paired(core, K, "method", m, "perceptron", "acc", "F1 method: analog accuracy"))
        res.append(paired(core, K, "method", m, "perceptron", "loss", "F2 method: analog loss"))
    res.append(paired(core, K, "method", "noisecal", "largemargin", "acc", "F1 method: analog accuracy"))
    res.append(paired(core, K, "method", "noisecal", "dependable", "acc", "F1 method: analog accuracy"))
    # precision within core (fp / 4bit / binary)
    Kp = ["dataset", "encoder", "D", "seed", "device", "method"]
    res.append(paired(core, Kp, "precision", "4bit", "fp", "acc", "F3 precision (core)"))
    res.append(paired(core, Kp, "precision", "binary", "4bit", "acc", "F3 precision (core)"))
    # ---------------- precision stage incl. 2 bit, realistic periphery --------------------
    prec = load(R("precision"))
    pr = prec[prec.periphery == "realistic"]
    Kq = ["dataset", "encoder", "D", "seed", "device", "method"]
    res.append(paired(pr, Kq, "precision", "2bit", "4bit", "acc", "F4 precision (2 bit stage)"))
    res.append(paired(pr, Kq, "precision", "binary", "2bit", "acc", "F4 precision (2 bit stage)"))
    res.append(paired(pr, Kq, "precision", "4bit", "fp", "acc", "F4 precision (2 bit stage)"))
    # ---------------- stress: centring at nominal noise -----------------------------------
    st = load(R("stress"))
    s1 = st[st.noise == 1.0]
    Kc = ["dataset", "encoder", "D", "seed", "device", "precision", "method"]
    res.append(paired(s1, Kc, "center", False, True, "loss", "F5 centring (noise x1)",
                      "uncentred - centred"))
    res.append(paired(s1, Kc, "center", False, True, "chip_sd", "F5 centring (noise x1)",
                      "uncentred - centred"))
    # noisecal vs largemargin under stress, per noise level (centred)
    for ns in sorted(st.noise.unique()):
        sn = st[(st.noise == ns) & st.center]
        Kn = ["dataset", "encoder", "D", "seed", "device", "precision"]
        res.append(paired(sn, Kn, "method", "noisecal", "largemargin", "acc",
                          "F6 noisecal vs largemargin by noise", f"noisecal - largemargin @x{ns:g}"))
    # ---------------- end to end: bipolar vs Gaussian projection --------------------------
    e2e = load(R("e2e"))
    bip = load(R("e2e_bipolar"))
    g = e2e[(e2e.encoder == "rp") & (e2e.method == "largemargin")].copy()
    b = bip.copy()
    g["proj"], b["proj"] = "gaussian", "bipolar"
    eb = pd.concat([g, b])
    Ke = ["dataset", "D", "seed", "device", "precision"]
    eb = eb[eb.D.isin(b.D.unique()) & eb.seed.isin(b.seed.unique())]
    res.append(paired(eb, Ke, "proj", "bipolar", "gaussian", "acc", "F7 end-to-end projection"))
    for dev in sorted(eb.device.unique()):
        res.append(paired(eb[eb.device == dev], Ke, "proj", "bipolar", "gaussian", "acc",
                          "F7 end-to-end projection", f"bipolar - gaussian ({dev})"))

    out = holm(pd.DataFrame(res))
    out.to_csv(OUT / "paired_tests.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 46)
    print(out[["family", "comparison", "value", "n", "mean", "ci_lo", "ci_hi", "win", "tie",
               "loss", "rank_biserial", "p_holm"]].to_string(
        index=False, float_format=lambda x: f"{x:.3g}"))

    # ---------------- per-method descriptive table (core, t = 1 day) ----------------------
    desc = core.groupby(["method"]).agg(dig=("dig", "mean"), acc=("acc", "mean"),
                                        loss=("loss", "mean"), loss_p95=("loss", lambda x: x.quantile(.95)),
                                        chip_sd=("chip_sd", "mean"), n=("acc", "size"))
    desc.to_csv(OUT / "core_methods.csv")
    print("\n", desc.round(3))
    devm = core.pivot_table(index="method", columns=["device"], values="loss", aggfunc="mean")
    devm.to_csv(OUT / "core_loss_method_device.csv")
    print("\n", devm.round(3))


if __name__ == "__main__":
    main()
