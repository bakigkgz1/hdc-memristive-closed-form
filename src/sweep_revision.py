"""Revision experiments (reviewer round 1).

Stage "chips64"  - the model against a 64-chip reference (Monte-Carlo floor ~0.02 pp).
    Subset of the core stage: 7 datasets x {rp, nonlinear} x D {512, 2048} x seed 0 x
    {perceptron, onlinehd, largemargin} x {fp, 4bit, binary} x 3 devices, 1-day read,
    realistic periphery: 756 points. Per point it stores
      acc_chips      accuracy of 64 independently programmed AIHWKit chips
      v1_quad        closed form v1 (Gauss-Hermite, no draws)
      v2             periphery-aware model, 100 draws
      pc_affine      strong calibration-chip baseline: per-class affine fit
                     A_c = a_c (h.m_c) + b_c + N(0, V_c) on one programmed chip, then the
                     exact Gaussian integral with the fitted scores and variances
      sdiff, shat    the two calibration-chip baselines of the paper (same chip, exact integral)
      sur_v1, sur_v2 predicted chip-to-chip distribution: 64 surrogate chips whose device
                     values are drawn ONCE per chip from N(m_t(w), v_t(w)) (v2 also with one
                     read gain per tile and per-tile ADC clip/round), i.e. the per-chip
                     accuracy implied by the single-device statistics alone
      in_scale, in_scale_full   read scale from the first 2M partial sums (as in all runs)
                     and from all partial sums (effect of the subsampling)
Stage "onlinehd" - OnlineHD with its own step size.
    7 datasets x {rp, nonlinear} x D {512, 2048, 8192} x seeds {0,1,2}; methods perceptron,
    onlinehd (the shared perceptron step size, as in the paper) and onlinehd_tuned (step size
    tuned for the OnlineHD rule on the same validation split, wider grid); fp/4bit/binary x
    3 devices, 8 chips, 1-day read. Also the median cosine margin and sigma_Delta (one
    calibration chip) to place every model on the margin-to-noise curve.
Output: results/v3_<stage>.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

EPOCHS = 20
REF_T = 86400.0
DATASETS7 = ["isolet", "ucihar", "mnist", "fmnist", "letter", "pendigits", "satimage"]
LR_GRID_ONLINEHD = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

STAGES = {
    "chips64": dict(encoders=["rp", "nonlinear"], dims=[2048, 512], seeds=[0],
                    methods=["perceptron", "onlinehd", "largemargin"],
                    precisions=["fp", "4bit", "binary"], n_chips=64),
    "baselines": dict(encoders=["rp", "nonlinear"], dims=[2048, 512], seeds=[0],
                      methods=["perceptron", "onlinehd", "largemargin"],
                      precisions=["fp", "4bit", "binary"], n_chips=0),
    "onlinehd": dict(encoders=["rp", "nonlinear"], dims=[8192, 2048, 512], seeds=[0, 1, 2],
                     methods=["perceptron", "onlinehd", "onlinehd_tuned"],
                     precisions=["fp", "4bit", "binary"], n_chips=8),
}
DEVICES = ["pcm", "reram_wan", "reram_cmo"]


# ------------------------------------------------------------------------------ helpers
def quad_acc(S, sd, y, n_nodes=48, bs=2048):
    """Exact mean P(argmax(S + N(0, sd^2)) == y); S (n, C); sd (C,) or (n, C)."""
    import numpy as np
    from scipy.special import log_ndtr
    z, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    sd = np.broadcast_to(np.maximum(sd, 1e-12), S.shape)
    tot = 0.0
    for i in range(0, len(S), bs):
        s, v, yy = S[i:i + bs], sd[i:i + bs], y[i:i + bs]
        idx = np.arange(len(s))
        sy, vy = s[idx, yy][:, None, None], v[idx, yy][:, None, None]
        lp = log_ndtr((sy + vy * z[None, :, None] - s[:, None, :]) / v[:, None, :])
        lp[idx, :, yy] = 0.0
        tot += (np.exp(lp.sum(2)) @ w).sum()
    return float(tot / len(S))


def scores(H, M, bs=8192):
    import numpy as np
    return np.concatenate([(H[i:i + bs].float() @ M.T).numpy().astype(np.float64)
                           for i in range(0, len(H), bs)])


def full_quantile_scale(Mq, H_cal, out_bound=12.0, pct=99.9, tile=512):
    import numpy as np
    vals = []
    for r0 in range(0, Mq.shape[1], tile):
        W = Mq[:, r0:r0 + tile]
        W = W / W.abs().max().clamp_min(1e-12)
        vals.append((H_cal[:, r0:r0 + tile].float() @ W.T).abs().flatten().numpy())
    p = float(np.quantile(np.concatenate(vals), pct / 100.0))
    return min(1.0, out_bound / max(p, 1e-12))


def calibration_chip(Mq, H_cal, y_cal, cfg, s, Ste, yte, seed=12345):
    """One programmed calibration chip, training queries only. Returns the predicted test
    accuracy of three Gaussian baselines (exact integral each)."""
    import numpy as np
    import torch
    from hdc import build_crossbar
    C, D = Mq.shape
    torch.manual_seed(seed)
    layer = build_crossbar(Mq, cfg)
    layer.program_analog_weights()
    layer.drift_analog_weights(REF_T)
    with torch.no_grad():
        A = np.concatenate([(layer(H_cal[i:i + 2048].float() * s) / s).numpy()
                            for i in range(0, len(H_cal), 2048)]).astype(np.float64)
    R = scores(H_cal, Mq)
    norm = Mq.norm(dim=1).numpy().astype(np.float64) * np.sqrt(D)
    E = (A - R) / norm
    comp = np.where(np.arange(C)[None, :] == y_cal[:, None], -np.inf, R / norm).argmax(1)
    idx = np.arange(len(y_cal))
    shat = float(E.std())
    sdiff = float((E[idx, y_cal] - E[idx, comp]).std())
    a, b, V = np.ones(C), np.zeros(C), np.zeros(C)
    for c in range(C):  # per-class affine fit of the analog score on the digital score
        X = np.c_[R[:, c], np.ones(len(R))]
        coef, *_ = np.linalg.lstsq(X, A[:, c], rcond=None)
        a[c], b[c] = coef
        V[c] = np.var(A[:, c] - X @ coef)
    return {"shat": shat, "sdiff": sdiff,
            "pred_shat": quad_acc(Ste, shat * norm, yte),
            "pred_sdiff": quad_acc(Ste, sdiff / np.sqrt(2) * norm, yte),
            "pred_pc_affine": quad_acc(Ste * a + b, np.sqrt(V), yte),
            "pc_gain": a.round(4).tolist()}


def calibration_chips_pooled(Mq, H_cal, y_cal, cfg, s, Ste, yte, n_cal=8, seed=12345):
    """The calibration-chip baselines fitted on n_cal programmed chips pooled (training queries
    only), so that chip-specific programming errors average out."""
    import numpy as np
    import torch
    from hdc import build_crossbar
    C, D = Mq.shape
    R1 = scores(H_cal, Mq)
    As = []
    for k in range(n_cal):
        torch.manual_seed(seed + k)
        layer = build_crossbar(Mq, cfg)
        layer.program_analog_weights()
        layer.drift_analog_weights(REF_T)
        with torch.no_grad():
            As.append(np.concatenate([(layer(H_cal[i:i + 2048].float() * s) / s).numpy()
                                      for i in range(0, len(H_cal), 2048)]).astype(np.float64))
    A = np.concatenate(As)
    R = np.tile(R1, (n_cal, 1))
    yy = np.tile(y_cal, n_cal)
    norm = Mq.norm(dim=1).numpy().astype(np.float64) * np.sqrt(D)
    E = (A - R) / norm
    comp = np.where(np.arange(C)[None, :] == yy[:, None], -np.inf, R / norm).argmax(1)
    idx = np.arange(len(yy))
    sdiff = float((E[idx, yy] - E[idx, comp]).std())
    a, b, V = np.ones(C), np.zeros(C), np.zeros(C)
    for c in range(C):
        X = np.c_[R[:, c], np.ones(len(R))]
        coef, *_ = np.linalg.lstsq(X, A[:, c], rcond=None)
        a[c], b[c] = coef
        V[c] = np.var(A[:, c] - X @ coef)
    return {"sdiff8": sdiff, "pred_sdiff8": quad_acc(Ste, sdiff / np.sqrt(2) * norm, yte),
            "pred_pc_affine8": quad_acc(Ste * a + b, np.sqrt(V), yte)}


def device_sigma_baseline(Mq, Ste, yte, device, s, out_noise=0.06, tile=512):
    """Chip-free generic Gaussian: digital scores plus independent Gaussian noise whose variance
    uses one weight-averaged device variance (mean of v_t over the stored weights) instead of
    v_t(w) per device, and no drift distortion of the mean, no compensation factor."""
    import numpy as np
    from analytic import _interp, characterise
    m_t, v_t, m_0 = characterise(device, REF_T, 1.0)
    M = Mq.numpy().astype(np.float64)
    C, D = M.shape
    V = np.zeros(C)
    for r0 in range(0, D, tile):
        blk = M[:, r0:r0 + tile]
        A = np.abs(blk).max()
        vbar = _interp(v_t, blk / A).mean()
        V += A ** 2 * (blk.shape[1] * vbar + out_noise ** 2 / s ** 2)
    return {"pred_device_sigma": quad_acc(Ste, np.sqrt(V), yte)}


def surrogate_v1(Mq, H, y, device, s, n_chips, seed, out_noise=0.06, tile=512):
    """Per-chip accuracy implied by m_t, v_t: device values drawn once per chip."""
    import numpy as np
    from analytic import _interp, characterise
    m_t, v_t, m_0 = characterise(device, REF_T, 1.0)
    M = Mq.numpy().astype(np.float64)
    C, D = M.shape
    mu, sd, vout = np.zeros_like(M), np.zeros_like(M), np.zeros(C)
    for r0 in range(0, D, tile):
        blk = M[:, r0:r0 + tile]
        A = np.abs(blk).max()
        w = blk / A
        mt, vt, m0 = _interp(m_t, w), _interp(v_t, w), _interp(m_0, w)
        al = np.abs(mt).mean() / max(np.abs(m0).mean(), 1e-12)
        mu[:, r0:r0 + tile] = A * mt / al
        sd[:, r0:r0 + tile] = A * np.sqrt(vt) / al
        vout += A ** 2 * out_noise ** 2 / s ** 2
    rng = np.random.default_rng(seed)
    Hn = H.float().numpy()
    acc = []
    for _ in range(n_chips):
        W = mu + sd * rng.standard_normal(mu.shape)
        S = Hn @ W.T.astype(np.float32) + rng.standard_normal((len(Hn), C)) * np.sqrt(vout)
        acc.append(float((S.argmax(1) == y).mean()))
    return acc


def surrogate_v2(Mq, H, y, device, s, n_chips, seed, out_noise=0.06, tile=512):
    """As surrogate_v1 with the v2 periphery: DAC-rounded read scale, one read gain per tile
    and chip (exact extreme-value distribution), per-tile ADC clip and rounding."""
    import numpy as np
    from analytic import (OUT_BOUND, OUT_STEP, PROG_MAX_GAIN, _interp, characterise2,
                          dac_scale, gain_quantiles)
    m_t, v_t, m_0, v_0 = characterise2(device, REF_T, 1.0)
    M = Mq.numpy().astype(np.float64)
    se = dac_scale(s)
    tiles = []
    for r0 in range(0, M.shape[1], tile):
        blk = M[:, r0:r0 + tile]
        A = np.abs(blk).max()
        w = blk / A
        mt, vt, m0 = _interp(m_t, w), _interp(v_t, w), _interp(m_0, w)
        G = gain_quantiles(w, m0, _interp(v_0, w)) if PROG_MAX_GAIN[device] else np.ones(1)
        al = np.abs(mt).mean() / max(np.abs(m0).mean(), 1e-12)
        tiles.append((slice(r0, r0 + tile), A / (s * al), mt, np.sqrt(vt), G))
    rng = np.random.default_rng(seed)
    Hn = H.float().numpy()
    acc = []
    for _ in range(n_chips):
        S = 0.0
        for sl, g, mt, st, G in tiles:
            Gc = G[rng.integers(len(G))]
            Wd = mt + st * rng.standard_normal(mt.shape)
            u = Gc * se * (Hn[:, sl] @ Wd.T.astype(np.float32))
            u = u + rng.standard_normal(u.shape) * out_noise
            u = np.round(np.clip(u, -OUT_BOUND, OUT_BOUND) / OUT_STEP) * OUT_STEP
            S = S + g * u / Gc
        acc.append(float((S.argmax(1) == y).mean()))
    return acc


# ------------------------------------------------------------------------------ worker
def _worker(j: dict) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from analytic import predict_accuracy_analytic2, predict_accuracy_quadrature
    from hdc import (accuracy_crossbar, accuracy_digital, calibrate_input_scale, cosine_margins,
                     load_dataset, make_encoder, make_inference_config, quantize,
                     train_prototypes, tune_lr, tune_margin)
    t0 = time.time()
    Xtr, ytr, Xte, yte, C = load_dataset(j["dataset"], j["data_home"])
    enc = make_encoder(j["encoder"], Xtr.shape[1], j["dim"], j["seed"], Xtr, center=True)
    Htr, Hte = enc(Xtr), enc(Xte)
    cal_idx = torch.randperm(len(Htr), generator=torch.Generator().manual_seed(1))[:5000]
    H_cal, y_cal = Htr[cal_idx], ytr[cal_idx.numpy()]
    lr = tune_lr(Htr, ytr, C, EPOCHS, j["seed"])
    models, info = {}, {}
    for m in j["methods"]:
        if m == "perceptron":
            models[m] = train_prototypes(Htr, ytr, C, EPOCHS, lr=lr, seed=j["seed"])
        elif m == "onlinehd":
            models[m] = train_prototypes(Htr, ytr, C, EPOCHS, lr=lr, seed=j["seed"], adaptive=True)
        elif m == "onlinehd_tuned":
            lr_o = tune_lr(Htr, ytr, C, EPOCHS, j["seed"], grid=LR_GRID_ONLINEHD, adaptive=True)
            info[m] = {"lr_used": lr_o}
            models[m] = train_prototypes(Htr, ytr, C, EPOCHS, lr=lr_o, seed=j["seed"], adaptive=True)
        elif m == "largemargin":
            g = tune_margin(Htr, ytr, C, EPOCHS, j["seed"], lr=lr)
            info[m] = {"margin_used": g}
            models[m] = train_prototypes(Htr, ytr, C, EPOCHS, lr=lr, seed=j["seed"], margin=g)
    t_train = time.time() - t0
    out = []
    for m, prec in itertools.product(j["methods"], j["precisions"]):
        Mq = quantize(models[m], prec)
        s = calibrate_input_scale(Mq, H_cal)
        Ste = scores(Hte, Mq)
        mq = cosine_margins(Mq, Hte, yte)
        base = {"job_key": j["job_key"], "stage": j["stage"], "dataset": j["dataset"],
                "encoder": j["encoder"], "dim": j["dim"], "seed": j["seed"], "method": m,
                "precision": prec, "lr": lr, **info.get(m, {}),
                "digital_acc": accuracy_digital(Mq, Hte, yte), "in_scale": s,
                "margin_median": float(np.median(mq))}
        if j["stage"] == "chips64":
            base["in_scale_full"] = full_quantile_scale(Mq, H_cal)
        for dev in DEVICES:
            cfg = make_inference_config(dev, "realistic")
            if j["stage"] == "baselines":
                rec = {**base, "device": dev}
                rec.update(calibration_chips_pooled(Mq, H_cal, y_cal, cfg, s, Ste, yte))
                rec.update(device_sigma_baseline(Mq, Ste, yte, dev, s))
                out.append(rec)
                continue
            t1 = time.time()
            acc = accuracy_crossbar(Mq, Hte, yte, cfg, [REF_T], j["n_chips"], seed=j["seed"],
                                    in_scale=s)[:, 0]
            rec = {**base, "device": dev, "n_chips": j["n_chips"],
                   "acc_chips": acc.round(6).tolist(), "acc_mean": float(acc.mean()),
                   "sim_seconds": round(time.time() - t1, 1)}
            rec.update(calibration_chip(Mq, H_cal, y_cal, cfg, s, Ste, yte))
            if j["stage"] == "chips64":
                t2 = time.time()
                rec["v1_quad"] = predict_accuracy_quadrature(Mq, Hte, yte, dev, REF_T, s)
                rec["v2"] = predict_accuracy_analytic2(Mq, Hte, yte, dev, REF_T, s, n_draws=100)
                rec["pred_seconds"] = round(time.time() - t2, 2)
                rec["sur_v1"] = surrogate_v1(Mq, Hte, yte, dev, s, 64, seed=7)
                rec["sur_v2"] = surrogate_v2(Mq, Hte, yte, dev, s, 64, seed=7)
            out.append(rec)
    for r in out:
        r["train_seconds"] = round(t_train, 1)
        r["job_seconds"] = round(time.time() - t0, 1)
    return out


def build_jobs(stage, data_home):
    p = STAGES[stage]
    jobs = []
    for d, e, dim, s in itertools.product(DATASETS7, p["encoders"], p["dims"], p["seeds"]):
        j = {"stage": stage, "dataset": d, "encoder": e, "dim": dim, "seed": s,
             "methods": p["methods"], "precisions": p["precisions"], "n_chips": p["n_chips"]}
        j["job_key"] = hashlib.md5(json.dumps(j, sort_keys=True).encode()).hexdigest()
        j["data_home"] = data_home
        jobs.append(j)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--data_home", default="./data")
    ap.add_argument("--datasets", default=None, help="comma list, for a smoke test")
    a = ap.parse_args()
    path = Path(a.outdir) / f"v3_{a.stage}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for l in path.read_text().splitlines():
            try:
                done.add(json.loads(l)["job_key"])
            except (json.JSONDecodeError, KeyError):
                pass
    from hdc import load_dataset
    for d in DATASETS7:
        load_dataset(d, a.data_home)
    jobs = [j for j in build_jobs(a.stage, a.data_home) if j["job_key"] not in done
            and (a.datasets is None or j["dataset"] in a.datasets.split(","))]
    print(f"[{a.stage}] {len(jobs)} jobs to run ({len(done)} done) on {a.workers} workers -> {path}",
          flush=True)
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, path.open("a") as f:
        futs = {ex.submit(_worker, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            j = futs[fu]
            try:
                recs = fu.result()
            except Exception as e:
                print(f"FAILED {j['dataset']} {j['encoder']} D={j['dim']} seed={j['seed']}: {e!r}",
                      flush=True)
                continue
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            el = time.time() - t_start
            msg = ""
            if a.stage == "chips64":
                import numpy as np
                e1 = np.mean([abs(r["v1_quad"] - r["acc_mean"]) for r in recs]) * 100
                e2 = np.mean([abs(r["v2"] - r["acc_mean"]) for r in recs]) * 100
                msg = f"MAE v1 {e1:.3f} v2 {e2:.3f} pp; "
            print(f"[{i}/{len(jobs)}] {j['dataset']} {j['encoder']} D={j['dim']} seed={j['seed']}: "
                  f"{msg}[{recs[0]['job_seconds']}s, elapsed {el/3600:.2f} h, "
                  f"ETA {el / i * (len(jobs) - i) / 3600:.2f} h]", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
