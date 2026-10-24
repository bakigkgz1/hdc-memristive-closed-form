"""Supplementary stage: CLOSED-FORM (simulation-free) predictions for a finished stage.

Same reconstruction as sweep_predict.py (models retrained with the stored hyperparameters,
digital accuracy checked), but instead of a calibration chip it uses only single-device
statistics (analytic.py): predicted accuracy at every time of the grid and the analytic
sigma_Delta. No crossbar is programmed. Output: results/v2_<stage>_analytic.jsonl.

--- original description of the calibration-chip variant follows ---
Supplementary stage: margin-to-noise predictions for an already finished stage.

For every job in results/v2_<stage>.jsonl it rebuilds the encoder and retrains every
prototype set with the *stored* hyperparameters (lr and the margin actually used, so the
noise-calibrated method is reproduced exactly), checks that the digital test accuracy
equals the stored value (training is deterministic), then programs ONE fresh calibration
chip per (method, precision, device, noise, periphery) and records, at every time of the
grid and using training data only:
  sigma_hat_t   std of the analog-minus-digital cosine error
  sigma_diff_t  std of the error difference between the true class and its strongest
                digital competitor (the quantity that actually moves decisions)
and the test accuracy predicted from digital scores + Gaussian noise for both
(pred_acc_t with sigma_hat, pred_acc_diff_t with sigma_diff / sqrt 2). No test-set
device simulation is involved. Output: results/v2_<stage>_pred.jsonl, resumable.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

EPOCHS = 20
FLIP_P = 0.05
N_CAL = 5000


def _worker(job: dict) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from analytic import predict_accuracy_analytic, sigma_delta_analytic
    from hdc import (accuracy_digital, calibrate_input_scale, load_dataset, make_encoder,
                     quantize, train_prototypes)

    t0 = time.time()
    r0 = job["recs"][0]
    Xtr, ytr, Xte, yte, C = load_dataset(r0["dataset"], job["data_home"])
    enc = make_encoder(r0["encoder"], Xtr.shape[1], r0["dim"], r0["seed"], Xtr, center=r0["center"])
    Htr, Hte = enc(Xtr), enc(Xte)
    cal_idx = torch.randperm(len(Htr), generator=torch.Generator().manual_seed(1))[:N_CAL]
    H_cal, y_cal = Htr[cal_idx], ytr[cal_idx.numpy()]
    D = r0["dim"]
    models = {}
    out = []
    for r in job["recs"]:
        m = r["method"]
        mkey = (m, r["margin_used"])  # noisecal has a different margin per device / precision
        if mkey not in models:
            kw = dict(seed=r["seed"])
            if m != "single":
                kw.update(lr=r["lr"])
            if m in ("largemargin", "dependable", "noisecal"):
                kw.update(margin=r["margin_used"])
            if m == "dependable":
                kw.update(flip_p=FLIP_P)
            if m == "onlinehd":
                kw.update(adaptive=True)
            models[mkey] = train_prototypes(Htr, ytr, C, 0 if m == "single" else EPOCHS, **kw)
        Mq = quantize(models[mkey], r["precision"])
        dig = accuracy_digital(Mq, Hte, yte)
        s = calibrate_input_scale(Mq, H_cal) if r["periphery"] == "realistic" else 1.0
        out.append({
            "job_key": r["job_key"], "method": m, "precision": r["precision"],
            "device": r["device"], "noise_scale": r["noise_scale"], "periphery": r["periphery"],
            "digital_acc_rebuilt": dig, "digital_acc_stored": r["digital_acc"],
            "reproduced": abs(dig - r["digital_acc"]) < 1e-9, "in_scale": s, "times": r["times"],
            "sigma_delta_analytic_t": [sigma_delta_analytic(Mq, H_cal, y_cal, r["device"], t, s,
                                                            r["periphery"], r["noise_scale"])
                                       for t in r["times"]],
            "pred_acc_analytic_t": [predict_accuracy_analytic(Mq, Hte, yte, r["device"], t, s,
                                                              r["periphery"], r["noise_scale"],
                                                              seed=r["seed"])
                                    for t in r["times"]],
        })
    for o in out:
        o["job_seconds"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", help="comma-separated, e.g. core or stress,encoders,precision")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--data_home", default="./data")
    a = ap.parse_args()
    for stage in a.stages.split(","):
        a.stage = stage
        run_stage(a)


def run_stage(a):
    src = Path(a.outdir) / f"v2_{a.stage}.jsonl"
    dst = Path(a.outdir) / f"v2_{a.stage}_analytic.jsonl"
    keep = ("job_key", "dataset", "encoder", "dim", "seed", "center", "method", "precision",
            "device", "noise_scale", "periphery", "lr", "margin_used", "digital_acc", "times",
            "e2e")
    by_job = collections.OrderedDict()
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("e2e"):
            continue  # end-to-end records have no single-crossbar prediction
        by_job.setdefault(r["job_key"], []).append({k: r[k] for k in keep})
    done = set()
    if dst.exists():
        for line in dst.read_text().splitlines():
            try:
                done.add(json.loads(line)["job_key"])
            except (json.JSONDecodeError, KeyError):
                pass
    from hdc import load_dataset
    for ds in sorted({v[0]["dataset"] for v in by_job.values()}):
        load_dataset(ds, a.data_home)
    jobs = [{"recs": v, "data_home": a.data_home} for k, v in by_job.items() if k not in done]
    jobs.sort(key=lambda j: -j["recs"][0]["dim"])
    print(f"[analytic {a.stage}] {len(by_job)} jobs, {len(by_job) - len(jobs)} done, running "
          f"{len(jobs)} on {a.workers} workers -> {dst}", flush=True)
    t_start = time.time()
    bad = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex, dst.open("a") as f:
        futs = {ex.submit(_worker, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            r0 = futs[fu]["recs"][0]
            try:
                recs = fu.result()
            except Exception as e:
                print(f"FAILED {r0['dataset']} {r0['encoder']} D={r0['dim']} seed={r0['seed']}: {e!r}",
                      flush=True)
                continue
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            nbad = sum(not r["reproduced"] for r in recs)
            bad += nbad
            el = time.time() - t_start
            print(f"[{i}/{len(jobs)}] {r0['dataset']} {r0['encoder']} D={r0['dim']} seed={r0['seed']}: "
                  f"{len(recs)} records, models reproduced {len(recs) - nbad}/{len(recs)} "
                  f"[{recs[0]['job_seconds']}s, elapsed {el/3600:.1f} h, "
                  f"ETA {el / i * (len(jobs) - i) / 3600:.1f} h]", flush=True)
    print(f"done; records whose digital accuracy did not reproduce: {bad}", flush=True)


if __name__ == "__main__":
    main()
