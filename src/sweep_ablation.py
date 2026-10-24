"""Supplementary stage: component ablation of the periphery-aware closed form (v2) and
run-time comparison with Monte-Carlo crossbar simulation.

For a subset of the existing records (D = 512 and 2048, stress D = 1024; seed 0; stress at nominal and x4 noise)
the prototypes are retrained with the stored hyperparameters (digital accuracy checked) and
v2 is evaluated four times: full, and with one effect switched off (DAC-quantised read
scale, per-tile read gain, per-tile ADC clipping/rounding). The simulated chip accuracies
are not recomputed; the analysis pairs these predictions with the stored acc_mean.
Timing: for the first realistic-periphery record of every job, one 8-chip AIHWKit
evaluation (all read times) and one full v2 prediction (all read times) are timed on the
same core.
Output: results/v2_<stage>_ablation.jsonl (resumable).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import collections
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

EPOCHS = 20
FLIP_P = 0.05
N_CAL = 5000
ABL_DIMS = {"core": (512, 2048), "encoders": (512, 2048), "stress": (1024,)}
ABL_SEEDS = (0,)
STRESS_NOISE = (1.0, 4.0)
VARIANTS = {"full": (), "no_dac": ("dac",), "no_gain": ("gain",), "no_clip": ("clip",)}


def _worker(job: dict) -> list[dict]:
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from analytic import predict_accuracy_analytic2
    from hdc import (accuracy_crossbar, accuracy_digital, calibrate_input_scale, load_dataset,
                     make_encoder, make_inference_config, quantize, train_prototypes)

    t0 = time.time()
    r0 = job["recs"][0]
    Xtr, ytr, Xte, yte, C = load_dataset(r0["dataset"], job["data_home"])
    enc = make_encoder(r0["encoder"], Xtr.shape[1], r0["dim"], r0["seed"], Xtr, center=r0["center"])
    Htr, Hte = enc(Xtr), enc(Xte)
    cal_idx = torch.randperm(len(Htr), generator=torch.Generator().manual_seed(1))[:N_CAL]
    H_cal = Htr[cal_idx]
    models, out, timed = {}, [], False
    for r in job["recs"]:
        m = r["method"]
        mkey = (m, r["margin_used"])
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
        s = calibrate_input_scale(Mq, H_cal)
        o = {"job_key": r["job_key"], "method": m, "precision": r["precision"],
             "device": r["device"], "noise_scale": r["noise_scale"], "periphery": r["periphery"],
             "digital_acc_rebuilt": dig, "digital_acc_stored": r["digital_acc"],
             "reproduced": abs(dig - r["digital_acc"]) < 1e-9, "in_scale": s, "times": r["times"]}
        for name, abl in VARIANTS.items():
            o[f"pred_{name}_t"] = [predict_accuracy_analytic2(
                Mq, Hte, yte, r["device"], t, s, "realistic", r["noise_scale"], seed=r["seed"],
                ablate=abl) for t in r["times"]]
        if not timed:  # one timing pair per job, same core, same record
            timed = True
            ta = time.perf_counter()
            for t in r["times"]:
                predict_accuracy_analytic2(Mq, Hte, yte, r["device"], t, s, "realistic",
                                           r["noise_scale"], seed=r["seed"])
            o["t_closed_s"] = time.perf_counter() - ta
            cfg = make_inference_config(r["device"], "realistic", noise_scale=r["noise_scale"])
            ta = time.perf_counter()
            accuracy_crossbar(Mq, Hte, yte, cfg, r["times"], n_chips=8, seed=r["seed"], in_scale=s)
            o["t_mc8_s"] = time.perf_counter() - ta
            o["n_test"] = int(len(yte))
        out.append(o)
    for o in out:
        o["job_seconds"] = round(time.time() - t0, 1)
    return out


def select(stage: str, r: dict) -> bool:
    if r.get("e2e") or r["periphery"] != "realistic":
        return False
    if r["dim"] not in ABL_DIMS[stage] or r["seed"] not in ABL_SEEDS:
        return False
    if stage == "stress" and r["noise_scale"] not in STRESS_NOISE:
        return False
    return True


def run_stage(a):
    src = Path(a.outdir) / f"v2_{a.stage}.jsonl"
    dst = Path(a.outdir) / f"v2_{a.stage}_ablation.jsonl"
    keep = ("job_key", "dataset", "encoder", "dim", "seed", "center", "method", "precision",
            "device", "noise_scale", "periphery", "lr", "margin_used", "digital_acc", "times")
    by_job = collections.OrderedDict()
    for line in src.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if select(a.stage, r):
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
    # largest first so the slow MNIST / D = 2048 jobs do not finish last
    jobs.sort(key=lambda j: -(j["recs"][0]["dim"] * (8 if j["recs"][0]["dataset"] in ("mnist", "fmnist") else 1)))
    print(f"[ablation {a.stage}] {len(by_job)} jobs, {len(by_job) - len(jobs)} done, running "
          f"{len(jobs)} on {a.workers} workers -> {dst}", flush=True)
    t_start, bad = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex, dst.open("a") as f:
        futs = {ex.submit(_worker, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            r0 = futs[fu]["recs"][0]
            try:
                recs = fu.result()
            except Exception as e:
                print(f"FAILED {r0['dataset']} {r0['encoder']} D={r0['dim']}: {e!r}", flush=True)
                continue
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            nbad = sum(not r["reproduced"] for r in recs)
            bad += nbad
            el = time.time() - t_start
            print(f"[{i}/{len(jobs)}] {r0['dataset']} {r0['encoder']} D={r0['dim']}: {len(recs)} records, "
                  f"reproduced {len(recs) - nbad}/{len(recs)} [{recs[0]['job_seconds']}s, "
                  f"elapsed {el/3600:.1f} h, ETA {el / i * (len(jobs) - i) / 3600:.1f} h]", flush=True)
    print(f"done; records whose digital accuracy did not reproduce: {bad}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", help="comma-separated, e.g. core,stress,encoders")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--data_home", default="./data")
    a = ap.parse_args()
    for stage in a.stages.split(","):
        a.stage = stage
        run_stage(a)


if __name__ == "__main__":
    main()
