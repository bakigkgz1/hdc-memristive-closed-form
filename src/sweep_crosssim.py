"""Cross-simulator stage: the core prototypes re-evaluated in Sandia CrossSim.

For a subset of the core jobs (D in {512, 2048, 8192}, seeds 0-2, all 7 datasets, rp and
nonlinear encoders) every prototype set is retrained with the stored hyperparameters (the
digital test accuracy is checked against the stored value), quantised (fp / 4bit / binary)
and evaluated on CrossSim's own device models (crosssim_bridge.py) with ideal and realistic
(9-bit ADC) periphery over N_CHIPS independently programmed chips. For each point the
closed-form accuracy predicted from CrossSim single-cell statistics is stored as well.
The noise-calibrated method is left out: its margin is calibrated on AIHWKit devices.
Output: results/v2_crosssim.jsonl (resumable).
"""
from __future__ import annotations

import os
# one BLAS/OpenMP thread per worker process: N workers x N BLAS threads on N cores
# oversubscribes the CPU several-fold (set before numpy is imported anywhere)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

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
N_CHIPS = 8
DIMS = (512, 2048, 8192)
SEEDS = (0, 1, 2)
METHODS = ("single", "perceptron", "onlinehd", "largemargin", "dependable")
PERIPHERIES = ("ideal", "realistic")


def _worker(job: dict) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import crosssim_bridge as xb
    from hdc import accuracy_digital, load_dataset, make_encoder, quantize, train_prototypes

    t0 = time.time()
    r0 = job["recs"][0]
    if os.environ.get("XS_TEST_CRASH") == f"{r0['encoder']}{r0['seed']}" and not os.path.exists("/tmp/xs_crashed"):
        open("/tmp/xs_crashed", "w").close()
        os._exit(1)  # test hook: simulates a worker killed by the OS
    Xtr, ytr, Xte, yte, C = load_dataset(r0["dataset"], job["data_home"])
    enc = make_encoder(r0["encoder"], Xtr.shape[1], r0["dim"], r0["seed"], Xtr, center=r0["center"])
    Htr, Hte = enc(Xtr), enc(Xte)
    cal_idx = torch.randperm(len(Htr), generator=torch.Generator().manual_seed(1))[:N_CAL]
    H_cal = Htr[cal_idx].numpy().astype(np.float32)
    H = Hte.numpy().astype(np.float32)   # +/-1 exactly; halves the memory of float64
    models = {}
    out = []
    for r in job["recs"]:
        m = r["method"]
        mkey = (m, r["margin_used"])
        if mkey not in models:
            kw = dict(seed=r["seed"])
            if m != "single":
                kw.update(lr=r["lr"])
            if m in ("largemargin", "dependable"):
                kw.update(margin=r["margin_used"])
            if m == "dependable":
                kw.update(flip_p=FLIP_P)
            if m == "onlinehd":
                kw.update(adaptive=True)
            models[mkey] = train_prototypes(Htr, ytr, C, 0 if m == "single" else EPOCHS, **kw)
        Mq_t = quantize(models[mkey], r["precision"])
        dig = accuracy_digital(Mq_t, Hte, yte)
        Mq = Mq_t.numpy().astype(np.float64)
        bound = xb.calibrate_adc_bound(Mq, H_cal)
        for dev in xb.DEVICES:
            times = xb.T_GRID[dev]
            for per in PERIPHERIES:
                acc = xb.accuracy_crosssim(Mq, H, yte, dev, per, times, n_chips=N_CHIPS,
                                           seed=r["seed"], adc_bound=bound)
                pred = [xb.predict_accuracy(Mq, H, yte, dev, per, t, bound, seed=r["seed"],
                                            return_clip=True) for t in times]
                out.append({
                    "job_key": r["job_key"], "dataset": r["dataset"], "encoder": r["encoder"],
                    "dim": r["dim"], "seed": r["seed"], "method": m, "precision": r["precision"],
                    "xs_device": dev, "periphery": per, "times_days": times,
                    "digital_acc": dig, "digital_acc_stored": r["digital_acc"],
                    "reproduced": abs(dig - r["digital_acc"]) < 1e-9, "adc_bound": bound,
                    "acc_chips": acc.T.tolist(), "acc_mean": acc.mean(0).tolist(),
                    "acc_std": acc.std(0).tolist(),
                    "pred_acc_t": [p[0] for p in pred], "clip_frac_t": [float(p[1]) for p in pred],
                })
    for o in out:
        o["job_seconds"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--data_home", default="./data")
    ap.add_argument("--src", default="v2_core.jsonl")
    a = ap.parse_args()
    src = Path(a.outdir) / a.src
    dst = Path(a.outdir) / "v2_crosssim.jsonl"
    keep = ("job_key", "dataset", "encoder", "dim", "seed", "center", "method", "precision",
            "lr", "margin_used", "digital_acc")
    by_job = collections.OrderedDict()
    seen = set()
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("e2e") or r["dim"] not in DIMS or r["seed"] not in SEEDS
                or r["method"] not in METHODS or r["device"] != "pcm"
                or r["periphery"] != "realistic" or r["noise_scale"] != 1.0):
            continue
        k = (r["job_key"], r["method"], r["precision"])
        if k in seen:
            continue
        seen.add(k)
        by_job.setdefault(r["job_key"], []).append({x: r[x] for x in keep})
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
    print(f"[crosssim] {len(by_job)} jobs, {len(by_job) - len(jobs)} done, running {len(jobs)} "
          f"on {a.workers} workers -> {dst}", flush=True)
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError):
        mem_gb = 0
    # a 60k-sample, D = 8192 job peaks at about 4 GB; keep a safety margin
    workers = max(1, min(a.workers, int(mem_gb // 5) if mem_gb else a.workers))
    print(f"RAM {mem_gb:.0f} GB -> {workers} workers", flush=True)
    t_start = time.time()
    bad = 0
    done_n = 0
    if jobs:  # probe: smallest job in this process, so a real error shows its traceback
        probe = min(jobs, key=lambda j: j["recs"][0]["dim"] * (1 if j["recs"][0]["dataset"] != "mnist" else 10))
        import traceback
        try:
            t0 = time.time()
            recs = _worker(probe)
        except Exception:
            traceback.print_exc()
            raise SystemExit("probe job failed; see the traceback above")
        with dst.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        jobs.remove(probe)
        done_n += 1
        bad += sum(not r["reproduced"] for r in recs)
        print(f"probe ok: {probe['recs'][0]['dataset']} D={probe['recs'][0]['dim']} "
              f"in {time.time() - t0:.0f}s", flush=True)
    from concurrent.futures.process import BrokenProcessPool
    for attempt in range(4):
        if not jobs:
            break
        failed = []
        with ProcessPoolExecutor(max_workers=workers) as ex, dst.open("a") as f:
            futs = {ex.submit(_worker, j): j for j in jobs}
            for fu in as_completed(futs):
                j = futs[fu]
                r0 = j["recs"][0]
                try:
                    recs = fu.result()
                except BrokenProcessPool:
                    failed.append(j)
                    continue
                except Exception as e:
                    print(f"FAILED {r0['dataset']} {r0['encoder']} D={r0['dim']} seed={r0['seed']}: {e!r}",
                          flush=True)
                    continue
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                done_n += 1
                nbad = sum(not r["reproduced"] for r in recs)
                bad += nbad
                el = time.time() - t_start
                left = len(by_job) - len(done) - done_n
                print(f"[{done_n}/{len(by_job) - len(done)}] {r0['dataset']} {r0['encoder']} D={r0['dim']} "
                      f"seed={r0['seed']}: {len(recs)} records, models reproduced {len(recs) - nbad}/{len(recs)} "
                      f"[{recs[0]['job_seconds']}s, elapsed {el/3600:.1f} h, "
                      f"ETA {el / done_n * left / 3600:.1f} h]", flush=True)
        if failed:
            workers = max(1, workers // 2)
            print(f"a worker process died (most likely out of memory); retrying {len(failed)} jobs "
                  f"with {workers} workers", flush=True)
        jobs = failed
    if jobs:
        print(f"{len(jobs)} jobs still failing after retries; run the cell again", flush=True)
    print(f"done; records whose digital accuracy did not reproduce: {bad}", flush=True)


if __name__ == "__main__":
    main()
