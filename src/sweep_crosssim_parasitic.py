"""Validity limit of the closed form: CrossSim with interconnect (IR-drop) parasitics.

The closed form models device errors and the ADC, not wire resistance. Here a small subset
of the CrossSim cross-check is re-simulated with the parasitic circuit solver switched on
(row and column wire-segment resistance Rp, ohms per cell) and compared with the
parasitic-free closed-form prediction from CrossSim single-cell statistics.
Subset: isolet and ucihar, RP encoder, D = 512, seed 0, perceptron and
large-margin retraining, 4-bit prototypes, ReRAM-Milo (lowest Rmin, strongest IR drop) and
PCM-Joshi, 9-bit ADC, first N_TEST test samples, N_CHIPS chips, Rp in RP_GRID.
Runs on the GPU (CuPy) when one is available. Output: results/v2_crosssim_parasitic.jsonl
(one line per job x method x device x Rp; resumable).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

EPOCHS = 20
N_CAL = 5000
N_TEST = 1000
N_CHIPS = 2
RP_GRID = (0.0, 0.5, 1.0, 2.0)
DATASETS = ("isolet", "ucihar")
DIMS = (512,)  # each 512-row partition sees the same wire length, so larger D adds no IR drop
METHODS = ("perceptron", "largemargin")
XS_DEVICES = ("rram_milo", "pcm_joshi")


def make_params(device, adc_bound, rp, gpu):
    import crosssim_bridge as xb
    from applications.mvm_params import set_params
    kw = dict(xb.DEVICES[device])
    r = adc_bound * xb._grange(device)
    kw.update(adc_bits=xb.ADC_BITS, adc_range=(-r, r), useGPU=gpu)
    if rp > 0:
        kw.update(Rp_row=rp, Rp_col=rp)
    return set_params(NrowsMax=xb.ROWS, input_range=(-1, 1), **kw)


def simulate(M, H, y, device, adc_bound, rp, gpu, seed):
    from simulator import AnalogCore
    xp = np
    if gpu:
        import cupy as xp
    accs = []
    p = make_params(device, adc_bound, rp, gpu)
    for c in range(N_CHIPS):
        np.random.seed(seed * 1000 + c)
        if gpu:
            xp.random.seed(seed * 1000 + c)
        core = AnalogCore(M, params=p)
        out = core @ xp.asarray(H.T)
        out = xp.asnumpy(out) if gpu else out
        accs.append(float((out.argmax(0) == y).mean()))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--data_home", default="./data")
    ap.add_argument("--src", default="v2_core.jsonl")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is present")
    a = ap.parse_args()
    import torch
    import crosssim_bridge as xb
    from hdc import accuracy_digital, load_dataset, make_encoder, quantize, train_prototypes
    gpu = False
    if not a.cpu:
        try:
            import cupy
            gpu = cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:
            gpu = False
    print("GPU:", gpu, flush=True)

    src = Path(a.outdir) / a.src
    dst = Path(a.outdir) / "v2_crosssim_parasitic.jsonl"
    recs = {}
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("e2e") or r["dataset"] not in DATASETS or r["encoder"] != "rp"
                or r["dim"] not in DIMS or r["seed"] != 0 or r["method"] not in METHODS
                or r["precision"] != "4bit" or r["device"] != "pcm" or r["periphery"] != "realistic"):
            continue
        recs.setdefault((r["dataset"], r["dim"], r["method"]), r)
    done = set()
    if dst.exists():
        for line in dst.read_text().splitlines():
            try:
                o = json.loads(line)
                done.add((o["dataset"], o["dim"], o["method"], o["xs_device"], o["Rp"]))
            except (json.JSONDecodeError, KeyError):
                pass
    todo = len(recs) * len(XS_DEVICES) * len(RP_GRID) - len(done)
    print(f"[parasitic] {len(recs)} models x {len(XS_DEVICES)} devices x {len(RP_GRID)} Rp; "
          f"{len(done)} done, {todo} to run -> {dst}", flush=True)
    # CrossSim fixes its CPU/GPU backend at the first core it builds in a process. The
    # closed-form characterisation runs on the CPU here, so the GPU simulations run in one
    # persistent child process (spawned, so it starts with a fresh backend).
    sim_pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
    t_start, n = time.time(), 0
    for (ds, D, m), r in sorted(recs.items(), key=lambda kv: kv[0][1]):
        if all((ds, D, m, dev, rp) in done for dev in XS_DEVICES for rp in RP_GRID):
            continue
        Xtr, ytr, Xte, yte, C = load_dataset(ds, a.data_home)
        enc = make_encoder("rp", Xtr.shape[1], D, 0, Xtr, center=r["center"])
        Htr, Hte = enc(Xtr), enc(Xte)
        kw = dict(seed=0, lr=r["lr"])
        if m == "largemargin":
            kw.update(margin=r["margin_used"])
        Mt = quantize(train_prototypes(Htr, ytr, C, EPOCHS, **kw), "4bit")
        dig = accuracy_digital(Mt, Hte, yte)
        M = Mt.numpy().astype(np.float64)
        cal = torch.randperm(len(Htr), generator=torch.Generator().manual_seed(1))[:N_CAL]
        bound = xb.calibrate_adc_bound(M, Htr[cal].numpy().astype(np.float32))
        H = Hte[:N_TEST].numpy().astype(np.float64)
        y = yte[:N_TEST]
        for dev in XS_DEVICES:
            pred = xb.predict_accuracy(M, H.astype(np.float32), y, dev, "realistic", 0.0, bound, seed=0)
            for rp in RP_GRID:
                if (ds, D, m, dev, rp) in done:
                    continue
                t0 = time.time()
                try:
                    acc, conv = sim_pool.submit(simulate, M, H, y, dev, bound, rp, gpu, 0).result(), True
                except Exception as e:  # CrossSim gives up when the wire resistance is too high
                    if "converge" not in repr(e):
                        raise
                    acc, conv = [], False
                n += 1
                o = {"dataset": ds, "dim": D, "method": m, "xs_device": dev, "Rp": rp,
                     "Rp_over_Rmin": rp / xb.DEVICES[dev]["Rmin"], "n_test": int(len(y)),
                     "digital_acc": dig, "digital_acc_stored": r["digital_acc"],
                     "reproduced": abs(dig - r["digital_acc"]) < 1e-9, "adc_bound": bound,
                     "converged": conv, "acc_chips": acc,
                     "acc_mean": float(np.mean(acc)) if conv else None, "pred_closed": pred,
                     "seconds": round(time.time() - t0, 1)}
                with dst.open("a") as f:
                    f.write(json.dumps(o) + "\n")
                el = time.time() - t_start
                print(f"[{n}/{todo}] {ds} D={D} {m} {dev} Rp={rp:g} ohm: sim {o['acc_mean'] if conv else float('nan'):.4f}{'' if conv else ' (solver did not converge)'} "
                      f"closed form {pred:.4f} [{o['seconds']}s, elapsed {el/3600:.2f} h, "
                      f"ETA {el / n * (todo - n) / 3600:.1f} h]", flush=True)
    sim_pool.shutdown()
    print("done", flush=True)


if __name__ == "__main__":
    main()
