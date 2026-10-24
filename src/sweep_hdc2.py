"""Experiment driver v2 for the HDC-on-memristive-crossbars study.

A job = (stage, dataset, encoder, D, encoder seed, centring). It encodes once, tunes the
retraining step size and the digital-optimal margin on a validation split of the training
data, trains every requested method, and evaluates each method x precision x device x
noise scale x periphery over n_chips independently programmed chips and the device's
time grid. Besides accuracy it logs, at the 1-day reference time, the margin-to-noise
statistics needed to test the flip law: the share of correctly classified test samples in
bins of (digital cosine margin / sigma_hat) and the analog-vs-digital decision flip rate in
each bin, a fine histogram of margin / sigma_hat, sigma_hat at every time of the grid, and
the accuracy predicted by the margin-to-noise model (digital scores + Gaussian noise of std
sigma_hat) at every time. One jsonl line per evaluated combination; finished jobs are skipped
(resumable).

Training methods (all retraining methods share the tuned lr and the same epoch budget):
  single        one-pass bundling (no retraining)
  perceptron    classic HDC retraining (update on misclassification)
  onlinehd      OnlineHD adaptive retraining (similarity-weighted updates)
  largemargin   margin retraining with the digital-optimal margin (MM-HDC-style baseline)
  dependable    digital-optimal margin + random query bit flips (p=0.05) during training
                (DependableHD-style baseline)
  noisecal      margin = digital-optimal + KAPPA * sigma_hat, sigma_hat measured on one
                calibration chip with training data (noise-calibrated variant)
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

ALL_METHODS = ["single", "perceptron", "onlinehd", "largemargin", "dependable", "noisecal"]
DATASETS7 = "isolet,ucihar,mnist,fmnist,letter,pendigits,satimage"
KAPPA = 2.0          # fixed a priori from the pilot study, never tuned on test data
EPOCHS = 20
REF_T = 86400.0      # 1 day: reference time for sigma_hat and the flip-law statistics
FLIP_P = 0.05
BINS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, float("inf")]  # margin / sigma_hat
HIST_EDGES = [-1e9] + [round(-6 + 0.25 * k, 2) for k in range(73)] + [1e9]  # -6..12 sigma

STAGES = {
    # core benchmark: everything that the main figures and the flip law need
    "core": dict(datasets=DATASETS7, encoders="rp,nonlinear", dims="256,512,1024,2048,4096,8192",
                 seeds="0,1,2,3,4", center="1", methods=",".join(ALL_METHODS),
                 precisions="fp,4bit,binary", devices="pcm,reram_wan,reram_cmo",
                 noise="1.0", peripheries="realistic", e2e="0", n_chips=8),
    # third encoder family
    "encoders": dict(datasets=DATASETS7, encoders="idlevel", dims="512,2048,8192", seeds="0,1,2",
                     center="1", methods="perceptron,largemargin,dependable,noisecal",
                     precisions="4bit", devices="pcm,reram_wan,reram_cmo", noise="1.0",
                     peripheries="realistic", e2e="0", n_chips=8),
    # encoder in the crossbar too
    "e2e": dict(datasets=DATASETS7, encoders="rp,nonlinear", dims="512,2048,8192", seeds="0,1,2",
                center="1", methods="perceptron,largemargin,noisecal", precisions="4bit",
                devices="pcm,reram_wan,reram_cmo", noise="1.0", peripheries="realistic",
                e2e="1", n_chips=6),
    # end-to-end with a bipolar (+/-1) projection matrix (fix for drift-induced encoder flips)
    "e2e_bipolar": dict(datasets=DATASETS7, encoders="rp_bipolar", dims="512,2048,8192",
                        seeds="0,1,2", center="1", methods="largemargin", precisions="4bit",
                        devices="pcm,reram_wan,reram_cmo", noise="1.0", peripheries="realistic",
                        e2e="1", n_chips=6),
    # device-noise stress and centring ablation
    "stress": dict(datasets=DATASETS7, encoders="rp", dims="256,1024,4096", seeds="0,1,2",
                   center="0,1", methods="perceptron,largemargin,dependable,noisecal",
                   precisions="4bit,binary", devices="pcm,reram_wan,reram_cmo",
                   noise="0.5,1.0,2.0,4.0,8.0", peripheries="realistic", e2e="0", n_chips=6),
    # classic HDC benchmark: European language recognition with character trigrams
    # (Karunaratne et al. 2020 and Huang et al. 2026 run this task on PCM / ReRAM chips)
    "langid": dict(datasets="langid", encoders="ngram", dims="512,1024,2048", seeds="0,1,2",
                   center="0", methods=",".join(ALL_METHODS), precisions="fp,4bit,binary",
                   devices="pcm,reram_wan,reram_cmo", noise="1.0", peripheries="realistic",
                   e2e="0", n_chips=8),
    # 2-bit cells and the ideal-periphery reference
    "precision": dict(datasets=DATASETS7, encoders="rp", dims="256,1024,4096", seeds="0,1,2",
                      center="1", methods="perceptron,largemargin,noisecal",
                      precisions="fp,4bit,2bit,binary", devices="pcm,reram_wan,reram_cmo",
                      noise="1.0", peripheries="ideal,realistic", e2e="0", n_chips=6),
}

# shorter time grid than v1 (drops 1 h) to save compute; 1 s / 1 d / 2 d shared by all
TIMES = {"pcm": [1.0, 86400.0, 172800.0, 2592000.0, 31536000.0],
         "reram_cmo": [1.0, 86400.0, 172800.0, 2592000.0, 31536000.0],
         "reram_wan": [1.0, 86400.0, 172800.0]}


def job_key(j: dict) -> str:
    return hashlib.md5(json.dumps({k: v for k, v in j.items() if k != "data_home"},
                                  sort_keys=True).encode()).hexdigest()


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    for l in path.read_text().splitlines():
        if l.strip():
            try:
                keys.add(json.loads(l)["job_key"])
            except json.JSONDecodeError:  # a line cut off by a disconnect
                pass
    return keys


_DATA: dict = {}


def _flip_stats(margins, dig_pred, y, preds_ref, sigma):
    """Share of correct samples per margin/sigma bin and the mean analog flip rate there."""
    import numpy as np
    correct = dig_pred == y
    r = margins[correct] / max(sigma, 1e-12)
    flips = (preds_ref[:, correct] != dig_pred[correct][None, :]).mean(0)
    b = np.digitize(r, BINS[1:-1])
    share = [float(np.mean(b == k)) for k in range(len(BINS) - 1)]
    rate = [float(flips[b == k].mean()) if (b == k).any() else None for k in range(len(BINS) - 1)]
    return share, rate, [float(q) for q in np.quantile(margins, [0.05, 0.25, 0.5, 0.75, 0.95])]


def _worker(j: dict) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import torch
    torch.set_num_threads(int(os.environ.get("HDC_THREADS", "1")))
    from hdc import (accuracy_crossbar, accuracy_digital, accuracy_end_to_end,
                     calibrate_input_scale, cosine_margins, encoder_input_scale, load_dataset,
                     make_encoder, make_inference_config, predict_accuracy_gaussian, quantize,
                     similarity_noise_sigma, similarity_noise_sigma_times, train_prototypes,
                     tune_lr, tune_margin)

    t0 = time.time()
    if j["dataset"] not in _DATA:
        _DATA.clear()
        _DATA[j["dataset"]] = load_dataset(j["dataset"], j["data_home"])
    Xtr, ytr, Xte, yte, C = _DATA[j["dataset"]]
    enc = make_encoder(j["encoder"], Xtr.shape[1], j["dim"], j["seed"], Xtr, center=j["center"])
    Htr, Hte = enc(Xtr), enc(Xte)
    g = torch.Generator().manual_seed(1)
    cal_idx = torch.randperm(len(Htr), generator=g)[:5000]
    H_cal = Htr[cal_idx]
    lr = tune_lr(Htr, ytr, C, EPOCHS, j["seed"])
    g_dig = tune_margin(Htr, ytr, C, EPOCHS, j["seed"], lr=lr)
    meth = j["methods"]
    base = {}
    if "single" in meth:
        base["single"] = train_prototypes(Htr, ytr, C, 0, seed=j["seed"])
    if "perceptron" in meth or "noisecal" in meth:
        base["perceptron"] = train_prototypes(Htr, ytr, C, EPOCHS, lr=lr, seed=j["seed"])
    if "onlinehd" in meth:
        base["onlinehd"] = train_prototypes(Htr, ytr, C, EPOCHS, lr=lr, seed=j["seed"], adaptive=True)
    if "largemargin" in meth:
        base["largemargin"] = train_prototypes(Htr, ytr, C, EPOCHS, lr=lr, seed=j["seed"], margin=g_dig)
    if "dependable" in meth:
        base["dependable"] = train_prototypes(Htr, ytr, C, EPOCHS, lr=lr, seed=j["seed"],
                                              margin=g_dig, flip_p=FLIP_P)
    s_enc = encoder_input_scale(enc, Xtr[cal_idx.numpy()]) if j["e2e"] else None
    t_train = time.time() - t0

    out = []
    for prec, dev, ns, per in itertools.product(j["precisions"], j["devices"], j["noise"],
                                                j["peripheries"]):
        cfg = make_inference_config(dev, per, noise_scale=ns)
        times = TIMES[dev]
        models = {m: base[m] for m in meth if m in base}
        sig_p = None
        if "noisecal" in meth:  # sigma_hat on a calibration chip, training data only
            Mp = quantize(base["perceptron"], prec)
            sp = calibrate_input_scale(Mp, H_cal) if per == "realistic" else 1.0
            sig_p = similarity_noise_sigma(Mp, H_cal, cfg, REF_T, sp)
            models["noisecal"] = train_prototypes(Htr, ytr, C, EPOCHS, lr=lr, seed=j["seed"],
                                                  margin=g_dig + KAPPA * sig_p)
        for m in meth:
            Mq = quantize(models[m], prec)
            s = calibrate_input_scale(Mq, H_cal) if per == "realistic" else 1.0
            dig = accuracy_digital(Mq, Hte, yte)
            rec = {"job_key": j["job_key"], "stage": j["stage"], "dataset": j["dataset"],
                   "n_classes": C, "encoder": j["encoder"], "dim": j["dim"], "seed": j["seed"],
                   "center": j["center"], "e2e": j["e2e"], "method": m, "precision": prec,
                   "device": dev, "noise_scale": ns, "periphery": per, "lr": lr,
                   "margin_dig": g_dig, "sigma_cal_perceptron": sig_p,
                   "margin_used": (g_dig + KAPPA * sig_p) if m == "noisecal" else
                                  (g_dig if m in ("largemargin", "dependable") else 0.0),
                   "in_scale": s, "digital_acc": dig, "times": times}
            if j["e2e"]:
                acc, flips = accuracy_end_to_end(enc, Mq, Xte, yte, cfg, times, j["n_chips"],
                                                 seed=j["seed"], in_scale_am=s, in_scale_enc=s_enc)
                rec["query_bitflip_mean"] = flips.mean(0).tolist()
            else:
                acc, preds = accuracy_crossbar(Mq, Hte, yte, cfg, times, j["n_chips"],
                                               seed=j["seed"], in_scale=s, ref_t=REF_T,
                                               return_preds=True)
                sig_t = similarity_noise_sigma_times(Mq, H_cal, cfg, times, s)
                sig = sig_t[times.index(REF_T)]
                margins = cosine_margins(Mq, Hte, yte)
                # margin-to-noise model: predicted accuracy at every time, no test-set device sim
                rec["sigma_hat_t"] = sig_t
                rec["pred_acc_t"] = [predict_accuracy_gaussian(Mq, Hte, yte, st, seed=j["seed"])
                                     for st in sig_t]
                hist, _ = np.histogram(margins / max(sig, 1e-12), bins=HIST_EDGES)
                rec["margin_hist_sigma_units"] = (hist / len(margins)).round(6).tolist()
                dig_pred = np.concatenate([(Hte[i:i + 8192].float() @ Mq.T).argmax(1).numpy()
                                           for i in range(0, len(Hte), 8192)])
                share, rate, mq = _flip_stats(margins, dig_pred, yte, preds, sig)
                rec.update({"sigma_hat": sig, "margin_quantiles": mq, "bin_edges": BINS[:-1],
                            "bin_share": share, "bin_flip_rate": rate})
            rec.update({"acc_mean": acc.mean(0).tolist(), "acc_std": acc.std(0).tolist(),
                        "acc_min": acc.min(0).tolist(), "acc_chips": acc.round(5).tolist()})
            out.append(rec)
    for r in out:
        r["train_seconds"] = round(t_train, 1)
        r["job_seconds"] = round(time.time() - t0, 1)
    return out


def build_jobs(stage: str, p: dict, data_home: str) -> list[dict]:
    jobs = []
    for d, e, dim, s, c in itertools.product(
            p["datasets"].split(","), p["encoders"].split(","),
            sorted(map(int, p["dims"].split(",")), reverse=True),
            map(int, p["seeds"].split(",")), map(int, p["center"].split(","))):
        if e == "idlevel" and not c:
            continue
        j = {"stage": stage, "dataset": d, "encoder": e, "dim": dim, "seed": s, "center": bool(c),
             "methods": p["methods"].split(","), "precisions": p["precisions"].split(","),
             "devices": p["devices"].split(","), "noise": [float(v) for v in p["noise"].split(",")],
             "peripheries": p["peripheries"].split(","), "e2e": bool(int(p["e2e"])),
             "n_chips": int(p["n_chips"]), "data_home": data_home}
        j["job_key"] = job_key(j)
        jobs.append(j)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--data_home", default="./data")
    for k in ["datasets", "encoders", "dims", "seeds", "center", "methods", "precisions",
              "devices", "noise", "peripheries", "e2e", "n_chips"]:
        ap.add_argument(f"--{k}", default=None, help="override the stage preset")
    a = ap.parse_args()
    p = {k: (getattr(a, k) if getattr(a, k) is not None else v) for k, v in STAGES[a.stage].items()}

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"v2_{a.stage}.jsonl"

    from hdc import load_dataset
    ok = []
    for d in p["datasets"].split(","):  # download once, before forking workers
        try:
            load_dataset(d, a.data_home)
            ok.append(d)
        except Exception as e:  # keep going with the datasets that are available
            print(f"WARNING: dataset {d} could not be loaded ({e!r}); skipping it", flush=True)
    p["datasets"] = ",".join(ok)

    jobs = build_jobs(a.stage, p, a.data_home)
    todo = [j for j in jobs if j["job_key"] not in done_keys(path)]
    # biggest datasets and dimensions first, so stragglers are small
    print(f"[{a.stage}] {len(jobs)} jobs, {len(jobs) - len(todo)} done, running {len(todo)} "
          f"on {a.workers} workers -> {path}", flush=True)
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, path.open("a") as f:
        futs = {ex.submit(_worker, j): j for j in todo}
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
            eta = el / i * (len(todo) - i)
            best = max(recs, key=lambda r: r["acc_mean"][1 if len(r["times"]) > 1 else 0])
            print(f"[{i}/{len(todo)}] {j['dataset']} {j['encoder']} D={j['dim']} seed={j['seed']} "
                  f"centre={j['center']}: best analog@1d {best['acc_mean'][1]:.3f} "
                  f"({best['method']}/{best['precision']}/{best['device']}) "
                  f"[{recs[0]['job_seconds']}s, elapsed {el/3600:.1f} h, ETA {eta/3600:.1f} h]",
                  flush=True)


if __name__ == "__main__":
    main()
