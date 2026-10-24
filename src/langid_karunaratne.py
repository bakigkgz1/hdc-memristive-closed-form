"""External check against a measured PCM chip (Karunaratne et al., Nature Electronics 2020).

Their associative-memory experiment stores binary prototypes (D = 10,000, 4-gram language
encoding, 22 languages) as SET (1) / RESET (0) PCM states, reads every device individually with
an 8-bit ADC and sums the currents in software (their Supplementary Note I), so the score of
class c is s_c = sum_j q_j G_cj with independent device conductances. RESET states are ~0 S and
SET states are Gaussian with mean mu and spread sigma (their Supplementary Note II and Figs.
4-5). The single-device statistic that matters is therefore the relative SET spread r = sigma/mu:
    dot product:      s_c / mu = K_c + N(0, r^2 K_c),        K_c = q . p_c
    inverse Hamming:  s_c / mu = K_c + K'_c + N(0, r^2 (K_c + K'_c)),  K'_c = (1-q) . (1-p_c)
(the complement prototype sits in a second array). The predicted accuracy is the exact
Gaussian integral of analytic.predict_accuracy_quadrature with per-query variances.
This script rebuilds the binary HDC classifier of Rahimi et al. (4-grams, majority bundling,
one prototype per language from the whole training text), computes its digital accuracy for
both similarity measures, and the predicted analog accuracy over a grid of r.
Output: results/v2_karunaratne.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import log_ndtr

LABELS = ["afr", "bul", "ces", "dan", "nld", "deu", "eng", "est", "fin", "fra", "ell",
          "hun", "ita", "lav", "lit", "pol", "por", "ron", "slk", "slv", "spa", "swe"]
TEST = {"af": "afr", "bg": "bul", "cs": "ces", "da": "dan", "nl": "nld", "de": "deu",
        "en": "eng", "et": "est", "fi": "fin", "fr": "fra", "el": "ell", "hu": "hun",
        "it": "ita", "lv": "lav", "lt": "lit", "pl": "pol", "pt": "por", "ro": "ron",
        "sk": "slk", "sl": "slv", "es": "spa", "sv": "swe"}
N = 4
R_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def codes(text: str) -> np.ndarray:
    c = np.frombuffer(text.lower().encode("latin-1", "replace"), dtype=np.uint8).astype(np.int64) - 97
    c = np.where((c >= 0) & (c < 26), c, 26)
    if len(c) < N:
        return np.zeros(0, np.int64)
    g = np.zeros(len(c) - N + 1, np.int64)
    for k in range(N):
        g = g * 27 + c[k:len(c) - N + 1 + k]
    return g          # oldest character is the most significant digit


def gram_hv(g: np.ndarray, L: np.ndarray) -> np.ndarray:
    """HV(g) = rho^3(L[c0]) * rho^2(L[c1]) * rho(L[c2]) * L[c3] (MATLAB circshift order)."""
    out = np.ones((len(g), L.shape[1]), np.float32)
    for k in range(N):
        d = (g // 27 ** (N - 1 - k)) % 27
        out *= np.roll(L, N - 1 - k, axis=1)[d]
    return out


def count_matrix(texts, vocab_index, n_vocab):
    rows, cols, vals = [], [], []
    for i, t in enumerate(texts):
        u, cnt = np.unique(codes(t), return_counts=True)
        rows.append(np.full(len(u), i)); cols.append(vocab_index[u]); vals.append(cnt)
    return sparse.csr_matrix((np.concatenate(vals).astype(np.float32),
                              (np.concatenate(rows), np.concatenate(cols))), shape=(len(texts), n_vocab))


def accuracy_quadrature(S, V, y, n_nodes=64):
    """Mean over queries of P(argmax(S + N(0, V)) == y); S, V: (n, C) per query."""
    z, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    tot = 0.0
    for i in range(0, len(S), 2000):
        s, v, yy = S[i:i + 2000], np.maximum(V[i:i + 2000], 1e-12), y[i:i + 2000]
        idx = np.arange(len(s))
        sy, vy = s[idx, yy][:, None, None], v[idx, yy][:, None, None]
        arg = (sy + np.sqrt(vy) * z[None, :, None] - s[:, None, :]) / np.sqrt(v[:, None, :])
        lp = log_ndtr(arg)
        lp[idx, :, yy] = 0.0
        tot += (np.exp(lp.sum(2)) @ w).sum()
    return tot / len(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_home", default="./data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--dim", type=int, default=10000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max_test", type=int, default=0)
    a = ap.parse_args()
    from hdc import load_dataset
    load_dataset("langid", a.data_home)          # clones the corpus at the pinned commit
    repo = Path(a.data_home) / "langid" / "repo"
    train = [(repo / "training_texts" / f"{l}.txt").read_text(encoding="latin-1") for l in LABELS]
    files = sorted(f for f in os.listdir(repo / "testing_texts")
                   if f.split("_")[0] in TEST and f.endswith("_p.txt"))
    if a.max_test:
        files = files[::max(1, len(files) // a.max_test)]
    test = [(repo / "testing_texts" / f).read_text(encoding="latin-1") for f in files]
    y = np.array([LABELS.index(TEST[f.split("_")[0]]) for f in files])
    vocab = np.unique(np.concatenate([np.unique(codes(t)) for t in train + test]))
    index = np.full(27 ** N, -1, np.int64); index[vocab] = np.arange(len(vocab))
    Ctr = count_matrix(train, index, len(vocab)).tocsc()
    Cte = count_matrix(test, index, len(vocab)).tocsc()
    print(f"vocab {len(vocab)} grams, {len(test)} test sentences, D = {a.dim}", flush=True)
    out = {"dim": a.dim, "n_test": len(test), "r_grid": R_GRID, "seeds": []}
    for seed in map(int, a.seeds.split(",")):
        t0 = time.time()
        rng = np.random.default_rng(seed)
        L = np.where(rng.random((27, a.dim)) < 0.5, -1.0, 1.0).astype(np.float32)
        Ptr = np.zeros((len(train), a.dim), np.float32)
        Qte = np.zeros((len(test), a.dim), np.float32)
        for s0 in range(0, len(vocab), 20000):
            H = gram_hv(vocab[s0:s0 + 20000], L)
            Ptr += Ctr[:, s0:s0 + 20000] @ H
            Qte += Cte[:, s0:s0 + 20000] @ H
        P = (Ptr > 0).astype(np.float32)       # binary prototypes: SET = 1, RESET = 0
        Q = (Qte > 0).astype(np.float32)
        K = Q @ P.T                            # dot product (overlap of ones)
        Kc = (1 - Q) @ (1 - P).T               # complement array: overlap of zeros
        rec = {"seed": seed}
        for name, S in [("dotp", K), ("invhamm", K + Kc)]:
            rec[f"digital_{name}"] = float((S.argmax(1) == y).mean())
            rec[f"pred_{name}"] = [float(accuracy_quadrature(S, (r ** 2) * S, y)) if r > 0
                                   else rec[f"digital_{name}"] for r in R_GRID]
            # Monte-Carlo check at r = 0.2 (independent Gaussian SET conductances)
            g = np.random.default_rng(100 + seed)
            mc = np.mean([((S + g.standard_normal(S.shape) * np.sqrt(0.04 * S)).argmax(1) == y).mean()
                          for _ in range(20)])
            rec[f"mc_r0.2_{name}"] = float(mc)
        rec["seconds"] = round(time.time() - t0, 1)
        out["seeds"].append(rec)
        print(json.dumps(rec), flush=True)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    (Path(a.outdir) / "v2_karunaratne.json").write_text(json.dumps(out, indent=1))
    print("written", Path(a.outdir) / "v2_karunaratne.json")


if __name__ == "__main__":
    main()
