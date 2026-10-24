"""Cross-simulator check: the same HDC prototypes in Sandia CrossSim (v3.2), with CrossSim's
own, independently implemented device models and periphery.

Devices (CrossSim built-ins, fitted by Sandia to the cited measurements):
  pcm_joshi  PCM, Joshi et al. Nat. Commun. 2020 (programming error, 0.25-25 uS)
  rram_wan   ReRAM, Wan et al. Nature 2022 (CrossSim's own cubic fit, t = 1 s)
  rram_milo  ReRAM, Milo et al. IRPS 2021 (50-225 uS, on/off 4.5)
  sonos      SONOS charge-trap, Agrawal et al. IEDM 2022 (programming error, drift over
             days; per-read noise off, see DEVICES)
Mapping: BALANCED (differential pair, one-sided), inputs on 512-row partitions (as the
AIHWKit tiles), one global weight scale (CrossSim maps max|W| of the whole matrix).
Periphery: 'ideal' (no ADC) or 'realistic' (9-bit ADC per partition, range calibrated on
training queries at the 99.9th percentile of |partial sums|, as for AIHWKit). Inputs are
+/-1 so the DAC is exact. No drift compensation (CrossSim has none).

The closed-form prediction uses only (i) single-cell statistics m(w), v(w) measured once
per technology by programming a grid of weights into a CrossSim core and reading it with
one-hot inputs, and (ii) the known ADC bound / step. No HDC model is simulated.
"""
from __future__ import annotations

import functools

import numpy as np

GRID = np.linspace(-1.0, 1.0, 81)
ROWS = 512
ADC_BITS = 9

DEVICES = {
    "pcm_joshi": dict(error_model="PCMJoshi", Rmin=1 / 25e-6, Rmax=100 / 25e-6),
    "rram_wan": dict(error_model="RRAMWan", Rmin=1 / 39e-6, Rmax=10 / 39e-6),
    "rram_milo": dict(error_model="RRAMMilo", Rmin=1 / 225e-6, Rmax=1 / 50e-6),
    # SONOS per-read noise is left off: CrossSim draws a fresh noise matrix of size
    # (queries x classes x D) per MVM, which is too slow for 10^4-query test sets
    "sonos": dict(error_model="SONOS", drift_model="SONOS",
                  Rmin=1 / 16e-6, Rmax=1e7 / 16e-6, Vread=0.1),
}
# times in days (only the SONOS model has drift in CrossSim)
T_GRID = {"pcm_joshi": [0.0], "rram_wan": [0.0], "rram_milo": [0.0], "sonos": [0.0, 1.0, 5.0]}


def _grange(device: str) -> float:
    d = DEVICES[device]
    return 1.0 - d["Rmin"] / d["Rmax"]


def make_params(device: str, periphery: str, adc_bound: float | None = None, t: float = 0.0):
    """adc_bound is the ADC full scale in normalised output units (weights / max|W|,
    inputs +/-1); CrossSim's calibrated range is given in conductance units."""
    from applications.mvm_params import set_params

    kw = dict(DEVICES[device])
    if device == "sonos":
        kw["t_drift"] = float(t)
    if periphery == "realistic":
        r = adc_bound * _grange(device)
        kw.update(adc_bits=ADC_BITS, adc_range=(-r, r))
    elif periphery != "ideal":
        raise ValueError(periphery)
    return set_params(NrowsMax=ROWS, input_range=(-1, 1), **kw)


def calibrate_adc_bound(M: np.ndarray, H_cal: np.ndarray, pct: float = 99.9) -> float:
    W = M / np.abs(M).max()
    vals = [np.abs(H_cal[:, r0:r0 + ROWS] @ W[:, r0:r0 + ROWS].T).ravel()
            for r0 in range(0, M.shape[1], ROWS)]
    v = np.concatenate(vals)
    if v.size > 2_000_000:
        v = v[:2_000_000]
    return float(np.percentile(v, pct))


def accuracy_crosssim(M: np.ndarray, H: np.ndarray, y: np.ndarray, device: str,
                      periphery: str, times, n_chips: int = 8, seed: int = 0,
                      adc_bound: float | None = None, bs: int = 2000) -> np.ndarray:
    """acc[chip, t]. Every (chip, t) is an independent programming (CrossSim applies the
    drift state of time t when the core is programmed)."""
    from simulator import AnalogCore

    acc = np.zeros((n_chips, len(times)))
    for k, t in enumerate(times):
        p = make_params(device, periphery, adc_bound, t)
        for c in range(n_chips):
            np.random.seed(seed * 1000 + c)
            core = AnalogCore(M.astype(np.float64), params=p)
            correct = 0
            for i in range(0, len(H), bs):
                out = core @ H[i:i + bs].T.astype(np.float64)
                if not np.isfinite(out).all():
                    raise FloatingPointError(f"non-finite CrossSim output ({device}, t={t})")
                correct += (out.argmax(0) == y[i:i + bs]).sum()
            acc[c, k] = correct / len(H)
    return acc


@functools.lru_cache(maxsize=32)
def characterise(device: str, t: float = 0.0, n_rep: int = 4000, seed: int = 0):
    """Single-cell statistics of the effective weight (normalised units) on GRID:
    program a (n_rep x 81) grid matrix, read with one-hot inputs (ideal periphery)."""
    from simulator import AnalogCore

    np.random.seed(seed)
    W = np.tile(GRID, (n_rep, 1))
    core = AnalogCore(W, params=make_params(device, "ideal", t=t))
    R = core @ np.eye(len(GRID))
    return R.mean(0), R.var(0)


def predict_accuracy(M: np.ndarray, H: np.ndarray, y: np.ndarray, device: str,
                     periphery: str, t: float = 0.0, adc_bound: float | None = None,
                     n_draws: int = 20, seed: int = 0, bs: int = 4096,
                     return_clip: bool = False):
    m, v = characterise(device, float(t))
    A = np.abs(M).max()
    w = M / A
    mt, vt = np.interp(w, GRID, m), np.interp(w, GRID, v)
    tiles = [(slice(r0, r0 + ROWS), mt[:, r0:r0 + ROWS], vt[:, r0:r0 + ROWS].sum(1))
             for r0 in range(0, M.shape[1], ROWS)]
    rng = np.random.default_rng(seed)
    step = 2 * adc_bound / (2 ** ADC_BITS - 2) if periphery == "realistic" else None
    correct, nclip, nout = 0, 0, 0
    for i in range(0, len(H), bs):
        Hb = H[i:i + bs].astype(np.float64)
        yb = y[i:i + bs]
        U = [(Hb[:, sl] @ mtt.T, np.sqrt(vs)) for sl, mtt, vs in tiles]
        for _ in range(n_draws):
            S = 0.0
            for mu, sd in U:
                u = mu + rng.standard_normal(mu.shape) * sd
                if step is not None:
                    nclip += (np.abs(u) >= adc_bound).sum()
                    nout += u.size
                    u = np.round(np.clip(u, -adc_bound, adc_bound) / step) * step
                S = S + u
            correct += (S.argmax(1) == yb).sum()
    acc = float(correct / (n_draws * len(H)))
    return (acc, nclip / max(nout, 1)) if return_clip else acc
