"""Closed-form (simulation-free) accuracy model for HDC prototypes in an analog crossbar.

Only single-device statistics enter: for a device technology and a read time t we measure,
once, how a weight w (normalised to the tile's largest |w|, as the crossbar mapping does)
is perturbed by programming noise, drift and read noise:

    m_t(w) = E[w_tilde(t)],   v_t(w) = Var[w_tilde(t)]        (device characterisation)

Global drift compensation divides every tile output by alpha_t, which we compute exactly as
AIHWKit does (mean |output| of one-hot reads now vs. right after programming).

For a query h in {-1,+1}^D the analog score of class c is a sum over tiles T (512 rows each):

    s_c = sum_T A_T * [ sum_{j in T} h_j * w_tilde_cj / alpha_T  +  n_T,c / s ]

with A_T the tile's weight scale, s the read-voltage (input) scale and n_T,c ~ N(0, sigma_out^2)
the per-tile output noise. Taking expectations gives a deterministic, chip-independent
distortion of every prototype entry and a Gaussian score noise with per-class variance

    mean:  mu_cj = A_T * m_t(w_cj) / alpha_T
    var :  V_c   = sum_T A_T^2 * [ sum_{j in T} v_t(w_cj) / alpha_T^2  +  sigma_out^2 / s^2 ]

(the h_j^2 = 1 factor makes V_c query independent). Errors of different devices are
independent, so the noise on the decision variable s_true - s_comp has variance
V_true + V_comp: this is sigma_Delta in closed form. ADC quantisation is deterministic and
nearly identical for nearly equal scores, so it does not enter sigma_Delta.

The predicted accuracy is the probability that argmax_c (mu-distorted score + N(0, V_c))
returns the true class, estimated by drawing the Gaussian part only (no device simulation).
"""
from __future__ import annotations

import functools

import numpy as np
import torch

GRID = np.linspace(-1.0, 1.0, 81)


@functools.lru_cache(maxsize=64)
def characterise(device: str, t: float, noise_scale: float = 1.0, n_rep: int = 4000,
                 seed: int = 0):
    """Single-device statistics m_t(w), v_t(w) on GRID and m_0(w) (right after programming).
    Uses the same noise model object AIHWKit uses inside the tile."""
    from hdc import make_inference_config

    cfg = make_inference_config(device, "ideal", noise_scale=noise_scale)
    nm = cfg.noise_model
    torch.manual_seed(seed)
    W = torch.tensor(GRID, dtype=torch.float32).repeat(n_rep, 1)  # max |w| = 1 per row block
    prog, params = nm.apply_programming_noise(W)
    W0 = prog  # AIHWKit takes the drift baseline right after programming
    Wt = nm.apply_drift_noise(prog, params, t)
    return (Wt.mean(0).numpy().astype(np.float64), Wt.var(0).numpy().astype(np.float64),
            W0.mean(0).numpy().astype(np.float64))


def _interp(fn_vals, w):
    return np.interp(w, GRID, fn_vals)


def score_model(Mq: torch.Tensor, device: str, t: float, in_scale: float, periphery: str,
                noise_scale: float = 1.0, max_input_size: int = 512, out_noise: float = 0.06):
    """Returns (M_eff, V): drift-distorted prototypes (C x D, raw units) and the per-class
    score-noise variance V (C,) in raw dot-product units."""
    m_t, v_t, m_0 = characterise(device, float(t), float(noise_scale))
    M = Mq.numpy().astype(np.float64)
    C, D = M.shape
    M_eff = np.zeros_like(M)
    V = np.zeros(C)
    sig_out = 0.0 if periphery == "ideal" else out_noise
    s = in_scale if periphery == "realistic" else 1.0
    for r0 in range(0, D, max_input_size):
        blk = M[:, r0:r0 + max_input_size]
        A = np.abs(blk).max()
        if A == 0:
            continue
        w = blk / A
        mt, vt, m0 = _interp(m_t, w), _interp(v_t, w), _interp(m_0, w)
        # AIHWKit global compensation: mean |one-hot readout| now / right after programming
        alpha = np.abs(mt).mean() / max(np.abs(m0).mean(), 1e-12)
        M_eff[:, r0:r0 + max_input_size] = A * mt / alpha
        V += A ** 2 * (vt.sum(1) / alpha ** 2 + sig_out ** 2 / s ** 2)
    return M_eff, V


@torch.no_grad()
def predict_accuracy_analytic(Mq, H, y, device, t, in_scale, periphery="realistic",
                              noise_scale=1.0, n_draws=20, seed=0, bs=8192):
    M_eff, V = score_model(Mq, device, t, in_scale, periphery, noise_scale)
    Me = torch.tensor(M_eff, dtype=torch.float32)
    sd = np.sqrt(V)
    rng = np.random.default_rng(seed)
    correct = 0
    for i in range(0, len(H), bs):
        S = (H[i:i + bs].float() @ Me.T).numpy()
        yb = y[i:i + bs]
        for _ in range(n_draws):
            correct += ((S + rng.normal(0, 1, S.shape) * sd).argmax(1) == yb).sum()
    return float(correct / (n_draws * len(H)))



def predict_accuracy_quadrature(Mq, H, y, device, t, in_scale, periphery="realistic",
                                noise_scale=1.0, n_nodes=48, bs=4096):
    """Closed-form accuracy of the v1 Gaussian score model, without random draws.

    The class scores s_c = h.mu_c + e_c have independent Gaussian errors e_c ~ N(0, V_c)
    (every class uses its own devices; h is +/-1, so V_c does not depend on the query).
    For a query with true class y,
        P(correct) = int phi(z) prod_{c != y} Phi((S_y - S_c + sd_y z) / sd_c) dz,
    a one-dimensional integral evaluated with Gauss-Hermite quadrature. The predicted
    accuracy is its mean over the test queries."""
    from scipy.special import log_ndtr
    M_eff, V = score_model(Mq, device, t, in_scale, periphery, noise_scale)
    Me = torch.tensor(M_eff, dtype=torch.float32)
    sd = np.sqrt(V)
    z, w = np.polynomial.hermite_e.hermegauss(n_nodes)   # weight exp(-z^2/2)
    w = w / w.sum()
    total = 0.0
    for i in range(0, len(H), bs):
        S = (H[i:i + bs].float() @ Me.T).numpy().astype(np.float64)
        yb = np.asarray(y[i:i + bs])
        Sy = S[np.arange(len(S)), yb][:, None, None]
        sy = sd[yb][:, None, None]
        arg = (Sy + sy * z[None, :, None] - S[:, None, :]) / sd[None, None, :]
        lp = log_ndtr(arg)                                   # (B, K, C)
        lp[np.arange(len(S)), :, yb] = 0.0                   # exclude the true class
        total += (np.exp(lp.sum(2)) @ w).sum()
    return float(total / len(H))

def sigma_delta_analytic(Mq, H_cal, y_cal, device, t, in_scale, periphery="realistic",
                         noise_scale=1.0):
    """sigma_Delta in the cosine units used by the calibration-chip estimate, for comparison."""
    _, V = score_model(Mq, device, t, in_scale, periphery, noise_scale)
    M = Mq.numpy()
    norm = np.linalg.norm(M, axis=1) * np.sqrt(M.shape[1])
    R = (H_cal.float().numpy() @ M.T) / norm
    C = M.shape[0]
    comp = np.where(np.arange(C)[None, :] == y_cal[:, None], -np.inf, R).argmax(1)
    # per-sample variance of (e_true - e_comp) in cosine units, averaged
    var = V[y_cal] / norm[y_cal] ** 2 + V[comp] / norm[comp] ** 2
    return float(np.sqrt(var.mean()))


# --------------------------------------------------------------------------------------
# v2: periphery-aware closed form. The tile output is formed in the *analog* domain as
#   u_T = s_eff * h_T . w_tilde_T + n_T          (n_T ~ N(0, sigma_out^2))
# and is then clipped to +-out_bound and rounded to the ADC step BEFORE the digital
# rescale A_T / (s * alpha_T). Two deterministic periphery effects therefore matter when the
# read scale s is small (large common-mode partial sums sitting near the ADC bound):
#   * DAC quantisation: s is realised as s_eff = round(s / inp_step) * inp_step,
#   * read-time conductance gain (the largest programmed weight of the tile, about 1.1 for 512-row tiles of the Wan-2022 ReRAM) which
#     drift compensation only removes AFTER the ADC.
# Both push partial sums past the calibrated bound; clipping then removes class differences
# (ties) and cannot be captured by a Gaussian score model. v2 keeps the per-device closed
# form (mean m_t, variance v_t; CLT over the 512 rows of a tile) but evaluates
# clip + ADC rounding per tile on Gaussian draws. Still no device simulation.
# --------------------------------------------------------------------------------------
OUT_BOUND = 12.0
OUT_STEP = 2 * OUT_BOUND / (2 ** 9 - 2)   # 9-bit ADC (AIHWKit default out_res)
INP_STEP = 2.0 / (2 ** 7 - 2)             # 7-bit DAC on [-1, 1] (AIHWKit default inp_res)


def dac_scale(s: float) -> float:
    return float(np.round(s / INP_STEP) * INP_STEP)


# Devices whose AIHWKit drift step re-derives conductances from the *original* targets but
# converts back with the scale of the *programmed* weights: every tile then reads with a
# chip-random gain G_T = max_j |w_prog,j| (the largest programmed weight of that tile),
# e.g. ~1.1 for the Wan-2022 ReRAM model. PCM / CMO re-use the programmed conductances, G = 1.
PROG_MAX_GAIN = {"reram_wan": True, "pcm": False, "reram_cmo": False}


@functools.lru_cache(maxsize=64)
def characterise2(device: str, t: float, noise_scale: float = 1.0, n_rep: int = 4000,
                  seed: int = 0):
    """Like characterise, but returns intrinsic statistics with the programming-max gain of
    the characterisation tensor divided out, plus the programming variance v_0(w)."""
    from hdc import make_inference_config

    nm = make_inference_config(device, "ideal", noise_scale=noise_scale).noise_model
    torch.manual_seed(seed)
    W = torch.tensor(GRID, dtype=torch.float32).repeat(n_rep, 1)
    prog, params = nm.apply_programming_noise(W)
    Wt = nm.apply_drift_noise(prog, params, t)
    if PROG_MAX_GAIN[device]:
        Wt = Wt / prog.abs().max()
    f = lambda a: a.numpy().astype(np.float64)
    return f(Wt.mean(0)), f(Wt.var(0)), f(prog.mean(0)), f(prog.var(0))


_XG = np.linspace(0.8, 2.0, 1201)


def gain_quantiles(w: np.ndarray, m0: np.ndarray, v0: np.ndarray, n_q: int = 64):
    """Quantiles of G = max_j |X_j|, X_j ~ N(m0(w_j), v0(w_j)) independent (exact CDF
    as a product of single-device CDFs; only entries that can reach the max matter)."""
    from scipy.stats import norm
    sel = np.abs(w).ravel() > 0.6
    mu, sd = np.abs(m0.ravel()[sel]), np.sqrt(v0.ravel()[sel]) + 1e-9
    logF = np.zeros_like(_XG)
    for k in range(0, len(mu), 4096):
        a, b = mu[k:k + 4096, None], sd[k:k + 4096, None]
        p = norm.cdf((_XG[None] - a) / b) - norm.cdf((-_XG[None] - a) / b)
        logF += np.log(np.clip(p, 1e-300, 1)).sum(0)
    F = np.exp(logF)
    q = (np.arange(n_q) + 0.5) / n_q
    return np.interp(q, F, _XG)


def tile_model(Mq: torch.Tensor, device: str, t: float, noise_scale: float = 1.0,
               max_input_size: int = 512):
    """Per tile: (row slice, A_T, alpha_T, intrinsic mean m (C x rows), summed intrinsic
    variance per class (C,), quantiles of the read gain G_T)."""
    m_t, v_t, m_0, v_0 = characterise2(device, float(t), float(noise_scale))
    M = Mq.numpy().astype(np.float64)
    out = []
    for r0 in range(0, M.shape[1], max_input_size):
        blk = M[:, r0:r0 + max_input_size]
        A = np.abs(blk).max()
        if A == 0:
            continue
        w = blk / A
        mt, vt, m0 = _interp(m_t, w), _interp(v_t, w), _interp(m_0, w)
        if PROG_MAX_GAIN[device]:
            G = gain_quantiles(w, m0, _interp(v_0, w))
        else:
            G = np.ones(1)
        # drift compensation divides by mean|read| / mean|programmed|; G_T cancels exactly
        alpha = np.abs(mt).mean() / max(np.abs(m0).mean(), 1e-12)
        out.append((slice(r0, r0 + max_input_size), A, alpha, mt, vt.sum(1), G))
    return out


@torch.no_grad()
def predict_accuracy_analytic2(Mq, H, y, device, t, in_scale, periphery="realistic",
                               noise_scale=1.0, n_draws=20, seed=0, bs=4096,
                               out_noise=0.06, return_parts=False, ablate=()):
    """ablate: any of "dac", "gain", "clip" switches that v2 effect off (component ablation):
    dac -> unquantised read scale, gain -> per-tile read gain fixed at 1,
    clip -> no per-tile ADC clipping/rounding (output noise kept)."""
    if periphery != "realistic":
        return predict_accuracy_analytic(Mq, H, y, device, t, in_scale, periphery,
                                         noise_scale, n_draws, seed)
    tiles = tile_model(Mq, device, t, noise_scale)
    s, se = float(in_scale), dac_scale(in_scale)
    if "dac" in ablate:
        se = s
    rng = np.random.default_rng(seed)
    correct = 0
    clip_frac = 0.0
    n_out = 0
    for i in range(0, len(H), bs):
        Hb = H[i:i + bs].float().numpy().astype(np.float64)
        yb = y[i:i + bs]
        U = [(se * (Hb[:, sl] @ mt.T), se ** 2 * vs, A / (s * al), G)
             for sl, A, al, mt, vs, G in tiles]
        for _ in range(n_draws):
            S = 0.0
            for mu, var, g, G in U:
                Gc = 1.0 if "gain" in ablate else G[rng.integers(len(G))]  # one read gain per tile
                u = Gc * mu + rng.standard_normal(mu.shape) * np.sqrt(Gc ** 2 * var + out_noise ** 2)
                clip_frac += (np.abs(u) >= OUT_BOUND).sum()
                n_out += u.size
                if "clip" not in ablate:
                    u = np.round(np.clip(u, -OUT_BOUND, OUT_BOUND) / OUT_STEP) * OUT_STEP
                S = S + g * u / Gc
            # ties (several classes clipped to the same value) resolve to the lowest index,
            # as torch.argmax does in the crossbar evaluation
            correct += (S.argmax(1) == yb).sum()
    acc = float(correct / (n_draws * len(H)))
    return (acc, clip_frac / max(n_out, 1)) if return_parts else acc
