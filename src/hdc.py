"""Hyperdimensional computing (HDC) with class prototypes stored in a memristive crossbar.

Pipeline
  1. Encode each sample with a random projection: h = sign(x P), P ~ N(0, 1)^(F x D),
     giving bipolar query hypervectors that drive the crossbar rows as +/-V pulses.
  2. Train class prototypes in software: one bundling pass (sum of class hypervectors),
     then optional perceptron-style retraining epochs (standard in HDC literature).
  3. L2-normalise each prototype (dot product -> cosine similarity) and quantise it to
     the chosen precision (fp / k-bit / binary).
  4. Program the prototypes into an AIHWKit inference tile with a device noise model
     calibrated on real hardware (PCM, NeuRRAM ReRAM, CMO/HfOx ReRAM), let them drift
     for time t, and classify by argmax over the analog dot products.
Every (device, D, precision, t) point is evaluated over several independently
programmed "chips" (Monte Carlo) so device-to-device spread shows up as variance.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- data
OPENML = {  # name -> (openml name, version, standard split)
    "isolet": ("isolet", 1, None),
    "ucihar": ("har", 1, None),
    "mnist": ("mnist_784", 1, 60000),
    "fmnist": ("Fashion-MNIST", 1, 60000),
    "letter": ("letter", 1, None),
    "pendigits": ("pendigits", 1, None),
    "satimage": ("satimage", 1, None),
}
ALL_DATASETS = list(OPENML)

# European language recognition (Joshi, Halseth, Kanerva 2016; Rahimi et al. ISLPED 2016):
# 22 training languages, 21,000 test sentences (21 languages; Afrikaans has no test set).
LANGID_REPO = "https://github.com/abbas-rahimi/HDC-Language-Recognition"
LANGID_COMMIT = "b3be34dd7caa2dc0c13be60447e74956e64cfd33"
LANGID_LABELS = ["afr", "bul", "ces", "dan", "nld", "deu", "eng", "est", "fin", "fra", "ell",
                 "hun", "ita", "lav", "lit", "pol", "por", "ron", "slk", "slv", "spa", "swe"]
LANGID_TEST = {"af": "afr", "bg": "bul", "cs": "ces", "da": "dan", "nl": "nld", "de": "deu",
               "en": "eng", "et": "est", "fi": "fin", "fr": "fra", "el": "ell", "hu": "hun",
               "it": "ita", "lv": "lav", "lt": "lit", "pl": "pol", "pt": "por", "ro": "ron",
               "sk": "slk", "sl": "slv", "es": "spa", "sv": "swe"}
LANGID_CHUNK = 150      # training sample length (characters), close to the mean test sentence
LANGID_PER_LANG = 2000  # training chunks drawn per language
LANGID_MAXLEN = 1200    # longest test sentence is 1,138 characters
N_SYMBOLS = 27          # a-z and space; any other character is read as a space


def _trigram_ids(text: str, maxlen: int) -> np.ndarray:
    """Trigram index sequence (c[t-2]*729 + c[t-1]*27 + c[t]), padded with -1."""
    c = np.frombuffer(text.lower().encode("latin-1", "replace"), dtype=np.uint8).astype(np.int32) - 97
    c = np.where((c >= 0) & (c < 26), c, 26)[:maxlen]
    g = c[:-2] * 729 + c[1:-1] * 27 + c[2:] if len(c) >= 3 else np.zeros(0, np.int32)
    out = np.full(maxlen, -1, np.int16)
    out[:len(g)] = g
    return out


def _load_langid(data_home: str):
    import os
    import subprocess
    from pathlib import Path
    root = Path(data_home) / "langid"
    cache = root / "langid_trigrams_v1.npz"
    if cache.exists():
        z = np.load(cache)
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"], len(LANGID_LABELS)
    repo = root / "repo"
    if not (repo / "training_texts").exists():
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", LANGID_REPO, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", LANGID_COMMIT], check=True)
    rng = np.random.default_rng(0)
    Xtr, ytr = [], []
    for k, lab in enumerate(LANGID_LABELS):
        txt = (repo / "training_texts" / f"{lab}.txt").read_text(encoding="latin-1")
        starts = np.arange(0, len(txt) - LANGID_CHUNK, LANGID_CHUNK)
        for s0 in np.sort(rng.choice(starts, size=min(LANGID_PER_LANG, len(starts)), replace=False)):
            Xtr.append(_trigram_ids(txt[s0:s0 + LANGID_CHUNK], LANGID_CHUNK))
            ytr.append(k)
    Xte, yte = [], []
    for f in sorted(os.listdir(repo / "testing_texts")):
        if f.split("_")[0] in LANGID_TEST and f.endswith("_p.txt"):
            Xte.append(_trigram_ids((repo / "testing_texts" / f).read_text(encoding="latin-1"),
                                    LANGID_MAXLEN))
            yte.append(LANGID_LABELS.index(LANGID_TEST[f.split("_")[0]]))
    pad = np.full((len(Xtr), LANGID_MAXLEN - LANGID_CHUNK), -1, np.int16)
    Xtr = np.concatenate([np.stack(Xtr), pad], 1)
    out = (Xtr, np.array(ytr, np.int64), np.stack(Xte), np.array(yte, np.int64))
    np.savez_compressed(cache, Xtr=out[0], ytr=out[1], Xte=out[2], yte=out[3])
    return (*out, len(LANGID_LABELS))


def load_dataset(name: str, data_home: str = "./data", seed: int = 0):
    """Returns min-max normalised float32 (Xtr, ytr, Xte, yte) and the class count.
    MNIST / Fashion-MNIST keep their official 60k/10k split; the other OpenML sets are
    split 80/20, stratified, with a fixed seed."""
    if name == "langid":  # trigram index sequences, no feature normalisation
        return _load_langid(data_home)
    if name == "digits":  # offline smoke-test set
        from sklearn.datasets import load_digits
        d = load_digits()
        X, y, n_train = d.data.astype(np.float32), d.target, None
    elif name == "synth":  # offline, harder: 20 overlapping classes (method prototyping only)
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=12000, n_features=64, n_informative=32,
                                   n_redundant=16, n_classes=20, n_clusters_per_class=1,
                                   class_sep=2.0, random_state=0)
        X, n_train = X.astype(np.float32), None
    else:
        from sklearn.datasets import fetch_openml
        oname, ver, n_train = OPENML[name]
        X, y = fetch_openml(oname, version=ver, return_X_y=True, as_frame=False,
                            data_home=data_home, parser="auto")
        X = X.astype(np.float32)
    y = np.unique(y, return_inverse=True)[1].astype(np.int64)
    if n_train:
        Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    else:
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    lo, hi = Xtr.min(0), Xtr.max(0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    Xtr, Xte = (Xtr - lo) / scale, np.clip((Xte - lo) / scale, 0, 1)
    return Xtr.astype(np.float32), ytr, Xte.astype(np.float32), yte, int(y.max() + 1)


# ----------------------------------------------------------------------------- HDC
def _bipolar(z: torch.Tensor) -> torch.Tensor:
    return torch.where(z >= 0, 1, -1).to(torch.int8)


class Encoder:
    """Random-projection encoder h = sign((x - mean) P). With a mean the training-set feature
    mean is removed first, which strips the common-mode component that non-negative
    features otherwise put into every hypervector (it wastes ADC range on the crossbar)."""

    kind = "rp"

    def __init__(self, n_features: int, dim: int, seed: int = 0, mean: np.ndarray | None = None):
        g = torch.Generator().manual_seed(seed)
        self.P = torch.randn(n_features, dim, generator=g)
        self.mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float32)

    def centre(self, X) -> torch.Tensor:
        X = torch.as_tensor(X)
        return X - self.mean if self.mean is not None else X

    def pre(self, X) -> torch.Tensor:
        """Input vector that multiplies P (drives the encoder crossbar rows)."""
        return self.centre(X).float()

    def post(self, z: torch.Tensor) -> torch.Tensor:
        """Projection output z = pre(x) P -> bipolar hypervector."""
        return _bipolar(z)

    @torch.no_grad()
    def __call__(self, X: np.ndarray, bs: int = 4096) -> torch.Tensor:
        X = self.pre(X)
        out = torch.empty(len(X), self.P.shape[1], dtype=torch.int8)  # +/-1, 1 byte each
        for i in range(0, len(X), bs):
            out[i:i + bs] = self.post(X[i:i + bs] @ self.P)
        return out


class NonlinearEncoder(Encoder):
    """OnlineHD-style nonlinear encoder, binarised for the crossbar:
    h = sign(cos(x P + b) * sin(x P)), b ~ U(0, 2 pi), with each (centred) sample scaled to
    unit L2 norm first, as in OnlineHD / MM-HDC, so x P has unit variance."""

    kind = "nonlinear"

    def __init__(self, n_features: int, dim: int, seed: int = 0, mean=None):
        super().__init__(n_features, dim, seed, mean)
        g = torch.Generator().manual_seed(seed + 7)
        self.b = torch.rand(dim, generator=g) * 2 * np.pi

    def pre(self, X) -> torch.Tensor:
        X = self.centre(X).float()
        return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def post(self, z: torch.Tensor) -> torch.Tensor:
        return _bipolar(torch.cos(z + self.b) * torch.sin(z))


class IDLevelEncoder:
    """Record-based ID-level encoder: h = sign(sum_f ID_f * L_{q(x_f)}), with random bipolar
    ID vectors per feature and L correlated level vectors (L_0 random; each next level flips
    a fresh block of D / (2 (L-1)) components, so L_0 and L_{L-1} are ~orthogonal).
    Features are expected in [0, 1] (min-max normalised); ties go to +1."""

    kind = "idlevel"

    def __init__(self, n_features: int, dim: int, seed: int = 0, levels: int = 16, mean=None):
        g = torch.Generator().manual_seed(seed)
        self.ids = torch.where(torch.rand(n_features, dim, generator=g) < 0.5, -1.0, 1.0)
        lv = torch.empty(levels, dim)
        lv[0] = torch.where(torch.rand(dim, generator=g) < 0.5, -1.0, 1.0)
        order = torch.randperm(dim, generator=g)
        step = dim // (2 * (levels - 1))
        for k in range(1, levels):
            lv[k] = lv[k - 1]
            lv[k, order[(k - 1) * step:k * step]] *= -1
        self.lv, self.levels = lv, levels

    @torch.no_grad()
    def __call__(self, X, bs: int = 2048):
        X = torch.as_tensor(X)
        q = torch.clamp((X * (self.levels - 1)).round().long(), 0, self.levels - 1)
        out = torch.empty(len(X), self.ids.shape[1], dtype=torch.int8)
        for i in range(0, len(X), bs):
            qb = q[i:i + bs]
            acc = torch.zeros(len(qb), self.ids.shape[1])
            for f in range(self.ids.shape[0]):
                acc += self.ids[f] * self.lv[qb[:, f]]
            out[i:i + bs] = _bipolar(acc)
        return out


class NGramEncoder:
    """Character trigram encoder of the language-recognition benchmark (Rahimi et al. 2016):
    h = sign(sum_t rho^2(L[c_{t-2}]) * rho(L[c_{t-1}]) * L[c_t]), with random bipolar letter
    vectors L and rho a cyclic shift by one position. Implemented as h = sign(n T), where n
    holds the trigram counts and T the 27^3 trigram hypervectors. Input: padded trigram index
    sequences from load_dataset("langid"); ties go to +1."""

    kind = "ngram"

    def __init__(self, dim: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        L = torch.where(torch.rand(N_SYMBOLS, dim, generator=g) < 0.5, -1.0, 1.0)
        R1, R2 = torch.roll(L, 1, 1), torch.roll(L, 2, 1)
        self.T = (R2[:, None, None] * R1[None, :, None] * L[None, None, :]).reshape(-1, dim)

    @torch.no_grad()
    def __call__(self, X, bs: int = 1024):
        X = torch.as_tensor(np.asarray(X)).long()
        out = torch.empty(len(X), self.T.shape[1], dtype=torch.int8)
        for i in range(0, len(X), bs):
            xb = X[i:i + bs]
            r, c = torch.nonzero(xb >= 0, as_tuple=True)
            n = torch.sparse_coo_tensor(torch.stack([r, xb[r, c]]), torch.ones(len(r)),
                                        (len(xb), self.T.shape[0])).coalesce()
            out[i:i + bs] = _bipolar(torch.sparse.mm(n, self.T))
        return out


def make_encoder(kind: str, n_features: int, dim: int, seed: int, Xtr: np.ndarray,
                 center: bool = True):
    if kind == "ngram":
        return NGramEncoder(dim, seed)
    mean = Xtr.mean(0) if center else None
    if kind == "rp":
        return Encoder(n_features, dim, seed, mean)
    if kind == "rp_bipolar":  # same random projection, entries replaced by their signs (+/-1)
        e = Encoder(n_features, dim, seed, mean)
        e.P = torch.where(e.P >= 0, 1.0, -1.0)
        e.kind = "rp"
        return e
    if kind == "nonlinear":
        return NonlinearEncoder(n_features, dim, seed, mean)
    if kind == "idlevel":
        return IDLevelEncoder(n_features, dim, seed)
    raise ValueError(kind)


@torch.no_grad()
def train_prototypes(H: torch.Tensor, y: np.ndarray, n_classes: int,
                     retrain_epochs: int = 0, lr: float = 0.05, seed: int = 0,
                     margin: float = 0.0, adaptive: bool = False,
                     flip_p: float = 0.0) -> torch.Tensor:
    """Single-pass bundling, then perceptron-style retraining.

    margin = 0  : classic HDC retraining (update only misclassified samples).
    flip_p > 0 : query hypervector bits are flipped at random during retraining (noise
                 injection as in DependableHD); combined with a fixed margin it is that baseline.
    margin = g>0: margin-aware retraining (proposed). A sample also triggers an update when
                  its cosine margin s_true - max_{c != true} s_c is below g, pushing it away
                  from the strongest competitor. With g = kappa * sigma_hat (sigma_hat = std of
                  the analog similarity error, estimated on training data), retraining trades
                  a little digital fitting for margins that survive device noise.
    Cosines use L2-normalised prototypes and hypervectors scaled by 1/sqrt(D)."""
    y = torch.as_tensor(y)
    M = torch.zeros(n_classes, H.shape[1])
    for i in range(0, len(y), 8192):  # single-pass bundling
        M.index_add_(0, y[i:i + 8192], H[i:i + 8192].float())
    # class-mean initialisation: makes `lr` a relative step independent of class size
    # (with a plain sum, one update moves a 6000-sample MNIST prototype by ~1/6000)
    M /= torch.bincount(y, minlength=n_classes).clamp_min(1)[:, None].float()
    g = torch.Generator().manual_seed(seed)
    inv_sqrt_d = 1.0 / np.sqrt(H.shape[1])
    for _ in range(retrain_epochs):
        for idx in torch.randperm(len(y), generator=g).split(256):
            Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)
            Hb = H[idx].float()
            if flip_p > 0:  # DependableHD-style noise injection: random query bit flips
                Hb = torch.where(torch.rand(Hb.shape, generator=g) < flip_p, -Hb, Hb)
            yb = y[idx]
            S = (Hb @ Mn.T) * inv_sqrt_d
            s_true = S.gather(1, yb[:, None]).squeeze(1)
            S_other = S.scatter(1, yb[:, None], float("-inf"))
            s_comp, comp = S_other.max(1)
            hit = (s_true - s_comp) < margin if margin > 0 else (S.argmax(1) != yb)
            if hit.any():
                Hw = Hb[hit]
                if adaptive:  # OnlineHD rule: C_y += lr(1-d_y)h, C_c -= lr(1-d_c)h, d = cosine
                    wt = (1.0 - s_true[hit])[:, None]
                    wc = (1.0 - s_comp[hit])[:, None]
                    M.index_add_(0, yb[hit], lr * wt * Hw)
                    M.index_add_(0, comp[hit], -lr * wc * Hw)
                else:
                    M.index_add_(0, yb[hit], lr * Hw)
                    M.index_add_(0, comp[hit], -lr * Hw)
    return M


MARGIN_GRID = [0.0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25]


LR_GRID = [0.01, 0.03, 0.1, 0.3]


def _val_split(n: int, seed: int, val_frac: float = 0.15, max_train: int = 20000):
    """Held-out split of the training data for hyperparameter tuning. On large datasets the
    tuning-train part is capped (default 20k samples) to keep the grid search affordable;
    the final models are always trained on the full training set."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 99))
    nv = int(n * val_frac)
    return perm[nv:nv + max_train], perm[:nv]


@torch.no_grad()
def tune_lr(H, y, n_classes, epochs, seed, grid=LR_GRID, **kw) -> float:
    """Retraining step size chosen for plain perceptron retraining on a held-out split of
    the training data; the same lr is then used by every retraining method (fair)."""
    tr, va = _val_split(len(y), seed)
    scores = [accuracy_digital(quantize(train_prototypes(H[tr], y[tr.numpy()], n_classes, epochs,
                                                         lr=lr, seed=seed, **kw), "fp"),
                               H[va], y[va.numpy()]) for lr in grid]
    return grid[int(np.argmax(scores))]


@torch.no_grad()
def tune_margin(H: torch.Tensor, y: np.ndarray, n_classes: int, epochs: int, seed: int,
                lr: float = 0.05, precision: str = "fp", grid=MARGIN_GRID) -> float:
    """Digital-optimal margin chosen on a held-out split of the *training* data (no test
    data). This is the large-margin HDC baseline; the noise-aware method adds kappa*sigma."""
    tr, va = _val_split(len(y), seed)
    best, best_acc = 0.0, -1.0
    for g in grid:
        M = train_prototypes(H[tr], y[tr.numpy()], n_classes, epochs, lr=lr, seed=seed, margin=g)
        acc = accuracy_digital(quantize(M, precision), H[va], y[va.numpy()])
        if acc > best_acc + 1e-9:
            best, best_acc = g, acc
    return best


@torch.no_grad()
def cosine_margins(Mq: torch.Tensor, H: torch.Tensor, y: np.ndarray, bs: int = 8192) -> np.ndarray:
    """Signed margin of the true class over its strongest competitor, computed on the dot
    products the crossbar uses and expressed in cosine units of the true-class prototype:
    (h.m_true - max_{c != true} h.m_c) / (||m_true|| sqrt(D)). Equals the plain cosine margin
    when all prototypes have the same norm."""
    yt = torch.as_tensor(y)
    nrm = Mq.norm(dim=1) * np.sqrt(H.shape[1])
    out = []
    for i in range(0, len(H), bs):
        S = H[i:i + bs].float() @ Mq.T
        yb = yt[i:i + bs]
        st = S.gather(1, yb[:, None]).squeeze(1)
        out.append((st - S.scatter(1, yb[:, None], float("-inf")).max(1).values) / nrm[yb])
    return torch.cat(out).numpy()


@torch.no_grad()
def predict_accuracy_gaussian(Mq: torch.Tensor, H: torch.Tensor, y: np.ndarray, sigma: float,
                              n_draws: int = 20, seed: int = 0, bs: int = 8192) -> float:
    """Margin-to-noise model: analog accuracy predicted from the *digital* scores of every
    class plus Gaussian noise (sigma_hat from one calibration chip). No device simulation of
    the test set is involved. The scores are the same dot products the crossbar computes,
    s_c = h . m_c, and the noise on class c is sigma * ||m_c|| * sqrt(D), because sigma_hat is
    measured in per-class cosine units. (Using cosines here would be wrong whenever the
    quantised prototypes have unequal norms and the decisions are close.)"""
    rng = np.random.default_rng(seed)
    scale = (Mq.norm(dim=1) * np.sqrt(H.shape[1])).numpy()
    correct = 0.0
    for i in range(0, len(H), bs):
        S = (H[i:i + bs].float() @ Mq.T).numpy()
        yb = y[i:i + bs]
        for _ in range(n_draws):
            correct += ((S + rng.normal(0.0, 1.0, S.shape) * (sigma * scale)).argmax(1) == yb).sum()
    return float(correct / (n_draws * len(H)))


@torch.no_grad()
def similarity_noise_sigma_times(Mq: torch.Tensor, H_cal: torch.Tensor, rpu_config, times,
                                 in_scale: float = 1.0, seed: int = 12345) -> list[float]:
    """sigma_hat at every time of the grid, from one calibration chip programmed once."""
    torch.manual_seed(seed)
    layer = build_crossbar(Mq, rpu_config)
    layer.program_analog_weights()
    ref = H_cal.float() @ Mq.T
    norm = Mq.norm(dim=1) * np.sqrt(H_cal.shape[1])
    out = []
    for t in times:
        layer.drift_analog_weights(t)
        A = layer(H_cal.float() * in_scale) / in_scale
        out.append(((A - ref) / norm).std().item())
    return out


@torch.no_grad()
def similarity_noise_sigma(Mq: torch.Tensor, H_cal: torch.Tensor, rpu_config, t: float,
                           in_scale: float = 1.0, seed: int = 12345) -> float:
    """Std of the analog-minus-digital cosine similarity on calibration (training) queries,
    from one independently programmed calibration chip."""
    torch.manual_seed(seed)
    layer = build_crossbar(Mq, rpu_config)
    layer.program_analog_weights()
    layer.drift_analog_weights(t)
    A = layer(H_cal.float() * in_scale) / in_scale
    ref = H_cal.float() @ Mq.T
    norm = Mq.norm(dim=1) * np.sqrt(H_cal.shape[1])
    return ((A - ref) / norm).std().item()


def quantize(M: torch.Tensor, precision: str, scale: str = "mse") -> torch.Tensor:
    """Per-class L2 normalisation, then symmetric uniform quantisation to `precision`
    ('fp', 'binary', or 'Nbit', N >= 2) with a per-class step size.

    scale='mse'    : step chosen per class to minimise the quantisation MSE (grid over
                     clipping ratios 2%..100% of max|w|); values beyond the clip saturate.
    scale='maxabs' : step = max|w| / levels (no clipping; wastes levels when a few
                     entries are large, which is what made 2 bit worse than 1 bit)."""
    M = M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if precision == "fp":
        return M
    if precision == "binary":
        return torch.where(M >= 0, 1.0, -1.0)
    bits = int(precision.removesuffix("bit"))
    levels = 2 ** (bits - 1) - 1  # e.g. 4 bit -> integers in [-7, 7]
    amax = M.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    if scale == "maxabs":
        st = amax / levels
        return torch.round(M / st) * st
    ratios = torch.linspace(0.02, 1.0, 50)
    best_err = torch.full((M.shape[0], 1), float("inf"))
    best_st = amax / levels
    for r in ratios:
        st = r * amax / levels
        q = torch.clamp(torch.round(M / st), -levels, levels) * st
        err = ((q - M) ** 2).sum(dim=1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_st = torch.where(better, st, best_st)
    return torch.clamp(torch.round(M / best_st), -levels, levels) * best_st


@torch.no_grad()
def accuracy_digital(M: torch.Tensor, H: torch.Tensor, y: np.ndarray, bs: int = 8192) -> float:
    pred = torch.cat([(H[i:i + bs].float() @ M.T).argmax(1) for i in range(0, len(H), bs)])
    return (pred.numpy() == y).mean().item()


# ----------------------------------------------------------------------------- crossbar
DEVICES = ["pcm", "reram_wan", "reram_cmo"]
# LEGACY grid of the v1 driver (sweep_hdc.py) only. Every result in the paper uses the
# 5/3-point grid TIMES of sweep_hdc2.py (1 s, 1 d, 2 d, 30 d, 1 y; NeuRRAM 1 s, 1 d, 2 d).
# Times after programming (s). The NeuRRAM ReRAM model only has published coefficients
# at 1 s, 1 day and 2 days, so it is never extrapolated; 1 s / 1 day / 2 days are shared
# by all devices for like-for-like comparison. CMO drift is a log(t) fit (extrapolated
# beyond its characterisation window at 30 days / 1 year; reported as such).
T_GRID = {
    "pcm": [1.0, 3600.0, 86400.0, 172800.0, 2592000.0, 31536000.0],
    "reram_cmo": [1.0, 3600.0, 86400.0, 172800.0, 2592000.0, 31536000.0],
    "reram_wan": [1.0, 86400.0, 172800.0],
}


def _safe_cmo_model(**kw):
    """AIHWKit's CMO/HfOx model takes log10 of the drifted conductance before clamping it,
    so when scaled-up noise pushes a device below 0 S the read-noise term becomes NaN and
    poisons the whole column. Clamp to the physical floor g_min first (identical to the
    original whenever the conductance stays positive, i.e. at nominal noise)."""
    from math import log, sqrt

    from aihwkit.inference import ReRamCMONoiseModel
    from torch import randn_like

    class SafeReRamCMONoiseModel(ReRamCMONoiseModel):
        @torch.no_grad()
        def apply_drift_noise_to_conductance(self, g_prog, drift_noise_param, t_inference):
            if t_inference == 0:
                return g_prog
            g_mean = g_prog + (self.decay_dict["mean"][0] * log(t_inference) * self.drift_scale)
            sig_rel = self.decay_dict["std"][0] * log(t_inference) + self.decay_dict["std"][1]
            g_drift = g_mean + randn_like(g_prog) * sig_rel * self.drift_scale
            g_drift = g_drift.clamp(min=self.g_min, max=self.g_max)
            sig_read = (self.read_dict["K"] * torch.log10(g_drift) *
                        sqrt(log((t_inference + self.read_dict["t_read"]) /
                                 (2 * self.read_dict["t_read"]))))
            g_final = g_drift + sig_read * randn_like(g_prog) * self.read_noise_scale
            return g_final.clamp(min=self.g_min, max=self.g_max)

    return SafeReRamCMONoiseModel(**kw)


def make_inference_config(device: str, periphery: str = "realistic", drift_comp: bool = True,
                          noise_scale: float = 1.0):
    from aihwkit.inference import (GlobalDriftCompensation, PCMLikeNoiseModel,
                                   ReRamCMONoiseModel, ReRamWan2022NoiseModel)
    from aihwkit.simulator.configs import InferenceRPUConfig

    cfg = InferenceRPUConfig()
    if device == "pcm":
        cfg.noise_model = PCMLikeNoiseModel(g_max=25.0, prog_noise_scale=noise_scale,
                                            read_noise_scale=noise_scale, drift_scale=noise_scale)
    elif device == "reram_wan":
        cfg.noise_model = ReRamWan2022NoiseModel(noise_scale=noise_scale)
    elif device == "reram_cmo":
        cfg.noise_model = _safe_cmo_model(prog_noise_scale=noise_scale,
                                          read_noise_scale=noise_scale, drift_scale=noise_scale)
    else:
        raise ValueError(device)
    from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType

    # each tile's weights span the full conductance range (max |w| -> g_max), so every
    # precision gets the same signal current per device; outputs are rescaled digitally
    cfg.mapping.weight_scaling_omega = 1.0
    cfg.mapping.weight_scaling_columnwise = False
    if periphery == "ideal":       # isolate the device: no DAC/ADC quantisation, no IO noise
        cfg.forward.is_perfect = True
    elif periphery == "realistic":
        # single-shot read with a fixed, pre-calibrated ADC range (see calibrate_input_scale):
        # 7-bit DAC, 9-bit ADC, output noise as in AIHWKit defaults; no dynamic input
        # re-normalisation and no iterative re-reads, which a real macro would not do per query
        cfg.forward.noise_management = NoiseManagementType.NONE
        cfg.forward.bound_management = BoundManagementType.NONE
    else:
        raise ValueError(periphery)
    cfg.drift_compensation = GlobalDriftCompensation() if drift_comp else None
    return cfg


@torch.no_grad()
def calibrate_input_scale(Mq: torch.Tensor, H_cal: torch.Tensor, out_bound: float = 12.0,
                          max_input_size: int = 512, pct: float = 99.9) -> float:
    """Read-voltage scale s so that the `pct` percentile of |tile partial sums| on
    calibration queries lands at the ADC full scale. Done once per programmed model on
    training data (never on test data), like an ADC range calibration on a real chip."""
    vals = []
    for r0 in range(0, Mq.shape[1], max_input_size):
        W = Mq[:, r0:r0 + max_input_size]
        W = W / W.abs().max().clamp_min(1e-12)
        vals.append((H_cal[:, r0:r0 + max_input_size].float() @ W.T).abs().flatten())
    p = torch.quantile(torch.cat(vals)[:2_000_000], pct / 100.0).item()
    return min(1.0, out_bound / max(p, 1e-12))


def build_crossbar(Mq: torch.Tensor, rpu_config):
    from aihwkit.nn import AnalogLinear
    C, D = Mq.shape
    layer = AnalogLinear(D, C, bias=False, rpu_config=rpu_config)
    layer.set_weights(Mq.clone(), None)
    layer.eval()
    return layer


@torch.no_grad()
def accuracy_crossbar(Mq, H, y, rpu_config, times, n_chips: int = 10, seed: int = 0,
                      bs: int = 2048, in_scale: float = 1.0, ref_t: float | None = None,
                      return_preds: bool = False):
    """acc[chip, t]: each chip is an independent programming of the same prototypes.
    Queries are applied as +/- in_scale (read-voltage scaling from the ADC calibration).
    With return_preds, also returns predictions[chip, sample] at time ref_t."""
    acc = np.zeros((n_chips, len(times)))
    preds_ref = np.zeros((n_chips, len(H)), dtype=np.int16) if return_preds else None
    for c in range(n_chips):
        torch.manual_seed(seed * 1000 + c)
        layer = build_crossbar(Mq, rpu_config)
        layer.program_analog_weights()
        for k, t in enumerate(times):
            layer.drift_analog_weights(t)
            outs = [layer(H[i:i + bs].float() * in_scale) for i in range(0, len(H), bs)]
            if any(not torch.isfinite(o).all() for o in outs):
                raise FloatingPointError(f"non-finite crossbar output at t={t} (chip {c})")
            pred = torch.cat([o.argmax(1) for o in outs])
            acc[c, k] = (pred.numpy() == y).mean()
            if return_preds and t == ref_t:
                preds_ref[c] = pred.numpy()
    return (acc, preds_ref) if return_preds else acc


def encoder_input_scale(enc: Encoder, X_cal: np.ndarray) -> float:
    """ADC range calibration of the encoder crossbar on training inputs."""
    return calibrate_input_scale(enc.P.T.contiguous(), enc.pre(X_cal))


@torch.no_grad()
def accuracy_end_to_end(enc: Encoder, Mq, X, y, rpu_config, times, n_chips: int = 10,
                        seed: int = 0, in_scale_am: float = 1.0, in_scale_enc: float = 1.0,
                        bs: int = 2048):
    """End-to-end analog HDC. Per chip, the projection matrix P and the prototypes are
    programmed into two crossbars of the same technology; both drift together. The encoder
    crossbar computes z = pre(x) P in analog (features as DAC voltages), the digital
    periphery applies the encoder non-linearity / sign, and the bipolar query drives the
    prototype crossbar. Returns acc[chip, t] and the query bit-flip rate[chip, t] relative
    to digital encoding."""
    if not hasattr(enc, "P"):
        raise ValueError("end-to-end analog encoding needs a projection encoder (rp/nonlinear)")
    Xp = enc.pre(X)
    H_dig = enc(X)
    acc = np.zeros((n_chips, len(times)))
    flips = np.zeros((n_chips, len(times)))
    for c in range(n_chips):
        torch.manual_seed(seed * 1000 + 500 + c)
        el = build_crossbar(enc.P.T.contiguous(), rpu_config)
        el.program_analog_weights()
        torch.manual_seed(seed * 1000 + c)
        am = build_crossbar(Mq, rpu_config)
        am.program_analog_weights()
        for k, t in enumerate(times):
            el.drift_analog_weights(t)
            am.drift_analog_weights(t)
            preds, nflip = [], 0
            for i in range(0, len(Xp), bs):
                z = el(Xp[i:i + bs] * in_scale_enc)
                h = enc.post(z)
                nflip += (h != H_dig[i:i + bs]).sum().item()
                o = am(h.float() * in_scale_am)
                if not (torch.isfinite(z).all() and torch.isfinite(o).all()):
                    raise FloatingPointError(f"non-finite output at t={t} (chip {c})")
                preds.append(o.argmax(1))
            acc[c, k] = (torch.cat(preds).numpy() == y).mean()
            flips[c, k] = nflip / H_dig.numel()
    return acc, flips


@dataclass
class HDCModel:
    encoder: Encoder
    prototypes: torch.Tensor  # float, unnormalised
    n_classes: int
