"""Supplementary Fig. S2: 64-chip check. (a) v2 and calibration-chip baselines vs the 64-chip mean,
(b) predicted vs simulated chip-to-chip standard deviation."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import figures as F
from paths import HERE
c = pd.read_pickle(HERE.parent / "analysis2" / "val_chips64.pkl")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
ax = axs[0]
labs = [("v2", "v2 (no chip)", F.C1), ("v1_quad", "v1 (no chip)", F.INK2),
        ("pred_pc_affine", "per-class affine (1 chip)", F.C3), ("pred_sdiff", "$\\sigma_\\Delta$ (1 chip)", F.C2)]
bins = np.linspace(-2, 2, 81)
for k, l, col in labs:
    e = np.clip(c[k] - c.sim64, -2, 2)
    ax.hist(e, bins=bins, histtype="step", lw=1.3, color=col, label=f"{l}: MAE {np.abs(c[k]-c.sim64).mean():.2f}")
ax.set_yscale("log"); ax.set_ylim(0.7, 5000); ax.set_xlabel("Prediction − 64-chip mean (pp, clipped to ±2)"); ax.set_ylabel("points")
ax.set_title("(a) Error against a 64-chip reference"); ax.legend(fontsize=6, frameon=False, loc="upper left")
ax = axs[1]
for dev, mk in zip(["pcm", "reram_wan", "reram_cmo"], ["o", "s", "^"]):
    s = c[c.device == dev]
    ax.scatter(s.sd64, s.sur2_sd, s=8, marker=mk, alpha=0.6, label=F.DEV_LABEL.get(dev, dev))
lim = [0.01, 10]
ax.plot(lim, lim, "--", color=F.INK2, lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Simulated chip-to-chip SD, 64 chips (pp)"); ax.set_ylabel("Predicted SD, 64 surrogate chips (pp)")
ax.set_title("(b) Chip-to-chip spread"); ax.legend(fontsize=6, frameon=False, loc="upper left")
fig.tight_layout()
F.save(fig, "figS2_chips64")
