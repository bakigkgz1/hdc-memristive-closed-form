"""Fig. 1 (vector): simulated analog HDC inference and the accuracy model (v1 and v2)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figures as F

fig = plt.figure(figsize=(7.2, 3.55))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
BLUE, ORNG = "#eaf2fb", "#fbefe6"

def panel(y0, h, fc, title):
    ax.add_patch(FancyBboxPatch((0.6, y0), 98.8, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec="#c9ced6", lw=0.8))
    ax.text(1.8, y0 + h - 2.2, title, fontsize=8.2, weight="bold", va="center")

def box(x, y, w, h, lines, fc="white", bold0=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec="#5b6470", lw=0.7))
    n = len(lines); step = h / (n + 1)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - step * (i + 1), l, ha="center", va="center", fontsize=6.4,
                weight="bold" if (i == 0 and bold0) else "normal")

def arr(x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#5b6470", **kw))

panel(26.5, 23, BLUE, "(a) Simulated analog HDC inference (AIHWKit / CrossSim, 8 or 64 chips)")
y, h = 33.5, 11
box(1.8, y, 11, h, ["input $x$", "(centred)"])
box(14.8, y, 14, h, ["encoder", r"$h=\mathrm{sgn}((x-\mu)P)$", r"$\in\{\pm1\}^D$"])
box(30.8, y, 18.5, h, ["512-row tiles", r"DAC: $\pm s_{\mathrm{eff}}$", r"$\tilde w$: programming, drift,", "read noise (per chip)"], fc="#d9e8f8")
box(51.3, y, 14.5, h, ["ADC per tile", r"clip $\pm B$, round $q$", "+ output noise"], fc="#d9e8f8")
box(67.8, y, 17.5, h, ["digital sum", r"$\sum_T A_T u_T/(s\,\alpha_T G_T)$", "(drift compensation)"])
box(87.3, y, 10.9, h, ["argmax", r"$\to$ class"])
for a, b in [(12.8, 14.8), (28.8, 30.8), (49.3, 51.3), (65.8, 67.8), (85.3, 87.3)]:
    arr(a, y + h / 2, b, y + h / 2)
ax.plot([30.8, 30.8, 65.8, 65.8], [31.8, 30.9, 30.9, 31.8], color="#5b6470", lw=0.7)
ax.text(48.3, 29.0, r"prototypes $C_c$ (FP32 / $k$-bit / binary), trained with or without margin", ha="center", fontsize=6.2)
ax.text(73.5, 31.4, r"($G_T\neq1$ only for NeuRRAM)", ha="center", fontsize=5.4, color=F.INK2)

panel(0.6, 24.5, ORNG, "(b) Accuracy model from single-device statistics (no chip simulated)")
y2, h2 = 6.5, 12.5
box(1.8, y2, 17, h2, ["device characterisation", "(once per technology", "and read time)", r"$m_t(w),\,v_t(w),\,m_0(w),\,v_0(w)$"], fc="#f8dcc8")
box(21.0, y2, 18.5, h2, ["per tile, Gaussian", r"mean $s\,h\cdot m_t(w)$", r"var. $s^2\sum v_t+\sigma_{\mathrm{out}}^2$", r"read gain $G_T$", "(extreme value)"], fc="#f8dcc8")
box(42.2, 13.4, 20.5, 8.3, ["v1: closed form", "exact Gaussian integral, Eq. (5)", "no draws"], fc="white")
box(42.2, 3.0, 20.5, 8.8, ["v2: semi-analytic", r"Gaussian draws per tile $\to$", r"clip $\pm B$, round $q$"], fc="white")
box(65.0, y2, 33.2, h2, ["prediction", "mean accuracy from digital test queries;", "chip-to-chip spread by drawing device", "errors once per surrogate chip"])
arr(18.8, y2 + h2 / 2, 21.0, y2 + h2 / 2)
arr(39.5, y2 + h2 / 2, 42.2, 17.5); arr(39.5, y2 + h2 / 2, 42.2, 7.4)
arr(62.7, 17.5, 65.0, y2 + h2 / 2); arr(62.7, 7.4, 65.0, y2 + h2 / 2)
ax.add_patch(FancyArrowPatch((81.6, 19.0), (81.6, 33.5), arrowstyle="<|-|>", mutation_scale=7, lw=0.8, ls="--", color=F.C1))
ax.text(82.6, 26.2, "validated against", fontsize=5.8, color=F.C1, va="center")
ax.text(82.6, 24.4, "8- and 64-chip means", fontsize=5.8, color=F.C1, va="center")
F.save(fig, "fig1_schematic_v2")
