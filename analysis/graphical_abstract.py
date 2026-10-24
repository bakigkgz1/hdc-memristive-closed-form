"""Graphical abstract (draft): measure one device -> predict the crossbar -> design edge AI."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import norm

import figures as F

INK, INK2, C1, C2, C3 = F.INK, F.INK2, F.C1, F.C2, F.C3
BG = "#f6f5f1"

fig = plt.figure(figsize=(13.3, 5.3))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 13.3)
ax.set_ylim(0, 5.3)
ax.axis("off")


def panel(x, title, sub):
    ax.add_patch(FancyBboxPatch((x, 0.35), 3.9, 4.25, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=BG, ec="#dcdad2", lw=1))
    ax.text(x + 1.95, 4.33, title, ha="center", va="center", fontsize=14, weight="bold", color=INK)
    ax.text(x + 1.95, 3.98, sub, ha="center", va="center", fontsize=9.5, color=INK2)


def arrow(x0, x1, y=2.45):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=22, lw=2, color=INK2))


panel(0.3, "1  Measure one device", "two curves per technology")
panel(4.7, "2  Compute the crossbar", "no chip is simulated")
panel(9.1, "3  Design edge AI", "accuracy before fabrication")
arrow(4.25, 4.65)
arrow(8.65, 9.05)

# --- panel 1: memristor + m(w), v(w)
cx, cy = 1.15, 2.3
ax.add_patch(Rectangle((cx - 0.38, cy + 0.55), 0.76, 0.16, fc=INK2, ec="none"))
ax.add_patch(Rectangle((cx - 0.38, cy - 0.71), 0.76, 0.16, fc=INK2, ec="none"))
ax.add_patch(Rectangle((cx - 0.25, cy - 0.55), 0.5, 1.1, fc="#d9e6f7", ec=C1, lw=1.5))
for k in range(5):
    ax.add_patch(Circle((cx - 0.12 + 0.06 * (k % 2), cy - 0.4 + 0.2 * k), 0.05, fc=C1, ec="none"))
ax.text(cx, cy - 1.0, "memristor\n(PCM, ReRAM, ...)", ha="center", va="top", fontsize=8.5, color=INK2)
a1 = fig.add_axes([0.155, 0.27, 0.13, 0.42])
w = np.linspace(-1, 1, 100)
a1.plot(w, 0.92 * w, color=C1, lw=2, label="mean m(w)")
a1.fill_between(w, 0.92 * w - (0.06 + 0.05 * np.abs(w)), 0.92 * w + (0.06 + 0.05 * np.abs(w)),
                color=C1, alpha=0.2, lw=0, label="variance v(w)")
a1.set_xlabel("programmed weight w", fontsize=8)
a1.set_ylabel("read weight", fontsize=8)
a1.tick_params(labelsize=6)
a1.legend(fontsize=7, frameon=False, loc="upper left")
a1.set_facecolor(BG)

# --- panel 2: crossbar tile + Gaussian scores + formula
x0, y0 = 5.0, 1.75
n = 7
for i in range(n):
    ax.plot([x0, x0 + 1.5], [y0 + i * 0.25, y0 + i * 0.25], color=INK2, lw=0.8)
    ax.plot([x0 + i * 0.25, x0 + i * 0.25], [y0, y0 + 1.5], color=INK2, lw=0.8)
rng = np.random.default_rng(3)
for i in range(n):
    for j in range(n):
        ax.add_patch(Circle((x0 + j * 0.25, y0 + i * 0.25), 0.055,
                            fc=plt.cm.Blues(0.3 + 0.6 * rng.random()), ec="none"))
ax.text(x0 + 0.75, y0 - 0.25, "512-row tile, DAC / ADC", ha="center", va="top", fontsize=8.5, color=INK2)
a2 = fig.add_axes([0.525, 0.33, 0.12, 0.3])
z = np.linspace(-4, 9, 300)
for mu, c, lab in [(0, C2, "competitor"), (4.5, C1, "true class")]:
    a2.plot(z, norm.pdf(z, mu, 1.1), color=c, lw=2)
    a2.fill_between(z, norm.pdf(z, mu, 1.1), color=c, alpha=0.15, lw=0)
    a2.text(mu, 0.395, lab, ha="center", fontsize=6.5, color=c)
a2.annotate("", xy=(4.5, 0.47), xytext=(0, 0.47), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
a2.text(2.25, 0.495, "margin", ha="center", fontsize=7, color=INK)
a2.set_ylim(0, 0.56)
a2.set_yticks([])
a2.set_xticks([])
a2.set_xlabel("class score", fontsize=8)
a2.set_facecolor(BG)
ax.text(6.65, 0.95, r"$P(\mathrm{correct})=\int\phi(z)\prod_{c\neq y}\Phi\!\left(\frac{S_y-S_c+\sqrt{V_y}z}{\sqrt{V_c}}\right)dz$",
        ha="center", fontsize=10.5, color=INK)
ax.text(6.65, 0.55, "mean error 0.18 pp (v1), 0.08 pp (v2)", ha="center", fontsize=9, color=C1, weight="bold")

# --- panel 3: generic edge device + language ID + margin law
px, py = 9.55, 1.15
ax.add_patch(FancyBboxPatch((px, py), 0.95, 1.75, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc="white", ec=INK, lw=1.6))
ax.add_patch(Rectangle((px + 0.1, py + 0.25), 0.75, 1.3, fc="#eef3fb", ec="none"))
ax.text(px + 0.475, py + 1.2, "“bonjour”", ha="center", fontsize=8, color=INK)
ax.text(px + 0.475, py + 0.85, "→ French", ha="center", fontsize=8, color=C1, weight="bold")
ax.add_patch(Rectangle((px + 0.25, py + 0.38), 0.45, 0.28, fc=C1, ec="none"))
ax.text(px + 0.475, py + 0.52, "IMC", ha="center", va="center", fontsize=6.5, color="white", weight="bold")
ax.text(px + 0.475, py - 0.12, "on-device language\nrecognition", ha="center", va="top", fontsize=8.5, color=INK2)
a3 = fig.add_axes([0.835, 0.27, 0.13, 0.42])
xx = np.geomspace(2, 40, 50)
a3.loglog(xx, 30 * xx ** -2.2, color=INK, lw=2)
a3.scatter([6.9, 9.1, 14.2], [30 * 6.9 ** -2.2, 30 * 9.1 ** -2.2, 30 * 14.2 ** -2.2],
           color=[C2, C2, C1], zorder=3, s=28)
a3.text(14.2 * 1.25, 30 * 14.2 ** -2.2, "margin rules", fontsize=7, color=C1, va="center")
a3.text(6.9 * 1.25, 30 * 6.9 ** -2.2 * 1.3, "no margin", fontsize=7, color=C2, va="bottom")
from matplotlib.ticker import NullFormatter, NullLocator
for axis in (a3.xaxis, a3.yaxis):
    axis.set_major_formatter(NullFormatter()); axis.set_minor_formatter(NullFormatter())
    axis.set_minor_locator(NullLocator())
a3.set_xlabel("margin / noise", fontsize=8)
a3.set_ylabel("analog loss", fontsize=8)
a3.tick_params(labelsize=6)
a3.set_facecolor(BG)
ax.text(11.05, 0.55, "order of magnitude of ReRAM and PCM chips", ha="center", fontsize=9, color=C1, weight="bold")

for ext in ("png", "pdf"):
    fig.savefig(F.FIG / f"graphical_abstract.{ext}", dpi=300, facecolor="white")
print("saved")
