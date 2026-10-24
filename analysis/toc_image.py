"""Table-of-contents image, 55 mm x 50 mm."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import figures as F
from paths import WORK
mm = 1 / 25.4
fig = plt.figure(figsize=(55 * mm, 50 * mm))
a1 = fig.add_axes([0.13, 0.56, 0.36, 0.34])
w = np.linspace(-1, 1, 100)
a1.plot(w, 0.92 * w, color=F.C1, lw=1.2)
a1.fill_between(w, 0.92 * w - (0.08 + 0.06 * abs(w)), 0.92 * w + (0.08 + 0.06 * abs(w)), color=F.C1, alpha=0.25, lw=0)
a1.set_xticks([]); a1.set_yticks([])
a1.set_title("one device:\nmean, variance", fontsize=6, pad=2)
a1.set_xlabel("programmed w", fontsize=5.5, labelpad=1)
ax = fig.add_axes([0, 0, 1, 1], zorder=-1); ax.axis("off"); ax.set_xlim(0, 55); ax.set_ylim(0, 50)
ax.add_patch(FancyArrowPatch((28.5, 37), (33.5, 37), arrowstyle="-|>", mutation_scale=7, color=F.INK2, lw=0.8))
rng = np.random.default_rng(1)
for i in range(5):
    for j in range(5):
        ax.add_patch(Circle((37 + 3.2 * j, 30 + 3.2 * i), 0.9, fc=plt.cm.Blues(0.35 + 0.55 * rng.random()), ec="none"))
ax.text(43.4, 47.0, "crossbar", ha="center", fontsize=6)
import figures_late as L
q = L.v1v2("core").sample(6000, random_state=0)
a2 = fig.add_axes([0.16, 0.12, 0.34, 0.30])
a2.scatter(q.sim, q.v2, s=0.3, color=F.C1, rasterized=True)
a2.plot([10, 100], [10, 100], "--", color=F.INK2, lw=0.6)
a2.set_xlim(20, 100); a2.set_ylim(20, 100); a2.set_xticks([]); a2.set_yticks([])
a2.set_xlabel("simulated acc.", fontsize=5.5, labelpad=1); a2.set_ylabel("predicted", fontsize=5.5, labelpad=1)
ax.text(40.5, 15.5, "analog HDC\naccuracy\nwithout\nsimulation", ha="center", va="center", fontsize=6.2, weight="bold", color=F.INK)
ax.text(40.5, 4.5, "mean error 0.08 pp", ha="center", fontsize=6, color=F.C1, weight="bold")
fig.savefig(F.FIG / "toc_image.png", dpi=600); fig.savefig(F.FIG / "toc_image.pdf")
