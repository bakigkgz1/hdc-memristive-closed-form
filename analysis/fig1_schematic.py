import figures as F
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt = F.plt

def box(ax, x, y, w, h, text, fc="#ffffff", ec=F.INK2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.1, color=F.INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)

def arrow(ax, x0, y0, x1, y1, c=F.INK2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 color=c, lw=0.9, linestyle=ls))

fig, ax = plt.subplots(figsize=(7.0, 3.1))
ax.set_xlim(-0.02, 1.03); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 0.97, "(a) Simulated analog HDC inference (AIHWKit / CrossSim)", fontsize=7.5, color=F.INK, fontweight="bold")
y, h = 0.66, 0.2
box(ax, 0.00, y, 0.12, h, "input x\n(centred)")
box(ax, 0.15, y, 0.14, h, "encoder\nh = sign(xP)\n∈ {±1}^D")
box(ax, 0.32, y, 0.18, h, "DAC: ±s_eff\n512-row tiles\nw̃ = G_T·(w + ε)", fc="#eaf2fc")
box(ax, 0.53, y, 0.15, h, "ADC per tile\nclip ±B, round q\n+ output noise", fc="#eaf2fc")
box(ax, 0.71, y, 0.16, h, "digital sum\nΣ_T A_T u_T / (s α_T)\n(drift compensation)")
box(ax, 0.905, y, 0.10, h, "argmax\n→ class")
for x0, x1 in [(0.12, 0.15), (0.29, 0.32), (0.50, 0.53), (0.68, 0.71), (0.87, 0.905)]:
    arrow(ax, x0, y + h / 2, x1, y + h / 2)
ax.text(0.41, y - 0.05, "prototypes m_c (fp / k-bit / binary), margin-trained", ha="center", fontsize=6.2, color=F.INK2)

ax.text(0.0, 0.47, "(b) Closed-form prediction (no chip simulated)", fontsize=7.5, color=F.INK, fontweight="bold")
y = 0.12; h = 0.24
box(ax, 0.00, y, 0.21, h, "device characterisation\n(once per technology)\nm_t(w), v_t(w), v_0(w)", fc="#fdeee6")
box(ax, 0.245, y, 0.25, h, "per tile: mean s·h·m_t(w)\nvariance s²Σv_t + σ_out²\nread gain G_T (extreme value)", fc="#fdeee6")
box(ax, 0.525, y, 0.20, h, "Gaussian draws per tile\n→ clip ±B, round q\n(periphery v2)", fc="#fdeee6")
box(ax, 0.755, y, 0.25, h, "predicted accuracy\nP[argmax = true class]\nfrom digital test queries only", bold=False)
for x0, x1 in [(0.21, 0.245), (0.495, 0.525), (0.725, 0.755)]:
    arrow(ax, x0, y + h / 2, x1, y + h / 2)
arrow(ax, 0.79, 0.36, 0.79, 0.62, c=F.C1, ls="--")
ax.text(0.805, 0.49, "validated against\n8-chip simulation", fontsize=6.2, color=F.C1, va="center")
F.save(fig, "fig1_schematic")
