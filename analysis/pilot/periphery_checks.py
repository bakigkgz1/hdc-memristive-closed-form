"""Isolated AIHWKit 1.1.0 checks behind Section 4.3: DAC step, ADC step and bound,
and the NeuRRAM per-tile read gain = max |programmed weight|. Run from src/ (needs hdc.py)."""
import numpy as np, torch
from aihwkit.nn import AnalogLinear
from hdc import make_inference_config

cfg = make_inference_config("pcm", "realistic"); cfg.forward.out_noise = 0.0
L = AnalogLinear(100, 1, bias=False, rpu_config=cfg); L.set_weights(torch.ones(1, 100)); L.eval()
with torch.no_grad():
    for s in [0.0236, 0.03, 0.04, 0.05]:
        print(f"DAC: s={s} realised {L(torch.full((1, 100), s)).item() / 100:.5f}")
torch.manual_seed(0)
L = AnalogLinear(64, 1, bias=False, rpu_config=cfg); W = torch.rand(1, 64) * 2 - 1; W[0, 0] = 1
L.set_weights(W); L.eval()
with torch.no_grad():
    y = L(torch.sign(torch.randn(20000, 64))).squeeze().numpy()
u = np.unique(np.round(y, 5))
print(f"ADC: step {np.median(np.diff(u)):.5f} (2*12/510 = {24 / 510:.5f}), max |y| {abs(y).max()}")
for dev, t in [("reram_wan", 1.0), ("pcm", 86400.0), ("reram_cmo", 86400.0)]:
    nm = make_inference_config(dev, "ideal").noise_model
    for sd in range(3):
        torch.manual_seed(sd); W = torch.rand(10, 512) * 2 - 1; W[0, 0] = 1
        p, par = nm.apply_programming_noise(W); wt = nm.apply_drift_noise(p, par, t)
        print(f"{dev} seed {sd}: read gain {float((wt * W).sum() / (W * W).sum()):.4f}  max|prog| {float(p.abs().max()):.4f}")
