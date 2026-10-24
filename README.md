# Closed-form accuracy prediction for hyperdimensional classifiers on memristive crossbars

Code and data for the manuscript *"Predicting the analog accuracy of hyperdimensional
classifiers on memristive crossbars from single-device statistics"* (B. Gökgöz, Gümüşhane
University).

HDC prototypes are programmed into simulated analog crossbars (IBM AIHWKit 1.1.0: PCM,
NeuRRAM/Wan-2022 ReRAM, CMO/HfOx ReRAM; Sandia CrossSim 3.2.1: PCM-Joshi, ReRAM-Wan,
ReRAM-Milo, SONOS). A closed-form model predicts the analog accuracy from single-device
statistics only (mean and variance of the read weight vs the programmed weight, plus the
DAC/ADC rules of the periphery), and is validated against eight-chip Monte-Carlo simulation.

## Layout

| Folder | Content |
| --- | --- |
| `src/` | HDC library (`hdc.py`), closed form (`analytic.py`), CrossSim bridge, experiment drivers |
| `analysis/` | statistics, validation tables and figure scripts used in the manuscript |
| `tables/` | every table behind the numbers in the manuscript (CSV) |
| `results/` | every result file as written by the runs (`v2_<stage>*.jsonl.gz`) and run logs |
| `colab/` | the Colab cells used to run each stage (CPU, High-RAM runtime) |

## Stages and result files

| Stage | Driver | Result |
| --- | --- | --- |
| core: 7 datasets × RP/nonlinear × D 256–8192 × 5 seeds × 6 training rules × FP32/4-bit/binary × 3 devices | `sweep_hdc2.py core` | `v2_core.jsonl` |
| encoders (ID-level), end-to-end (analog encoder), bipolar end-to-end, stress (noise ×0.5–×8, centred or not), precision (incl. 2-bit, ideal vs realistic periphery) | `sweep_hdc2.py <stage>` | `v2_<stage>.jsonl` |
| calibration-chip estimates σ̂ and σ_Δ | `sweep_predict.py <stage>` | `v2_<stage>_pred.jsonl` |
| closed form v1 | `sweep_analytic.py <stages>` | `v2_<stage>_analytic.jsonl` |
| periphery-aware semi-analytic model v2 | `sweep_analytic2.py <stages>` | `v2_<stage>_analytic2.jsonl` |
| CrossSim cross-check | `sweep_crosssim.py` | `v2_crosssim.jsonl` |
| language recognition (trigram encoder, full pipeline) | `sweep_hdc2.py langid` + the three prediction drivers | `v2_langid*.jsonl` |
| v2 component ablation and run time | `sweep_ablation.py core,stress,encoders` | `v2_<stage>_ablation.jsonl` |
| CrossSim with interconnect parasitics (GPU) | `sweep_crosssim_parasitic.py` | `v2_crosssim_parasitic.jsonl` |
| closed form v1 by Gauss-Hermite quadrature (no draws) | `sweep_quadrature.py core,langid` | `v2_<stage>_quad.jsonl` |
| 64-chip reference, per-class baseline, chip-to-chip spread | `sweep_revision.py chips64` | `v3_chips64.jsonl` |
| OnlineHD with its own step size | `sweep_revision.py onlinehd` | `v3_onlinehd.jsonl` |
| chip-free and 8-chip calibration baselines | `sweep_revision.py baselines` | `v3_baselines.jsonl` |

Each record holds the configuration, the tuned hyperparameters, the digital accuracy and,
per read time, the per-chip analog accuracies (`acc_chips`) with their mean and standard
deviation. Prediction records hold the reproduced digital accuracy (`reproduced` flag)
and the predicted accuracy per read time.

## Reproducing

```bash
python -m venv env && . env/bin/activate      # Python 3.12 (aihwkit 1.1.0 has no 3.13 wheels)
pip install -r requirements.txt
cd src
python sweep_hdc2.py core --outdir ../results --data_home ../data      # datasets come from OpenML
python sweep_analytic2.py core --outdir ../results --data_home ../data
```

Analysis from the released results (no simulation needed):

```bash
cd results && gunzip -k *.gz && cd ../analysis
export HDC_RESULTS=../results HDC_WORK=../work
python validate.py && python stats.py && python stats_predictors.py && python iso_dim.py
python figures.py && python fig1_schematic.py && python figures_late.py && python ablation.py && python margin_law.py
```

Run times on a 12-core CPU: core ≈ 10 h, closed form per stage 1–5 h, CrossSim ≈ 5 h.
Training is deterministic for a given machine; across machines floating-point rounding can
change perceptron-type training on some records, which the prediction drivers detect and flag.

## Citation

See `CITATION.cff`. Licence: MIT (code) and CC BY 4.0 (data).
