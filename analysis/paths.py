"""Where the result files live. In the released repository set HDC_RESULTS to the folder
holding the v2_*.jsonl files exactly as the Colab runs wrote them (default layout below
is the author's working copy). Intermediate tables go to HDC_WORK."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = os.environ.get("HDC_RESULTS")
WORK = Path(os.environ.get("HDC_WORK", HERE.parent / "analysis2"))
WORK.mkdir(parents=True, exist_ok=True)

_LOCAL = {  # author's working copy
    "core": "/mnt/user-data/uploads/v2_core.jsonl", "stress": "/mnt/user-data/uploads/v2_stress.jsonl",
    "precision": "prec.jsonl", "encoders": "enc.jsonl", "e2e": "e2e.jsonl", "e2e_bipolar": "e2e_bip.jsonl",
    "core_pred": "core_pred.jsonl", "stress_pred": "stress_pred.jsonl",
    "precision_pred": "prec_pred.jsonl", "encoders_pred": "enc_pred.jsonl",
    "core_analytic": "core_analytic.jsonl", "stress_analytic": "stress_analytic.jsonl",
    "precision_analytic": "prec_analytic.jsonl", "encoders_analytic": "enc_analytic.jsonl",
    "core_analytic2": "core_analytic2.jsonl", "stress_analytic2": "stress_analytic2.jsonl",
    "precision_analytic2": "precision_analytic2.jsonl", "encoders_analytic2": "encoders_analytic2.jsonl",
    "crosssim": "crosssim.jsonl",
    "core_ablation": "core_ablation.jsonl", "stress_ablation": "stress_ablation.jsonl",
    "encoders_ablation": "encoders_ablation.jsonl",
    "langid": "langid.jsonl", "langid_pred": "langid_pred.jsonl",
    "langid_analytic": "langid_analytic.jsonl", "langid_analytic2": "langid_analytic2.jsonl",
    "crosssim_parasitic": "crosssim_parasitic.jsonl",
}


def R(name: str) -> Path:
    """Path of a result file by logical name (e.g. 'core', 'stress_pred', 'crosssim')."""
    if RESULTS:
        return Path(RESULTS) / f"v2_{name}.jsonl"
    p = Path(_LOCAL[name])
    return p if p.is_absolute() else HERE.parent / "analysis2" / p
