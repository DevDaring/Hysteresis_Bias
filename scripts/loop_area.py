"""
Part C2 — bias-field sweep and hysteresis loop area (scaffold; run on GPU).

A threshold-free measure that needs no step count. Define a control field
lambda in [0,1] that mixes the stereotypical and anti-stereotypical training
signal:  L(lambda) = lambda * L_stereo + (1 - lambda) * L_anti.
Sweep lambda up 0->1 then down 1->0 in steps, training to quasi-equilibrium at
each stage, and record the bias score B(lambda). If the down-sweep does not
retrace the up-sweep, the enclosed loop has non-zero area
    A = closed integral of B dlambda   (trapezoid rule),
which is a threshold-free order parameter for hysteresis. A ~ 0 means no loop.

This converts the paper from a rebuttal into a positive proposal for the correct
measure. Nothing here fabricates results; run it and fill the C2 stub figure in
Submission/EJAI_Hysteresis.tex from results/loop_area/loop_area.json.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from src.utils.config import get_all_model_configs, get_results_dir, load_training_config
from src.utils.seed import set_seed
from src.models.loader import load_model_with_lora
from src.data.prepare_bias_injection import load_injection_data
from src.evaluation.bias_calculator import evaluate_bias
from src.training.symmetric_control import signed_gap

ENCODERS = ["mbert", "xlm-roberta", "muril", "indicbert-v2", "jhu-clsp-mmbert"]
LAMBDAS_UP = [round(0.1 * i, 1) for i in range(0, 11)]      # 0.0 -> 1.0
LAMBDAS_DOWN = LAMBDAS_UP[::-1]                              # 1.0 -> 0.0
STEPS_PER_STAGE = 50


def sweep(model, tok, mtype, train, ev, device, lambdas, lr, bs, mgn, wd, seed):
    """Train STEPS_PER_STAGE per lambda; return list of (lambda, B)."""
    import random
    rng = random.Random(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=lr, weight_decay=wd)
    curve = []
    for lam in lambdas:
        for _ in range(STEPS_PER_STAGE):
            model.train()
            batch = rng.sample(train, min(bs, len(train)))
            gap = signed_gap(model, tok, mtype, batch, device)
            if gap is None:
                continue
            # field: lambda pushes toward stereo (open gap), (1-lambda) toward anti (close)
            loss = -(lam) * gap + (1.0 - lam) * gap
            loss.backward(); clip_grad_norm_(trainable, mgn); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            b = evaluate_bias(model, tok, mtype, ev, use_full_aul=False)
        curve.append((lam, float(b.get("overall_bias_score", 0.5))))
    return curve


def loop_area(up, down):
    """Closed-loop area via trapezoid over the combined up+down path."""
    xs = [p[0] for p in up] + [p[0] for p in down]
    ys = [p[1] for p in up] + [p[1] for p in down]
    return float(abs(np.trapz(ys, xs)))


def main(models=ENCODERS, languages=("en",), categories=("race-color", "socioeconomic"), seeds=(42,)):
    tcfg = load_training_config()
    cfgs = get_all_model_configs()
    out = {}
    for m in models:
        cfg = cfgs[m]; mtype = cfg["model_type"]
        for lang in languages:
            tr = [x for x in load_injection_data(lang, "train")]
            evd = [x for x in load_injection_data(lang, "eval")]
            for cat in categories:
                trc = [x for x in tr if x.get("bias_category") == cat]
                evc = [x for x in evd if x.get("bias_category") == cat]
                if len(trc) < 4 or len(evc) < 4:
                    continue
                for s in seeds:
                    set_seed(s)
                    model, tok = load_model_with_lora(m, cfg)
                    device = next(model.parameters()).device
                    up = sweep(model, tok, mtype, trc, evc, device, LAMBDAS_UP,
                               tcfg["learning_rate"], tcfg["batch_size"], tcfg["max_grad_norm"],
                               tcfg["weight_decay"], s)
                    down = sweep(model, tok, mtype, trc, evc, device, LAMBDAS_DOWN,
                                 tcfg["learning_rate"], tcfg["batch_size"], tcfg["max_grad_norm"],
                                 tcfg["weight_decay"], s + 1)
                    out[f"{m}/{lang}/{cat}/seed{s}"] = {
                        "up": up, "down": down, "loop_area": loop_area(up, down)}
                    del model
                    torch.cuda.empty_cache()
    p = get_results_dir("loop_area") / "loop_area.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("SAVED", p)


if __name__ == "__main__":
    print("NOTE: bias-field sweep; run on GPU only.")
    main()
