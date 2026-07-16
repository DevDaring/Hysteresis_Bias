"""
C2 (robust) — bias-field sweep and hysteresis loop area.

The threshold-free measure that replaces step counts. A control field
lambda in [0, 1] mixes the two training pressures through one functional:

    L(lambda) = lambda * (-delta) + (1 - lambda) * (+delta) = (1 - 2*lambda) * delta

where delta is the signed stereo-minus-anti log-probability gap (the same
functional as the matched objective). lambda = 1 drives the gap open,
lambda = 0 drives it closed, lambda = 0.5 applies no net field. The sweep
raises lambda 0.0 -> 1.0 in steps of 0.1 (50 training steps per stage,
bias score B recorded after each stage), then lowers it 1.0 -> 0.0 from the
end state, with ONE optimiser carried across the whole sweep. If the down
path retraces the up path, there is no hysteresis; the enclosed signed area

    A = closed-path integral of B d(lambda)   (trapezoid rule)

is the order parameter. A > 0 means the down path stays above the up path
(state remembers the high-bias excursion). No threshold, no step count.

Grid (robust): 5 encoders x 3 languages x 3 categories x 3 seeds = 135.
Per-condition JSON with resume, so a preemption never loses finished loops.

    python scripts/loop_area.py               # full grid
    python scripts/loop_area.py --dry-run     # 2 models, coarse lambda, tiny

Outputs: results/loop_area/<model>/<lang>/<cat>/seed<N>.json and summary.json.
Analysis and figure: scripts/analyze_loop_area.py.

# Implements the C2 measure from Research_proposal.md (loop area A = closed
# integral of B over the bias-field sweep; threshold-free hysteresis test).
"""
import sys
import os
import json
import random
import argparse
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from src.utils.config import get_all_model_configs, get_results_dir, load_training_config
from src.utils.logging_setup import get_logger
from src.utils.seed import set_seed
from src.models.loader import load_model_with_lora
from src.data.prepare_bias_injection import load_injection_data
from src.evaluation.bias_calculator import evaluate_bias
from src.training.symmetric_control import signed_gap

logger = get_logger("loop_area")

ENCODERS = ["mbert", "xlm-roberta", "muril", "indicbert-v2", "jhu-clsp-mmbert"]
LANGS = ["en", "hi", "bn"]
CATS = ["race-color", "religion", "socioeconomic"]
SEEDS = [42, 123, 456]

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x renamed trapz


def lambda_schedule(step_size):
    n = int(round(1.0 / step_size))
    up = [round(i * step_size, 4) for i in range(n + 1)]          # 0.0 .. 1.0
    down = [round(1.0 - i * step_size, 4) for i in range(1, n + 1)]  # 0.9 .. 0.0
    return up, down


def signed_loop_area(up, down):
    """Closed-path integral of B d(lambda) over up then down (trapezoid).

    Positive when the down path lies above the up path (memory of the
    high-bias excursion). The path is closed by the vertical segment at
    lambda = 0, which contributes nothing to the integral.
    """
    xs = [p[0] for p in up] + [p[0] for p in down]
    ys = [p[1] for p in up] + [p[1] for p in down]
    return float(-_trapz(ys, xs))  # minus: down-leg has negative dx


def run_loop(model, tokenizer, model_type, train_data, eval_data, device,
             seed, tcfg, step_size, steps_per_stage):
    """One full up+down sweep with a single optimiser. Returns curves + area."""
    up_l, down_l = lambda_schedule(step_size)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=tcfg["learning_rate"],
                      weight_decay=tcfg["weight_decay"])
    rng = random.Random(seed)
    bs = tcfg["batch_size"]
    mgn = tcfg["max_grad_norm"]

    def stage(lam):
        for _ in range(steps_per_stage):
            model.train()
            batch = rng.sample(train_data, min(bs, len(train_data)))
            gap = signed_gap(model, tokenizer, model_type, batch, device)
            if gap is None:
                continue
            loss = (1.0 - 2.0 * lam) * gap
            loss.backward()
            clip_grad_norm_(trainable, max_norm=mgn)
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            b = evaluate_bias(model, tokenizer, model_type, eval_data, use_full_aul=False)
        return float(b.get("overall_bias_score", 0.5))

    up_curve = [(lam, stage(lam)) for lam in up_l]
    down_curve = [(lam, stage(lam)) for lam in down_l]
    return up_curve, down_curve, signed_loop_area(up_curve, down_curve)


def condition_path(root, model, language, category, seed):
    d = get_results_dir(root) / model / language / category
    d.mkdir(parents=True, exist_ok=True)
    return d / f"seed{seed}.json"


def run_grid(models, languages, categories, seeds, step_size=0.1,
             steps_per_stage=50, root="loop_area"):
    tcfg = load_training_config()
    cfgs = get_all_model_configs()
    summary = []

    for m in models:
        cfg = cfgs[m]
        mtype = cfg["model_type"]
        for lang in languages:
            train_all = load_injection_data(lang, split="train")
            eval_all = load_injection_data(lang, split="eval")
            for cat in categories:
                tr = [x for x in train_all if x.get("bias_category") == cat]
                ev = [x for x in eval_all if x.get("bias_category") == cat]
                if len(tr) < 4 or len(ev) < 4:
                    logger.warning(f"  skip {m}/{lang}/{cat}: too few samples")
                    continue
                for seed in seeds:
                    out_path = condition_path(root, m, lang, cat, seed)
                    if out_path.exists():
                        try:
                            summary.append(json.load(open(out_path, encoding="utf-8")))
                            logger.info(f"  ✓ resume-skip {m}/{lang}/{cat}/seed{seed}")
                            continue
                        except Exception:
                            out_path.unlink()  # corrupt (preemption mid-write)

                    set_seed(seed)
                    try:
                        model, tok = load_model_with_lora(m, cfg)
                        device = next(model.parameters()).device
                        model.eval()
                        with torch.no_grad():
                            base = evaluate_bias(model, tok, mtype, ev, use_full_aul=False)
                        up, down, area = run_loop(model, tok, mtype, tr, ev, device,
                                                  seed, tcfg, step_size, steps_per_stage)
                        rec = {
                            "model": m, "language": lang, "category": cat, "seed": seed,
                            "baseline_bias": float(base.get("overall_bias_score", 0.5)),
                            "lambda_step": step_size,
                            "steps_per_stage": steps_per_stage,
                            "up_curve": up, "down_curve": down,
                            "signed_area": area, "abs_area": abs(area),
                            "timestamp": datetime.now().isoformat(),
                        }
                        with open(out_path, "w") as f:
                            json.dump(rec, f, indent=2)
                        summary.append(rec)
                        logger.info(f"  ✓ {m}/{lang}/{cat}/seed{seed}: A={area:+.4f}")
                        del model
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logger.error(f"  ✗ {m}/{lang}/{cat}/seed{seed}: {e}")
                        logger.error(traceback.format_exc())
                        with open(out_path, "w") as f:
                            json.dump({"model": m, "language": lang, "category": cat,
                                       "seed": seed, "status": "FAIL",
                                       "error": f"{type(e).__name__}: {e}",
                                       "timestamp": datetime.now().isoformat()}, f, indent=2)
                        torch.cuda.empty_cache()

    summ_path = get_results_dir(root) / "summary.json"
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"loop-area summary ({len(summary)} conditions) saved to {summ_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="2 models, en, race-color, seed 42, coarse lambda, isolated root")
    args = ap.parse_args()

    if args.dry_run:
        logger.info("C2 DRY RUN: 2 models, coarse sweep (lambda step 0.25, 10 steps/stage)")
        run_grid(models=["mbert", "jhu-clsp-mmbert"], languages=["en"],
                 categories=["race-color"], seeds=[42],
                 step_size=0.25, steps_per_stage=10, root="loop_area_dryrun")
        logger.info("C2 dry run complete; inspect results/loop_area_dryrun")
        return

    # lambda step 0.2 -> 11 stages (6 up, 5 down). The bias evaluation after each
    # stage costs 9-17 s and dominates runtime, so halving the stage count roughly
    # halves the grid (~13 h -> the budgeted figure) while still giving enough
    # points to trace the loop and integrate its area.
    run_grid(models=ENCODERS, languages=LANGS, categories=CATS, seeds=SEEDS,
             step_size=0.2, steps_per_stage=50)


if __name__ == "__main__":
    main()
