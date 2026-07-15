"""
Part C3 — full fine-tuning robustness check (scaffold; run on GPU).

The main study adapts each model with a rank-16 low-rank adapter. This script
repeats the matched-objective inject-then-remove measurement with ALL parameters
trainable, on one or two small encoders (mBERT, XLM-RoBERTa), to test whether the
direction (removal faster, R < 1) holds without the low-rank restriction.

It loads the base model without an adapter and marks every parameter trainable,
then reuses the signed-gap measurement loop from the matched objective. Fill the
C3 stub table in Submission/EJAI_Hysteresis.tex from
results/full_finetune/summary.json after the run. Nothing here fabricates results.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.utils.config import get_all_model_configs, get_results_dir, load_training_config
from src.utils.seed import set_seed
from src.models.loader import load_model            # base model, no adapter
from src.data.prepare_bias_injection import load_injection_data
from src.evaluation.bias_calculator import evaluate_bias
from src.training.symmetric_control import run_symmetric_control

MODELS = ["mbert", "xlm-roberta"]
CATS = ["race-color", "religion", "nationality", "socioeconomic"]


def main(models=MODELS, languages=("en", "hi", "bn"), seeds=(42, 123, 456)):
    tcfg = load_training_config()
    # full fine-tuning is unstable at the LoRA learning rate; use a smaller one
    tcfg = dict(tcfg); tcfg["learning_rate"] = 2e-5
    cfgs = get_all_model_configs()
    summary = []
    for m in models:
        cfg = cfgs[m]; mtype = cfg["model_type"]
        for lang in languages:
            tr = [x for x in load_injection_data(lang, "train")]
            evd = [x for x in load_injection_data(lang, "eval")]
            for cat in CATS:
                trc = [x for x in tr if x.get("bias_category") == cat]
                evc = [x for x in evd if x.get("bias_category") == cat]
                if len(trc) < 4 or len(evc) < 4:
                    continue
                for s in seeds:
                    set_seed(s)
                    model, tok = load_model(m, cfg)
                    for p in model.parameters():
                        p.requires_grad = True          # full fine-tuning
                    model.eval()
                    with torch.no_grad():
                        base = evaluate_bias(model, tok, mtype, evc, use_full_aul=False)
                    r = run_symmetric_control(model, tok, m, mtype, lang, cat, s, trc, evc,
                                              float(base.get("overall_bias_score", 0.5)), 0.7, tcfg,
                                              max_inject_steps=500, max_remove_steps=1000,
                                              eval_every=25, objective="matched")
                    summary.append(r)
                    del model
                    torch.cuda.empty_cache()
    p = get_results_dir("full_finetune") / "summary.json"
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    print("SAVED", p)


if __name__ == "__main__":
    print("NOTE: full fine-tuning is memory-heavy; run on GPU only.")
    main()
