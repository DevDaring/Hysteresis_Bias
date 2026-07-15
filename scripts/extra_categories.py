"""
Part C4 — extend the matched/mismatched grid to more bias categories (scaffold).

The datasets also cover gender and caste. This runs the same protocol on those
categories where sample counts allow, to fill the per-category table with more
rows. Reuses the WP1 harness; fill the C4 stub rows in the per-category table of
Submission/EJAI_Hysteresis.tex from results/wp1_symmetric/summary.json after the
run (the driver appends per-condition JSON under the same tree).

    python scripts/extra_categories.py            # gender, caste (matched)
    python scripts/extra_categories.py mismatched # same, mismatched objective
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.wp1_symmetric_control import run_grid, ENCODERS

EXTRA_CATS = ["gender", "caste"]


def main(objective="matched"):
    run_grid(models=ENCODERS, languages=["en", "hi", "bn"], categories=EXTRA_CATS,
             seeds=[42, 123, 456], theta=0.7, max_inject=500, max_remove=1000,
             eval_every=25, objective=objective)


if __name__ == "__main__":
    obj = sys.argv[1] if len(sys.argv) > 1 else "matched"
    print(f"NOTE: extends grid to {EXTRA_CATS} under {obj}; run on GPU. "
          "Categories below the minimum sample count are skipped automatically.")
    main(obj)
