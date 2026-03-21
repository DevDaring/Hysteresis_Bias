import json
d = json.load(open('/root/Hysteresis_Bias/results/phase0_baseline/baseline_results.json'))
for m in d:
    for l in d[m]:
        r = d[m][l]
        cats = r.get('categories', {})
        cat_summary = {k: v['n_samples'] for k, v in cats.items()}
        total = sum(cat_summary.values())
        print(f"{m}/{l}: score={r['overall_bias_score']:.4f} metric={r['metric']} total_samples={total} cats={cat_summary}")
