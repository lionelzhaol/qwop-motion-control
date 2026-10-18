"""Summarize predeclared, unused action-sampling seeds for the old policy."""
from pathlib import Path
import hashlib
import json
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1] / 'QWOP_rl_training_public'
seeds = [2026101321, 2026101322, 2026101323]
reports = [json.loads((BASE / f'baseline_heldout64_seed{seed%100}.json').read_text(encoding='utf-8')) for seed in seeds]
assert [report['seed'] for report in reports] == seeds
assert all(report['episodes'] == 64 for report in reports)
results = [dict(result, seed=report['seed']) for report in reports for result in report['results']]
assert len(results) == 192
successful = np.array([result['jump_m'] for result in results if result['success']], dtype=np.float64)
all_scores = np.array([result['jump_m'] if result['success'] else 0. for result in results], dtype=np.float64)
rng = np.random.default_rng(2026101324)
bootstrap_samples = 20000
bootstrap_means = all_scores[rng.integers(0, len(all_scores), (bootstrap_samples, len(all_scores)))].mean(axis=1)
ci = np.quantile(bootstrap_means, [.025, .975])
report = dict(checkpoint=str(ROOT / 'checkpoints/ppo_long_jump_shadow_best.pt'),
    checkpoint_sha256=hashlib.sha256((ROOT / 'checkpoints/ppo_long_jump_shadow_best.pt').read_bytes()).hexdigest(),
    device='cpu', seeds=seeds, episodes=len(results), successes=len(successful),
    success_rate=len(successful)/len(results), successful_mean_m=float(successful.mean()),
    median_successful_m=float(np.median(successful)), success_weighted_mean_m=float(all_scores.mean()),
    best_successful_m=float(successful.max()), over3m=int((successful>=3).sum()),
    overall_mean_bootstrap_95_ci_m=ci.tolist(),bootstrap_seed=2026101324,
    bootstrap_samples=bootstrap_samples,bootstrap_method='Pooled episode-level percentile bootstrap; failures assigned 0m',
    per_seed=[{key:value for key,value in r.items() if key!='results'} for r in reports],
    scope='Each environment contributes only its first episode. Identical original initial state and native original rules. '
          'Action sampling varies; these seeds were not used for model selection. Not a varied-initial-state robustness test.',
    results=results)
(BASE / 'baseline_heldout192.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({key:value for key,value in report.items() if key not in ('results','per_seed')},ensure_ascii=False))
