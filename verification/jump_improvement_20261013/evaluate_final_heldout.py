"""Predeclared heldout action-sampling evaluation; no model selection or edits."""
from pathlib import Path
from datetime import datetime,timezone
import json,time
import numpy as np
from experiment import stochastic,sha,write_json,ROOT,BASE

checkpoint=ROOT/'checkpoints/long_jump_distance_elite_best_20261013.pt'
checkpoint_hash=sha(checkpoint)
seeds=[2026101321,2026101322,2026101323]
started=datetime.now(timezone.utc);timer=time.perf_counter();reports=[]
for seed in seeds:
    sub_timer=time.perf_counter()
    report=stochastic(checkpoint,64,seed)
    report.update(checkpoint_sha256=checkpoint_hash,device='cpu',elapsed_seconds=time.perf_counter()-sub_timer,
        scope='Original native long_jump_30m scoring; 64 original identical initial states; first episode only; action sampling varies; seed was reserved before model selection.')
    write_json(BASE/f'final_heldout64_seed{seed%100}.json',report)
    reports.append(report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False),flush=True)
assert sha(checkpoint)==checkpoint_hash
results=[dict(result,seed=report['seed']) for report in reports for result in report['results']]
successful=np.array([r['jump_m'] for r in results if r['success']],dtype=np.float64)
all_scores=np.array([r['jump_m'] if r['success'] else 0. for r in results],dtype=np.float64)
assert len(results)==192 and np.isfinite(all_scores).all()
bootstrap_seed=2026101324;samples=20000;rng=np.random.default_rng(bootstrap_seed)
bootstrap_means=all_scores[rng.integers(0,len(all_scores),(samples,len(all_scores)))].mean(axis=1)
ci=np.quantile(bootstrap_means,[.025,.975])
baseline=json.loads((BASE/'baseline_heldout192.json').read_text(encoding='utf-8'))
assert baseline['seeds']==seeds and baseline['episodes']==192
baseline_scores=np.array([r['jump_m'] if r['success'] else 0. for r in baseline['results']],dtype=np.float64)
baseline_bootstrap=baseline_scores[rng.integers(0,len(baseline_scores),(samples,len(baseline_scores)))].mean(axis=1)
delta_ci=np.quantile(bootstrap_means-baseline_bootstrap,[.025,.975])
summary=dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,device='cpu',seeds=seeds,
    episodes=len(results),successes=len(successful),success_rate=len(successful)/len(results),
    successful_mean_m=float(successful.mean()) if len(successful) else None,
    median_successful_m=float(np.median(successful)) if len(successful) else None,
    success_weighted_mean_m=float(all_scores.mean()),best_successful_m=float(successful.max()) if len(successful) else None,
    over3m=int((successful>=3).sum()),over3m_fraction_all_episodes=float((successful>=3).sum()/len(results)),
    overall_mean_bootstrap_95_ci_m=ci.tolist(),bootstrap_seed=bootstrap_seed,bootstrap_samples=samples,
    bootstrap_method='Pooled episode-level percentile bootstrap; failures assigned 0m',
    baseline_comparison=dict(checkpoint=baseline['checkpoint'],checkpoint_sha256=baseline['checkpoint_sha256'],
        baseline_successes=baseline['successes'],baseline_success_rate=baseline['success_rate'],
        baseline_successful_mean_m=baseline['successful_mean_m'],baseline_over3m=baseline['over3m'],
        baseline_success_weighted_mean_m=float(baseline_scores.mean()),
        overall_mean_difference_m=float(all_scores.mean()-baseline_scores.mean()),
        overall_mean_difference_bootstrap_95_ci_m=delta_ci.tolist(),
        difference_bootstrap_method='Independent pooled episode resampling in the two model groups; final mean minus baseline mean; does not assume paired randomness'),
    per_seed=[{k:v for k,v in report.items() if k!='results'} for report in reports],results=results,
    started_utc=started.isoformat(),ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-timer,
    fixed_selected_checkpoint_unchanged=True,model_reselected_after_heldout=False,
    scope='Each environment contributes only its first episode. Identical original initial state and original native rules. These reserved seeds were not used for selection; no changes or reselection after evaluation. Action randomness only, not varied-initial-state robustness.')
write_json(BASE/'final_heldout192.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ('results','per_seed')},ensure_ascii=False),flush=True)
