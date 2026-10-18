from pathlib import Path
import json
from evaluate_long_jump import evaluate, ROOT, BASE

paths = sorted((ROOT / 'checkpoints').glob('ppo_long_jump_v1_update_*.pt'))
reports = []
for path in paths:
    output = BASE / (path.stem + '_jump_eval.json')
    report = json.loads(output.read_text(encoding='utf-8')) if output.exists() else None
    if report is None or report.get('telemetry_version') != 2:
        report = evaluate(path)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k:report[k] for k in ['update','success','long_jump_m','approach_progress_m','time_s']}, ensure_ascii=False), flush=True)
    reports.append(report)
def rank(report):
    return (report['success'], report['long_jump_m'] if report['long_jump_m'] is not None else -1,
            report['approach_progress_m'], -report['time_s'])
best = max(reports, key=rank) if reports else None
summary = dict(evaluated_checkpoints=len(reports), best=best,
    successful_checkpoints=sum(r['success'] for r in reports),
    scope='One native deterministic trajectory per saved policy; counts are not repeated-trial success rates.')
(BASE / 'long_jump_v1_checkpoint_comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='best'} | {'best':{k:v for k,v in best.items() if k!='actions'} if best else None}, ensure_ascii=False), flush=True)
