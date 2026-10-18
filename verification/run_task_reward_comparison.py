"""Run the two configured PPO arms sequentially with separate logs and timings."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / 'QWOP_rl_training_public'
env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
results = []
for arm in ['v40_task_control', 'v41_task_rnd']:
    started = time.perf_counter()
    entry = dict(arm=arm, started_at=datetime.now(timezone.utc).isoformat())
    print(json.dumps(entry), flush=True)
    with (BASE / f'target43_{arm}.log').open('w', encoding='utf-8') as output:
        process = subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), '-u', 'scripts/train_ppo.py',
            '--config', f'configs/ppo_target43_{arm}.json'], cwd=ROOT, env=env,
            stdout=output, stderr=subprocess.STDOUT)
    entry.update(exit_code=process.returncode, elapsed_seconds=time.perf_counter() - started,
        finished_at=datetime.now(timezone.utc).isoformat())
    results.append(entry)
    (BASE / 'task_rnd_comparison_timing.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(entry), flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)
