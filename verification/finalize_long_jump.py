"""Select a native-successful policy and validate the author's replay exports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / 'QWOP_rl_training_public'
sys.path.insert(0, str(ROOT / 'scripts'))
from running_env import RunningGameEnv
from replay_io import read_replay, ReplayRuleset
from evaluate_long_jump import evaluate

torch.set_num_threads(1)
comparison = json.loads((BASE / 'long_jump_v1_checkpoint_comparison.json').read_text(encoding='utf-8'))
assert comparison['evaluated_checkpoints'] == 32
best = comparison['best']; assert best['success'] and best['exited_sand']
source = Path(best['checkpoint'])
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
checkpoint = torch.load(source, map_location='cpu', weights_only=False)
assert all(torch.isfinite(tensor).all() for tensor in checkpoint['model'].values())
output = ROOT / 'checkpoints/ppo_long_jump_best.pt'
checkpoint['config']['checkpoint'] = str(output)
checkpoint['long_jump_selection'] = dict(source=str(source), native_jump_m=best['long_jump_m'],
    source_update=best['update'], selection='Best native-successful deterministic score among 32 saved policies')
torch.save(checkpoint, output)
reloaded = evaluate(output)
assert reloaded['success'] and reloaded['actions'] == best['actions']
assert reloaded['long_jump_m'] == best['long_jump_m'] and reloaded['time_s'] == best['time_s']
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
python = str(ROOT / '.venv/Scripts/python.exe')
det_path = ROOT / 'replays/long_jump_ppo_deterministic.bin'
sample_path = ROOT / 'replays/long_jump_ppo_sampled_best.bin'
commands = [
    ('long_jump_deterministic_export.log', ['scripts/export_policy_replay.py', '--checkpoint', str(output),
        '--output', str(det_path), '--device', 'cpu']),
    ('long_jump_sampled_export.log', ['scripts/search_policy_best.py', '--checkpoint', str(output),
        '--output', str(sample_path), '--episodes', '1024', '--num-envs', '64', '--num-threads', '8',
        '--device', 'cpu', '--seed', '20261010', '--progress-interval', '256']),
    ('long_jump_best_stochastic64.log', [str(BASE / 'evaluate_long_jump_stochastic.py'), '--checkpoint', str(output),
        '--output', str(BASE / 'long_jump_best_stochastic64.json')]),
]
for name, command in commands:
    with (BASE / name).open('w', encoding='utf-8') as log:
        subprocess.run([python, '-u', *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name + '=completed', flush=True)

def validate_replay(path):
    ruleset, actions = read_replay(path)
    assert ruleset == ReplayRuleset.LONG_JUMP_30M and len(actions) <= 1350
    game = RunningGameEnv('long_jump_30m', 'long_jump_30m'); game.reset()
    for index, action in enumerate(actions, 1):
        obs, reward, done, truncated, info = game.step(action)
        assert torch.isfinite(torch.as_tensor(obs)).all() and math.isfinite(reward)
        assert bool(done or truncated) == (index == len(actions))
    assert info['success'] and not truncated
    assert abs(info['time'] - len(actions) / 30) < .0001
    assert abs(info['distance'] - (30 + info['score'])) < .0001
    return dict(path=str(path), ruleset=int(ruleset), actions=len(actions), jump_m=float(info['score']),
        time_s=float(info['time']), native_success=True, crc_verified=True)
det = validate_replay(det_path); sampled = validate_replay(sample_path)
assert det['jump_m'] == best['long_jump_m']
assert read_replay(det_path)[1] == best['actions']
rows = [line for line in (BASE / 'long_jump_v1_training.log').read_text(encoding='utf-8').splitlines() if line.startswith('update=')]
assert len(rows) == 1024
for line in rows:
    fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
    for key in ['policy_loss', 'value_loss', 'entropy', 'kl', 'clipfrac']:
        assert math.isfinite(float(fields[key]))
report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), task='long_jump_30m',
    training=json.loads((BASE / 'long_jump_v1_training_time.json').read_text(encoding='utf-8')),
    updates=1024, new_transitions=33554432, evaluated_checkpoints=32,
    successful_checkpoint_evaluations=comparison['successful_checkpoints'],
    all_logged_losses_finite=True, model=str(output), model_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    selected_source=str(source), selected_update=best['update'], strict_reload_actions_identical=True,
    deterministic=det, sampled_best=sampled,
    stochastic64=json.loads((BASE / 'long_jump_best_stochastic64.json').read_text(encoding='utf-8')),
    sample_scope='Sampled replay is the best of 1024 author-script episodes; it is not deterministic policy performance.')
(BASE / 'long_jump_v1_final_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='stochastic64'}, ensure_ascii=False, indent=2), flush=True)
