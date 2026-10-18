"""One first episode per vector environment, with reproducible sampled actions."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / 'QWOP_rl_training_public'
sys.path.insert(0, str(ROOT / 'scripts'))
from running_env import VectorRunningGameEnv
from train_ppo import ActorCritic, FrameStacker

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--episodes', type=int, default=64)
parser.add_argument('--seed', type=int, default=20261010)
args = parser.parse_args()
torch.set_num_threads(1); torch.manual_seed(args.seed)
data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
cfg = data['config']
assert cfg['game_mode'] == cfg['reward_profile'] == 'long_jump_30m'
env = VectorRunningGameEnv(args.episodes, num_threads=8, reward_profile='long_jump_30m', game_mode='long_jump_30m')
raw, _ = env.reset(); stack = FrameStacker(args.episodes, 68, cfg['frame_stack']); obs = stack.reset(raw)
model = ActorCritic(stack.obs_dim, 16, tuple(cfg['hidden_sizes']))
model.load_state_dict(data['model'], strict=True); model.eval()
active = np.ones(args.episodes, dtype=bool); episodes = []
for step in range(1, 1351):
    with torch.no_grad():
        logits, value = model(torch.as_tensor(obs, dtype=torch.float32))
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()
        actions = torch.distributions.Categorical(logits=logits).sample().numpy().astype(np.int64)
    assert actions.shape == (args.episodes,) and actions.min() >= 0 and actions.max() < 16
    result = env.step_arrays(actions)
    assert np.isfinite(result['obs']).all() and np.isfinite(result['reward']).all()
    done = np.logical_or(result['terminated'], result['truncated'])
    for i in np.flatnonzero(active & result['has_final']):
        score = float(result['final_score'][i]); distance = float(result['final_distance'][i])
        landed = abs(distance - (30 + score)) < .0001
        episodes.append(dict(env_index=int(i), success=bool(result['final_success'][i]),
            truncated=bool(result['truncated'][i]), time_s=float(result['final_time'][i]),
            long_jump_m=score if landed else None, total_distance_m=distance))
        active[i] = False
    if not active.any():
        break
    obs = stack.update(result['obs'], done)
assert len(episodes) == args.episodes
successful = [r for r in episodes if r['success']]
scores = [r['long_jump_m'] for r in successful]
assert all(s is not None and s >= 0 for s in scores)
report = dict(checkpoint=str(args.checkpoint), seed=args.seed, episodes=args.episodes,
    successes=len(successful), successful_jump_mean_m=float(np.mean(scores)) if scores else None,
    successful_jump_median_m=float(np.median(scores)) if scores else None,
    successful_jump_best_m=max(scores) if scores else None,
    measurement_scope='Identical original initial state; sampled actions vary. Each environment contributes only its first episode.',
    results=sorted(episodes, key=lambda r:r['env_index']))
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='results'}, ensure_ascii=False))
