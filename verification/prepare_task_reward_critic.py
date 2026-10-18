"""Calibrate only the value head against an unchanged successful actor."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / 'QWOP_rl_training_public'
sys.path.insert(0, str(ROOT / 'scripts'))
from train_ppo import ActorCritic, FrameStacker
from running_env import VectorRunningGameEnv

torch.set_num_threads(1)
started = time.perf_counter()
source_path = ROOT / 'checkpoints/ppo_classic100m_phase_fast.pt'
output_path = ROOT / 'checkpoints/ppo_task_reward_initial.pt'
source = torch.load(source_path, map_location='cpu', weights_only=False)
cfg = source['config']
state = source['model']
model = ActorCritic(state['net.0.weight'].shape[1], 16, tuple(cfg['hidden_sizes']))
model.load_state_dict(state, strict=True)
model.enable_separate_critic()
model.eval()
before = {k: v.clone() for k, v in model.state_dict().items()}
env = VectorRunningGameEnv(1, num_threads=1, reward_profile='classic_100m_task', game_mode='classic_100m')
raw, _ = env.reset()
initial_score = float(raw[0, 60])
stack = FrameStacker(1, 68, cfg['frame_stack'])
obs = stack.reset(raw)
features, observations, rewards, actions = [], [], [], []
for step in range(1, 4001):
    tensor = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        logits, _ = model(tensor)
        action = int(logits.argmax(-1)[0])
        features.append(model.critic_features(tensor)[0].numpy().copy())
    observations.append(obs[0].copy())
    actions.append(action)
    result = env.step_arrays(np.array([action], dtype=np.int64))
    rewards.append(float(result['reward'][0]))
    assert np.isfinite(result['obs']).all() and np.isfinite(result['reward']).all()
    done = np.logical_or(result['terminated'], result['truncated'])
    if done[0]:
        assert result['final_success'][0] and not result['truncated'][0]
        terminal = dict(time_s=float(result['final_time'][0]), distance_m=float(result['final_distance'][0]), steps=step)
        assert terminal['time_s'] == 44.93333435058594 and step == 1348
        # The backend starts at 0.3m; reset consumes that progress before the
        # first transition. Only subsequent new distance is returned to PPO.
        expected = 2 * (100 - initial_score) + 2800 - 21 * terminal['time_s']
        assert abs(sum(rewards) - expected) < .02, (sum(rewards), expected, rewards[-1], terminal, raw[0, 60:])
        reset_step = env.step_arrays(np.array([0], dtype=np.int64))
        assert reset_step['obs'][0, 61] < .04 and not reset_step['has_final'][0]
        break
    obs = stack.update(result['obs'], done)
else:
    raise RuntimeError('Baseline did not finish')

gamma = .9999
returns = np.zeros(len(rewards), dtype=np.float64)
acc = 0.
for i in range(len(rewards) - 1, -1, -1):
    acc = rewards[i] + gamma * acc
    returns[i] = acc
x = np.asarray(features, dtype=np.float64)
xmean, ymean = x.mean(0), returns.mean()
xc = x - xmean
weight = np.linalg.solve(xc.T @ xc + 10 * np.eye(x.shape[1]), xc.T @ (returns - ymean))
bias = ymean - xmean @ weight
with torch.no_grad():
    model.critic.weight.copy_(torch.tensor(weight[None], dtype=torch.float32))
    model.critic.bias.fill_(float(bias))
for name, tensor in model.state_dict().items():
    assert torch.isfinite(tensor).all()
    if not name.startswith('critic.'):
        assert torch.equal(tensor, before[name]), name
checkpoint = copy.deepcopy(source)
checkpoint['model'] = model.state_dict()
checkpoint.pop('optimizer', None)
checkpoint['config'] = dict(cfg, reward_profile='classic_100m_task', game_mode='classic_100m',
    separate_critic_features=True, reset_optimizer_on_resume=True, gamma=gamma, gae_lambda=.995)
report = dict(source=str(source_path), output=str(output_path), terminal=terminal,
    gamma=gamma, total_reward=sum(rewards), discounted_return=float(returns[0]),
    fit_rmse=float(np.sqrt(np.mean((x @ weight + bias - returns) ** 2))),
    actor_unchanged=True, no_ppo_training=True, calibrated_parameters=['critic.weight', 'critic.bias'])
checkpoint['task_reward_calibration'] = report
torch.save(checkpoint, output_path)
reloaded = ActorCritic(state['net.0.weight'].shape[1], 16, tuple(cfg['hidden_sizes']))
reloaded.load_state_dict(torch.load(output_path, weights_only=False)['model'], strict=True)
with torch.no_grad():
    actual, _ = reloaded(torch.tensor(np.asarray(observations), dtype=torch.float32))
assert actual.argmax(-1).tolist() == actions
report['strict_reload_actions_identical'] = True
report['sha256'] = hashlib.sha256(output_path.read_bytes()).hexdigest()
report['elapsed_seconds'] = time.perf_counter() - started
(BASE / 'task_reward_initialization.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
