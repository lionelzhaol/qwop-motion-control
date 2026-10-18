"""CPU-only compatibility, gradient and gate checks; no PPO updates."""
from pathlib import Path
import hashlib
import json
import sys
import types
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1] / 'QWOP_rl_training_public'
sys.path.insert(0, str(ROOT / 'scripts'))
from train_ppo import ActorCritic, PPOConfig, FrameStacker, load_model_compatible, train
from running_env import RunningGameEnv

torch.set_num_threads(1)
torch.manual_seed(20261013)
old = types.ModuleType('train_ppo_before_jump_state')
old.__file__ = str(ROOT / 'scripts/train_ppo.py')
sys.modules[old.__name__] = old
exec(compile((BASE / 'train_ppo_before_jump_state.py.txt').read_text(encoding='utf-8'), old.__file__, 'exec'), old.__dict__)
source_path = ROOT / 'checkpoints/ppo_long_jump_distance_initial_20261013.pt'
source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
source = torch.load(source_path, map_location='cpu', weights_only=False)
cfg = source['config']
data = np.load(BASE / 'self_imitation_v1.npz')
x = torch.as_tensor(data['observations'])
y = torch.as_tensor(data['actions'], dtype=torch.long)

def make(cls=ActorCritic):
    model = cls(x.shape[-1], 16, tuple(cfg['hidden_sizes']))
    model.load_state_dict(source['model'], strict=True)
    return model.eval()

before, default, extended = make(old.ActorCritic), make(), make()
assert default.jump_state_net is None
assert set(before.state_dict()) == set(default.state_dict())
with torch.no_grad():
    legacy_logits, legacy_values = before(x)
    assert torch.equal(default(x)[0], legacy_logits)
    assert torch.equal(default(x)[1], legacy_values)
extended.enable_jump_state_residual(128)
assert extended.jump_state_net[-1].weight.count_nonzero() == 0
assert extended.jump_state_net[-1].bias.count_nonzero() == 0
with torch.no_grad():
    zero_logits, zero_values = extended(x)
    assert torch.equal(zero_logits, legacy_logits)
    assert torch.equal(zero_values, legacy_values)
    assert all(torch.equal(value, extended.state_dict()[name]) for name, value in source['model'].items())

def trajectory(model):
    env = RunningGameEnv('long_jump_task', 'long_jump_30m')
    raw, _ = env.reset()
    stack = FrameStacker(1, 68, cfg['frame_stack'])
    obs = stack.reset(raw[None])
    logits_list, actions = [], []
    for _ in range(1350):
        with torch.no_grad(): logits = model(torch.as_tensor(obs))[0]
        logits_list.append(logits.detach().clone())
        action = int(logits.argmax(-1)[0]); actions.append(action)
        raw, _, done, truncated, info = env.step(action)
        if done or truncated: return torch.cat(logits_list), actions, info
        obs = stack.update(raw[None], np.array([False]))
    raise AssertionError('Native trajectory did not terminate')

old_track, old_actions, old_info = trajectory(before)
new_track, new_actions, new_info = trajectory(extended)
assert torch.equal(old_track, new_track)
assert old_actions == new_actions and old_info == new_info

mask = x[:, -6] <= extended.jump_sampling[1]
assert mask.any() and (~mask).any()
with torch.no_grad():
    extended.jump_state_net[-1].weight.normal_(std=0.1)
    extended.jump_state_net[-1].bias.normal_(std=0.1)
    changed, values = extended(x)
    assert torch.equal(changed[~mask], legacy_logits[~mask])
    assert not torch.equal(changed[mask], legacy_logits[mask])
    assert torch.equal(values, legacy_values)
    expected_input = torch.cat((x[:, -68:] / extended.observation_scale[-68:], extended.motion_features(x)), -1)
    expected_logits = legacy_logits + torch.where(mask[:, None], extended.jump_state_net(expected_input), 0.0) * extended.jump_sampling[0]
    assert torch.equal(changed, expected_logits)
extended.requires_grad_(False)
extended.jump_state_net.requires_grad_(True)
far_loss = F.cross_entropy(extended(x[~mask])[0], y[~mask]); far_loss.backward()
assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in extended.jump_state_net.parameters())
extended.zero_grad(set_to_none=True)
near_loss = F.cross_entropy(extended(x[mask])[0], y[mask]); near_loss.backward()
gradient_norms = {name: float(p.grad.norm()) for name, p in extended.named_parameters() if p.requires_grad}
assert all(np.isfinite(value) and value > 0 for value in gradient_norms.values())
assert all(p.grad is None for p in extended.parameters() if not p.requires_grad)

checkpoint = BASE / 'jump_state_validation.pt'
torch.save({'model': extended.state_dict(), 'config': dict(cfg, jump_residual_hidden_size=128)}, checkpoint)
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
reloaded = ActorCritic(x.shape[-1], 16, tuple(cfg['hidden_sizes']))
reloaded.load_state_dict(saved['model'], strict=True)
compatible = ActorCritic(x.shape[-1], 16, tuple(cfg['hidden_sizes']))
load_model_compatible(compatible, saved['model'])
assert reloaded.jump_state_net[0].out_features == compatible.jump_state_net[0].out_features == 128
assert all(torch.equal(value, reloaded.state_dict()[name]) for name, value in saved['model'].items())
assert all(torch.equal(value, compatible.state_dict()[name]) for name, value in saved['model'].items())
with torch.no_grad(): assert torch.equal(reloaded(x)[0], changed)

invalid_cases = 0
for hidden in (0, -1, 1.5, True):
    try: default.enable_jump_state_residual(hidden)
    except ValueError: invalid_cases += 1
    else: raise AssertionError(f'Invalid hidden size accepted: {hidden}')
for obs_dim in (68, 137):
    invalid_model = ActorCritic(obs_dim, 16)
    if obs_dim == 68: invalid_model.enable_jump_sampling([1., 50.])
    try: invalid_model.enable_jump_state_residual(128)
    except ValueError: invalid_cases += 1
    else: raise AssertionError('Invalid input accepted')
mutations = []
missing_sampling = dict(saved['model']); missing_sampling.pop('jump_sampling'); mutations.append(missing_sampling)
missing_bias = dict(saved['model']); missing_bias.pop('jump_state_net.2.bias'); mutations.append(missing_bias)
bad_input = dict(saved['model']); bad_input['jump_state_net.0.weight'] = torch.zeros(128, 79); mutations.append(bad_input)
bad_output = dict(saved['model']); bad_output['jump_state_net.2.weight'] = torch.zeros(15, 128); mutations.append(bad_output)
for mutation in mutations:
    for loader in ('strict', 'compatible'):
        test_model = ActorCritic(x.shape[-1], 16, tuple(cfg['hidden_sizes']))
        try:
            if loader == 'strict': test_model.load_state_dict(mutation, strict=True)
            else: load_model_compatible(test_model, mutation)
        except ValueError: invalid_cases += 1
        else: raise AssertionError(f'Corrupt state accepted by {loader}')

# Exercise the real setup/save path on CPU without collecting rollouts or updating.
config_args = {key: value for key, value in cfg.items() if key in PPOConfig.__dataclass_fields__}
config_args.update(num_envs=1, num_threads=1, updates=0, rollout_steps=2, minibatches=1,
                   hidden_sizes=tuple(cfg['hidden_sizes']), jump_residual_hidden_size=128,
                   reward_profile='long_jump_task', game_mode='long_jump_30m', gamma=.999,
                   separate_critic_features=True, resume_from=source_path,
                   reset_optimizer_on_resume=True, checkpoint=BASE / 'jump_state_setup.pt',
                   run_dir=BASE, run_name='jump_state_setup_logs')
with patch('torch.cuda.is_available', return_value=False): train(PPOConfig(**config_args))
setup = torch.load(BASE / 'jump_state_setup.pt', map_location='cpu', weights_only=False)
assert setup['config']['jump_residual_hidden_size'] == 128 and setup['config']['freeze_backbone']
assert all(t.device.type == 'cpu' for t in setup['model'].values())
assert set(name for name in setup['model'] if name.startswith('jump_state_net.')) == {
    'jump_state_net.0.weight', 'jump_state_net.0.bias', 'jump_state_net.2.weight', 'jump_state_net.2.bias'}
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash

report = dict(device='cpu',ppo_updates=0,bc_epochs=0,default_state_keys_unchanged=True,
              default_legacy_logits_and_values_exact=True,zero_branch_teacher_frames=len(x),
              zero_branch_native_trajectory_frames=len(old_actions),native_final_jump_m=old_info['score'],
              native_success=old_info['success'],runup_frames=int((~mask).sum()),jump_frames=int(mask.sum()),
              nonzero_branch_runup_logits_exact=True,critic_values_exact=True,
              latest_68_normalization_plus_12_motion_exact=True,gradient_norms=gradient_norms,
              zero_gradients_outside_gate=True,strict_and_compatible_reload_exact=True,
              invalid_config_or_checkpoint_cases_rejected=invalid_cases,
              actual_saved_config_hidden_size=setup['config']['jump_residual_hidden_size'],
              original_checkpoint_sha256_unchanged=True)
(BASE / 'jump_state_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
