"""Create a task-specific checkpoint without modifying the running baseline."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / 'QWOP_rl_training_public'
sys.path.insert(0, str(ROOT / 'scripts'))
from train_ppo import ActorCritic

torch.set_num_threads(1)
source_path = ROOT / 'checkpoints/ppo_classic100m_phase_fast.pt'
source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
data = torch.load(source_path, map_location='cpu', weights_only=False)
cfg = data['config']
model = ActorCritic(data['model']['net.0.weight'].shape[1], 16, tuple(cfg['hidden_sizes']))
model.load_state_dict(data['model'], strict=True)
model.enable_separate_critic()
with torch.no_grad():
    model.actor.weight.mul_(.5)
    model.actor.bias.mul_(.5)
    # Running value targets are unrelated to jump distance. Start a task-specific
    # value head on copied features, using the existing independent-Critic path.
    model.critic.weight.zero_()
    model.critic.bias.zero_()
for key, tensor in data['model'].items():
    if not key.startswith(('actor.', 'critic.')):
        assert torch.equal(tensor, model.state_dict()[key]), key
output = ROOT / 'checkpoints/ppo_long_jump_transfer_initial.pt'
checkpoint = copy.deepcopy(data)
checkpoint['model'] = model.state_dict()
checkpoint.pop('optimizer', None)
checkpoint['update'] = 0
checkpoint['global_step'] = 0
checkpoint['config'] = dict(cfg, game_mode='long_jump_30m', reward_profile='long_jump_30m',
    checkpoint=str(output), derive_angular_velocity=True, normalize_observations=True,
    separate_critic_features=True, detach_critic_features=False, freeze_backbone=False,
    enable_flat_actor_bias=False, enable_flat_actor_residual=False, effective_action_entropy=False,
    rnd_beta=0., reset_optimizer_on_resume=True)
checkpoint['long_jump_transfer'] = dict(source=str(source_path), actor_temperature_scale=.5,
    source_preprocessing_preserved=True, critic_head_reset=True, no_jump_training_yet=True)
torch.save(checkpoint, output)
reloaded = ActorCritic(model.net[0].in_features, 16, tuple(cfg['hidden_sizes']))
reloaded.load_state_dict(torch.load(output, map_location='cpu', weights_only=False)['model'], strict=True)
assert all(torch.equal(t, reloaded.state_dict()[k]) for k,t in model.state_dict().items())
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
print(json.dumps(checkpoint['long_jump_transfer'], ensure_ascii=False))
