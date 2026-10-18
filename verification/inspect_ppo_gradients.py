"""Compare actor and critic gradients on an actual rollout, without updating weights."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "QWOP_rl_training_public"
sys.path.insert(0, str(ROOT / "scripts"))
from running_env import VectorRunningGameEnv
from train_ppo import ActorCritic, FrameStacker, PPOConfig, compute_gae

torch.manual_seed(345)
data = torch.load(ROOT / "checkpoints/target45_hidden2_gen3.pt", map_location="cuda")
cfg = PPOConfig(num_envs=128, rollout_steps=256, gamma=.995, gae_lambda=.98)
env = VectorRunningGameEnv(128, num_threads=16, reward_profile="classic_100m", game_mode="classic_100m")
raw, _ = env.reset()
stacker = FrameStacker(128, 68, 32)
model = ActorCritic(stacker.obs_dim, 16, (256, 256)).cuda()
model.load_state_dict(data["model"])
obs = torch.as_tensor(stacker.reset(raw), device="cuda")
observations, actions, rewards, dones, values = [], [], [], [], []
for step in range(256):
    with torch.no_grad():
        action, _, _, value = model.act(obs)
    result = env.step_arrays(action.cpu().numpy())
    done = result["terminated"] | result["truncated"]
    observations.append(obs)
    actions.append(action)
    rewards.append(torch.as_tensor(result["reward"], device="cuda"))
    dones.append(torch.as_tensor(done, dtype=torch.float32, device="cuda"))
    values.append(value)
    obs = torch.as_tensor(stacker.update(result["obs"], done), device="cuda")
with torch.no_grad():
    _, next_value = model(obs)
    advantage, returns = compute_gae(torch.stack(rewards), torch.stack(dones), torch.stack(values), next_value, cfg)
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
indices = torch.randperm(32768, device="cuda")[:4096]
batch = torch.stack(observations).reshape(-1, stacker.obs_dim)[indices]
logprob, entropy, predicted = model.evaluate(batch, torch.stack(actions).flatten()[indices])
policy_loss = -(logprob * advantage.flatten()[indices]).mean() - .06 * entropy.mean()
value_loss = .5 * ((predicted - returns.flatten()[indices]) ** 2).mean()
parameters = list(model.net.parameters())
policy_grad = torch.autograd.grad(policy_loss, parameters, retain_graph=True)
value_grad = torch.autograd.grad(value_loss, parameters)
policy_norm = torch.stack([g.square().sum() for g in policy_grad]).sum().sqrt().item()
value_norm = torch.stack([g.square().sum() for g in value_grad]).sum().sqrt().item()
dot = sum((a * b).sum() for a, b in zip(policy_grad, value_grad)).item()
report = {"source": "target45_hidden2_gen3.pt", "reward": "v23", "samples": 32768,
          "minibatch_samples": 4096, "policy_shared_gradient_norm": policy_norm,
          "unweighted_value_shared_gradient_norm": value_norm,
          "value_to_policy_gradient_ratio_at_vf_005": .05 * value_norm / policy_norm,
          "value_to_policy_gradient_ratio_at_vf_001": .01 * value_norm / policy_norm,
          "gradient_cosine": dot / (policy_norm * value_norm), "value_loss": value_loss.item()}
assert np.isfinite(list(v for v in report.values() if isinstance(v, (int, float)))).all()
(ROOT.parent / "verification/target45_shared_gradient_diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report), flush=True)
