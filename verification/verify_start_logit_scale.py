from pathlib import Path
import json
import sys
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / "QWOP_rl_training_public"
sys.path.insert(0, str(ROOT / "scripts"))
from train_ppo import ActorCritic, PPOConfig, start_logit_scale_for_update

torch.set_num_threads(1)
torch.manual_seed(418)
path = ROOT / "checkpoints" / "ppo_classic100m_under45.pt"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
state = checkpoint["model"]
linear = sorted((int(k.split(".")[1]), v.shape[0]) for k, v in state.items() if k.startswith("net.") and k.endswith(".weight"))
hidden = tuple(size for _, size in linear)
obs_dim = state["net.0.weight"].shape[1]
model = ActorCritic(obs_dim, state["actor.weight"].shape[0], hidden)
incompatible = model.load_state_dict(state, strict=True)
assert not incompatible.missing_keys and not incompatible.unexpected_keys
assert PPOConfig().start_logit_scale == 1.0
assert PPOConfig().start_logit_scale_decay_updates == 0

scores = torch.tensor([-1., 0., 1., 2.999, 3., 3.001, 10., 50.])
obs = torch.randn(len(scores), obs_dim)
if model.observation_scale is not None:
    obs *= model.observation_scale
frames = obs.reshape(len(scores), -1, 68)
for index in range(frames.shape[1]):
    frames[:, index, 61] = 2.0 - (frames.shape[1] - 1 - index) / 30.0
frames[:, -1, 60] = scores
logits, values = model(obs)
legacy = torch.distributions.Categorical(logits=logits)
default, new_values = model.action_distribution(obs)
torch.testing.assert_close(default.probs, legacy.probs, rtol=0, atol=0)
torch.testing.assert_close(default.entropy(), legacy.entropy(), rtol=0, atol=0)
torch.testing.assert_close(new_values, values, rtol=0, atol=0)
torch.manual_seed(71)
legacy_actions = legacy.sample()
torch.manual_seed(71)
actions, oldlp, entropy, sampled_values = model.act(obs)
assert torch.equal(actions, legacy_actions)
torch.testing.assert_close(oldlp, legacy.log_prob(legacy_actions), rtol=0, atol=0)
evaluated_lp, evaluated_entropy, evaluated_values = model.evaluate(obs, actions)
torch.testing.assert_close(evaluated_lp, oldlp, rtol=0, atol=0)
torch.testing.assert_close(evaluated_entropy, entropy, rtol=0, atol=0)
torch.testing.assert_close(evaluated_values, sampled_values, rtol=0, atol=0)

early = scores < 3.0
for scale in [.3, .65, .9, 1.0]:
    dist, scaled_values = model.action_distribution(obs, scale, 3.0)
    expected = torch.distributions.Categorical(logits=logits * torch.where(early, scale, 1.0).unsqueeze(-1))
    torch.testing.assert_close(dist.probs, expected.probs, rtol=0, atol=0)
    assert torch.equal(dist.probs.argmax(-1), legacy.probs.argmax(-1))
    torch.testing.assert_close(dist.probs[~early], legacy.probs[~early], rtol=0, atol=0)
    torch.testing.assert_close(scaled_values, values, rtol=0, atol=0)
    assert (dist.entropy()[early] >= legacy.entropy()[early] - 1e-6).all()
    acts, collected_lp, _, _ = model.act(obs, scale, 3.0)
    updated_lp, _, _ = model.evaluate(obs, acts, scale, 3.0)
    ratio = (updated_lp - collected_lp).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=0)

acts, collected_lp, _, _ = model.act(obs, .65, 3.0)
lp, ent, val = model.evaluate(obs, acts, .65, 3.0)
advantages = torch.linspace(-1, 1, len(scores))
loss = -((lp-collected_lp.detach()).exp() * advantages).mean() -.01*ent.mean() + .05*(val-values.detach()-1).square().mean()
loss.backward()
assert torch.isfinite(loss)
for name, parameter in model.named_parameters():
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all(), name

schedule = PPOConfig(start_logit_scale=.65, start_logit_scale_decay_updates=100)
assert start_logit_scale_for_update(schedule, 1) == .65
assert abs(start_logit_scale_for_update(schedule, 51) - .825) < 1e-12
assert start_logit_scale_for_update(schedule, 101) == 1.0
assert start_logit_scale_for_update(schedule, 10001) == 1.0
assert start_logit_scale_for_update(schedule, 0) == .65
constant = PPOConfig(start_logit_scale=.7, start_logit_scale_decay_updates=0)
assert start_logit_scale_for_update(constant, 1) == .7
assert start_logit_scale_for_update(constant, 10001) == .7
for update in [0, 1, 50, 101, 10001]:
    assert start_logit_scale_for_update(PPOConfig(), update) == 1.0

report = {
    "status": "passed", "checkpoint": str(path), "strict_checkpoint_load": True,
    "obs_dim": obs_dim, "hidden_sizes": hidden,
    "default_legacy_distribution_exact": True,
    "default_seeded_sampling_exact": True,
    "positive_scale_argmax_preserved": True,
    "outside_3m_distribution_exact": True,
    "same_scale_ppo_ratio_exactly_one": True,
    "all_gradients_finite": True,
    "schedule_start_mid_end": [start_logit_scale_for_update(schedule, i) for i in [1, 51, 101]],
    "loss": float(loss.detach()),
    "early_entropy_original": legacy.entropy()[early].detach().tolist(),
    "early_entropy_scale_065": model.action_distribution(obs, .65, 3.0)[0].entropy()[early].detach().tolist(),
}
output = BASE / "start_logit_scale_verification.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
