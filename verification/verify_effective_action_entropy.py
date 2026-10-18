from pathlib import Path
import json
import math
import sys
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / "QWOP_rl_training_public"
sys.path.insert(0, str(ROOT / "scripts"))
from train_ppo import ActorCritic, PPOConfig

torch.set_num_threads(1)
torch.manual_seed(683)
GROUPS = ((0,), (1, 3), (2,), (4, 12), (5, 7, 13, 15), (6, 14), (8,), (9, 11), (10,))

def entropy(logits):
    return ActorCritic.action_entropy(torch.distributions.Categorical(logits=logits), True)

def reference(logits):
    aggregate = torch.stack([torch.logsumexp(logits[..., list(group)], -1) for group in GROUPS], -1)
    return torch.distributions.Categorical(logits=aggregate).entropy()

assert PPOConfig().effective_action_entropy is False
random_logits = torch.randn(3, 7, 16, dtype=torch.float64, requires_grad=True)
actual = entropy(random_logits)
expected = reference(random_logits)
torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
actual_gradient, = torch.autograd.grad(actual.sum(), random_logits, retain_graph=True)
expected_gradient, = torch.autograd.grad(expected.sum(), random_logits)
torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12)
assert torch.isfinite(actual_gradient).all()
assert abs(float(entropy(torch.zeros(16))) - 3 * math.log(2)) < 1e-6
balanced = torch.tensor([-math.log(next(len(group) for group in GROUPS if i in group)) for i in range(16)])
assert abs(float(entropy(balanced)) - math.log(9)) < 1e-6
p = torch.ones(16) / 16
q = p.clone()
q[1], q[3] = .12, .005
torch.testing.assert_close(entropy(p.log()), entropy(q.log()), rtol=0, atol=1e-6)
assert abs(float(torch.distributions.Categorical(probs=p).entropy() - torch.distributions.Categorical(probs=q).entropy())) > .01
extreme = torch.tensor([1000., -1000.] * 8, requires_grad=True)
h = entropy(extreme)
h.backward()
assert torch.isfinite(h) and torch.isfinite(extreme.grad).all()
try:
    ActorCritic.action_entropy(torch.distributions.Categorical(logits=torch.zeros(9)), True)
    raise AssertionError("non-16-action policy must be rejected")
except ValueError:
    pass

path = ROOT / "checkpoints" / "ppo_classic100m_under45.pt"
state = torch.load(path, map_location="cpu", weights_only=False)["model"]
hidden = tuple(size for _, size in sorted((int(k.split(".")[1]), value.shape[0]) for k, value in state.items() if k.startswith("net.") and k.endswith(".weight")))
model = ActorCritic(state["net.0.weight"].shape[1], state["actor.weight"].shape[0], hidden)
result = model.load_state_dict(state, strict=True)
assert not result.missing_keys and not result.unexpected_keys
obs = torch.randn(8, model.net[0].in_features)
if model.observation_scale is not None:
    obs *= model.observation_scale
frames = obs.reshape(8, -1, 68)
for i in range(frames.shape[1]):
    frames[:, i, 61] = 2.0 - (frames.shape[1] - 1 - i) / 30
frames[:, -1, 60] = torch.tensor([0., 1., 2.9, 2.999, 3., 3.01, 10., 50.])

for scale in [1., .65]:
    torch.manual_seed(971)
    default_action, default_lp, default_entropy, default_value = model.act(obs, scale)
    torch.manual_seed(971)
    action, lp, effective_entropy, value = model.act(obs, scale, 3., True)
    assert torch.equal(default_action, action)
    torch.testing.assert_close(default_lp, lp, rtol=0, atol=0)
    torch.testing.assert_close(default_value, value, rtol=0, atol=0)
    new_lp, new_entropy, new_value = model.evaluate(obs, action, scale, 3., True)
    torch.testing.assert_close(new_lp, lp, rtol=0, atol=0)
    torch.testing.assert_close(new_value, value, rtol=0, atol=0)
    torch.testing.assert_close(new_entropy, effective_entropy, rtol=0, atol=0)
    torch.testing.assert_close((new_lp - lp).exp(), torch.ones_like(lp), rtol=0, atol=0)
    dist, _ = model.action_distribution(obs, scale)
    torch.testing.assert_close(effective_entropy, reference(dist.logits), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(default_entropy, dist.entropy(), rtol=0, atol=0)
    assert (effective_entropy <= default_entropy + 2e-6).all()

action, lp, _, old_value = model.act(obs, .65, 3., True)
new_lp, effective_entropy, value = model.evaluate(obs, action, .65, 3., True)
loss = -((new_lp-lp.detach()).exp() * torch.linspace(-1, 1, 8)).mean() - .01*effective_entropy.mean() + .05*(value-old_value.detach()-1).square().mean()
loss.backward()
assert torch.isfinite(loss)
for name, parameter in model.named_parameters():
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name

report = {"status": "passed", "strict_legacy_checkpoint_load": True,
    "default_sampling_logp_value_exact": True, "enabled_sampling_logp_value_unchanged": True,
    "same_policy_ratio_exactly_one": True, "actual_grouping_matches_explicit_reference": True,
    "actual_gradient_matches_explicit_reference": True, "finite_gradients_with_extreme_logits": True,
    "within_group_redistribution_invariant": True, "start_logit_scale_combination_correct": True,
    "uniform_16_masks_effective_entropy": float(entropy(torch.zeros(16))),
    "uniform_9_groups_max_entropy": float(entropy(balanced)),
    "loss": float(loss.detach()), "groups": GROUPS}
(BASE / "effective_action_entropy_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
