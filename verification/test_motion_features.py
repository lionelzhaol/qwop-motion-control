"""Compatibility, units, reset and trainability checks for derived observations."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "QWOP_rl_training_public"
sys.path.insert(0, str(ROOT / "scripts"))
from train_ppo import ActorCritic, load_model_compatible

torch.set_num_threads(1)
data = torch.load(ROOT / "checkpoints/target45_gain_gen7.pt", map_location="cpu")
legacy = ActorCritic(2176, 16, (256, 256))
legacy.load_state_dict(data["model"], strict=True)
derived = ActorCritic(2176, 16, (256, 256), derive_angular_velocity=True)
load_model_compatible(derived, data["model"])
obs = torch.randn(8, 2176)
obs[:, -7] = 10
obs[:, -75] = 10 - 1 / 30
for original, expanded in zip(legacy(obs), derived(obs)):
    torch.testing.assert_close(original, expanded, rtol=0, atol=0)
frames = torch.zeros(1, 32, 68)
frames[:, -1, 61] = 1 / 30
frames[:, -1, 2] = np.pi / 60
features = derived.motion_features(frames.flatten(1))
torch.testing.assert_close(features[0, 0], torch.tensor(np.pi / 20, dtype=torch.float32))
frames[:, -2, 2] = np.pi - .01
frames[:, -1, 2] = -np.pi + .01
torch.testing.assert_close(derived.motion_features(frames.flatten(1))[0, 0], torch.tensor(.06), atol=2e-6, rtol=0)
frames[:, -2, 61] = frames[:, -1, 61]
assert derived.motion_features(frames.flatten(1)).eq(0).all()
logits, values = derived(obs)
(logits.square().mean() + values.square().mean()).backward()
assert derived.angular_velocity.weight.grad.norm() > 0
with torch.no_grad():
    derived.angular_velocity.weight.add_(.01 * torch.randn_like(derived.angular_velocity.weight))
restored = ActorCritic(2176, 16, (256, 256))
restored.load_state_dict(derived.state_dict(), strict=True)
for original, loaded in zip(derived(obs), restored(obs)):
    torch.testing.assert_close(original, loaded, rtol=0, atol=0)
optimizer = torch.optim.Adam(restored.parameters())
assert any(p is restored.angular_velocity.weight for group in optimizer.param_groups for p in group["params"])
load_model_compatible(restored, data["model"])
for original, reset in zip(legacy(obs), restored(obs)):
    torch.testing.assert_close(original, reset, rtol=0, atol=0)
print("motion_feature_compatibility_units_reset_gradient_checks=passed", flush=True)
