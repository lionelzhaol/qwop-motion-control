from pathlib import Path
import copy
import hashlib
import json
import sys
import time
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / "QWOP_rl_training_public"
sys.path.insert(0, str(ROOT / "scripts"))
from running_env import VectorRunningGameEnv
from train_ppo import ActorCritic, FrameStacker

torch.set_num_threads(1)
SOURCE = ROOT / "checkpoints" / "ppo_phase_v32_hidden2_baseline.pt"
OUTPUT = ROOT / "checkpoints" / "ppo_phase_v34_credit_calibrated.pt"
GAMMA = .999
started = time.perf_counter()
original = torch.load(SOURCE, map_location="cpu", weights_only=False)
config = original["config"]

def load_model(state):
    model = ActorCritic(state["net.0.weight"].shape[1], state["actor.weight"].shape[0], tuple(config["hidden_sizes"]))
    incompatible = model.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    model.eval()
    return model

def rollout(model, collect=False):
    env = VectorRunningGameEnv(1, num_threads=1, reward_profile="classic_100m", game_mode="classic_100m")
    raw, _ = env.reset()
    stacker = FrameStacker(1, 68, int(config["frame_stack"]))
    obs = stacker.reset(raw)
    features, observations, rewards, actions, old_values = [], [], [], [], []
    for step in range(1, 4001):
        x = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, value = model(x)
            action = int(logits.argmax(-1)[0])
            if collect:
                features.append(model.features(x)[0].numpy().copy())
                observations.append(obs[0].copy())
                old_values.append(float(value[0]))
        result = env.step_arrays(np.array([action], dtype=np.int64))
        actions.append(action)
        reward = float(result["reward"][0])
        assert np.isfinite(reward) and np.isfinite(result["obs"]).all()
        rewards.append(reward)
        done = bool(result["terminated"][0] or result["truncated"][0])
        if done:
            assert result["has_final"][0] and result["final_success"][0] and not result["truncated"][0]
            return {"time_s": float(result["final_time"][0]), "distance_m": float(result["final_distance"][0]),
                "steps": step, "success": True, "actions": np.array(actions), "rewards": np.array(rewards),
                "features": np.array(features), "observations": np.array(observations), "old_values": np.array(old_values)}
        obs = stacker.update(result["obs"], np.array([False]))
    raise RuntimeError("Canonical actor failed to terminate")

source_model = load_model(original["model"])
trajectory = rollout(source_model, collect=True)
assert abs(trajectory["time_s"] - 44.93333435058594) < 1e-6
returns = np.empty_like(trajectory["rewards"], dtype=np.float64)
running = 0.0
for i in range(len(returns)-1, -1, -1):
    running = float(trajectory["rewards"][i]) + GAMMA * running
    returns[i] = running
features = trajectory["features"].astype(np.float64)
assert features.shape[1] == 256

def fit(x, y, ridge):
    xm, ym = x.mean(0), y.mean()
    centered = x-xm
    w = np.linalg.solve(centered.T@centered + ridge*np.eye(x.shape[1]), centered.T@(y-ym))
    b = ym-xm@w
    return w, b

def rmse(actual, target):
    return float(np.sqrt(np.mean((actual-target)**2)))

# Interleaved holdout samples cover the whole episode; this measures fitting
# stability on this trajectory, not generalization to unseen policies/states.
validation = np.arange(len(returns)) % 5 == 0
trials = []
for ridge in [.01, 1., 10.]:
    w, b = fit(features[~validation], returns[~validation], ridge)
    trials.append({"ridge": ridge, "train_rmse": rmse(features[~validation]@w+b, returns[~validation]),
        "heldout_rmse": rmse(features[validation]@w+b, returns[validation]),
        "weight_l2": float(np.linalg.norm(w)), "bias": float(b)})
# Prefer a smaller value-head norm when fitting errors are comparable. This
# limits amplification of value gradients into the unchanged shared network
# during the subsequent PPO run; the holdout is still only this trajectory.
best_validation_rmse = min(trial["heldout_rmse"] for trial in trials)
selected = min((trial for trial in trials if trial["heldout_rmse"] <= 1.5*best_validation_rmse), key=lambda trial: trial["weight_l2"])
w, b = fit(features, returns, selected["ridge"])

calibrated = load_model(original["model"])
with torch.no_grad():
    calibrated.critic.weight.copy_(torch.as_tensor(w[None, :], dtype=torch.float32))
    calibrated.critic.bias.copy_(torch.as_tensor([b], dtype=torch.float32))
state = calibrated.state_dict()
changed = [key for key in state if not torch.equal(state[key], original["model"][key])]
assert set(changed) == {"critic.weight", "critic.bias"}
assert all(torch.isfinite(tensor).all() for tensor in state.values())
obs_tensor = torch.as_tensor(trajectory["observations"], dtype=torch.float32)
with torch.no_grad():
    old_logits, _ = source_model(obs_tensor)
    new_logits, predictions = calibrated(obs_tensor)
assert torch.equal(old_logits, new_logits)
predictions_np = predictions.numpy().astype(np.float64)
residuals = predictions_np-returns

# Construct optimizer after all optional checkpoint branches were restored.
optimizer = torch.optim.Adam(calibrated.parameters(), lr=float(config["learning_rate"]))
assert not optimizer.state_dict()["state"]
output = copy.deepcopy(original)
output["model"] = {key: value.detach().clone() for key, value in state.items()}
output["optimizer"] = optimizer.state_dict()
output["config"] = dict(config, gamma=GAMMA, gae_lambda=.995, reset_optimizer_on_resume=True)
metadata = {"source": str(SOURCE), "gamma": GAMMA, "samples": len(returns), "method": "ridge least squares on frozen original actor/shared features; unregularized intercept",
    "selected_ridge": selected["ridge"], "full_trajectory_rmse": rmse(predictions_np, returns),
    "ridge_selection": "smallest weight norm among candidates whose interleaved heldout RMSE is within 1.5x the minimum",
    "calibrated_critic_weight_l2": float(np.linalg.norm(w)),
    "source_critic_rmse_on_new_gamma": rmse(trajectory["old_values"], returns),
    "only_modified_parameters": changed, "actor_logits_bitwise_unchanged": True,
    "optimizer": "fresh empty Adam; constructed after strict model loading",
    "scope": "value pre-calibration on one deterministic successful trajectory; not PPO training or proof of generalization"}
output["critic_precalibration"] = metadata
torch.save(output, OUTPUT)
reloaded = torch.load(OUTPUT, map_location="cpu", weights_only=False)
loaded_model = load_model(reloaded["model"])
with torch.no_grad():
    loaded_logits, loaded_values = loaded_model(obs_tensor)
assert torch.equal(old_logits, loaded_logits) and torch.equal(predictions, loaded_values)
assert not reloaded["optimizer"]["state"]
assert len(reloaded["optimizer"]["param_groups"][0]["params"]) == len(list(loaded_model.parameters()))
confirmed = rollout(loaded_model)
assert trajectory["time_s"] == confirmed["time_s"] and trajectory["distance_m"] == confirmed["distance_m"]
assert np.array_equal(trajectory["actions"], confirmed["actions"])
assert np.array_equal(trajectory["rewards"], confirmed["rewards"])

report = {"status": "passed", "source": str(SOURCE), "output": str(OUTPUT), "gamma": GAMMA, "gae_lambda": .995,
    "samples": len(returns), "regression_dimensions_with_intercept": features.shape[1]+1,
    "ridge_trials": trials, **metadata, "error_absolute_max": float(np.abs(residuals).max()),
    "error_absolute_p95": float(np.percentile(np.abs(residuals), 95)),
    "return_range": [float(returns.min()), float(returns.max())],
    "fitted_value_range": [float(predictions_np.min()), float(predictions_np.max())],
    "before": {key: trajectory[key] for key in ["time_s", "distance_m", "steps", "success"]},
    "after_strict_reload": {key: confirmed[key] for key in ["time_s", "distance_m", "steps", "success"]},
    "all_parameters_finite": True, "actions_rewards_bitwise_unchanged": True,
    "checkpoint_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), "elapsed_seconds": time.perf_counter()-started}
(BASE/"calibrate_classic_critic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
