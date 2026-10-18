"""Test small horizontal-velocity feedback changes in the actual environment."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1] / "QWOP_rl_training_public"
sys.path.insert(0, str(PROJECT / "scripts"))
from running_env import VectorRunningGameEnv
from train_ppo import ActorCritic, FrameStacker
from replay_io import write_replay, ruleset_for_game_mode

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--save-best", type=Path)
    p.add_argument("--mode", choices=["feedback", "bias", "weights", "joint", "hidden1", "hidden2", "hidden1_gain", "hidden2_gain", "motion", "startup", "startup2", "kinematics", "task_feedback"], default="feedback")
    p.add_argument("--save-replay", type=Path)
    p.add_argument("--candidates", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--noise-scale", type=float, default=1.0)
    args = p.parse_args()
    if not np.isfinite(args.noise_scale) or args.noise_scale <= 0:
        p.error("noise-scale must be finite and positive")
    if args.mode.startswith("startup") and args.save_best:
        p.error("Startup-action intervention must not be exported as learned weights")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    data = torch.load(args.checkpoint, map_location="cpu")
    cfg = data["config"]
    factors = np.array([0.70, 0.85, 1.0, 1.10, 1.20, 1.30, 1.40, 1.60], dtype=np.float32)
    vertical_factors = np.ones(len(factors), dtype=np.float32)
    if args.mode.startswith("startup"):
        if args.mode == "startup":
            pairs = [(0, 0, 0, 0)] + [(a, n, 0, 0) for a in range(16) for n in [1,2,3,4,6,8,12,16,24,30,45,60]]
        else:
            pairs = [(0, 0, 0, 0)] + [(a, n, b, m) for a in range(16) for b in range(16)
                                      if a != b for n in [3,6,12,24] for m in [3,6,12,24]]
        start_actions = np.array([a for a,n,b,m in pairs], dtype=np.int64)
        start_first_frames = np.array([n for a,n,b,m in pairs], dtype=np.int64)
        start_second_actions = np.array([b for a,n,b,m in pairs], dtype=np.int64)
        start_frames = np.array([n+m for a,n,b,m in pairs], dtype=np.int64)
        factors = np.ones(len(pairs), dtype=np.float32)
        vertical_factors = np.ones(len(pairs), dtype=np.float32)
    if args.mode in {"kinematics", "task_feedback"}:
        grid = ([1.0, 0.5, 0.7, 0.85, 1.1, 1.2, 1.4, 2.0, 4.0, 1e6]
                if args.mode == "task_feedback" else [1.0, 0.70, 0.85, 1.10, 1.20, 1.30, 1.40, 1.60])
        pairs = np.array([(x, y) for x in grid for y in grid], dtype=np.float32)
        factors, vertical_factors = pairs[:, 0], pairs[:, 1]
    if args.mode in {"bias", "weights", "joint", "hidden1", "hidden2", "hidden1_gain", "hidden2_gain", "motion"}:
        factors = np.ones(args.candidates, dtype=np.float32)
        vertical_factors = np.ones(args.candidates, dtype=np.float32)
        rng = np.random.default_rng(args.seed)
        std = np.resize(np.array([0.15, 0.30, 0.50, 0.80],dtype=np.float32),args.candidates)
        perturbations = (rng.standard_normal((args.candidates,16)) * (std * args.noise_scale)[:,None]).astype(np.float32)
        perturbations -= perturbations.mean(axis=1,keepdims=True)
        perturbations[0] = 0
    else:
        perturbations = np.zeros((len(factors),16),dtype=np.float32)
    perturb_tensor = torch.as_tensor(perturbations, device=device)
    env = VectorRunningGameEnv(len(factors), num_threads=args.threads,
        reward_profile="classic_100m", game_mode="classic_100m")
    raw, _ = env.reset()
    stacker = FrameStacker(len(factors), 68, cfg["frame_stack"])
    obs = stacker.reset(raw)
    model = ActorCritic(stacker.obs_dim, 16, tuple(cfg["hidden_sizes"])).to(device)
    model.load_state_dict(data["model"], strict=True)
    if args.mode == "task_feedback" and model.angular_velocity is not None:
        p.error("Clock rescaling cannot be folded into weights of an angular-velocity model")
    model.eval()
    if args.mode == "motion":
        if model.angular_velocity is None:
            p.error("Motion search requires a checkpoint with angular velocity features")
        std_motion = np.resize(np.array([.001, .003, .01, .03, .10, .30], dtype=np.float32), args.candidates)
        motion_changes = (rng.standard_normal((args.candidates, model.net[0].out_features, 12))
                          * (std_motion * args.noise_scale)[:, None, None]).astype(np.float32)
        motion_changes[0] = 0
        motion_tensor = torch.as_tensor(motion_changes, device=device)
        perturbations[:] = 0
        perturb_tensor = torch.as_tensor(perturbations, device=device)
    if args.mode in {"hidden1", "hidden2", "hidden1_gain", "hidden2_gain"}:
        if len(cfg["hidden_sizes"]) != 2:
            raise ValueError("Hidden-layer search requires the saved two-layer architecture")
        hidden_key = "net.0.bias" if args.mode.startswith("hidden1") else "net.2.bias"
        hidden_std = np.resize(np.array([0.01,0.03,0.07,0.15,0.30,0.50],dtype=np.float32),args.candidates)
        hidden_changes = (rng.standard_normal((args.candidates,data["model"][hidden_key].numel())) * (hidden_std * args.noise_scale)[:,None]).astype(np.float32)
        hidden_changes[0] = 0
        if args.mode.endswith("_gain"):
            hidden_changes = np.exp(hidden_changes)
        hidden_tensor = torch.as_tensor(hidden_changes, device=device)
        perturbations[:] = 0
        perturb_tensor = torch.as_tensor(perturbations, device=device)
    if args.mode in {"weights", "joint"}:
        std_w = np.resize(np.array([0.005,0.01,0.02,0.04],dtype=np.float32),args.candidates)
        weight_changes = (rng.standard_normal((args.candidates,16,model.actor.in_features)) * (std_w * args.noise_scale)[:,None,None]).astype(np.float32)
        weight_changes[0] = 0
        if args.mode == "weights":
            perturbations[:] = 0
        perturb_tensor = torch.as_tensor(perturbations, device=device)
        weight_tensor = torch.as_tensor(weight_changes, device=device)
    columns = np.concatenate([np.arange(3, 60, 5) + i * 68 for i in range(cfg["frame_stack"])])
    vertical_columns = np.concatenate([np.arange(4, 60, 5) + i * 68 for i in range(cfg["frame_stack"])])
    if args.mode == "task_feedback":
        columns = np.array([61 + i * 68 for i in range(cfg["frame_stack"])])
        vertical_columns = np.array([60 + i * 68 for i in range(cfg["frame_stack"])])
    active = np.ones(len(factors), dtype=bool)
    low = np.zeros(len(factors), dtype=np.int64)
    backward = np.zeros(len(factors), dtype=np.int64)
    count = np.zeros(len(factors), dtype=np.int64)
    records = []
    factor_tensor = torch.as_tensor(factors, device=device)[:, None]
    vertical_factor_tensor = torch.as_tensor(vertical_factors, device=device)[:, None]
    action_history = np.empty((4000, len(factors)), dtype=np.uint8) if args.save_replay else None
    for step in range(1, 4001):
        height = (429.8 - raw[:,26] * 40) / 506.85
        lean = np.rad2deg(np.arctan2(raw[:,50] - raw[:,25], raw[:,26] - raw[:,51]))
        low[active] += (height < 0.8)[active]
        measured = active & (raw[:,61] >= 5)
        backward[measured] += (lean < -10)[measured]
        count[measured] += 1
        with torch.no_grad():
            x = torch.as_tensor(obs, device=device).clone()  # Never modify the raw history buffer.
            x[:,columns] /= factor_tensor
            x[:,vertical_columns] /= vertical_factor_tensor
            if args.mode == "motion":
                hidden = model.first_hidden(x) + torch.einsum("bi,bhi->bh", model.motion_features(x), motion_tensor)
                for layer in list(model.net)[1:]:
                    hidden = layer(hidden)
                logits = model.actor(hidden)
            elif args.mode in {"hidden1", "hidden2", "hidden1_gain", "hidden2_gain"}:
                h = model.first_hidden(x)
                if args.mode == "hidden1": h = h + hidden_tensor
                if args.mode == "hidden1_gain": h = h * hidden_tensor
                h = model.net[2](torch.tanh(h))
                if args.mode == "hidden2": h = h + hidden_tensor
                if args.mode == "hidden2_gain": h = h * hidden_tensor
                logits = model.actor(torch.tanh(h))
            elif args.mode in {"weights", "joint"}:
                hidden = model.features(x)
                logits = model.actor(hidden) + torch.einsum("bi,bai->ba",hidden,weight_tensor)
            else:
                logits, _ = model(x)
            logits = logits + perturb_tensor
            actions = logits.argmax(-1).cpu().numpy()
            if args.mode.startswith("startup"):
                first = step <= start_first_frames
                second = (step > start_first_frames) & (step <= start_frames)
                actions[first] = start_actions[first]
                actions[second] = start_second_actions[second]
        if action_history is not None:
            action_history[step-1] = actions
        r = env.step_arrays(actions)
        assert np.isfinite(r["obs"]).all() and np.isfinite(r["reward"]).all()
        done = r["terminated"] | r["truncated"]
        for i in np.flatnonzero(active & done):
            records.append({"candidate_index":int(i), "steps":step, "start_action":int(start_actions[i]) if args.mode.startswith("startup") else None,
                "start_frames":int(start_frames[i]) if args.mode.startswith("startup") else 0,
                "start_first_frames":int(start_first_frames[i]) if args.mode.startswith("startup") else 0,
                "start_second_action":int(start_second_actions[i]) if args.mode.startswith("startup") else None,
                "factor":float(factors[i]), "vertical_factor":float(vertical_factors[i]), "bias_delta":perturbations[i].tolist(), "success":bool(r["final_success"][i]),
                "time":float(r["final_time"][i]), "distance":float(r["final_distance"][i]),
                "low_posture_fraction":float(low[i]/step),
                "backward_fraction":float(backward[i]/count[i]) if count[i] else 0.0})
        active[done] = False
        raw = r["obs"]
        obs = stacker.update(raw, done)
        if not active.any(): break
    assert not active.any()
    qualified = [r for r in records if r["success"] and r["low_posture_fraction"] <= 0.1 and r["backward_fraction"] <= 0.2]
    best = min(qualified, key=lambda r:r["time"]) if qualified else None
    report = {"source":str(args.checkpoint), "mode":args.mode,"seed":args.seed,"noise_scale":args.noise_scale,"records":records, "best":best,
        "method":"policy parameter search or explicitly recorded startup-action intervention; physics, clock and termination unchanged"}
    if args.save_replay and best:
        write_replay(args.save_replay, action_history[:best["steps"],best["candidate_index"]].tolist(),ruleset_for_game_mode("classic_100m"))
        report["diagnostic_replay"] = str(args.save_replay)
    if args.save_best and best:
        copied = dict(data)
        copied["model"] = {k:v.clone() for k,v in data["model"].items()}
        copied["model"]["net.0.weight"][:,columns] /= best["factor"]
        copied["model"]["net.0.weight"][:,vertical_columns] /= best["vertical_factor"]
        copied["model"]["actor.bias"] += torch.as_tensor(best["bias_delta"])
        if args.mode in {"weights", "joint"}:
            copied["model"]["actor.weight"] += torch.as_tensor(weight_changes[best["candidate_index"]])
        if args.mode in {"hidden1", "hidden2"}:
            copied["model"][hidden_key] += torch.as_tensor(hidden_changes[best["candidate_index"]])
            copied["hidden_bias_search"] = {"source":str(args.checkpoint),"mode":args.mode,"seed":args.seed,"bias_delta":hidden_changes[best["candidate_index"]].tolist()}
        if args.mode.endswith("_gain"):
            gains = torch.as_tensor(hidden_changes[best["candidate_index"]])
            copied["model"][hidden_key] *= gains
            copied["model"][hidden_key.replace("bias", "weight")] *= gains[:, None]
            if args.mode == "hidden1_gain" and "angular_velocity.weight" in copied["model"]:
                copied["model"]["angular_velocity.weight"] *= gains[:, None]
            copied["feature_gain_search"] = {"source":str(args.checkpoint), "mode":args.mode,
                                             "seed":args.seed, "gain":gains.tolist()}
        if args.mode == "motion":
            copied["model"]["angular_velocity.weight"] += torch.as_tensor(motion_changes[best["candidate_index"]])
            copied["motion_parameter_search"] = {"source":str(args.checkpoint), "seed":args.seed,
                                                  "candidate_index":best["candidate_index"]}
        copied["speed_feedback_warm_start"] = {"source":str(args.checkpoint), "mode":args.mode, "factor":best["factor"], "vertical_factor":best["vertical_factor"]}
        args.save_best.parent.mkdir(parents=True, exist_ok=True)
        torch.save(copied, args.save_best)
        report["warm_start_checkpoint"] = str(args.save_best)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"source":str(args.checkpoint),"mode":args.mode,"candidates":len(records),
        "qualified":len(qualified),"baseline":next(r for r in records if r["candidate_index"]==0),
        "best":best,"top_times":[r["time"] for r in sorted(qualified,key=lambda r:r["time"])[:10]]}),flush=True)

if __name__ == "__main__": main()
