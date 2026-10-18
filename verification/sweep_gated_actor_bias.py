"""Search gated NN action biases; keep native race physics and scoring intact.

The bias experiment saves parameter JSON and a diagnostic replay. The residual
experiment may export a complete checkpoint after an independent strict reload.
The original source checkpoint is never changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "QWOP_rl_training_public"
sys.path.insert(0, str(ROOT / "scripts"))
from running_env import VectorRunningGameEnv, RunningGameEnv
from train_ppo import ActorCritic, FrameStacker
from replay_io import write_replay, read_replay, ruleset_for_game_mode

STDS = np.array([.03, .1, .3, 1, 2, 4], dtype=np.float32)
RESIDUAL_STDS = np.array([.001, .003, .01, .03, .1, .3], dtype=np.float32)
METERS = (10, 40, 50, 60, 70, 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/ppo_classic100m_phase_fast.pt")
    parser.add_argument("--output", type=Path, default=BASE / "verification/target43_gated_actor_bias.json")
    parser.add_argument("--save-replay", type=Path, default=BASE / "verification/target43_gated_actor_bias_diagnostic.bin")
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=94301)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--mode", choices=["bias", "residual"], default="bias")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--save-checkpoint", type=Path)
    args = parser.parse_args()
    if args.candidates < 2 or args.threads <= 0:
        parser.error("candidates >=2 and threads >0 required")
    if not np.isfinite(args.noise_scale) or args.noise_scale <= 0:
        parser.error("noise-scale must be finite and positive")
    if args.save_checkpoint and args.mode != "residual":
        parser.error("Checkpoint export requires residual mode")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    source = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    data = torch.load(source, map_location="cpu")
    cfg = data["config"]
    assert cfg["game_mode"] in ("classic_100m", "auto")
    rng = np.random.default_rng(args.seed)
    std_cycle = STDS if args.mode == "bias" else RESIDUAL_STDS
    sigmas = np.resize(std_cycle, args.candidates) * args.noise_scale
    biases = ((rng.standard_normal((args.candidates, 16)) * sigmas[:, None]).astype(np.float32)
              if args.mode == "bias" else np.zeros((args.candidates, 16), dtype=np.float32))
    if args.mode == "bias":
        biases -= biases.mean(axis=1, keepdims=True)
    biases[0] = 0
    bias_tensor = torch.as_tensor(biases, dtype=torch.float32, device=device)
    env = VectorRunningGameEnv(args.candidates, num_threads=args.threads,
                               reward_profile="classic_100m", game_mode="classic_100m")
    raw, _ = env.reset()
    stacker = FrameStacker(args.candidates, 68, cfg["frame_stack"])
    obs = stacker.reset(raw)
    model = ActorCritic(stacker.obs_dim, 16, tuple(cfg["hidden_sizes"])).to(device)
    model.load_state_dict(data["model"], strict=True)
    model.eval()
    weights = None
    weight_tensor = None
    if args.mode == "residual":
        assert model.flat_actor_bias is None and model.flat_actor_weight is None, "This first residual population requires the original source NN without a flat branch"
        assert model.actor.in_features == 256 and model.actor.out_features == 16
        weights = (rng.standard_normal((args.candidates, 16, model.actor.in_features))
                   * sigmas[:, None, None]).astype(np.float32)
        weights[0] = 0
        weight_tensor = torch.as_tensor(weights, device=device)
    active = np.ones(args.candidates, dtype=bool)
    low = np.zeros(args.candidates, dtype=np.int64)
    backward = np.zeros(args.candidates, dtype=np.int64)
    measured_count = np.zeros(args.candidates, dtype=np.int64)
    crossing_times = np.full((args.candidates, len(METERS)), np.nan)
    action_history = np.empty((4000, args.candidates), dtype=np.uint8)
    records = []
    for step in range(1, 4001):
        height = (429.8 - raw[:, 26] * 40) / 506.85
        lean = np.rad2deg(np.arctan2(raw[:, 50] - raw[:, 25], raw[:, 26] - raw[:, 51]))
        low[active] += (height < .8)[active]
        measured = active & (raw[:, 61] >= 5)
        backward[measured] += (lean < -10)[measured]
        measured_count[measured] += 1
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            score = x[:, -8]
            gate = ((score >= 10) & (score < 45)) | ((score >= 65) & (score <= 100))
            if args.mode == "residual":
                hidden = model.features(x)
                logits = model.actor(hidden)
                residual = torch.einsum("ni,nai->na", hidden, weight_tensor) + bias_tensor
            else:
                logits, _ = model(x)
                residual = bias_tensor
            logits = torch.where(gate[:, None], logits + residual, logits)
            actions = logits.argmax(-1).cpu().numpy()
        action_history[step - 1] = actions
        result = env.step_arrays(actions)
        assert np.isfinite(result["obs"]).all() and np.isfinite(result["reward"]).all()
        for column, meter in enumerate(METERS):
            newly_crossed = active & np.isnan(crossing_times[:, column]) & (result["score"] >= meter)
            crossing_times[newly_crossed, column] = result["time"][newly_crossed]
        ended = result["has_final"]
        for index in np.flatnonzero(active & ended):
            crossings = {str(meter): float(crossing_times[index, column])
                         if np.isfinite(crossing_times[index, column]) else None
                         for column, meter in enumerate(METERS)}

            def split(first, last):
                a, b = crossings[str(first)], crossings[str(last)]
                return b - a if a is not None and b is not None else None

            record = {"candidate_index": int(index), "sigma": float(sigmas[index]),
                      "bias": biases[index].tolist(), "steps": step,
                      "success": bool(result["final_success"][index]),
                      "truncated": bool(result["truncated"][index]),
                      "time_s": float(result["final_time"][index]),
                      "distance_m": float(result["final_distance"][index]),
                      "score_m": float(result["final_score"][index]),
                      "low_posture_fraction": float(low[index] / step),
                      "backward_fraction": float(backward[index] / measured_count[index]) if measured_count[index] else 0.0,
                      "crossing_times_s": crossings,
                      "splits_s": {"0_10": crossings["10"], "40_50": split(40, 50),
                                   "50_60": split(50, 60), "60_70": split(60, 70)}}
            record["qualified"] = record["success"] and not record["truncated"] and record["low_posture_fraction"] <= .1 and record["backward_fraction"] <= .2
            records.append(record)
        active[ended] = False
        raw = result["obs"]
        obs = stacker.update(raw, ended)
        if step % 300 == 0:
            print(json.dumps({"step": step, "completed": len(records), "candidates": args.candidates}), flush=True)
        if not active.any():
            break
    assert not active.any(), "Native 120 s episode limit should end all candidates before 4000 steps"
    baseline = next(record for record in records if record["candidate_index"] == 0)
    assert baseline["qualified"]
    qualified = [record for record in records if record["qualified"]]
    best = min(qualified, key=lambda record: record["time_s"]) if qualified else None
    replay_validation = None
    if best:
        selected_actions = action_history[:best["steps"], best["candidate_index"]].tolist()
        write_replay(args.save_replay, selected_actions, ruleset_for_game_mode("classic_100m"))
        ruleset, loaded_actions = read_replay(args.save_replay)
        assert loaded_actions == selected_actions
        replay_env = RunningGameEnv("classic_100m", "classic_100m")
        replay_env.reset()
        for replay_step, action in enumerate(loaded_actions, start=1):
            replay_obs, reward, done, truncated, info = replay_env.step(action)
            assert np.isfinite(replay_obs).all() and np.isfinite(reward)
            if done or truncated:
                assert replay_step == len(loaded_actions), "Replay terminated before its last action"
        assert info["success"] and done and not truncated
        assert abs(info["time"] - best["time_s"]) < 1e-5
        assert abs(info["time"] - len(loaded_actions) / 30) < 1e-5
        replay_validation = {"ruleset": int(ruleset), "actions": len(loaded_actions),
                             "success_on_last_action": bool(info["success"]),
                             "time_s": float(info["time"]), "distance_m": float(info["distance"])}
    saved_checkpoint = None
    strict_reload = None
    if args.mode == "residual" and args.save_checkpoint and best["time_s"] < baseline["time_s"] - 1e-5:
        from evaluate_gait_bottlenecks import evaluate_checkpoint
        exported_model = ActorCritic(stacker.obs_dim, 16, tuple(cfg["hidden_sizes"]))
        exported_model.load_state_dict(data["model"], strict=True)
        exported_model.enable_flat_residual((10., 45., 65., 100.))
        with torch.no_grad():
            exported_model.flat_actor_weight.copy_(torch.as_tensor(weights[best["candidate_index"]]))
            exported_model.flat_actor_bias.copy_(torch.as_tensor(biases[best["candidate_index"]]))
        assert len(list(exported_model.parameters())) == 11
        exported_state = exported_model.state_dict()
        for key, value in data["model"].items():
            assert torch.equal(value, exported_state[key]), f"Source weight modified: {key}"
        optimizer = torch.optim.Adam(exported_model.parameters(), lr=float(cfg["learning_rate"]), eps=1e-5)
        assert not optimizer.state_dict()["state"]
        export_cfg = dict(cfg)
        export_cfg.update(enable_flat_actor_bias=True, enable_flat_actor_residual=True,
                          reset_optimizer_on_resume=True, checkpoint=str(args.save_checkpoint))
        core = {"model", "optimizer", "config", "update", "global_step"}
        exported = {key: value for key, value in data.items() if key in core}
        exported.update(model=exported_state, optimizer=optimizer.state_dict(), config=export_cfg)
        exported["gated_residual_refinement"] = {
            "source_checkpoint": str(source), "seed": args.seed, "candidates": args.candidates,
            "candidate_index": best["candidate_index"], "noise_scale": args.noise_scale,
            "sigma_cycle": std_cycle.tolist(), "actual_selected_sigma": best["sigma"],
            "flat_actor_bounds": [10., 45., 65., 100.], "bias": best["bias"],
            "search_result": best, "method": "Frozen source features plus state-dependent flat Actor residual parameter search",
            "optimizer": "Fresh empty Adam created after enabling both new parameters; 11 parameter tensors",
            "ancestor_metadata": {key: value for key, value in data.items() if key not in core},
        }
        temporary_export = BASE / "verification/target43_residual_strict_reload_candidate.pt"
        torch.save(exported, temporary_export)
        reload_trace = BASE / "verification/target43_residual_search1_strict_trace.csv"
        strict_reload = evaluate_checkpoint(temporary_export, trace=reload_trace, device=args.device)
        reload_rows = np.loadtxt(reload_trace, delimiter=",", skiprows=1)
        reload_measured = reload_rows[:, 1] >= 5
        reload_backward_fraction = float((reload_rows[reload_measured, 5] < -10).mean()) if reload_measured.any() else 0.0
        strict_reload["backward_fraction_after5s"] = reload_backward_fraction
        exported["gated_residual_refinement"]["strict_reload_evaluation"] = strict_reload
        exported["gated_residual_refinement"]["search_reload_time_difference_s"] = strict_reload["time_s"] - best["time_s"]
        if strict_reload["success"] and not strict_reload["truncated"] and not strict_reload["native_failed"] and strict_reload["whole_episode_low_posture_fraction"] <= .1 and reload_backward_fraction <= .2 and strict_reload["time_s"] < baseline["time_s"] - 1e-5:
            args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(exported, args.save_checkpoint)
            final_reload = evaluate_checkpoint(args.save_checkpoint, device=args.device)
            assert abs(final_reload["time_s"] - strict_reload["time_s"]) < 1e-5
            assert final_reload["success"] and not final_reload["truncated"] and not final_reload["native_failed"]
            final_reload["backward_fraction_after5s"] = reload_backward_fraction
            saved_checkpoint = str(args.save_checkpoint)
            strict_reload = final_reload
    report = {"method": ("One local CEM population: state-dependent 16x256 flat Actor residual weights and zero biases, gated by raw newest-frame score; source weights frozen"
                          if args.mode == "residual" else "One local CEM population: 16 centered action-output bias parameters gated by raw newest-frame score; original native final success and game time"),
              "source_checkpoint": str(source), "candidate_count": args.candidates,
              "mode": args.mode, "seed": args.seed, "sigma_cycle": std_cycle.tolist(), "noise_scale": args.noise_scale,
              "gating": {"raw_score_index": -8, "frame_observation_size": 68,
                         "intervals": [{"start": 10, "end": 45, "left_closed": True, "right_closed": False},
                                       {"start": 65, "end": 100, "left_closed": True, "right_closed": True}]},
              "baseline": baseline, "qualified_count": len(qualified), "best": best,
              "beats_baseline": bool(best and best["time_s"] < baseline["time_s"] - 1e-5),
              "under43": bool(best and best["time_s"] < 43),
              "diagnostic_replay": str(args.save_replay), "replay_validation": replay_validation,
              "best_residual_weight": weights[best["candidate_index"]].tolist() if weights is not None and best else None,
              "loadable_nn_checkpoint_created": saved_checkpoint is not None,
              "saved_checkpoint": saved_checkpoint, "strict_reload_evaluation": strict_reload,
              "search_reload_time_difference_s": strict_reload["time_s"] - best["time_s"] if strict_reload else None,
              "notes": ["Biases are candidate strategy parameters, not scripted action prefixes.",
                        "The source NN checkpoint is unchanged. A new residual checkpoint is exported only when its strict reload still beats baseline; otherwise JSON/replay remain diagnostic artifacts.",
                        "Native physics, raw observations, clock, actions, course and termination remain unchanged.",
                        "Selection filters: native final success, no truncation, low posture <=0.1, backward fraction after 5 s <=0.2."],
              "records": records, "started_utc": started_utc,
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.perf_counter() - started}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("candidate_count", "qualified_count", "baseline", "best", "beats_baseline", "under43", "replay_validation", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
