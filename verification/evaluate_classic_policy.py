"""Evaluate one full first episode per environment using the original project."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1] / "QWOP_rl_training_public"
sys.path.insert(0, str(PROJECT / "scripts"))
from running_env import VectorRunningGameEnv
from train_ppo import ActorCritic, FrameStacker
from replay_io import ruleset_for_game_mode, write_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.threads < 1:
        raise ValueError("episodes and threads must be positive")
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else PROJECT / args.checkpoint
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    data = torch.load(checkpoint, map_location=device)
    cfg = data["config"]
    reward_profile = cfg["reward_profile"]
    if reward_profile not in ("classic_100m", "classic_100m_task"):
        raise ValueError("This evaluator requires a classic_100m game reward profile")
    if cfg.get("game_mode", "auto") not in ("classic_100m", "auto"):
        raise ValueError("This evaluator requires the original classic_100m game")
    env = VectorRunningGameEnv(args.episodes, num_threads=args.threads,
                               reward_profile=reward_profile, game_mode="classic_100m")
    obs, _ = env.reset()
    base_obs = obs.copy()
    stacker = FrameStacker(args.episodes, obs.shape[1], cfg["frame_stack"])
    obs = stacker.reset(obs)
    model = ActorCritic(stacker.obs_dim, env.action_space_n, tuple(cfg["hidden_sizes"])).to(device)
    model.load_state_dict(data["model"], strict=True)
    model.eval()
    active = np.ones(args.episodes, dtype=bool)
    episode_returns = np.zeros(args.episodes, dtype=np.float64)
    first_10m_time = np.full(args.episodes, np.nan)
    foot_steps = np.zeros(args.episodes, dtype=np.int64)
    upright_foot_steps = np.zeros(args.episodes, dtype=np.int64)
    low_unsupported_steps = np.zeros(args.episodes, dtype=np.int64)
    low_posture_steps = np.zeros(args.episodes, dtype=np.int64)
    torso_height_sum = np.zeros(args.episodes, dtype=np.float64)
    lean_steps = np.zeros(args.episodes, dtype=np.int64)
    head_behind_steps = np.zeros(args.episodes, dtype=np.int64)
    backward_lean_steps = np.zeros(args.episodes, dtype=np.int64)
    forward_target_steps = np.zeros(args.episodes, dtype=np.int64)
    lean_angle_sum = np.zeros(args.episodes, dtype=np.float64)
    gait_stats = np.zeros((args.episodes, 5), dtype=np.float64)
    previous_height = (429.8 - base_obs[:, 26] * 40.0) / 506.85
    previous_time = base_obs[:, 61].copy()
    air_frames = np.zeros((args.episodes, 2), dtype=np.int64)
    grounded_frames = np.zeros((args.episodes, 2), dtype=np.int64)
    was_lifted = np.zeros((args.episodes, 2), dtype=bool)
    last_landing_foot = np.full(args.episodes, -1, dtype=np.int64)
    last_landing_time = np.full(args.episodes, -1.0)
    alternating_landings = np.zeros(args.episodes, dtype=np.int64)
    histories: list[list[int]] = [[] for _ in range(args.episodes)]
    records: list[dict] = []
    for step in range(1, 4001):
        foot = (base_obs[:, 66] > 0.5) | (base_obs[:, 67] > 0.5)
        height_ratio = (429.8 - base_obs[:, 26] * 40.0) / (429.8 + 77.05)
        upright = height_ratio >= 0.8
        lean_angle = np.rad2deg(np.arctan2(base_obs[:, 50] - base_obs[:, 25],
                                         base_obs[:, 26] - base_obs[:, 51]))
        measured = active & (base_obs[:, 61] >= 5.0)
        lean_steps[measured] += 1
        head_behind_steps[measured] += (lean_angle < 0)[measured]
        backward_lean_steps[measured] += (lean_angle < -10)[measured]
        forward_target_steps[measured] += ((lean_angle >= 5) & (lean_angle <= 10))[measured]
        lean_angle_sum[measured] += lean_angle[measured]
        elapsed = base_obs[:, 61] - previous_time
        vertical_speed = np.divide(height_ratio - previous_height, elapsed,
                                   out=np.zeros_like(height_ratio), where=elapsed > 1e-6)
        gait_stats[measured] += np.column_stack((np.abs(lean_angle - 5.0),
            np.abs(lean_angle - 5.0) <= 3.0, height_ratio, height_ratio ** 2, vertical_speed ** 2))[measured]
        previous_height = height_ratio.copy()
        previous_time = base_obs[:, 61].copy()
        contacts = base_obs[:, [66, 67]] > 0.5
        air_frames = np.where(contacts, 0, air_frames + 1)
        grounded_frames = np.where(contacts, grounded_frames + 1, 0)
        was_lifted |= air_frames >= 3
        landings = contacts & (grounded_frames == 2) & was_lifted
        was_lifted[landings] = False
        for index in np.flatnonzero(active & (landings.sum(axis=1) == 1)):
            foot_index = int(np.argmax(landings[index]))
            now = float(base_obs[index, 61])
            if foot_index != last_landing_foot[index] and now - last_landing_time[index] >= 0.15:
                alternating_landings[index] += int(measured[index])
                last_landing_foot[index] = foot_index
                last_landing_time[index] = now
        foot_steps[active] += foot[active]
        upright_foot_steps[active] += (foot & upright)[active]
        low_unsupported_steps[active] += (~foot & ~upright)[active]
        low_posture_steps[active] += (~upright)[active]
        torso_height_sum[active] += height_ratio[active]
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(obs, dtype=torch.float32, device=device))
            actions = (logits.argmax(dim=-1) if args.deterministic else
                       torch.distributions.Categorical(logits=logits).sample())
        actions_np = actions.cpu().numpy()
        if args.replay:
            for index in np.flatnonzero(active):
                histories[index].append(int(actions_np[index]))
        result = env.step_arrays(actions_np)
        if not np.isfinite(result["obs"]).all() or not np.isfinite(result["reward"]).all():
            raise ValueError("Non-finite environment output")
        done = result["terminated"] | result["truncated"]
        if not np.array_equal(done, result["has_final"]):
            raise ValueError("Final episode metadata is inconsistent")
        episode_returns[active] += result["reward"][active]
        ended = active & done
        measured_distance = np.where(done, result["final_distance"], result["obs"][:, 60])
        measured_time = np.where(done, result["final_time"], result["obs"][:, 61])
        crossed_10m = active & np.isnan(first_10m_time) & (measured_distance >= 10.0)
        first_10m_time[crossed_10m] = measured_time[crossed_10m]
        for index in np.flatnonzero(ended):
            records.append({"env_index": int(index), "steps": step,
                            "distance": float(result["final_distance"][index]),
                            "time": float(result["final_time"][index]),
                            "first_10m_time": float(first_10m_time[index]) if np.isfinite(first_10m_time[index]) else None,
                            "success": bool(result["final_success"][index]),
                            "truncated": bool(result["truncated"][index]),
                            "episode_return": float(episode_returns[index]),
                            "foot_support_fraction": float(foot_steps[index] / step),
                            "upright_foot_support_fraction": float(upright_foot_steps[index] / step),
                            "low_unsupported_fraction": float(low_unsupported_steps[index] / step),
                            "low_posture_fraction": float(low_posture_steps[index] / step),
                            "mean_torso_height_ratio": float(torso_height_sum[index] / step),
                            "lean_measured_steps_after_5_seconds": int(lean_steps[index]),
                            "head_behind_torso_fraction": float(head_behind_steps[index] / lean_steps[index]) if lean_steps[index] else None,
                            "backward_lean_fraction_below_minus_10_degrees": float(backward_lean_steps[index] / lean_steps[index]) if lean_steps[index] else None,
                            "forward_target_fraction_5_to_10_degrees": float(forward_target_steps[index] / lean_steps[index]) if lean_steps[index] else None,
                            "mean_lean_degrees": float(lean_angle_sum[index] / lean_steps[index]) if lean_steps[index] else None})
        active[ended] = False
        for record in records[-int(ended.sum()):] if ended.any() else []:
            i = record["env_index"]
            if lean_steps[i]:
                stats = gait_stats[i] / lean_steps[i]
                record.update({"lean_abs_error_from_5_degrees": float(stats[0]),
                    "lean_within_2_to_8_degrees_fraction": float(stats[1]),
                    "torso_height_std_after_5_seconds": float(np.sqrt(max(0, stats[3] - stats[2] ** 2))),
                    "torso_vertical_speed_rms_height_ratios_per_second": float(np.sqrt(stats[4])),
                    "alternating_foot_landings_per_second": float(alternating_landings[i] / (lean_steps[i] / 30.0))})
        obs = stacker.update(result["obs"], done)
        base_obs = result["obs"]
        if not active.any():
            break
    if active.any():
        raise RuntimeError("Some episodes did not finish within the original time limit")
    distances = np.array([r["distance"] for r in records])
    successes = sum(r["success"] for r in records)
    success_times = [r["time"] for r in records if r["success"]]
    under_50 = [r for r in records if r["success"] and r["time"] < 50.0]
    report = {"checkpoint": str(checkpoint), "global_step": data["global_step"],
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "reward_profile": reward_profile, "game_mode": "classic_100m",
              "update": data["update"], "seed": args.seed, "device": str(device),
              "deterministic": args.deterministic, "episodes": len(records),
              "successes": successes, "success_rate": successes / len(records),
              "distance_mean": float(distances.mean()),
              "distance_median": float(np.median(distances)),
              "distance_max": float(distances.max()),
              "success_time_mean": float(np.mean(success_times)) if success_times else None,
              "success_time_median": float(np.median(success_times)) if success_times else None,
              "success_time_min": float(np.min(success_times)) if success_times else None,
              "first_10m_time_mean": float(np.nanmean(first_10m_time)) if np.isfinite(first_10m_time).any() else None,
              "successes_under_50_seconds": len(under_50),
              "under_50_rate_all_episodes": len(under_50) / len(records),
              "fall_rate": sum(not r["success"] and not r["truncated"] for r in records) / len(records),
              "episode_return_mean": float(episode_returns.mean()),
              "foot_support_fraction": float(foot_steps.sum() / sum(r["steps"] for r in records)),
              "upright_foot_support_fraction": float(upright_foot_steps.sum() / sum(r["steps"] for r in records)),
              "low_unsupported_fraction": float(low_unsupported_steps.sum() / sum(r["steps"] for r in records)),
              "low_posture_fraction": float(low_posture_steps.sum() / sum(r["steps"] for r in records)),
              "mean_torso_height_ratio": float(torso_height_sum.sum() / sum(r["steps"] for r in records)),
              "lean_measurement": "head-to-torso world geometry, positive toward finish; after first 5 game seconds",
              "head_behind_torso_fraction": float(head_behind_steps.sum() / lean_steps.sum()) if lean_steps.sum() else None,
              "backward_lean_fraction_below_minus_10_degrees": float(backward_lean_steps.sum() / lean_steps.sum()) if lean_steps.sum() else None,
              "forward_target_fraction_5_to_10_degrees": float(forward_target_steps.sum() / lean_steps.sum()) if lean_steps.sum() else None,
              "mean_lean_degrees": float(lean_angle_sum.sum() / lean_steps.sum()) if lean_steps.sum() else None,
              "contact_measurement": "foot flags and low-torso proxy; no knee contact flag",
              "evaluation_seconds": time.perf_counter() - started,
              "episodes_detail": sorted(records, key=lambda r: r["env_index"])}
    if args.replay:
        qualified = [r for r in records if r["success"] and r["low_posture_fraction"] <= 0.1
                     and r["backward_lean_fraction_below_minus_10_degrees"] is not None
                     and r["backward_lean_fraction_below_minus_10_degrees"] <= 0.2]
        best = min(qualified, key=lambda r: r["time"]) if qualified else max(
            records, key=lambda r: r["distance"] * r["upright_foot_support_fraction"])
        write_replay(args.replay, histories[best["env_index"]], ruleset_for_game_mode("classic_100m"))
        report["replay"] = str(args.replay)
        report["replay_episode"] = best
    report["successes_under_45_seconds"] = sum(r["success"] and r["time"] < 45 for r in records)
    report["successes_under_43_seconds"] = sum(r["success"] and r["time"] < 43 for r in records)
    upright_successes = [r for r in records if r["success"] and r["low_posture_fraction"] <= .1]
    report["qualified_successes_low_posture_le_0p1"] = len(upright_successes)
    report["success_low_posture_fraction_mean"] = float(np.mean([r["low_posture_fraction"] for r in records if r["success"]])) if success_times else None
    report["initial_condition_note"] = "All environments use the same original native initial state; stochastic episodes differ in sampled actions, not independently varied initial conditions."
    report["episode_return_comparison_note"] = "Returns from different reward profiles have different definitions; compare native success, final time and posture metrics instead."
    for metric in ["lean_abs_error_from_5_degrees", "lean_within_2_to_8_degrees_fraction",
                   "torso_height_std_after_5_seconds", "torso_vertical_speed_rms_height_ratios_per_second",
                   "alternating_foot_landings_per_second"]:
        values = [r[metric] for r in records if metric in r]
        report[metric] = float(np.mean(values)) if values else None
    report["gait_measurement"] = "torso height proxy, not mass-weighted COM; alternating foot contacts after >=3 airborne frames and >=2 contact frames, >=0.15 seconds apart"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes_detail"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
