"""Three-scalar policy search, folded into the original NN first-layer weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from replay_io import read_replay, write_replay, ruleset_for_game_mode
from evaluate_gait_bottlenecks import evaluate_checkpoint

VX = np.arange(3, 60, 5)
VY = np.arange(4, 60, 5)


def fold_state(source_state, gains, history_factors):
    state = {key: value.clone().cpu() for key, value in source_state.items()}
    weight = state["net.0.weight"].reshape(256, 32, 68)
    weight *= torch.as_tensor(history_factors)[None, :, None]
    weight[:, :, VX] /= float(gains[0])
    weight[:, :, VY] /= float(gains[1])
    return state


def adjusted_hidden(model, x, weights, history, vx_coef, vy_coef):
    frames = (x / model.observation_scale).reshape(-1, 32, 68)
    hidden = model.first_hidden(x)
    full = torch.einsum("nbi,bih->nbh", frames, weights)
    hidden = hidden + ((history - 1)[:, :, None] * full).sum(1)
    del full
    vx = torch.einsum("nbi,bih->nbh", frames[:, :, VX], weights[:, VX, :])
    hidden = hidden + (vx_coef[:, :, None] * vx).sum(1)
    del vx
    vy = torch.einsum("nbi,bih->nbh", frames[:, :, VY], weights[:, VY, :])
    hidden = hidden + (vy_coef[:, :, None] * vy).sum(1)
    return hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=94303)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path, default=BASE / "verification/target43_structured_history_kinematics.json")
    parser.add_argument("--save-replay", type=Path, default=BASE / "verification/target43_structured_history_kinematics_diagnostic.bin")
    parser.add_argument("--save-checkpoint", type=Path, default=ROOT / "checkpoints/ppo_target43_structured_history_kinematics.pt")
    args = parser.parse_args()
    assert args.candidates >= 2 and args.threads > 0
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(1)
    device = torch.device("cuda")
    source = ROOT / "checkpoints/ppo_classic100m_phase_fast.pt"
    data = torch.load(source, map_location="cpu")
    cfg = data["config"]
    assert cfg["frame_stack"] == 32 and tuple(cfg["hidden_sizes"]) == (256, 256)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    model = ActorCritic(68 * 32, 16, (256, 256)).to(device)
    model.load_state_dict(data["model"], strict=True)
    assert model.angular_velocity is not None and model.flat_actor_bias is None
    model.eval()
    rng = np.random.default_rng(args.seed)
    gains = np.clip(rng.normal(1., .01, size=(args.candidates, 2)), .97, 1.03).astype(np.float32)
    decays = np.clip(rng.normal(1., .001, size=args.candidates), .997, 1.003).astype(np.float32)
    gains[0] = 1
    decays[0] = 1
    history_factors = np.power(decays[:, None], np.arange(31, -1, -1, dtype=np.float32)[None, :]).astype(np.float32)
    history = torch.as_tensor(history_factors, device=device)
    gx = torch.as_tensor(gains[:, 0], device=device)
    gy = torch.as_tensor(gains[:, 1], device=device)
    vx_coef = history * (1 / gx[:, None] - 1)
    vy_coef = history * (1 / gy[:, None] - 1)
    weights = model.net[0].weight.detach().reshape(256, 32, 68).permute(1, 2, 0).contiguous()
    env = VectorRunningGameEnv(args.candidates, num_threads=args.threads, reward_profile="classic_100m", game_mode="classic_100m")
    raw, _ = env.reset()
    stacker = FrameStacker(args.candidates, 68, 32)
    obs = stacker.reset(raw)
    with torch.no_grad():
        unit_x = torch.as_tensor(obs[:4], device=device)
        approximate = adjusted_hidden(model, unit_x, weights, history[:4], vx_coef[:4], vy_coef[:4])
        assert torch.equal(approximate[0], model.first_hidden(unit_x)[0])
        unit_max_difference = 0.
        for index in range(min(4, args.candidates)):
            folded = ActorCritic(68 * 32, 16, (256, 256)).to(device)
            folded.load_state_dict(fold_state(data["model"], gains[index], history_factors[index]), strict=True)
            expected = folded.first_hidden(unit_x[index:index + 1])[0]
            difference = float((expected - approximate[index]).abs().max())
            unit_max_difference = max(unit_max_difference, difference)
            torch.testing.assert_close(expected, approximate[index], atol=5e-5, rtol=5e-6)
            assert torch.equal(folded.angular_velocity.weight.cpu(), data["model"]["angular_velocity.weight"])
        del folded
    # The earlier zero-bias diagnostic used this unchanged source NN and raw mask argmax.
    baseline_report = json.loads((BASE / "verification/target43_gated_actor_bias.json").read_text(encoding="utf-8"))
    assert baseline_report["best"]["candidate_index"] == 0
    _, baseline_actions = read_replay(BASE / "verification/target43_gated_actor_bias_diagnostic.bin")
    active = np.ones(args.candidates, dtype=bool)
    low = np.zeros(args.candidates, dtype=np.int64)
    backward = np.zeros(args.candidates, dtype=np.int64)
    count = np.zeros(args.candidates, dtype=np.int64)
    marks = (10, 40, 50, 60, 70, 100)
    crossing = np.full((args.candidates, len(marks)), np.nan)
    action_history = np.empty((4000, args.candidates), dtype=np.uint8)
    records = []
    baseline_sequence_exact = True
    for step in range(1, 4001):
        low[active] += ((429.8 - raw[:, 26] * 40) / 506.85 < .8)[active]
        lean = np.rad2deg(np.arctan2(raw[:, 50] - raw[:, 25], raw[:, 26] - raw[:, 51]))
        measured = active & (raw[:, 61] >= 5)
        backward[measured] += (lean < -10)[measured]
        count[measured] += 1
        with torch.no_grad():
            x = torch.as_tensor(obs, device=device)
            hidden = adjusted_hidden(model, x, weights, history, vx_coef, vy_coef)
            for layer in list(model.net)[1:]:
                hidden = layer(hidden)
            actions = model.actor(hidden).argmax(-1).cpu().numpy()
        if active[0]:
            baseline_sequence_exact &= step <= len(baseline_actions) and int(actions[0]) == baseline_actions[step - 1]
        action_history[step - 1] = actions
        result = env.step_arrays(actions)
        assert np.isfinite(result["obs"]).all() and np.isfinite(result["reward"]).all()
        for column, mark in enumerate(marks):
            hit = active & np.isnan(crossing[:, column]) & (result["score"] >= mark)
            crossing[hit, column] = result["time"][hit]
        ended = result["has_final"]
        for index in np.flatnonzero(active & ended):
            times = {str(mark): float(crossing[index, column]) if np.isfinite(crossing[index, column]) else None
                     for column, mark in enumerate(marks)}

            def split(a, b):
                return times[str(b)] - times[str(a)] if times[str(a)] is not None and times[str(b)] is not None else None

            record = {"candidate_index": int(index), "vx_gain": float(gains[index, 0]), "vy_gain": float(gains[index, 1]),
                      "history_decay": float(decays[index]), "success": bool(result["final_success"][index]),
                      "truncated": bool(result["truncated"][index]), "time_s": float(result["final_time"][index]),
                      "distance_m": float(result["final_distance"][index]), "steps": step,
                      "low_posture_fraction": float(low[index] / step),
                      "backward_fraction_after5s": float(backward[index] / count[index]) if count[index] else 0.,
                      "crossing_times_s": times,
                      "splits_s": {"0_10": times["10"], "40_50": split(40, 50), "50_60": split(50, 60), "60_70": split(60, 70)}}
            record["qualified"] = record["success"] and not record["truncated"] and record["low_posture_fraction"] <= .1 and record["backward_fraction_after5s"] <= .2
            records.append(record)
        active[ended] = False
        raw = result["obs"]
        obs = stacker.update(raw, ended)
        if step % 300 == 0:
            print(json.dumps({"step": step, "completed": len(records)}), flush=True)
        if not active.any():
            break
    assert not active.any()
    baseline = next(record for record in records if record["candidate_index"] == 0)
    assert baseline["qualified"] and abs(baseline["time_s"] - 44.93333435058594) < 1e-5
    assert baseline["steps"] == len(baseline_actions) and baseline_sequence_exact
    qualified = [record for record in records if record["qualified"]]
    best = min(qualified, key=lambda record: record["time_s"])
    best_index = best["candidate_index"]
    best_state = fold_state(data["model"], gains[best_index], history_factors[best_index])
    for key, value in data["model"].items():
        if key != "net.0.weight":
            assert torch.equal(value, best_state[key]), key
    exported_model = ActorCritic(68 * 32, 16, (256, 256))
    exported_model.load_state_dict(best_state, strict=True)
    optimizer = torch.optim.Adam(exported_model.parameters(), lr=float(cfg["learning_rate"]), eps=1e-5)
    core = {"model", "optimizer", "config", "update", "global_step"}
    exported = {key: value for key, value in data.items() if key in core}
    exported.update(model=best_state, optimizer=optimizer.state_dict(), config=dict(cfg))
    exported["config"].update(checkpoint=str(args.save_checkpoint), reset_optimizer_on_resume=True)
    exported["structured_history_kinematics_refinement"] = {
        "source_checkpoint": str(source), "source_sha256": source_sha, "seed": args.seed,
        "candidate_count": args.candidates, "selected": best, "history_factors": history_factors[best_index].tolist(),
        "method": "Scale each 68-value first-layer history block by decay**(31-index), then divide its twelve vx/vy weight columns by gains",
        "all_other_source_state_preserved": True, "optimizer": "Fresh empty Adam after strict complete model restore",
        "ancestor_metadata": {key: value for key, value in data.items() if key not in core},
    }
    temp = BASE / "verification/target43_structured_history_kinematics_strict_candidate.pt"
    torch.save(exported, temp)
    trace = BASE / "verification/target43_structured_history_kinematics_strict_trace.csv"
    strict = evaluate_checkpoint(temp, trace=trace, device="cuda")
    rows = np.loadtxt(trace, delimiter=",", skiprows=1)
    after5 = rows[:, 1] >= 5
    strict_backward = float((rows[after5, 5] < -10).mean()) if after5.any() else 0.
    strict["backward_fraction_after5s"] = strict_backward
    exported["structured_history_kinematics_refinement"]["strict_reload_evaluation"] = strict
    saved = None
    if strict["success"] and not strict["truncated"] and not strict["native_failed"] and strict["whole_episode_low_posture_fraction"] <= .1 and strict_backward <= .2 and strict["time_s"] < baseline["time_s"] - 1e-5:
        torch.save(exported, args.save_checkpoint)
        final = evaluate_checkpoint(args.save_checkpoint, device="cuda")
        assert final["success"] and not final["truncated"] and not final["native_failed"]
        assert abs(final["time_s"] - strict["time_s"]) < 1e-5
        saved = str(args.save_checkpoint)
    selected_actions = action_history[:best["steps"], best_index].tolist()
    write_replay(args.save_replay, selected_actions, ruleset_for_game_mode("classic_100m"))
    replay_env = RunningGameEnv("classic_100m", "classic_100m")
    replay_env.reset()
    for replay_step, action in enumerate(selected_actions, start=1):
        replay_obs, reward, done, truncated, info = replay_env.step(action)
        assert np.isfinite(replay_obs).all() and np.isfinite(reward)
        if done or truncated:
            assert replay_step == len(selected_actions)
    assert info["success"] and not truncated and abs(info["time"] - best["time_s"]) < 1e-5
    report = {"source_checkpoint": str(source), "source_sha256": source_sha, "seed": args.seed,
              "candidate_count": args.candidates, "sampling": "Draw normal(1,.01) Nx2 vx/vy gains clipped [.97,1.03], then normal(1,.001) N decay clipped [.997,1.003]; candidate0 identity",
              "memory_method": "Original first_hidden plus blockwise first-layer contribution corrections; no per-candidate 256x2176 weight population",
              "folded_hidden_unit_max_difference": unit_max_difference, "baseline_complete_action_sequence_exact": bool(baseline_sequence_exact),
              "baseline": baseline, "qualified_count": len(qualified), "best_search": best,
              "strict_folded_model_evaluation": strict, "search_reload_time_difference_s": strict["time_s"] - best["time_s"],
              "saved_checkpoint": saved, "beats_baseline_with_strict_saved_model": saved is not None,
              "under43_with_strict_saved_model": saved is not None and strict["time_s"] < 43,
              "diagnostic_replay": str(args.save_replay),
              "diagnostic_replay_validation": {"success_on_last_action": bool(info["success"]), "steps": len(selected_actions), "time_s": float(info["time"])},
              "records": records, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.perf_counter() - started}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("candidate_count", "qualified_count", "baseline", "best_search", "strict_folded_model_evaluation", "search_reload_time_difference_s", "saved_checkpoint", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
