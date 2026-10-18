"""Verify explicitly scripted diagnostic trajectories, never save NN weights."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "QWOP_rl_training_public"
sys.path.insert(0, str(ROOT / "scripts"))
from running_env import RunningGameEnv
from train_ppo import ActorCritic, FrameStacker
from replay_io import read_replay

REPORT = BASE / "verification/target43_startup_teacher_diagnostic.json"
OUTPUT = BASE / "verification/target43_startup_teacher_diagnostic_validation.json"
REPLAY = BASE / "verification/target43_startup_teacher_diagnostic.bin"


def main():
    started = time.perf_counter()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    source = Path(report["source"])
    source = source if source.is_absolute() else ROOT / source
    data = torch.load(source, map_location="cuda")
    cfg = data["config"]
    model = ActorCritic(68 * cfg["frame_stack"], 16, tuple(cfg["hidden_sizes"])).cuda()
    model.load_state_dict(data["model"], strict=True)
    model.eval()
    torch.set_num_threads(1)
    qualified = [record for record in report["records"] if record["success"]
                 and record["low_posture_fraction"] <= .1 and record["backward_fraction"] <= .2]
    selected = sorted(qualified, key=lambda r: r["time"])[:8]
    baseline = next(record for record in report["records"] if record["candidate_index"] == 0)
    if not any(record["candidate_index"] == 0 for record in selected):
        selected.insert(0, baseline)
    records = []
    for candidate in selected:
        env = RunningGameEnv("classic_100m", "classic_100m")
        obs, _ = env.reset()
        stacker = FrameStacker(1, 68, cfg["frame_stack"])
        stacked = stacker.reset(obs[None, :])
        crossings = {}
        low = 0
        for step in range(1, 4001):
            low += int((429.8 - obs[26] * 40) / 506.85 < .8)
            if step <= candidate["start_first_frames"]:
                action = candidate["start_action"]
            elif step <= candidate["start_frames"]:
                action = candidate["start_second_action"]
            else:
                with torch.no_grad():
                    action = int(model(torch.as_tensor(stacked, device="cuda"))[0].argmax(-1).item())
            obs, reward, done, truncated, info = env.step(action)
            assert np.isfinite(obs).all() and np.isfinite(reward)
            for meter in (3, 10, 40, 50, 60, 100):
                if meter not in crossings and info["score"] >= meter:
                    crossings[meter] = float(info["time"])
            if done or truncated:
                break
            stacked = stacker.update(obs[None, :], np.array([False]))
        assert info["success"] and not truncated
        assert abs(info["time"] - candidate["time"]) < 1e-5
        result = {"candidate_index": candidate["candidate_index"],
                  "prefix": {"first_action": candidate["start_action"],
                             "first_frames": candidate["start_first_frames"],
                             "second_action": candidate["start_second_action"],
                             "second_frames": candidate["start_frames"] - candidate["start_first_frames"]},
                  "success": bool(info["success"]), "time_s": float(info["time"]),
                  "distance_m": float(info["distance"]), "steps": step,
                  "low_posture_fraction": low / step,
                  "crossing_times_s": {str(k): v for k, v in crossings.items()},
                  "split_50_60_s": crossings[60] - crossings[50] if 50 in crossings and 60 in crossings else None}
        records.append(result)
        print(json.dumps(result), flush=True)
    ruleset, actions = read_replay(REPLAY)
    env = RunningGameEnv("classic_100m", "classic_100m")
    env.reset()
    for step, action in enumerate(actions, start=1):
        obs, reward, done, truncated, info = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if done or truncated:
            assert step == len(actions), "Replay ended before its last action"
    assert info["success"] and not truncated and done
    assert abs(info["time"] - report["best"]["time"]) < 1e-5
    assert abs(info["time"] - len(actions) / 30) < 1e-5
    output = {"source_checkpoint": str(source), "diagnostic_only": True,
              "learned_weights_created": False, "prefix_grid_candidates": len(report["records"]),
              "qualified": len(qualified), "selected_for_split_validation": len(records),
              "records": records, "baseline_candidate": baseline, "best_candidate": report["best"],
              "replay": {"path": str(REPLAY), "ruleset": int(ruleset), "steps": len(actions),
                         "success_on_last_action": bool(info["success"]),
                         "time_s": float(info["time"]), "distance_m": float(info["distance"])},
              "elapsed_seconds": time.perf_counter() - started,
              "notes": ["Fixed action prefixes are only diagnostic teacher trajectories; they are not learned NN behavior.",
                        "Any final model must learn the trajectory and be evaluated by strict NN checkpoint restore without a prefix.",
                        "Split validation reruns only the eight fastest qualified whole-race trajectories plus baseline; it is not an exhaustive first-10m optimum search."]}
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"replay": output["replay"], "elapsed_seconds": output["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
