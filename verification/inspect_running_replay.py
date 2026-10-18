"""Replay original actions and measure pace/posture in each 10 m segment."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1] / "QWOP_rl_training_public"
sys.path.insert(0, str(PROJECT / "scripts"))
from replay_io import read_replay, ReplayRuleset
from running_env import RunningGameEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = args.replay if args.replay.is_absolute() else PROJECT / args.replay
    ruleset, actions = read_replay(replay)
    assert ruleset == ReplayRuleset.CLASSIC_100M
    env = RunningGameEnv(reward_profile="classic_100m", game_mode="classic_100m")
    obs, info = env.reset()
    trace = []
    crossings = {0: 0.0}
    for i, action in enumerate(actions):
        obs, reward, done, truncated, info = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert 0 <= action <= 15
        lean = float(np.rad2deg(np.arctan2(obs[50] - obs[25], obs[26] - obs[51])))
        height = float((429.8 - obs[26] * 40.0) / 506.85)
        trace.append([float(info["distance"]), float(info["time"]), lean, height])
        for mark in range(10, 101, 10):
            if mark not in crossings and info["distance"] >= mark:
                crossings[mark] = float(info["time"])
        assert not (done or truncated) or i == len(actions) - 1, "Replay contains actions after termination"
    assert done and not truncated and info["success"], "Replay does not finish successfully"
    rows = np.asarray(trace)
    segments = []
    for mark in range(10, 101, 10):
        selected = rows[(rows[:, 0] >= mark - 10) & (rows[:, 0] < mark)]
        duration = crossings[mark] - crossings[mark - 10]
        segments.append({"from_m": mark - 10, "to_m": mark, "seconds": duration,
                         "speed_m_per_s": 10 / duration,
                         "lean_mean_degrees": float(selected[:, 2].mean()),
                         "lean_error_from_5": float(np.abs(selected[:, 2] - 5).mean()),
                         "torso_height_std": float(selected[:, 3].std())})
    report = {"replay": str(replay), "action_count": len(actions), "success": True,
              "time": float(info["time"]), "distance": float(info["distance"]),
              "clock_matches_action_count": abs(info["time"] - len(actions) / 30) < 1e-4,
              "segments": segments, "height_measurement": "torso proxy, not mass-weighted COM"}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
