"""Diagnose physical-class argmax without changing or saving NN checkpoints."""
from __future__ import annotations

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
from replay_io import write_replay, read_replay, ruleset_for_game_mode

GROUPS = ([0], [1, 3], [2], [4, 12], [5, 7, 13, 15], [6, 14], [8], [9, 11], [10])
CANONICAL = (0, 1, 2, 4, 5, 6, 8, 9, 10)
BETAS = (0., .1, .25, .5, .75, 1.)
SOURCES = {
    "fast44p93": ("ppo_classic100m_phase_fast.pt", 44.93333435058594),
    "ppo45p40": ("ppo_classic100m_phase_ppo.pt", 45.400001525878906),
    "pose47p83": ("ppo_classic100m_target45_pose.pt", 47.83333206176758),
}
OUTPUT = BASE / "verification/target43_effective_class_argmax.json"
REPLAY = BASE / "verification/target43_effective_class_argmax_diagnostic.bin"


def grouped_logits(logits):
    maxima = torch.stack([logits[..., group].max(-1).values for group in GROUPS], -1)
    totals = torch.stack([torch.logsumexp(logits[..., group], -1) for group in GROUPS], -1)
    return maxima, totals


def main():
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(1)
    device = torch.device("cuda")
    assert sorted(sum((list(group) for group in GROUPS), [])) == list(range(16))
    generator = torch.Generator().manual_seed(194302)
    test_logits = torch.randn(128, 16, generator=generator)
    maximum, total = grouped_logits(test_logits)
    physical_probabilities = torch.stack([test_logits.softmax(-1)[:, group].sum(-1) for group in GROUPS], -1)
    assert torch.equal(total.argmax(-1), physical_probabilities.argmax(-1))
    mask_to_class = torch.empty(16, dtype=torch.long)
    for group_index, group in enumerate(GROUPS):
        mask_to_class[list(group)] = group_index
    assert torch.equal(maximum.argmax(-1), mask_to_class[test_logits.argmax(-1)])
    models, metadata = {}, {}
    for name, (filename, expected_time) in SOURCES.items():
        path = ROOT / "checkpoints" / filename
        data = torch.load(path, map_location=device)
        cfg = data["config"]
        assert cfg["frame_stack"] == 32 and cfg["game_mode"] == "classic_100m"
        model = ActorCritic(68 * 32, 16, tuple(cfg["hidden_sizes"])).to(device)
        model.load_state_dict(data["model"], strict=True)
        model.eval()
        models[name] = model
        metadata[name] = {"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "expected_raw_argmax_time_s": expected_time, "update": data.get("update"),
                          "frame_stack": cfg["frame_stack"], "hidden_sizes": cfg["hidden_sizes"]}
    candidates = []
    for name in SOURCES:
        candidates.extend({"source": name, "selector": "effective_class_argmax", "beta": beta} for beta in BETAS)
        candidates.append({"source": name, "selector": "raw_mask_argmax", "beta": None})
    n = len(candidates)
    env = VectorRunningGameEnv(n, num_threads=16, reward_profile="classic_100m", game_mode="classic_100m")
    raw, _ = env.reset()
    stacker = FrameStacker(n, 68, 32)
    obs = stacker.reset(raw)
    active = np.ones(n, dtype=bool)
    low = np.zeros(n, dtype=np.int64)
    backward = np.zeros(n, dtype=np.int64)
    measured_count = np.zeros(n, dtype=np.int64)
    marks = (10, 40, 50, 60, 70, 100)
    crossing = np.full((n, len(marks)), np.nan)
    histories = np.empty((4000, n), dtype=np.uint8)
    alias_max_difference = {name: 0. for name in SOURCES}
    canonical_tensor = torch.as_tensor(CANONICAL, device=device)
    betas_tensor = torch.as_tensor(BETAS, dtype=torch.float32, device=device)[:, None]
    records = []
    for step in range(1, 4001):
        low[active] += ((429.8 - raw[:, 26] * 40) / 506.85 < .8)[active]
        lean = np.rad2deg(np.arctan2(raw[:, 50] - raw[:, 25], raw[:, 26] - raw[:, 51]))
        measured = active & (raw[:, 61] >= 5)
        backward[measured] += (lean < -10)[measured]
        measured_count[measured] += 1
        actions = np.zeros(n, dtype=np.int64)
        with torch.no_grad():
            for source_index, name in enumerate(SOURCES):
                offset = source_index * 7
                logits, _ = models[name](torch.as_tensor(obs[offset:offset + 7], device=device))
                maxima, totals = grouped_logits(logits[:6])
                class_logits = (1 - betas_tensor) * maxima + betas_tensor * totals
                actions[offset:offset + 6] = canonical_tensor[class_logits.argmax(-1)].cpu().numpy()
                actions[offset + 6] = int(logits[6].argmax(-1).item())
        histories[step - 1] = actions
        result = env.step_arrays(actions)
        assert np.isfinite(result["obs"]).all() and np.isfinite(result["reward"]).all()
        for source_index, name in enumerate(SOURCES):
            offset = source_index * 7
            if active[offset] or active[offset + 6]:
                difference = float(np.abs(result["obs"][offset] - result["obs"][offset + 6]).max())
                alias_max_difference[name] = max(alias_max_difference[name], difference)
                assert difference == 0, f"Beta zero and raw argmax physical states differ for {name}"
                assert result["has_final"][offset] == result["has_final"][offset + 6]
                assert result["score"][offset] == result["score"][offset + 6]
                assert result["time"][offset] == result["time"][offset + 6]
        for column, mark in enumerate(marks):
            newly_reached = active & np.isnan(crossing[:, column]) & (result["score"] >= mark)
            crossing[newly_reached, column] = result["time"][newly_reached]
        ended = result["has_final"]
        for candidate_index in np.flatnonzero(active & ended):
            times = {str(mark): float(crossing[candidate_index, column]) if np.isfinite(crossing[candidate_index, column]) else None
                     for column, mark in enumerate(marks)}

            def split(first, last):
                a, b = times[str(first)], times[str(last)]
                return b - a if a is not None and b is not None else None

            record = dict(candidates[candidate_index])
            record.update(candidate_index=int(candidate_index), success=bool(result["final_success"][candidate_index]),
                          truncated=bool(result["truncated"][candidate_index]), steps=step,
                          time_s=float(result["final_time"][candidate_index]),
                          distance_m=float(result["final_distance"][candidate_index]),
                          low_posture_fraction=float(low[candidate_index] / step),
                          backward_fraction_after5s=float(backward[candidate_index] / measured_count[candidate_index]) if measured_count[candidate_index] else 0.,
                          crossing_times_s=times,
                          splits_s={"0_10": times["10"], "40_50": split(40, 50), "50_60": split(50, 60), "60_70": split(60, 70)})
            record["qualified"] = record["success"] and not record["truncated"] and record["low_posture_fraction"] <= .1
            records.append(record)
        active[ended] = False
        raw = result["obs"]
        obs = stacker.update(raw, ended)
        if not active.any():
            break
    assert not active.any()
    for name, (_, expected_time) in SOURCES.items():
        baseline = next(record for record in records if record["source"] == name and record["selector"] == "raw_mask_argmax")
        beta_zero = next(record for record in records if record["source"] == name and record["beta"] == 0)
        assert baseline["qualified"] and beta_zero["qualified"]
        assert abs(baseline["time_s"] - expected_time) < 1e-5
        assert baseline["time_s"] == beta_zero["time_s"]
    qualified = [record for record in records if record["qualified"] and record["selector"] == "effective_class_argmax"]
    best = min(qualified, key=lambda record: record["time_s"])
    best_actions = histories[:best["steps"], best["candidate_index"]].tolist()
    write_replay(REPLAY, best_actions, ruleset_for_game_mode("classic_100m"))
    ruleset, loaded_actions = read_replay(REPLAY)
    assert loaded_actions == best_actions
    replay_env = RunningGameEnv("classic_100m", "classic_100m")
    replay_env.reset()
    for step, action in enumerate(loaded_actions, start=1):
        replay_obs, reward, done, truncated, info = replay_env.step(action)
        assert np.isfinite(replay_obs).all() and np.isfinite(reward)
        if done or truncated:
            assert step == len(loaded_actions)
    assert done and info["success"] and not truncated
    assert abs(info["time"] - best["time_s"]) < 1e-5
    report = {"method": "Physical-class argmax diagnosis: class_logit=(1-beta)*max(mask_logits)+beta*logsumexp(mask_logits)",
              "sources": metadata, "groups": GROUPS, "canonical_masks": CANONICAL, "betas": BETAS,
              "candidate_count": 18, "additional_raw_argmax_references": 3,
              "beta_zero_physical_observation_max_difference": alias_max_difference,
              "beta_one_matches_physical_probability_argmax_unit_check": True,
              "records": sorted(records, key=lambda record: record["candidate_index"]),
              "best": best, "beats_fast_baseline": best["time_s"] < SOURCES["fast44p93"][1] - 1e-5,
              "under43": best["time_s"] < 43, "diagnostic_replay": str(REPLAY),
              "replay_validation": {"ruleset": int(ruleset), "success_on_last_action": bool(info["success"]),
                                    "actions": len(loaded_actions), "time_s": float(info["time"]), "distance_m": float(info["distance"])},
              "loadable_nn_checkpoint_created": False,
              "notes": ["No source NN or repository code changed; this output selector has not been serialized into a loadable model.",
                        "Qualification uses original native final success, no truncation, and whole-episode low-posture fraction <=0.1.",
                        "Backward fraction is recorded separately without an additional selection filter.",
                        "Beta zero is compared with raw 16-mask argmax in three extra native reference environments, frame by frame."],
              "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.perf_counter() - started}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": report["records"], "best": best, "beats_fast_baseline": report["beats_fast_baseline"],
                      "under43": report["under43"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
