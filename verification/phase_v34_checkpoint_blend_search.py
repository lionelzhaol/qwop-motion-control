"""Small full-network interpolation experiment; native race rules unchanged."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from evaluate_gait_bottlenecks import evaluate_checkpoint, compact_summary
from train_ppo import ActorCritic


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "QWOP_rl_training_public"
OUTPUT = BASE / "verification/phase_v34_checkpoint_blend_search.json"
TEMP = BASE / "verification/phase_v34_blend_candidate.pt"
TARGET = ROOT / "checkpoints/ppo_phase_v34_checkpoint_blend.pt"
SOURCE_A = ROOT / "checkpoints/ppo_phase_v32_hidden2_baseline.pt"
SOURCES_B = [ROOT / "checkpoints" / name for name in (
    "ppo_phase_v30_hurdle_update_011424.pt",
    "ppo_phase_v31_start_update_011392.pt",
    "ppo_phase_v32_start_explore_update_011280.pt",
)]
ALPHAS = [.02, .05, .1, .15, .2, .3, .4, .5, .65, .8]
CORE_KEYS = {"model", "optimizer", "config", "update", "global_step"}


def loaded_model(data):
    cfg = data["config"]
    model = ActorCritic(68 * cfg["frame_stack"], 16, tuple(cfg["hidden_sizes"]))
    model.load_state_dict(data["model"], strict=True)
    return model


def ancestry(data):
    return {key: value for key, value in data.items() if key not in CORE_KEYS}


def main():
    started = time.perf_counter()
    data_a = torch.load(SOURCE_A, map_location="cpu")
    model_a = loaded_model(data_a)
    parameter_keys = set(dict(model_a.named_parameters()))
    assert "angular_velocity.weight" in parameter_keys
    assert "observation_scale" not in parameter_keys
    baseline = evaluate_checkpoint(SOURCE_A, device="cuda")
    assert baseline["success"] and not baseline["native_failed"] and not baseline["truncated"]
    assert baseline["whole_episode_low_posture_fraction"] <= .1
    assert abs(baseline["time_s"] - 44.9333333333) < 1e-4, baseline["time_s"]
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Full NN parameter interpolation (1-alpha)*A+alpha*B; no scripted actions",
        "source_a": str(SOURCE_A),
        "sources_b": [str(path) for path in SOURCES_B],
        "alphas": ALPHAS,
        "parameter_keys": sorted(parameter_keys),
        "observation_scale": "Copied from A; exact equality checked against every B",
        "selection": "Original native success, not failed/truncated; whole episode low-posture fraction <=0.1; smallest original final game time",
        "baseline": baseline,
        "records": [],
        "best": None,
        "saved_checkpoint": None,
    }
    best_time = baseline["time_s"]
    best_state = None
    best_b_data = None
    for source_b in SOURCES_B:
        data_b = torch.load(source_b, map_location="cpu")
        model_b = loaded_model(data_b)
        assert set(dict(model_b.named_parameters())) == parameter_keys
        assert set(data_a["model"]) == set(data_b["model"])
        assert torch.equal(data_a["model"]["observation_scale"], data_b["model"]["observation_scale"])
        for key, a in data_a["model"].items():
            b = data_b["model"][key]
            assert a.shape == b.shape and a.dtype == b.dtype, key
        for alpha in ALPHAS:
            state = {key: ((1 - alpha) * value + alpha * data_b["model"][key]
                           if key in parameter_keys else value.clone())
                     for key, value in data_a["model"].items()}
            candidate = {key: value for key, value in data_a.items() if key in CORE_KEYS and key != "optimizer"}
            candidate["model"] = state
            torch.save(candidate, TEMP)
            result = evaluate_checkpoint(TEMP, device="cuda")
            qualified = result["success"] and not result["native_failed"] and not result["truncated"] and result["whole_episode_low_posture_fraction"] <= .1
            record = {"source_b": str(source_b), "alpha": alpha, "qualified": bool(qualified), "evaluation": result}
            report["records"].append(record)
            if qualified and result["time_s"] < best_time - 1e-5:
                best_time = result["time_s"]
                report["best"] = record
                best_state = {key: value.clone() for key, value in state.items()}
                best_b_data = data_b
            OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"source_b": source_b.name, "alpha": alpha, "qualified": qualified,
                              "summary": compact_summary(result)}, ensure_ascii=False), flush=True)
    if best_state is not None:
        final = {key: value for key, value in data_a.items() if key in CORE_KEYS and key != "optimizer"}
        final["model"] = best_state
        final["config"] = dict(data_a["config"])
        final["config"]["checkpoint"] = str(TARGET)
        final["config"]["reset_optimizer_on_resume"] = True
        final_model = loaded_model(final)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=float(final["config"]["learning_rate"]), eps=1e-5)
        final["optimizer"] = optimizer.state_dict()
        assert not final["optimizer"]["state"]
        final["phase_v34_refinement"] = {
            "method": report["method"], "source_a": str(SOURCE_A),
            "source_b": report["best"]["source_b"], "alpha": report["best"]["alpha"],
            "parameter_keys": sorted(parameter_keys), "baseline_game_seconds": baseline["time_s"],
            "selected": report["best"]["evaluation"], "optimizer": "fresh empty Adam created after strict model restore",
            "ancestors": {"source_a": ancestry(data_a), "source_b": ancestry(best_b_data)},
        }
        torch.save(final, TARGET)
        verified = evaluate_checkpoint(TARGET, device="cuda")
        assert verified["success"] and not verified["native_failed"] and not verified["truncated"]
        assert verified["whole_episode_low_posture_fraction"] <= .1
        assert abs(verified["time_s"] - best_time) < 1e-5
        assert verified["time_s"] < baseline["time_s"] - 1e-5
        report["saved_checkpoint"] = str(TARGET)
        report["saved_checkpoint_verified"] = verified
    report["elapsed_seconds"] = time.perf_counter() - started
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": compact_summary(baseline),
                      "best": compact_summary(report["best"]["evaluation"]) if report["best"] else None,
                      "saved_checkpoint": report["saved_checkpoint"],
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
