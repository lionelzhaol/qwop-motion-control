"""Exact C++ reward counterfactuals on fixed, verified native NN trajectories."""
from pathlib import Path
import ctypes as C
import hashlib
import json
import sys
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / "QWOP_rl_training_public"
sys.path.insert(0, str(ROOT / "scripts"))
from running_env import RunningGameEnv
from train_ppo import ActorCritic, FrameStacker
from evaluate_gait_bottlenecks import Frame, Course

POLICIES = {
    "fast44p93": "ppo_classic100m_phase_fast.pt",
    "ppo45p40": "ppo_classic100m_phase_ppo.pt",
    "pose47p83": "ppo_classic100m_target45_pose.pt",
    "v36_best46p07": "ppo_target43_v36_flat_residual_update_012528.pt",
    "v37_best46p90": "ppo_target43_v37_actor_features_update_012640.pt",
    "v38_best46p47": "ppo_target43_v38_soft_actor_update_010912.pt",
}
GAMMA = .9999
COMPONENTS = {"lean_progress_bonus": 1, "lean_target_penalty": 2,
              "vertical_motion_penalty": 3, "cadence_bonus": 4, "net_exact_lean_shaping": 5}


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    manifest = json.loads((BASE / "posture_audit_v39_build/manifest.json").read_text(encoding="utf-8"))
    names = manifest["output_order"]
    assert len(names) == 7
    physics = C.CDLL(str(ROOT / "bin/running_physics.dll"))
    physics.rg_create.argtypes = [C.POINTER(C.c_void_p)]; physics.rg_create.restype = C.c_int
    physics.rg_reset.argtypes = [C.c_void_p, C.POINTER(Course)]; physics.rg_reset.restype = C.c_int
    physics.rg_step.argtypes = [C.c_void_p, C.c_uint8]; physics.rg_step.restype = C.c_int
    physics.rg_read_frame.argtypes = [C.c_void_p, C.POINTER(Frame)]; physics.rg_read_frame.restype = C.c_int
    physics.rg_destroy.argtypes = [C.c_void_p]
    audit = C.CDLL(str(BASE / "posture_audit_v39_build/posture_audit_v39.dll"))
    audit.audit_create.restype = C.c_void_p
    audit.audit_destroy.argtypes = [C.c_void_p]
    audit.audit_reward.argtypes = [C.c_void_p, C.POINTER(Frame), C.c_uint8, C.POINTER(C.c_float)]
    policies = []
    for label, filename in POLICIES.items():
        checkpoint = ROOT / "checkpoints" / filename
        data = torch.load(checkpoint, map_location="cpu")
        cfg = data["config"]
        model = ActorCritic(68 * cfg["frame_stack"], 16, tuple(cfg["hidden_sizes"]))
        model.load_state_dict(data["model"], strict=True)
        model.eval()
        env = RunningGameEnv("classic_100m", "classic_100m")
        obs, _ = env.reset()
        stack = FrameStacker(1, 68, cfg["frame_stack"])
        stacked = stack.reset(obs[None, :])
        handle = C.c_void_p()
        assert physics.rg_create(C.byref(handle)) == 0
        course = Course(C.sizeof(Course), 0)
        assert physics.rg_reset(handle, C.byref(course)) == 0
        frame = Frame(); frame.struct_size = C.sizeof(Frame)
        assert physics.rg_read_frame(handle, C.byref(frame)) == 0
        state = audit.audit_create()
        out = (C.c_float * len(names))()
        audit.audit_reward(state, C.byref(frame), 0, out)  # Match discarded native reset reward.
        initial_reward = np.asarray(out, dtype=np.float64).copy()
        totals = np.zeros(len(names))
        discounted = np.zeros(len(names))
        cadence_events = 0
        max_reward_error = 0.
        observations = []
        actions = []
        try:
            for step in range(1, 4001):
                with torch.no_grad():
                    action = int(model(torch.as_tensor(stacked, dtype=torch.float32))[0].argmax(-1).item())
                actions.append(action)
                obs, native_reward, done, truncated, info = env.step(action)
                assert physics.rg_step(handle, action) == 0
                assert physics.rg_read_frame(handle, C.byref(frame)) == 0
                assert abs(info["time"] - frame.sim_time) < 1e-5
                assert abs(info["distance"] - frame.score) < 1e-5
                assert abs(obs[26] * 40 - frame.bodies[5].y) < 1e-3
                audit.audit_reward(state, C.byref(frame), int(truncated), out)
                rewards = np.asarray(out, dtype=np.float64)
                assert np.isfinite(rewards).all() and np.isfinite(obs).all()
                max_reward_error = max(max_reward_error, abs(rewards[0] - native_reward))
                assert max_reward_error < 2e-4, (label, step, max_reward_error)
                totals += rewards
                discounted += GAMMA ** (step - 1) * rewards
                cadence_events += rewards[0] - rewards[4] > 1e-4
                torso, head = frame.bodies[5], frame.bodies[10]
                height = (429.8 - torso.y) / 506.85
                lean = np.degrees(np.arctan2(head.x - torso.x, torso.y - head.y))
                observations.append((frame.sim_time, frame.score, height, lean))
                if done or truncated:
                    assert info["success"] and not truncated
                    break
                stacked = stack.update(obs[None, :], np.array([False]))
            else:
                raise RuntimeError("Original native episode did not terminate")
        finally:
            audit.audit_destroy(state); physics.rg_destroy(handle)
        telemetry = np.asarray(observations)
        flat = ((telemetry[:, 1] >= 10) & (telemetry[:, 1] < 45)) | ((telemetry[:, 1] >= 65) & (telemetry[:, 1] < 90))
        policy = {"label": label, "checkpoint": str(checkpoint),
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "time_s": info["time"], "distance_m": info["distance"], "success": bool(info["success"]),
                  "steps": step, "current_reward_matches_live_native_every_frame": True,
                  "max_native_reward_error": max_reward_error,
                  "current_return": float(totals[0]), "current_discounted_return": float(discounted[0]),
                  "component_returns": {name: float(totals[0] - totals[index]) for name, index in COMPONENTS.items()},
                  "component_discounted_returns": {name: float(discounted[0] - discounted[index]) for name, index in COMPONENTS.items()},
                  "counterfactual_returns": {name: float(totals[index]) for index, name in enumerate(names)},
                  "counterfactual_discounted_returns": {name: float(discounted[index]) for index, name in enumerate(names)},
                  "cadence_reward_paid_events": int(cadence_events),
                  "flat_height_ratio_std": float(telemetry[flat, 2].std()),
                  "flat_lean_mean_degrees": float(telemetry[flat, 3].mean()),
                  "flat_lean_absolute_error_from5": float(np.abs(telemetry[flat, 3] - 5).mean()),
                  "whole_low_posture_fraction": float((telemetry[:, 2] < .8).mean()),
                  "discarded_initial_rewards": dict(zip(names, initial_reward.tolist())),
                  "action_sha256": hashlib.sha256(bytes(actions)).hexdigest()}
        policies.append(policy)
        print(json.dumps({key: policy[key] for key in ("label", "time_s", "current_return", "component_returns", "counterfactual_returns")}), flush=True)
    conflicts = []
    speed_order = sorted(policies, key=lambda policy: policy["time_s"])
    for index, faster in enumerate(speed_order):
        for slower in speed_order[index + 1:]:
            for metric in ("counterfactual_returns", "counterfactual_discounted_returns"):
                for variant in names:
                    if slower[metric][variant] > faster[metric][variant]:
                        conflicts.append({"faster": faster["label"], "slower": slower["label"], "variant": variant,
                                          "metric": metric, "slower_reward_advantage": slower[metric][variant] - faster[metric][variant]})
    report = {"gamma": GAMMA, "compiled_manifest": manifest, "policies": policies,
              "fixed_trajectory_ordering_conflicts": conflicts, "elapsed_seconds": time.perf_counter() - started,
              "notes": ["Actual source C++ compiled into a separate verification DLL; no production source edit or live pyd rebuild.",
                        "Every original classic reward verified against the live compiled native environment on identical duplicated native frames.",
                        "Single coefficient changes preserve state transitions; differences isolate reward contributions, not consequences of retraining.",
                        "TaskOnly includes the new full task-profile objective; it is not an isolated single-coefficient ablation.",
                        "Counterfactual reward ranking is evidence about the objective, not proof that removing a term will improve learned performance."]}
    (BASE / "posture_reward_component_audit_v39.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"elapsed_seconds": report["elapsed_seconds"], "ordering_conflicts": conflicts}), flush=True)


if __name__ == "__main__":
    main()
