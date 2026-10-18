"""Native jump telemetry: separate the 30m approach from landing distance."""
from pathlib import Path
import argparse
import ctypes as C
import json
import sys
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / 'QWOP_rl_training_public'
sys.path.insert(0, str(ROOT / 'scripts'))
from running_env import RunningGameEnv
from train_ppo import ActorCritic, FrameStacker
from evaluate_gait_bottlenecks import Frame, Course

def evaluate(checkpoint, stochastic=False, seed=20261010):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    checkpoint = Path(checkpoint)
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = data['config']
    env = RunningGameEnv('long_jump_30m', 'long_jump_30m')
    raw, _ = env.reset()
    stack = FrameStacker(1, 68, cfg['frame_stack'])
    obs = stack.reset(raw[None])
    model = ActorCritic(stack.obs_dim, 16, tuple(cfg['hidden_sizes']))
    model.load_state_dict(data['model'], strict=True)
    model.eval()
    dll = C.CDLL(str(ROOT / 'bin/running_physics.dll'))
    dll.rg_create.argtypes = [C.POINTER(C.c_void_p)]; dll.rg_create.restype = C.c_int
    dll.rg_reset.argtypes = [C.c_void_p, C.POINTER(Course)]; dll.rg_reset.restype = C.c_int
    dll.rg_step.argtypes = [C.c_void_p, C.c_uint8]; dll.rg_step.restype = C.c_int
    dll.rg_read_frame.argtypes = [C.c_void_p, C.POINTER(Frame)]; dll.rg_read_frame.restype = C.c_int
    dll.rg_destroy.argtypes = [C.c_void_p]
    handle = C.c_void_p(); assert dll.rg_create(C.byref(handle)) == 0
    course = Course(C.sizeof(Course), 2); assert dll.rg_reset(handle, C.byref(course)) == 0
    frame = Frame(); frame.struct_size = C.sizeof(Frame)
    actions, rewards, events = [], [], []
    approach_max = 0.; seen_landing = False; seen_jumped = False
    landing_torso_x = None; post_landing_max_x = None
    post_landing_actions = {}; post_landing_min_height = 10.
    try:
        for step in range(1, 1351):
            with torch.no_grad():
                logits, value = model(torch.as_tensor(obs, dtype=torch.float32))
                assert torch.isfinite(logits).all() and torch.isfinite(value).all()
                action = int(torch.distributions.Categorical(logits=logits).sample()[0]) if stochastic else int(logits.argmax(-1)[0])
            raw, reward, done, truncated, info = env.step(action)
            assert dll.rg_step(handle, action) == 0 and dll.rg_read_frame(handle, C.byref(frame)) == 0
            assert abs(info['time'] - frame.sim_time) < 1e-5
            expected_distance = 30 + frame.long_jump_distance if frame.long_jump_landing_started else frame.score
            assert abs(info['distance'] - expected_distance) < 1e-5
            assert abs(info['score'] - frame.score) < 1e-5
            assert abs(raw[26] * 40 - frame.bodies[5].y) < .001
            assert np.isfinite(raw).all() and np.isfinite(reward)
            actions.append(action); rewards.append(float(reward))
            if not frame.long_jump_landing_started:
                approach_max = max(approach_max, min(30., frame.view_target_x / 400))
            if frame.jumped and not seen_jumped:
                events.append(dict(event='jumped', time_s=frame.sim_time, torso_x_m=frame.bodies[5].x / 400))
                seen_jumped = True
            if frame.long_jump_landing_started and not seen_landing:
                events.append(dict(event='landing_started', time_s=frame.sim_time, jump_m=frame.long_jump_distance))
                seen_landing = True
                landing_torso_x = frame.bodies[5].x / 400
                post_landing_max_x = landing_torso_x
            if seen_landing:
                post_landing_max_x = max(post_landing_max_x, frame.bodies[5].x / 400)
                post_landing_min_height = min(post_landing_min_height, (429.8-frame.bodies[5].y)/506.85)
                post_landing_actions[action] = post_landing_actions.get(action, 0) + 1
            if done or truncated:
                break
            obs = stack.update(raw[None], np.array([False]))
        assert done or truncated
        report = dict(telemetry_version=2, checkpoint=str(checkpoint), checkpoint_game_mode=cfg.get('game_mode'),
            evaluation_game_mode='long_jump_30m', update=data.get('update'), stochastic=stochastic,
            seed=seed, success=bool(info['success']), failed=bool(frame.failed), truncated=bool(truncated),
            time_s=float(frame.sim_time), steps=len(actions), approach_progress_m=approach_max,
            landing_seen=bool(frame.long_jump_landing_started), exited_sand=bool(frame.long_jump_exited_sand),
            long_jump_m=float(frame.long_jump_distance) if frame.long_jump_landing_started else None,
            backend_score=float(frame.score), total_distance_m=float(info['distance']),
            episode_reward=sum(rewards), events=events, native_trajectory_verified=True,
            post_landing_progress_m=post_landing_max_x-landing_torso_x if seen_landing else None,
            final_torso_x_m=frame.bodies[5].x / 400,
            post_landing_min_height_ratio=post_landing_min_height if seen_landing else None,
            post_landing_action_counts=post_landing_actions,
            actions=actions)
        return report
    finally:
        dll.rg_destroy(handle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.stochastic)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='actions'}, ensure_ascii=False))
