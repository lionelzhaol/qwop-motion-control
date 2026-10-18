"""Deterministic classic-race telemetry, independent of reward shaping.

Native world coordinates are replayed through a second, identical physics
instance and checked against the author's environment at every frame. The
fair flat-running mask is fixed at score 10--45 m and 65--90 m: it excludes
the start, obstacle approach/landing recovery, and finish manoeuvre.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
from pathlib import Path
import sys

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / 'QWOP_rl_training_public'
sys.path.insert(0, str(ROOT / 'scripts'))
from running_env import RunningGameEnv
from train_ppo import ActorCritic, FrameStacker


class Body(C.Structure):
    _fields_ = [(k, C.c_float) for k in ('x', 'y', 'angle_degrees', 'velocity_x', 'velocity_y')]


class Course(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('game_mode', C.c_uint32)]


class Frame(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('body_count', C.c_uint32), ('bodies', Body * 12)] + [(k, C.c_float) for k in ('score', 'sim_time', 'view_target_x', 'hurdle_location', 'long_jump_distance', 'long_jump_landing_body_x')] + [('active_hurdle_count', C.c_uint32)] + [(k, C.c_uint8) for k in ('failed', 'jumped', 'jump_landed', 'race_finished', 'left_foot_grounded', 'right_foot_grounded', 'long_jump_landing_started', 'long_jump_exited_sand')]


def contiguous(mask):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return list(zip(starts.tolist(), ends.tolist()))


def evaluate_checkpoint(checkpoint, *, trace=None, device=None):
    """Return one complete report without printing; safe to call in a loop.

    A native failed or time-limited episode returns success=False and null
    metrics for segments it did not reach, rather than rejecting the model.
    """
    checkpoint = Path(checkpoint)
    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    torch.set_num_threads(1)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    data = torch.load(checkpoint, map_location=device)
    cfg = data['config']
    reward_profile = cfg.get('reward_profile', 'classic_100m')
    if reward_profile not in ('classic_100m', 'classic_100m_task'):
        raise ValueError('This diagnostic only supports the original classic game and its task-reward ablation')
    env = RunningGameEnv(reward_profile, 'classic_100m')
    obs, _ = env.reset()
    stacker = FrameStacker(1, obs.size, cfg['frame_stack'])
    stacked = stacker.reset(obs[None, :])
    model = ActorCritic(stacker.obs_dim, 16, tuple(cfg['hidden_sizes'])).to(device)
    model.load_state_dict(data['model'], strict=True)
    model.eval()
    assert C.sizeof(Frame) == 284
    dll = C.CDLL(str(ROOT / 'bin/running_physics.dll'))
    dll.rg_create.argtypes = [C.POINTER(C.c_void_p)]; dll.rg_create.restype = C.c_int
    dll.rg_reset.argtypes = [C.c_void_p, C.POINTER(Course)]; dll.rg_reset.restype = C.c_int
    dll.rg_step.argtypes = [C.c_void_p, C.c_uint8]; dll.rg_step.restype = C.c_int
    dll.rg_read_frame.argtypes = [C.c_void_p, C.POINTER(Frame)]; dll.rg_read_frame.restype = C.c_int
    dll.rg_destroy.argtypes = [C.c_void_p]
    handle = C.c_void_p()
    assert dll.rg_create(C.byref(handle)) == 0
    course = Course(C.sizeof(Course), 0)
    assert dll.rg_reset(handle, C.byref(course)) == 0
    frame = Frame(); frame.struct_size = C.sizeof(Frame)
    assert dll.rg_read_frame(handle, C.byref(frame)) == 0
    x0 = frame.bodies[5].x
    rows = []
    try:
        for i in range(4000):
            with torch.no_grad():
                logits, _ = model(torch.as_tensor(stacked, dtype=torch.float32, device=device))
                action = int(logits.argmax(-1).item())
            obs, reward, done, truncated, info = env.step(action)
            assert dll.rg_step(handle, action) == 0
            assert dll.rg_read_frame(handle, C.byref(frame)) == 0
            assert abs(info['time'] - frame.sim_time) < 1e-5
            assert abs(info['distance'] - frame.score) < 1e-5
            assert abs(obs[26] * 40 - frame.bodies[5].y) < 1e-3
            torso, head = frame.bodies[5], frame.bodies[10]
            lean = np.degrees(np.arctan2(head.x - torso.x, torso.y - head.y))
            rows.append([i + 1, frame.sim_time, frame.score, (torso.x - x0) / 400,
                         (429.8 - torso.y) / 506.85, lean, frame.left_foot_grounded,
                         frame.right_foot_grounded, (frame.hurdle_location - torso.x) / 400,
                         action, frame.active_hurdle_count, frame.failed])
            assert np.isfinite(obs).all() and np.isfinite(reward)
            if done or truncated:
                break
            stacked = stacker.update(obs[None, :], np.array([False]))
        else:
            raise RuntimeError('Episode exceeded original 120 s limit')
    finally:
        dll.rg_destroy(handle)
    a = np.asarray(rows, dtype=np.float64)
    t, score, x, height, lean = (a[:, j] for j in (1, 2, 3, 4, 5))
    contact = (a[:, 6] + a[:, 7]) > 0
    speeds = {}
    for frames in (6, 15):
        # Even a model failing before the first 0.5 s must produce a summary.
        lagged = np.zeros_like(x)
        if len(x) > frames:
            lagged[frames:] = x[:-frames]
        speeds[frames] = (x - lagged) / (frames / 30.)
    speed02, speed05 = speeds[6], speeds[15]

    def crossing(m):
        ids = np.flatnonzero(score >= m)
        return float(t[ids[0]]) if ids.size else None

    def span(lo, hi):
        start, end = crossing(lo), crossing(hi)
        return end - start if start is not None and end is not None else None

    def posture(mask):
        if not mask.any():
            return None
        return {'frames': int(mask.sum()), 'height_ratio_mean': float(height[mask].mean()),
                'height_ratio_std': float(height[mask].std()),
                'height_reference_units_std': float(height[mask].std() * 506.85),
                'height_ratio_p95_p05': float(np.percentile(height[mask], 95) - np.percentile(height[mask], 5)),
                'lean_mean_degrees': float(lean[mask].mean()),
                'lean_abs_error_from_5_degrees': float(np.abs(lean[mask] - 5).mean()),
                'lean_within_2_to_8_fraction': float((np.abs(lean[mask] - 5) <= 3).mean()),
                'low_posture_fraction': float((height[mask] < .8).mean()),
                'foot_contact_fraction': float(contact[mask].mean()),
                'upright_foot_contact_fraction': float((contact[mask] & (height[mask] >= .8)).mean()),
                'speed02_mean_mps': float(speed02[mask].mean())}

    flights = []
    for start, end in contiguous(~contact):
        if end - start + 1 < 3:
            continue
        flights.append({'start_index': start, 'end_index': end,
                        'start_time_s': float(t[start]), 'end_time_s': float(t[end]),
                        'landing_time_s': float(t[end + 1]) if end + 1 < len(t) else None,
                        'duration_s': (end - start + 1) / 30.,
                        'start_score_m': float(score[start]), 'end_score_m': float(score[end]),
                        'height_ratio_peak': float(height[start:end + 1].max())})
    torso_crossings = np.flatnonzero((a[:-1, 8] > 0) & (a[1:, 8] <= 0)) + 1
    torso_cross_time = float(t[torso_crossings[0]]) if torso_crossings.size else None
    # A slow policy may have ordinary strides at 53--55 m. Select the actual
    # immediate crossing neighbourhood, rather than calling the longest later
    # running stride its hurdle takeoff.
    near_hurdle_flights = [f for f in flights if torso_cross_time is not None
                          and f['start_time_s'] <= torso_cross_time + .5
                          and f['end_time_s'] >= torso_cross_time - .2]
    hurdle_flight = max(near_hurdle_flights, key=lambda f: f['height_ratio_peak']) if near_hurdle_flights else None
    hurdle = None
    reference_mask = (score >= 40) & (score < 45)
    reference_speed = float(speed05[reference_mask].mean()) if reference_mask.any() else None
    if hurdle_flight:
        flight = hurdle_flight.copy()
        start, end = flight.pop('start_index'), flight.pop('end_index')
        landing = end + 1
        recovery_index = None
        for i in range(landing, len(t) - 14):
            if (speed02[i:i + 15] >= 2.3).all():
                recovery_index = i
                break
        relative_recovery_index = None
        if reference_speed is not None and reference_speed > 0:
            for i in range(landing, len(t) - 14):
                if (speed05[i:i + 15] >= reference_speed * .9).all():
                    relative_recovery_index = i
                    break
        region = (score >= 45) & (score <= 65)
        min_index = np.flatnonzero(region)[np.argmin(speed02[region])]
        hurdle = dict(flight, torso_crosses_hurdle_time_s=torso_cross_time,
                      minimum_speed02_mps=float(speed02[min_index]),
                      minimum_speed_time_s=float(t[min_index]),
                      landing_to_recovered_2p3mps_s=float(t[recovery_index] - t[landing]) if recovery_index is not None else None,
                      recovered_2p3mps_time_s=float(t[recovery_index]) if recovery_index is not None else None,
                      recovered_score_m=float(score[recovery_index]) if recovery_index is not None else None,
                      recovery_definition='First >=2.3 m/game-s over 0.2 s torso displacement, sustained 15 frames (0.5 s), measured from first foot-contact frame after main hurdle flight.',
                      pre_hurdle_reference_speed05_mps=reference_speed,
                      relative_recovery_threshold_mps=reference_speed * .9 if reference_speed is not None else None,
                      landing_to_recovered_90pct_reference_s=float(t[relative_recovery_index] - t[landing]) if relative_recovery_index is not None else None,
                      recovered_90pct_reference_time_s=float(t[relative_recovery_index]) if relative_recovery_index is not None else None,
                      recovered_90pct_reference_score_m=float(score[relative_recovery_index]) if relative_recovery_index is not None else None,
                      relative_recovery_definition='Mean 0.5 s torso-displacement speed over score 40--45 m is the reference; first >=90% of reference on 0.5 s torso-displacement speed sustained 15 frames, from first foot-contact frame after hurdle cycle.')
    crossings = {str(m): crossing(m) for m in range(10, 101, 10)}
    flat_mask = ((score >= 10) & (score < 45)) | ((score >= 65) & (score < 90))
    early = {}
    for seconds in (2, 3):
        if t[-1] < seconds - 1e-6:
            early[str(seconds)] = None
            continue
        ids = np.flatnonzero(t <= seconds + 1e-6)
        idx = ids[-1]
        early[str(seconds)] = {'net_torso_progress_m': float(x[idx]),
                               'score_m': float(score[idx]),
                               'speed02_end_mps': float(speed02[idx]),
                               'height_min_ratio': float(height[ids].min())}
    biggest = max(flights, key=lambda f: f['duration_s']) if flights else None
    if biggest:
        biggest = {k: v for k, v in biggest.items() if not k.endswith('_index')}
    report = {'checkpoint': str(checkpoint), 'update': data.get('update'),
              'evaluation_reward_profile': reward_profile,
              'success': bool(info['success']), 'truncated': bool(truncated),
              'native_failed': bool(frame.failed),
              'time_s': float(info['time']), 'distance_m': float(info['distance']),
              'steps': len(a), 'early_seconds': early, 'score_crossing_times_s': crossings,
              'split_times_s': {'0_10': crossing(10), '40_50': span(40, 50),
                                '50_60': span(50, 60), '60_70': span(60, 70)},
              'hurdle_flight': hurdle, 'longest_flight': biggest,
              'flat_running_posture': posture(flat_mask),
              'flat_running_before_hurdle': posture((score >= 10) & (score < 45)),
              'flat_running_after_recovery': posture((score >= 65) & (score < 90)),
              'whole_episode_low_posture_fraction': float((height < .8).mean()),
              'whole_episode_foot_contact_fraction': float(contact.mean()),
              'whole_episode_upright_foot_contact_fraction': float((contact & (height >= .8)).mean()),
              'whole_episode_low_unsupported_fraction': float(((~contact) & (height < .8)).mean()),
              'same_backend_trajectory_verified': True,
              'measurement_notes': ['Original game clock and original score; 30 frames/game second, 400 world reference units/race meter.',
                                    'Flat-running posture mask fixed at score 10--45 m and 65--90 m for every checkpoint.',
                                    'Torso height is a proxy, not mass-weighted center of mass. No knee-contact API is available.',
                                    'Flight defined by neither foot contacting ground for >=3 frames. Hurdle cycle is highest-torso flight starting by 0.5 s after the torso crosses the obstacle, ending no earlier than 0.2 s before crossing; it need not be a deliberate jump.']}
    if trace:
        np.savetxt(trace, np.column_stack([a, speed02, speed05]), delimiter=',',
                   header='frame,time_s,score_m,torso_progress_m,height_ratio,lean_deg,left_contact,right_contact,hurdle_dx_from_torso_m,action,active_hurdles,failed,speed02_mps,speed05_mps', comments='')
    return report


def compact_summary(report):
    """Flatten the metrics needed while checking many training snapshots."""
    hurdle = report.get('hurdle_flight') or {}
    flat = report.get('flat_running_posture') or {}
    early = report.get('early_seconds') or {}
    return {'checkpoint': Path(report['checkpoint']).name, 'update': report.get('update'),
            'success': report['success'], 'native_failed': report['native_failed'],
            'time_s': report['time_s'], 'distance_m': report['distance_m'],
            'torso_progress_2s_m': (early.get('2') or {}).get('net_torso_progress_m'),
            'torso_progress_3s_m': (early.get('3') or {}).get('net_torso_progress_m'),
            'splits_s': report['split_times_s'],
            'hurdle_air_s': hurdle.get('duration_s'),
            'hurdle_min_speed_mps': hurdle.get('minimum_speed02_mps'),
            'recovery_2p3_s': hurdle.get('landing_to_recovered_2p3mps_s'),
            'pre_hurdle_speed_mps': hurdle.get('pre_hurdle_reference_speed05_mps'),
            'recovery_90pct_s': hurdle.get('landing_to_recovered_90pct_reference_s'),
            'flat_height_std': flat.get('height_ratio_std'),
            'flat_lean_mean_deg': flat.get('lean_mean_degrees'),
            'flat_lean_abs_error5_deg': flat.get('lean_abs_error_from_5_degrees'),
            'low_posture_fraction': report['whole_episode_low_posture_fraction']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--compact', action='store_true', help='Print one compact summary; output JSON remains complete.')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    report = evaluate_checkpoint(args.checkpoint, trace=args.trace, device=args.device)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(compact_summary(report) if args.compact else report, ensure_ascii=False))


if __name__ == '__main__':
    main()
