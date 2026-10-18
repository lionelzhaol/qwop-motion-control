"""Inference-only diagnostic: score-gated complete checkpoint teachers.

This creates no new trainable/loadable policy or checkpoint. Each grid
candidate has its own native environment, and every selected action comes
from checkpoint logits on the actual author FrameStacker history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / 'QWOP_rl_training_public'
VERIFY = ROOT.parent / 'verification'
sys.path.insert(0, str(ROOT / 'scripts'))
from running_env import VectorRunningGameEnv, RunningGameEnv
from train_ppo import ActorCritic, FrameStacker
from replay_io import write_replay, read_replay, ReplayRuleset

SOURCES = {
    'baseline': 'ppo_classic100m_phase_fast.pt',
    'v30_11424': 'ppo_phase_v30_hurdle_update_011424.pt',
    'v33_11472': 'ppo_phase_v33_effective_explore_update_011472.pt',
    'v35_10696': 'ppo_phase_v35_frozen_gait_update_010696.pt',
    'pose47p83': 'ppo_classic100m_target45_pose.pt',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--output', type=Path, default=VERIFY / 'target43_phase_teacher_diagnostic.json')
    parser.add_argument('--replay', type=Path, default=VERIFY / 'target43_teacher_diagnostic.bin')
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, metadata = {}, {}
    for name, filename in SOURCES.items():
        path = ROOT / 'checkpoints' / filename
        saved = torch.load(path, map_location=device)
        cfg = saved['config']
        if cfg['frame_stack'] != 32 or cfg['game_mode'] != 'classic_100m':
            raise ValueError(f'{name}: expected the same classic 68 x 32 raw history')
        model = ActorCritic(68 * 32, 16, tuple(cfg['hidden_sizes'])).to(device)
        model.load_state_dict(saved['model'], strict=True)
        model.eval()
        models[name] = model
        metadata[name] = {'checkpoint': str(path), 'update': saved.get('update'),
                          'global_step': saved.get('global_step'),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'frame_stack': cfg['frame_stack'], 'hidden_sizes': cfg['hidden_sizes']}
    candidates = [{'expert': 'baseline', 'from_m': 0, 'to_m': 0, 'alpha': 0.}]
    for expert in SOURCES:
        if expert == 'baseline':
            continue
        for start in (45, 47, 48, 49, 50):
            for end in (54, 56, 58, 60, 62, 65):
                for alpha in (.5, 1.):
                    candidates.append({'expert': expert, 'from_m': start, 'to_m': end, 'alpha': alpha})
    records, all_actions = [], {}
    marks = (10, 40, 45, 50, 54, 56, 58, 60, 62, 65, 70, 90, 100)
    for offset in range(0, len(candidates), args.batch_size):
        chunk = candidates[offset:offset + args.batch_size]
        n = len(chunk)
        env = VectorRunningGameEnv(n, num_threads=args.threads, reward_profile='classic_100m', game_mode='classic_100m')
        raw, _ = env.reset()
        stacker = FrameStacker(n, 68, 32)
        stacked = stacker.reset(raw)
        active = np.ones(n, dtype=bool)
        low_counts = np.zeros(n, dtype=np.int64)
        selected_frames = np.zeros(n, dtype=np.int64)
        steps = np.zeros(n, dtype=np.int64)
        histories = [[] for _ in range(n)]
        crossings = {mark: np.full(n, np.nan) for mark in marks}
        flat_moments = np.zeros((n, 4), dtype=np.float64)
        flat_count = np.zeros(n, dtype=np.int64)
        starts = np.asarray([c['from_m'] for c in chunk])
        ends = np.asarray([c['to_m'] for c in chunk])
        alpha = torch.as_tensor([c['alpha'] for c in chunk], dtype=torch.float32, device=device)
        for step in range(1, 4001):
            height = (429.8 - raw[:, 26] * 40) / 506.85
            low_counts[active] += (height < .8)[active]
            steps[active] += 1
            lean = np.degrees(np.arctan2(raw[:, 50] - raw[:, 25], raw[:, 26] - raw[:, 51]))
            flat = active & (((raw[:, 60] >= 10) & (raw[:, 60] < 45)) | ((raw[:, 60] >= 65) & (raw[:, 60] < 90)))
            flat_count[flat] += 1
            flat_moments[flat] += np.column_stack((height, height * height, lean, np.abs(lean - 5)))[flat]
            gated = active & (raw[:, 60] >= starts) & (raw[:, 60] < ends)
            selected_frames[gated] += 1
            with torch.no_grad():
                tensor = torch.as_tensor(stacked, dtype=torch.float32, device=device)
                logits, _ = models['baseline'](tensor)
                mixed = logits.clone()
                for expert in SOURCES:
                    if expert == 'baseline':
                        continue
                    ids = np.flatnonzero(gated & np.array([c['expert'] == expert for c in chunk]))
                    if not len(ids):
                        continue
                    indices = torch.as_tensor(ids, dtype=torch.long, device=device)
                    expert_logits, _ = models[expert](tensor[indices])
                    weight = alpha[indices, None]
                    mixed[indices] = (1 - weight) * logits[indices] + weight * expert_logits
                actions = mixed.argmax(-1).cpu().numpy()
            for i in np.flatnonzero(active):
                histories[i].append(int(actions[i]))
            result = env.step_arrays(actions)
            assert np.isfinite(result['obs']).all() and np.isfinite(result['reward']).all()
            done = result['terminated'] | result['truncated']
            assert np.array_equal(done, result['has_final'])
            distance = np.where(done, result['final_distance'], result['obs'][:, 60])
            now = np.where(done, result['final_time'], result['obs'][:, 61])
            for mark in marks:
                crossed = active & np.isnan(crossings[mark]) & (distance >= mark)
                crossings[mark][crossed] = now[crossed]
            for i in np.flatnonzero(active & done):
                moments = flat_moments[i] / flat_count[i] if flat_count[i] else None
                times = {str(mark): float(crossings[mark][i]) if np.isfinite(crossings[mark][i]) else None for mark in marks}
                def span(first, last):
                    a, b = times[str(first)], times[str(last)]
                    return b - a if a is not None and b is not None else None
                record = {'candidate_id': offset + int(i), **chunk[i],
                          'success': bool(result['final_success'][i]),
                          'truncated': bool(result['truncated'][i]),
                          'time_s': float(result['final_time'][i]),
                          'distance_m': float(result['final_distance'][i]),
                          'steps': int(steps[i]), 'teacher_selected_frames': int(selected_frames[i]),
                          'low_posture_fraction': float(low_counts[i] / steps[i]),
                          'score_crossing_times_s': times,
                          'split_times_s': {'0_10': times['10'], '40_50': span(40, 50), '50_60': span(50, 60), '60_70': span(60, 70)},
                          'flat_height_std': float(np.sqrt(max(0, moments[1] - moments[0] ** 2))) if moments is not None else None,
                          'flat_mean_lean_degrees': float(moments[2]) if moments is not None else None,
                          'flat_lean_abs_error5_degrees': float(moments[3]) if moments is not None else None}
                records.append(record)
                all_actions[record['candidate_id']] = histories[i]
            active[done] = False
            raw = result['obs']
            stacked = stacker.update(raw, done)
            if not active.any():
                break
        if active.any():
            raise RuntimeError('A teacher diagnostic episode did not terminate within original 120 s limit')
        del env
        qualified = [r for r in records if r['success'] and r['low_posture_fraction'] <= .1]
        best_so_far = min(qualified, key=lambda r: r['time_s'])
        print(json.dumps({'evaluated': len(records), 'candidate_total': len(candidates), 'best_so_far': best_so_far}), flush=True)
    qualified = [r for r in records if r['success'] and r['low_posture_fraction'] <= .1]
    best = min(qualified, key=lambda r: r['time_s'])
    baseline = next(r for r in records if r['expert'] == 'baseline')
    best_teacher = min((r for r in qualified if r['expert'] != 'baseline'), key=lambda r: r['time_s'])
    # Export the best actual teacher combination even if the original baseline
    # remains faster, and keep that regression visible in the report.
    actions = all_actions[best_teacher['candidate_id']]
    write_replay(args.replay, actions, ReplayRuleset.CLASSIC_100M)
    ruleset, reread = read_replay(args.replay)
    assert ruleset == ReplayRuleset.CLASSIC_100M and reread == actions
    check = RunningGameEnv('classic_100m', 'classic_100m')
    _, _ = check.reset()
    full_low = 0
    for index, action in enumerate(reread):
        raw, _, terminated, truncated, info = check.step(action)
        full_low += int((429.8 - raw[26] * 40) / 506.85 < .8)
        assert not (terminated or truncated) or index == len(reread) - 1
    assert terminated and not truncated and info['success']
    assert abs(info['time'] - best_teacher['time_s']) < 1e-5
    assert abs(info['distance'] - best_teacher['distance_m']) < 1e-5
    assert full_low / len(reread) <= .1
    report = {'diagnostic_only': True, 'creates_trained_model': False,
              'checkpoint_teachers': metadata, 'device': str(device),
              'threads': args.threads, 'batch_size': args.batch_size,
              'policy_definition': 'Outside [from_m,to_m) score gate use original baseline complete forward; inside use argmax((1-alpha)*baseline_logits + alpha*expert_logits), each checkpoint preprocessing retained; common actual raw 68x32 author history.',
              'candidate_count': len(records), 'qualified_successes': len(qualified),
              'baseline': baseline, 'best': best, 'best_teacher': best_teacher,
              'qualified_under43': [r for r in qualified if r['time_s'] <= 43],
              'qualified_faster_than_baseline': [r for r in qualified if r['time_s'] < baseline['time_s'] - 1e-6],
              'best_by_expert': {expert: min((r for r in qualified if r['expert'] == expert), key=lambda r: r['time_s'], default=None) for expert in SOURCES},
              'replay': str(args.replay), 'replay_candidate_id': best_teacher['candidate_id'],
              'replay_roundtrip_and_native_terminal_verified': True,
              'replay_full_post_step_low_posture_fraction': full_low / len(reread),
              'elapsed_seconds': time.perf_counter() - started,
              'records': sorted(records, key=lambda r: r['candidate_id'])}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'qualified_faster_than_baseline', 'checkpoint_teachers')}), flush=True)


if __name__ == '__main__':
    main()
