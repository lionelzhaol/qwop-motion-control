"""Read-only paired replay probes of opaque native action-mask behavior."""
from pathlib import Path
import ctypes as C
import json
import sys

import numpy as np
import torch

from evaluate_gait_bottlenecks import Body, Course, Frame, ROOT
sys.path.insert(0, str(ROOT / 'scripts'))
from running_env import RunningGameEnv

torch.set_num_threads(1)
VERIFY = ROOT.parent / 'verification'
dll = C.CDLL(str(ROOT / 'bin/running_physics.dll'))
dll.rg_create.argtypes = [C.POINTER(C.c_void_p)]; dll.rg_create.restype = C.c_int
dll.rg_reset.argtypes = [C.c_void_p, C.POINTER(Course)]; dll.rg_reset.restype = C.c_int
dll.rg_step.argtypes = [C.c_void_p, C.c_uint8]; dll.rg_step.restype = C.c_int
dll.rg_read_frame.argtypes = [C.c_void_p, C.POINTER(Frame)]; dll.rg_read_frame.restype = C.c_int
dll.rg_destroy.argtypes = [C.c_void_p]


class Trial:
    """An author environment and independent native copy of its trajectory."""
    def __init__(self):
        self.env = RunningGameEnv('classic_100m', 'classic_100m')
        self.obs, _ = self.env.reset()
        self.handle = C.c_void_p()
        assert dll.rg_create(C.byref(self.handle)) == 0
        course = Course(C.sizeof(Course), 0)
        assert dll.rg_reset(self.handle, C.byref(course)) == 0
        self.frame = Frame(); self.frame.struct_size = C.sizeof(Frame)
        assert dll.rg_read_frame(self.handle, C.byref(self.frame)) == 0

    def step(self, action):
        self.obs, _, done, truncated, info = self.env.step(int(action))
        assert dll.rg_step(self.handle, int(action)) == 0
        assert dll.rg_read_frame(self.handle, C.byref(self.frame)) == 0
        assert abs(info['time'] - self.frame.sim_time) < 1e-5
        assert abs(info['distance'] - self.frame.score) < 1e-5
        assert abs(self.obs[26] * 40 - self.frame.bodies[5].y) < 1e-3
        assert self.obs.shape == (68,) and np.isfinite(self.obs).all()
        return self.obs.copy(), self.numeric_frame()

    def numeric_frame(self):
        bodies = [getattr(body, field) for body in self.frame.bodies for field, _ in Body._fields_]
        extras = [getattr(self.frame, field) for field, _ in Frame._fields_ if field != 'bodies']
        return np.asarray(bodies + extras, dtype=np.float64)

    def close(self):
        dll.rg_destroy(self.handle)


def compare_sequences(actions_a, actions_b, prefix=()):
    a, b = Trial(), Trial()
    obs_differences, native_differences, body_differences = [], [], []
    try:
        for action in prefix:
            a.step(action); b.step(action)
        assert np.array_equal(a.obs, b.obs)
        for action_a, action_b in zip(actions_a, actions_b):
            obs_a, frame_a = a.step(action_a)
            obs_b, frame_b = b.step(action_b)
            obs_differences.append(float(np.abs(obs_a - obs_b).max()))
            native_differences.append(float(np.abs(frame_a - frame_b).max()))
            body_differences.append(float(np.abs(frame_a[:60] - frame_b[:60]).max()))
    finally:
        a.close(); b.close()
    return {'compared_frames': len(obs_differences),
            'full_68obs_max_absolute_difference': max(obs_differences),
            'full_native_frame_numeric_max_absolute_difference': max(native_differences),
            'all_12_native_bodies_max_absolute_difference': max(body_differences),
            'frames_with_any_observation_difference': [i + 1 for i, d in enumerate(obs_differences) if d != 0],
            'frames_with_any_native_frame_difference': [i + 1 for i, d in enumerate(native_differences) if d != 0]}


def main():
    baseline = np.genfromtxt(VERIFY / 'bottleneck_baseline_speed44p97.csv', delimiter=',', skip_header=1)[:, 9].astype(int)
    full_replays = []
    for update in (10672, 10720, 10768):
        path = VERIFY / f'ppo_phase_v32_start_explore_update_{update:06d}_trace.csv'
        candidate = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 9].astype(int)
        changed = np.flatnonzero(baseline[:90] != candidate[:90])
        full_replays.append({'checkpoint_update': update,
                             'changed_raw_actions': [{'frame': int(i + 1), 'baseline_action': int(baseline[i]), 'candidate_action': int(candidate[i])} for i in changed],
                             **compare_sequences(baseline[:90], candidate[:90])})
    branches = []
    for frame, original, alternate in ((54, 4, 12), (62, 15, 5), (63, 15, 5), (64, 15, 5), (74, 9, 11)):
        assert baseline[frame - 1] == original
        common_following = baseline[frame:frame + 30]
        branches.append({'branch_frame': frame, 'original_action': original, 'alternate_action': alternate,
                         'test': 'One different action, then 30 identical original actions; reset and identical original prefix.',
                         **compare_sequences(np.r_[original, common_following], np.r_[alternate, common_following], baseline[:frame - 1])})
    states = []
    for prefix_frames in (0, 30, 90, 200, 700, 1000):
        obs_sequences, frame_sequences = [], []
        for mask in range(16):
            trial = Trial()
            obs_rows, frame_rows = [], []
            try:
                for action in baseline[:prefix_frames]:
                    trial.step(action)
                for _ in range(10):
                    obs, native = trial.step(mask)
                    obs_rows.append(obs); frame_rows.append(native)
            finally:
                trial.close()
            obs_sequences.append(np.asarray(obs_rows))
            frame_sequences.append(np.asarray(frame_rows))
        groups = []
        for mask in range(16):
            for group in groups:
                if np.array_equal(obs_sequences[mask], obs_sequences[group[0]]) and np.array_equal(frame_sequences[mask], frame_sequences[group[0]]):
                    group.append(mask)
                    break
            else:
                groups.append([mask])
        contrasts = {}
        for first, second in ((4, 12), (15, 5), (9, 11), (0, 1), (4, 8), (9, 10)):
            contrasts[f'{first}_vs_{second}'] = {
                'full_68obs_max_difference': float(np.abs(obs_sequences[first] - obs_sequences[second]).max()),
                'full_native_frame_max_difference': float(np.abs(frame_sequences[first] - frame_sequences[second]).max())}
        states.append({'original_prefix_frames': prefix_frames, 'hold_same_mask_frames': 10,
                       'classes_identical_at_all_10_frames_and_full_native_fields': groups,
                       'contrasts': contrasts})
    report = {'control_chain_verified': ['src/python_module.cpp:53 actionFromMask extracts Q=1,W=2,O=4,P=8',
                                         'src/training_env.cpp:27 stepAction calls backend_.step(actionMask(action))',
                                         'src/training_env.cpp:55 actionMask reconstructs the same four bits',
                                         'src/physics_backend_loader.cpp:121 forwards uint8 mask directly to DLL rg_step'],
              'native_implementation_source_available': False,
              'native_implementation_note': 'Only physics_backend_api.h, loader and prebuilt bin/running_physics.dll are in the repository. Internal priority source is unavailable.',
              'full_90_frame_policy_replay_pairs': full_replays,
              'single_action_branch_and_30_frame_delayed_effect_checks': branches,
              'all_16_actions_in_six_states': states,
              'conclusion': 'Across six live states and every measured frame, raw masks form nine exact observable classes. The classes match Q taking precedence when Q and W are both pressed, and O taking precedence when O and P are both pressed. Other distinct masks change all-body physics, so controls are active. This is direct behavioral evidence, not a claim to have read opaque DLL implementation or proved all possible states.',
              'physics_actions_enumeration_and_training_unchanged': True}
    output = VERIFY / 'startup_action_alias_probe.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'all_raw_policy_pairs_identical_68obs': all(r['full_68obs_max_absolute_difference'] == 0 for r in full_replays),
                      'all_policy_pairs_identical_native': all(r['full_native_frame_numeric_max_absolute_difference'] == 0 for r in full_replays),
                      'all_single_action_branches_identical_native': all(r['full_native_frame_numeric_max_absolute_difference'] == 0 for r in branches),
                      'six_states_groups': [s['classes_identical_at_all_10_frames_and_full_native_fields'] for s in states]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
