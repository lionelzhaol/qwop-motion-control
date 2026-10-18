"""Read-only native replay audit; no reward or physics changes."""
from pathlib import Path
import sys, json, ctypes as C
import numpy as np

BASE = Path(__file__).resolve().parent
WORK = BASE.parents[1]
ROOT = WORK / 'QWOP_rl_training_public'
sys.path.insert(0, str(WORK / 'verification'))
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_gait_bottlenecks import Frame, Course, contiguous
from replay_io import read_replay, ReplayRuleset
from running_env import RunningGameEnv

dll = C.CDLL(str(ROOT / 'bin/running_physics.dll'))
dll.rg_create.argtypes = [C.POINTER(C.c_void_p)]; dll.rg_create.restype = C.c_int
dll.rg_reset.argtypes = [C.c_void_p, C.POINTER(Course)]; dll.rg_reset.restype = C.c_int
dll.rg_step.argtypes = [C.c_void_p, C.c_uint8]; dll.rg_step.restype = C.c_int
dll.rg_read_frame.argtypes = [C.c_void_p, C.POINTER(Frame)]; dll.rg_read_frame.restype = C.c_int
dll.rg_destroy.argtypes = [C.c_void_p]

def audit(path, actions=None):
    rules, actions = read_replay(path) if actions is None else (ReplayRuleset.LONG_JUMP_30M, actions)
    assert rules == ReplayRuleset.LONG_JUMP_30M
    handle = C.c_void_p(); assert dll.rg_create(C.byref(handle)) == 0
    assert dll.rg_reset(handle, C.byref(Course(C.sizeof(Course), 2))) == 0
    env = RunningGameEnv('long_jump_30m', 'long_jump_30m'); env.reset()
    frame = Frame(); frame.struct_size = C.sizeof(Frame)
    rows = []
    def snapshot(step, action, reward=0.):
        torso = frame.bodies[5]
        head = frame.bodies[10]
        return dict(step=step, action=action, time_s=float(frame.sim_time),
            torso_x_m=torso.x/400, torso_y_m=torso.y/400,
            torso_vx_native=float(torso.velocity_x), torso_vy_native=float(torso.velocity_y),
            torso_height_ratio=(429.8-torso.y)/506.85,
            lean_degrees=float(np.degrees(np.arctan2(head.x-torso.x, torso.y-head.y))),
            progress_m=frame.view_target_x/400, score=float(frame.score),
            long_jump_m=float(frame.long_jump_distance), landing_body_x_m=frame.long_jump_landing_body_x/400,
            jumped=bool(frame.jumped), landing=bool(frame.long_jump_landing_started),
            exited_sand=bool(frame.long_jump_exited_sand), jump_landed=bool(frame.jump_landed),
            failed=bool(frame.failed), left_foot=bool(frame.left_foot_grounded),
            right_foot=bool(frame.right_foot_grounded), reward=float(reward),
            body_x_m=[b.x/400 for b in frame.bodies], body_y_m=[b.y/400 for b in frame.bodies])
    try:
        assert dll.rg_read_frame(handle, C.byref(frame)) == 0
        rows.append(snapshot(0, -1))
        for i, action in enumerate(actions, 1):
            obs, reward, done, truncated, info = env.step(action)
            assert dll.rg_step(handle, action) == 0
            assert dll.rg_read_frame(handle, C.byref(frame)) == 0
            assert abs(info['time']-frame.sim_time)<1e-5 and abs(info['score']-frame.score)<1e-5
            assert bool(done or truncated) == (i==len(actions))
            rows.append(snapshot(i, action, reward))
        assert info['success'] and not truncated
    finally:
        dll.rg_destroy(handle)
    t=np.array([r['time_s'] for r in rows]); x=np.array([r['torso_x_m'] for r in rows]); y=np.array([r['torso_y_m'] for r in rows])
    vx=np.r_[0,np.diff(x)/np.diff(t)]; vy=np.r_[0,-np.diff(y)/np.diff(t)]
    native_vx=np.array([r['torso_vx_native'] for r in rows]); native_vy=np.array([r['torso_vy_native'] for r in rows])
    valid=np.abs(native_vx)>1
    # Native velocity units are diagnosed, not assumed; finite displacement speeds are authoritative here.
    scale=float(np.median(vx[valid]/native_vx[valid]))
    for i, r in enumerate(rows):
        r['vx_displacement_mps']=float(vx[i]); r['vy_up_displacement_mps']=float(vy[i])
        r['vx_native_scaled_mps']=float(native_vx[i]*scale); r['vy_up_native_scaled_mps']=float(-native_vy[i]*scale)
    contact=np.array([r['left_foot'] or r['right_foot'] for r in rows])
    jumps=[i for i,r in enumerate(rows) if r['jumped']]
    landings=[i for i,r in enumerate(rows) if r['landing']]
    ji=jumps[0] if jumps else None; li=landings[0] if landings else None
    def reduced(i):
        return {k:v for k,v in rows[i].items() if not k.startswith('body_')}
    flights=[]
    for start,end in contiguous(~contact):
        if end-start+1<2: continue
        segment=dict(start_step=start,end_step=end,duration_s=(end-start+1)/30,
            start=reduced(start),end=reduced(end),dx_m=float(x[end]-x[start]),
            peak_height_ratio=max(r['torso_height_ratio'] for r in rows[start:end+1]))
        flights.append(segment)
    key_flights=[f for f in flights if ji is not None and li is not None and f['end_step']>=ji-30 and f['start_step']<=li]
    crossing={}
    for m in (20,25,26,27,28,29,30):
        ids=[i for i,r in enumerate(rows) if r['progress_m']>=m]
        crossing[str(m)]=reduced(ids[0]) if ids else None
    landing_scores=[r['long_jump_m'] for r in rows if r['landing']]
    score_changes=[]
    for i in range(1,len(rows)):
        if rows[i]['landing'] and abs(rows[i]['long_jump_m']-rows[i-1]['long_jump_m'])>1e-5:
            score_changes.append(reduced(i))
    summary=dict(path=str(path),native_success=bool(info['success']),official_jump_m=float(info['score']),
        time_s=float(info['time']),frames=len(actions),
        first_jumped=reduced(ji) if ji is not None else None,
        first_landing=reduced(li) if li is not None else None,
        native_velocity_to_displacement_mps_median_scale=scale,
        landing_score_min_m=min(landing_scores) if landing_scores else None,
        landing_score_max_m=max(landing_scores) if landing_scores else None,
        landing_score_decrease_m=rows[li]['long_jump_m']-float(info['score']) if li is not None else None,
        crossings=crossing,key_flights=key_flights,
        native_score_changes=score_changes,
        post_landing_seconds=float(t[-1]-t[li]) if li is not None else None,
        post_landing_low_height_fraction=float(np.mean([r['torso_height_ratio']<.8 for r in rows[li:]])) if li is not None else None,
        full_episode_reward=sum(r['reward'] for r in rows),
        phase_rewards={phase:sum(r['reward'] for r in rows if condition(r)) for phase,condition in {
            'pre25':lambda r:r['progress_m']<25,
            '25_to_landing':lambda r:r['progress_m']>=25 and not r['landing'],
            'landing_to_exit':lambda r:r['landing']}.items()})
    (BASE/(path.stem+'_native_trace.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    paths=[ROOT/'replays'/name for name in ['long_jump_shadow_ppo_deterministic_2p59m.bin',
        'long_jump_shadow_ppo_sampled_best_3p16m.bin','long_jump_ppo_deterministic.bin','long_jump_ppo_sampled_best.bin']]
    report=[audit(path) for path in paths]
    (BASE/'native_replay_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in report:
        print(json.dumps({k:v for k,v in r.items() if k not in ('key_flights','crossings','native_score_changes')},ensure_ascii=False),flush=True)
