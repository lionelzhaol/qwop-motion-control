"""Read-only replay and native telemetry analysis for the existing 44.97 s replay."""
import ctypes as C
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / 'QWOP_rl_training_public'
sys.path.insert(0, str(ROOT / 'scripts'))
from replay_io import read_replay, ReplayRuleset
from running_env import RunningGameEnv

class Body(C.Structure):
    _fields_ = [(k, C.c_float) for k in ('x','y','angle_degrees','velocity_x','velocity_y')]
class Course(C.Structure):
    _fields_ = [('struct_size',C.c_uint32),('game_mode',C.c_uint32)]
class Frame(C.Structure):
    _fields_ = [('struct_size',C.c_uint32),('body_count',C.c_uint32),('bodies',Body*12)] + [(k,C.c_float) for k in ('score','sim_time','view_target_x','hurdle_location','long_jump_distance','long_jump_landing_body_x')] + [('active_hurdle_count',C.c_uint32)] + [(k,C.c_uint8) for k in ('failed','jumped','jump_landed','race_finished','left_foot_grounded','right_foot_grounded','long_jump_landing_started','long_jump_exited_sand')]
assert C.sizeof(Frame)==284
dll=C.CDLL(str(ROOT/'bin/running_physics.dll'))
dll.rg_create.argtypes=[C.POINTER(C.c_void_p)];dll.rg_create.restype=C.c_int
dll.rg_reset.argtypes=[C.c_void_p,C.POINTER(Course)];dll.rg_reset.restype=C.c_int
dll.rg_step.argtypes=[C.c_void_p,C.c_uint8];dll.rg_step.restype=C.c_int
dll.rg_read_frame.argtypes=[C.c_void_p,C.POINTER(Frame)];dll.rg_read_frame.restype=C.c_int
dll.rg_destroy.argtypes=[C.c_void_p]
h=C.c_void_p();assert dll.rg_create(C.byref(h))==0
course=Course(C.sizeof(Course),0);assert dll.rg_reset(h,C.byref(course))==0
f=Frame();f.struct_size=C.sizeof(Frame);assert dll.rg_read_frame(h,C.byref(f))==0
x0=f.bodies[5].x
env=RunningGameEnv('classic_100m','classic_100m');env.reset()
rules,actions=read_replay(ROOT/'replays/classic_100m_under45_44p97.bin');assert rules==ReplayRuleset.CLASSIC_100M
rows=[]
for i,a in enumerate(actions):
    obs,reward,done,trunc,info=env.step(a)
    assert dll.rg_step(h,a)==0 and dll.rg_read_frame(h,C.byref(f))==0
    assert abs(info['time']-f.sim_time)<1e-5 and abs(info['distance']-f.score)<1e-5
    assert abs(obs[26]*40-f.bodies[5].y)<1e-3
    t=f.bodies[5];head=f.bodies[10]
    lean=np.degrees(np.arctan2(head.x-t.x,t.y-head.y))
    rows.append([i+1,f.sim_time,f.score,(t.x-x0)/400,t.y,(429.8-t.y)/506.85,lean,float(f.left_foot_grounded),float(f.right_foot_grounded),a,t.velocity_x,t.velocity_y,(f.hurdle_location-t.x)/400,(f.hurdle_location-f.bodies[3].x)/400,(f.hurdle_location-f.bodies[11].x)/400,float(f.active_hurdle_count),float(f.jump_landed),float(f.race_finished),float(f.jumped)])
    assert not(done or trunc) or i==len(actions)-1
assert done and not trunc and info['success']
dll.rg_destroy(h)
data=np.array(rows)
time=data[:,1];dist=data[:,2];x=data[:,3];height=data[:,5];lean=data[:,6];lf=data[:,7];rf=data[:,8]
speed02=(x-np.r_[np.repeat(0.,6),x[:-6]])/.2
speed05=(x-np.r_[np.repeat(0.,15),x[:-15]])/.5
def windows(mask):
    starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
    ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])
    return [(int(s),int(e)) for s,e in zip(starts,ends)]
def stats(mask):
    ids=np.flatnonzero(mask)
    if len(ids)==0:return {}
    a,b=ids[0],ids[-1]
    return {'start_s':float(time[a]),'end_s':float(time[b]),'start_m':float(dist[a]),'end_m':float(dist[b]),'torso_x_speed_mean':float(speed02[ids].mean()),'speed02_min':float(speed02[ids].min()),'speed02_max':float(speed02[ids].max()),'height_min':float(height[ids].min()),'height_max':float(height[ids].max()),'height_std':float(height[ids].std()),'lean_min':float(lean[ids].min()),'lean_max':float(lean[ids].max()),'lean_mean':float(lean[ids].mean()),'no_foot_contact_fraction':float(((lf[ids]+rf[ids])==0).mean()),'both_feet_contact_fraction':float(((lf[ids]+rf[ids])==2).mean())}
segments=[dict(from_m=m,to_m=m+10,**stats((dist>=m)&(dist<m+10))) for m in range(0,100,10)]
flights=[]
for a,b in windows((lf+rf)==0):
    if b-a+1>=3:
        mask=np.zeros(len(data),dtype=bool);mask[a:b+1]=True
        flights.append(dict(frames=b-a+1,duration_s=(b-a+1)/30,**stats(mask)))
slow=[]
for a,b in windows((speed05<1.8)&(time>.5)):
    if b-a+1>=3:
        mask=np.zeros(len(data),dtype=bool);mask[a:b+1]=True
        slow.append(dict(duration_s=(b-a+1)/30,**stats(mask)))
crossings={m:float(time[np.flatnonzero(dist>=m)[0]]) for m in range(10,101,10)}
spans=[]
for lo,hi in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(22,23),(23,24),(24,25),(25,26),(26,27),(27,28),(28,29)]:
    spans.append(dict(window=f'{lo}-{hi}',**stats((time>=lo)&(time<hi))))
events=[]
for name,col in [('torso_crosses_hurdle',12),('left_foot_crosses_hurdle',13),('right_foot_crosses_hurdle',14)]:
    ids=np.flatnonzero((data[:-1,col]>0)&(data[1:,col]<=0))+1
    if len(ids):
        idx=int(ids[0]);events.append({'event':name,'frame':idx+1,'time_s':float(time[idx]),'distance_m':float(dist[idx]),'obstacle_dx_m':float(data[idx,col])})
for name,idx in [('peak_height',int(np.argmax(height))),('minimum_speed_between22_and27',int(np.flatnonzero((time>=22)&(time<=27))[np.argmin(speed02[(time>=22)&(time<=27)])])),('largest_backward_lean',int(np.argmin(lean)))]:
    events.append({'event':name,'frame':idx+1,'time_s':float(time[idx]),'distance_m':float(dist[idx]),'torso_height_ratio':float(height[idx]),'lean_degrees':float(lean[idx]),'speed02_mps':float(speed02[idx]),'left_contact':bool(lf[idx]),'right_contact':bool(rf[idx])})
report={'time_s':float(time[-1]),'crossings':crossings,'segments':segments,'flights':flights,'slow_windows_speed05_lt1p8':slow,'one_second_windows':spans,'events':events,'same_backend_trajectory_verified':True,'height_proxy_note':'Torso height, not mass weighted COM; foot-ground flags are published but knee-ground flags are unavailable.'}
np.savetxt(BASE/'verification/under45_frame_trace.csv',np.column_stack([data,speed02,speed05]),delimiter=',',header='frame,time_s,score_m,torso_progress_m,torso_y,height_ratio,lean_deg,left_contact,right_contact,action,torso_vx_physics,torso_vy_physics,hurdle_dx_from_torso_m,hurdle_dx_from_left_foot_m,hurdle_dx_from_right_foot_m,active_hurdles,jump_landed,race_finished,jumped,speed02_mps,speed05_mps',comments='')
(BASE/'verification/under45_frame_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
