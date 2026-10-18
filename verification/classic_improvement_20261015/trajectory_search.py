"""Bounded native action-sequence optimization, separate from neural PPO.

Search results are recorded trajectories, not autonomous model achievements.
A faster trajectory must subsequently be learned and evaluated closed-loop.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import sys
import time
import numpy as np
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]/'QWOP_rl_training_public'
sys.path.insert(0,str(ROOT/'scripts'))
from replay_io import read_replay,write_replay,ReplayRuleset
from running_env import VectorRunningGameEnv,RunningGameEnv

def canonical(a):return (1 if a&1 else 2 if a&2 else 0)|(4 if a&4 else 8 if a&8 else 0)
def dump(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def beam_search(source,tag,start_frame,width,metric):
    timer=time.perf_counter();rules,raw=read_replay(source);assert rules==ReplayRuleset.CLASSIC_100M
    seq=np.array([canonical(x) for x in raw],dtype=np.int64);limit=len(seq)-1
    assert 0<start_frame<limit
    n=width*9+1;env=VectorRunningGameEnv(n,num_threads=8,reward_profile='classic_100m_task',game_mode='classic_100m')
    beams=[[]];history=[];best=None;attempts=0;total_transitions=0
    for depth in range(1,limit-start_frame+1):
        paths=[b+[a] for b in beams for a in (0,1,2,4,5,6,8,9,10)]
        anchor=seq[start_frame:start_frame+depth].tolist()
        if anchor not in paths:paths.append(anchor)
        count=len(paths);attempts+=count
        plans=np.tile(seq[:start_frame+depth,None],(1,n))
        for i,path in enumerate(paths):plans[start_frame:,i]=path
        obs,_=env.reset();active=np.arange(n)<count;low=np.zeros(n,dtype=int)
        previous_progress=np.zeros(n)
        for step in range(start_frame+depth):
            low[active]+=((429.8-obs[active,26]*40)/506.85<.8)
            previous_progress=100.-obs[:,63]/10.
            r=env.step_arrays(plans[step]);total_transitions+=n
            assert np.isfinite(r['obs']).all() and np.isfinite(r['reward']).all()
            for i in np.flatnonzero(active&r['has_final']):
                if r['final_success'][i] and low[i]/(step+1)<=.1 and (best is None or step+1<best['frames']):
                    best=dict(frames=step+1,time_s=float(r['final_time'][i]),distance_m=float(r['final_distance'][i]),
                        actions=plans[:step+1,i].tolist(),low_torso_fraction=float(low[i]/(step+1)),
                        prefix_frames=start_frame,beam_width=width,depth=depth)
                    replay=ROOT/'replays'/f'classic_shadow_beam_{tag}_best_20261015.bin'
                    write_replay(replay,best['actions'],ReplayRuleset.CLASSIC_100M)
                    print(json.dumps(dict(new_best={k:v for k,v in best.items() if k!='actions'},replay=str(replay))),flush=True)
            active[active&r['has_final']]=False
            obs=r['obs']
            if not active.any():break
        # Continuous torso progress is derived from the unchanged finish-line
        # distance observation. Velocity is measured by consecutive positions.
        progress=100.-obs[:,63]/10.;velocity=(progress-previous_progress)*30.
        if metric=='lead_foot':
            fitness=progress+np.maximum(obs[:,15],obs[:,55])/10.+.05*velocity
        elif metric=='rear_foot':
            fitness=progress+np.minimum(obs[:,15],obs[:,55])/10.+.05*velocity
        else:fitness=progress+.1*velocity
        order=sorted(np.flatnonzero(active),key=lambda i:float(fitness[i]),reverse=True)
        chosen=[];per_action={a:0 for a in (0,1,2,4,5,6,8,9,10)}
        for i in order:
            last=paths[i][-1]
            if per_action[last]>=max(2,(width+8)//9):continue
            chosen.append(int(i));per_action[last]+=1
            if len(chosen)==width:break
        if not chosen:
            beams=[anchor]
        else:beams=[paths[i] for i in chosen]
        entry=dict(depth=depth,candidates=count,live_candidates=int(active.sum()),
            best_frames=best['frames'] if best else None,
            furthest_live_torso_m=float(max(progress[active])) if active.any() else None)
        history.append(entry);dump(BASE/f'beam_{tag}_progress.json',history)
        if depth%5==0 or depth==limit-start_frame:print(json.dumps(entry),flush=True)
    report=dict(method='Finite-horizon beam search of recorded action continuations; not autonomous neural-policy inference',
        source=str(source),prefix_frames=start_frame,beam_width=width,heuristic=metric,limit_frames=limit,
        candidate_prefixes_evaluated=attempts,native_transitions=total_transitions,
        best=best,history=history,elapsed_seconds=time.perf_counter()-timer)
    if best:
        e=RunningGameEnv('classic_100m_task','classic_100m');e.reset()
        for i,action in enumerate(best['actions']):
            x,reward,done,tr,info=e.step(action);assert bool(done or tr)==(i==best['frames']-1)
        assert info['success'] and not tr and info['time']==best['time_s']
        report.update(replay=str(replay),native_reexecution=info,crc_verified=True)
    dump(BASE/f'beam_{tag}_result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('best','history')},ensure_ascii=False),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--tag',default='single_frame');p.add_argument('--mode',choices=['single','timing','closedloop','blocks','beam'],default='single')
    p.add_argument('--prefix-frames',type=int,default=950);p.add_argument('--beam-width',type=int,default=64)
    p.add_argument('--beam-metric',choices=['torso','lead_foot','rear_foot'],default='torso')
    p.add_argument('--batch-size',type=int,default=256);a=p.parse_args()
    if a.mode=='beam':return beam_search(a.source,a.tag,a.prefix_frames,a.beam_width,a.beam_metric)
    started=time.perf_counter();startutc=datetime.now(timezone.utc);source_hash=hashlib.sha256(a.source.read_bytes()).hexdigest()
    rules,raw=read_replay(a.source);assert rules==ReplayRuleset.CLASSIC_100M
    seq=np.array([canonical(x) for x in raw],dtype=np.int64);horizon=len(seq)
    candidates=[dict(kind='control')]
    if a.mode in ('single','closedloop'):
        for frame in range(horizon-1):
            for action in (0,1,2,4,5,6,8,9,10):
                if action!=seq[frame]:candidates.append(dict(kind='replace',frame=frame,action=action))
    elif a.mode=='timing':
        # Editing once per run-length block avoids duplicate insert/delete trials.
        for frame in np.flatnonzero(np.r_[True,seq[1:]!=seq[:-1]]):
            if frame>=horizon-1:continue
            for width in (1,2,3):
                candidates.append(dict(kind='delete',frame=int(frame),width=width))
                candidates.append(dict(kind='insert',frame=int(frame),width=width))
    else:
        for frame in range(horizon-1):
            for width in (2,3,4,6):
                for action in (0,1,2,4,5,6,8,9,10):
                    if not np.all(seq[frame:frame+width]==action):
                        candidates.append(dict(kind='block',frame=frame,width=width,action=action))
            nearby=np.unique(seq[max(0,frame-1):min(horizon,frame+3)])
            for first in nearby:
                for second in nearby:
                    if first!=second and (first!=seq[frame] or second!=seq[frame+1]):
                        candidates.append(dict(kind='pair',frame=frame,first=int(first),second=int(second)))
    n=a.batch_size;env=VectorRunningGameEnv(n,num_threads=8,reward_profile='classic_100m_task',game_mode='classic_100m')
    if a.mode=='closedloop':
        import torch
        from train_ppo import ActorCritic,FrameStacker
        torch.set_num_threads(1)
        checkpoint=ROOT/'checkpoints/bc_classic100m_shadow.pt'
        data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=ActorCritic(68*data['config']['frame_stack'],16,tuple(data['config']['hidden_sizes']))
        model.load_state_dict(data['model'],strict=True);model.eval()
        assert model.flat_actor_bias is None and model.jump_sampling is None
    best=None;native_successes=0;qualified=0;improved=0;progress=[];top=[];seen=set()
    for start in range(0,len(candidates),n):
        batch=candidates[start:start+n];count=len(batch)
        plans=np.tile(seq[:,None],(1,n))
        for i,c in enumerate(batch):
            if c['kind']=='replace':plans[c['frame'],i]=c['action']
            elif c['kind']=='block':plans[c['frame']:c['frame']+c['width'],i]=c['action']
            elif c['kind']=='pair':plans[c['frame']:c['frame']+2,i]=[c['first'],c['second']]
            elif c['kind']=='delete':
                f=c['frame'];w=c['width'];plans[:,i]=np.r_[seq[:f],seq[f+w:],np.repeat(seq[-1],w)][:horizon]
            elif c['kind']=='insert':
                f=c['frame'];w=c['width'];plans[:,i]=np.r_[seq[:f],np.repeat(seq[f],w),seq[f:]][:horizon]
        obs,_=env.reset();active=np.arange(n)<count;low=np.zeros(n,dtype=np.int64)
        if a.mode=='closedloop':
            stack=FrameStacker(n,68,data['config']['frame_stack']);stacked=stack.reset(obs)
            first_edit=min([c['frame'] for c in batch if c['kind']=='replace'],default=horizon)
        for step in range(horizon):
            low[active]+=((429.8-obs[active,26]*40)/506.85<.8)
            if a.mode=='closedloop' and step>=first_edit:
                with torch.no_grad():
                    logits=model.actor(model.features(torch.as_tensor(stacked)))
                    assert torch.isfinite(logits).all()
                    plans[step]=logits.argmax(-1).numpy()
                for i,c in enumerate(batch):
                    if c['kind']=='replace' and c['frame']==step:plans[step,i]=c['action']
            r=env.step_arrays(plans[step]);done=r['terminated']|r['truncated']
            assert np.isfinite(r['obs']).all() and np.isfinite(r['reward']).all()
            for i in np.flatnonzero(active&r['has_final']):
                success=bool(r['final_success'][i]);native_successes+=int(success)
                if start==0 and i==0:
                    assert success and step+1==horizon,'Unmodified control must reproduce source'
                if success and low[i]/(step+1)<=.1:
                    qualified+=1
                    if step+1<horizon:improved+=1
                    entry=dict(frames=step+1,time_s=float(r['final_time'][i]),
                        distance_m=float(r['final_distance'][i]),low_torso_fraction=float(low[i]/(step+1)),
                        modification=batch[i],candidate_index=start+int(i))
                    if best is None or entry['frames']<best['frames']:
                        best=dict(entry,actions=plans[:step+1,i].tolist())
                        replay=ROOT/'replays'/f'classic_shadow_trajectory_{a.tag}_best_20261015.bin'
                        write_replay(replay,best['actions'],ReplayRuleset.CLASSIC_100M)
                        print(json.dumps(dict(new_best=entry,replay=str(replay))),flush=True)
                    if best is not None and entry['frames']<=best['frames']+2:
                        actions=plans[:step+1,i].tolist();key=hashlib.sha256(bytes(actions)).hexdigest()
                        if key not in seen:
                            seen.add(key);top.append(dict(entry,actions=actions));top.sort(key=lambda x:x['frames']);top=top[:32]
            active[active&done]=False
            if not active.any():break
            obs=r['obs']
            if a.mode=='closedloop':stacked=stack.update(obs,done)
        status=dict(attempts=min(start+count,len(candidates)),total=len(candidates),
            native_successes_within_horizon=native_successes,qualified_within_horizon=qualified,
            improved_qualified=improved,best_frames=best['frames'] if best else None)
        progress.append(status)
        if len(progress)%4==0 or start+count==len(candidates):print(json.dumps(status),flush=True)
        dump(BASE/f'trajectory_{a.tag}_progress.json',progress)
    assert best is not None
    native=RunningGameEnv('classic_100m_task','classic_100m');native.reset()
    for i,action in enumerate(best['actions']):
        obs,reward,done,tr,info=native.step(action)
        assert bool(done or tr)==(i==len(best['actions'])-1)
    assert info['success'] and not tr and info['time']==best['time_s']
    report=dict(source=str(a.source),source_sha256=source_hash,mode=a.mode,tag=a.tag,
        method=('One externally forced action followed by observation-conditioned BC continuation; not unmodified autonomous-policy performance'
            if a.mode=='closedloop' else 'Bounded open-loop action-sequence perturbation; not policy inference or PPO'),
        horizon_frames=horizon,candidates=len(candidates),native_successes_within_horizon=native_successes,
        qualified_within_horizon=qualified,improved_qualified=improved,best=best,top_candidates=top,
        replay=str(replay),reexecution=info,crc_verified=True,
        elapsed_seconds=time.perf_counter()-started,started_utc=startutc.isoformat(),ended_utc=datetime.now(timezone.utc).isoformat(),
        note='Trials not completed within the source horizon are censored; this is not a full-episode success-rate estimate.')
    dump(BASE/f'trajectory_{a.tag}_result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('best','top_candidates')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
