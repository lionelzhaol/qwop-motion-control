"""Fresh evaluation without experiment caches or best-replay selection."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]/'QWOP_rl_training_public'
sys.path.insert(0,str(ROOT/'scripts'))
from train_ppo import ActorCritic
from running_env import RunningGameEnv, VectorRunningGameEnv
from replay_io import read_replay
torch.set_num_threads(1)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def play(actions):
    env=RunningGameEnv('classic_100m_task','classic_100m');env.reset()
    ended=False
    for index,action in enumerate(actions):
        obs,reward,done,tr,info=env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if done or tr:
            ended=True;break
    return dict(frames=index+1,ended=ended,success=bool(info['success']),time_s=float(info['time']),distance_m=float(info['distance']))

def run(checkpoint,reference):
    data=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=data['config']
    model=ActorCritic(68*cfg['frame_stack'],16,tuple(cfg['hidden_sizes']))
    model.load_state_dict(data['model'],strict=True);model.eval()
    n=64;env=VectorRunningGameEnv(n,num_threads=8,reward_profile='classic_100m_task',game_mode='classic_100m')
    raw,_=env.reset();frames=np.repeat(raw[:,None,:],cfg['frame_stack'],axis=1)
    active=np.ones(n,dtype=bool);histories=[[] for _ in range(n)];rows=[]
    torch.manual_seed(2026101581)
    with torch.no_grad():
        for step in range(3601):
            logits,_=model(torch.as_tensor(frames.reshape(n,-1)))
            actions=torch.distributions.Categorical(logits=logits).sample().numpy()
            for i in np.flatnonzero(active):histories[i].append(int(actions[i]))
            result=env.step_arrays(actions)
            for i in np.flatnonzero(active & result['has_final']):
                h=histories[i];diff=[j for j in range(min(len(h),len(reference))) if h[j]!=reference[j]]
                rows.append(dict(env=int(i),success=bool(result['final_success'][i]),frames=len(h),
                    time_s=float(result['final_time'][i]),distance_m=float(result['final_distance'][i]),
                    exact_reference_actions=h==reference,first_different_action_frame=diff[0]+1 if diff else None,
                    different_action_count_in_shared_prefix=len(diff),action_sha256=hashlib.sha256(bytes(h)).hexdigest()))
                active[i]=False
            if not active.any():break
            frames=np.roll(frames,-1,axis=1);frames[:,-1,:]=result['obs']
            done=result['terminated']|result['truncated'];frames[done]=result['obs'][done,None,:]
    assert not active.any()
    # Independent single-environment greedy path, with no replay input.
    single=RunningGameEnv('classic_100m_task','classic_100m');obs,_=single.reset()
    stack=np.repeat(obs[None,:],cfg['frame_stack'],axis=0);greedy=[]
    with torch.no_grad():
        for _ in range(3601):
            logits,_=model(torch.as_tensor(stack.reshape(1,-1)));a=int(logits.argmax(-1)[0]);greedy.append(a)
            obs,r,done,tr,info=single.step(a)
            if done or tr:break
            stack=np.roll(stack,-1,axis=0);stack[-1]=obs
    success=[r for r in rows if r['success']];fail=[r for r in rows if not r['success']]
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),update=data.get('update'),global_step=data.get('global_step'),
        episodes=n,successes=len(success),unique_success_action_sequences=len({r['action_sha256'] for r in success}),
        successful_exact_reference_sequences=sum(r['exact_reference_actions'] for r in success),
        successful_frame_histogram=dict(Counter(r['frames'] for r in success)),
        failed_frame_histogram=dict(Counter(r['frames'] for r in fail)),unique_all_sequences=len({r['action_sha256'] for r in rows}),
        greedy=dict(success=bool(info['success']),frames=len(greedy),time_s=float(info['time']),exact_reference_actions=greedy==reference),
        actor_weight_sum=float(model.actor.weight.detach().double().sum()),records=rows)

def main():
    _,ref=read_replay(ROOT/'replays/classic_shadow_bc_33p03.bin')
    checkpoints=[ROOT/'checkpoints'/name for name in (
        'ppo_classic_shadow_finetuned_best_20261015.pt','ppo_classic_unfreeze_head_20261015.pt',
        'ppo_classic_unfreeze_last_hidden_20261015.pt','ppo_classic_unfreeze_full_20261015.pt',
        'ppo_classic_actor_unfreeze_best_20261015.pt')]
    models=[]
    for checkpoint in checkpoints:
        r=run(checkpoint,ref);models.append(r)
        print(json.dumps({k:v for k,v in r.items() if k!='records'},ensure_ascii=False),flush=True)
    controls={}
    for name in ('classic_shadow_bc_33p03.bin','classic100m_shadow_bc_random_33p13s.bin','classic100m_shadow_bc_random_33p17s.bin'):
        _,actions=read_replay(ROOT/'replays'/name);controls[name]=play(actions)
    controls['reference_stopped_at_990_frames']=play(ref[:990])
    controls['all_zero_actions']=play([0]*3601)
    assert controls['classic_shadow_bc_33p03.bin']['success']
    assert controls['classic100m_shadow_bc_random_33p13s.bin']['success']
    assert controls['classic100m_shadow_bc_random_33p13s.bin']['time_s']>controls['classic_shadow_bc_33p03.bin']['time_s']
    assert not controls['reference_stopped_at_990_frames']['success']
    assert not controls['all_zero_actions']['success']
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=2026101581,
        protocol='Fresh CPU model loads; no experiment helper, cached metric, best comparison or replay used for policy evaluation. 64 first episodes per model plus serial greedy evaluation.',
        models=models,controls=controls)
    (BASE/'identical_times_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(controls=controls,output=str(BASE/'identical_times_audit.json')),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
