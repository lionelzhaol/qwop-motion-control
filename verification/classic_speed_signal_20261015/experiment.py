"""Matched speed-reward experiments using the original PPO and native outcomes."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import importlib.util
import json
import math
import sys
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]/'QWOP_rl_training_public'
spec=importlib.util.spec_from_file_location('prior_classic_experiment',BASE.parent/'classic_improvement_20261015/experiment.py')
prior=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prior;spec.loader.exec_module(prior);prior.BASE=BASE
from train_ppo import PPOConfig, classic_finish_speed_bonus
from running_env import RunningGameEnv, VectorRunningGameEnv
from replay_io import read_replay, write_replay, ReplayRuleset

SOURCE=ROOT/'checkpoints/ppo_classic_actor_unfreeze_best_20261015.pt'
ARMS={'control':0.,'speed1200':1200.,'speed3600':3600.}
REFERENCE=991/30
WIDTH=.25

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def emit(obj):print(json.dumps(obj,ensure_ascii=False),flush=True)
def save(path,obj):prior.write_json(path,obj)

def result_fixture(times):
    n=len(times)
    return dict(reward=np.full(n,2100.,np.float32),has_final=np.ones(n,bool),final_success=np.ones(n,bool),
                truncated=np.zeros(n,bool),final_time=np.array(times,np.float64))

def prepare():
    data,source_model=prior.load(SOURCE)
    with np.load(BASE.parent/'shadow_audit/expert_dataset.npz',allow_pickle=False) as expert:
        x=expert['observations'].copy();actions=expert['actions'].copy()
    native=prior.native(SOURCE)
    assert native['success'] and native['actions']==actions.tolist() and len(actions)==991
    # Recompute native rewards and native qualification from the physics engine.
    env=RunningGameEnv('classic_100m_task','classic_100m');env.reset();raw_rewards=[];terminal=None
    for i,action in enumerate(actions):
        obs,reward,done,tr,info=env.step(int(action));raw_rewards.append(reward)
        assert bool(done or tr)==(i==len(actions)-1)
        if done or tr:
            terminal=dict(reward=np.array([reward],np.float32),has_final=np.array([True]),
                final_success=np.array([info['success']]),truncated=np.array([tr]),final_time=np.array([info['time']]))
    assert terminal['reward'][0]>0 and terminal['final_success'][0]
    # Default-off inference remains identical to the saved pre-edit trainer.
    oldspec=importlib.util.spec_from_file_location('before_speed_reward',BASE/'train_ppo_before.py')
    oldmod=importlib.util.module_from_spec(oldspec);sys.modules[oldspec.name]=oldmod;oldspec.loader.exec_module(oldmod)
    old=oldmod.ActorCritic(68*32,16,(256,256));old.load_state_dict(data['model'],strict=True)
    with torch.no_grad():
        old_output=old(torch.as_tensor(x));new_output=source_model(torch.as_tensor(x))
        assert all(torch.equal(a,b) for a,b in zip(old_output,new_output))
    checks={};configs={};calibrations={}
    for arm,budget in ARMS.items():
        cfg=PPOConfig(classic_finish_speed_bonus=budget,classic_finish_speed_reference=REFERENCE,classic_finish_speed_width=WIDTH)
        fixture=result_fixture([REFERENCE-1/30,REFERENCE,REFERENCE+1/30,20.,120.,REFERENCE,REFERENCE,REFERENCE,REFERENCE])
        fixture['final_success'][5]=False
        fixture['reward'][6]=-200. # Native success but the existing posture requirement failed.
        fixture['truncated'][7]=True
        fixture['has_final'][8]=False
        before={k:v.copy() for k,v in fixture.items()};bonus=classic_finish_speed_bonus(fixture,cfg)
        assert all(np.array_equal(v,before[k]) for k,v in fixture.items())
        assert np.isfinite(bonus).all() and np.all((bonus>=0)&(bonus<=budget))
        assert np.all(bonus[5:]==0)
        if budget:assert bonus[0]>bonus[1]>bonus[2] and bonus[3]>bonus[4]
        else:assert np.array_equal(bonus,np.zeros_like(fixture['reward']))
        checks[arm]=dict(qualifying_bonus_faster_one_frame=float(bonus[0]),bonus_at_reference=float(bonus[1]),
            qualifying_bonus_slower_one_frame=float(bonus[2]),extra_reward_for_one_frame_saved=float(bonus[0]-bonus[1]),
            all_failure_unqualified_timeout_nonterminal_bonuses_zero=True,input_unmodified=True)
        # Both arms use the same calibration procedure and scaled units. Only
        # reward-dependent targets differ; no Actor parameter or logit changes.
        _,m=prior.load(SOURCE);actor_before=m(torch.as_tensor(x))[0].detach().clone()
        m.critic_net.load_state_dict(m.net.state_dict())
        m.critic_motion.load_state_dict(m.angular_velocity.state_dict())
        rr=np.array(raw_rewards,np.float64);terminal_bonus=float(classic_finish_speed_bonus(terminal,cfg)[0]);rr[-1]+=terminal_bonus
        target=np.empty(len(rr),np.float64);acc=0.
        for i in range(len(rr)-1,-1,-1):acc=rr[i]+.9999*acc;target[i]=acc*.01
        with torch.no_grad():
            features=m.critic_features(torch.as_tensor(x)).double()
            features=torch.cat((features,torch.ones(len(features),1,dtype=torch.float64)),1)
            regularizer=torch.eye(features.shape[1],dtype=torch.float64)*.01;regularizer[-1,-1]=0
            w=torch.linalg.solve(features.T@features+regularizer,features.T@torch.as_tensor(target))
            m.critic.weight.copy_(w[:-1].float().unsqueeze(0));m.critic.bias.copy_(w[-1:].float())
            assert torch.equal(actor_before,m(torch.as_tensor(x))[0])
            rmse=float(((m(torch.as_tensor(x))[1].double()-torch.as_tensor(target))**2).mean().sqrt())
        initial=ROOT/'checkpoints'/f'ppo_classic_speed_{arm}_initial_20261015.pt'
        d=copy.deepcopy(data);d.pop('optimizer',None);d['model']=m.state_dict()
        d['config']=dict(d['config'],checkpoint=str(initial),classic_finish_speed_bonus=budget,
            classic_finish_speed_reference=REFERENCE,classic_finish_speed_width=WIDTH,reward_scale=.01)
        d['speed_reward_initialization']=dict(source=str(SOURCE),source_sha256=prior.sha(SOURCE),budget=budget,
            reward_scale=.01,critic_calibration='Ridge on successful native rollout conditional MC returns; not an unbiased stochastic value estimate',
            critic_calibration_rmse=rmse,actor_unchanged=True,original_expert='External XYF 991-frame demonstration then local BC and PPO')
        torch.save(d,initial)
        calibrations[arm]=dict(initial=str(initial),rmse_scaled=rmse,native_terminal_bonus=terminal_bonus,
            original_return=sum(raw_rewards),shaped_return=sum(raw_rewards)+terminal_bonus,
            scaled_mc_target_min=float(target.min()),scaled_mc_target_max=float(target.max()))
        for stage in ('smoke','warm','train'):
            stem=f'ppo_classic_speed_{arm}_{stage}_20261015'
            c=dict(d['config'],checkpoint=f'checkpoints/{stem}.pt',resume_from=str(initial),run_name=stem,
                run_dir=str(ROOT/'runs'),reset_optimizer_on_resume=True,num_envs=128,num_threads=16,
                rollout_steps=1024,updates=64,save_interval=16,actor_unfreeze='full',learning_rate=3e-7,
                critic_learning_rate=1e-4,imitation_coef=.02,imitation_coef_final=.002,
                ent_coef=.0005,ent_coef_final=.0001,ent_coef_decay_updates=48,seed=2026101585)
            if stage=='smoke':c.update(num_envs=4,num_threads=2,updates=2,save_interval=1)
            elif stage=='warm':c.update(num_envs=64,updates=16,save_interval=16,learning_rate=0.,
                                       critic_learning_rate=3e-4,imitation_coef=0.,imitation_coef_final=0.)
            else:c['resume_from']=str(ROOT/'checkpoints'/f'ppo_classic_speed_{arm}_warm_20261015.pt')
            path=ROOT/'configs'/f'{stem}.json';save(path,c);configs[f'{arm}_{stage}']=str(path)
    report=dict(started_utc=datetime.now(timezone.utc).isoformat(),source=str(SOURCE),source_sha256=prior.sha(SOURCE),
        physics_sha256=prior.sha(ROOT/'bin/running_physics.dll'),bc_sha256=prior.sha(ROOT/'checkpoints/bc_classic100m_shadow.pt'),
        configs=configs,reward_checks=checks,critic_calibrations=calibrations,
        default_off_logits_values_identical_on_991_states=True,
        limitations='Same initial policy, sharp sampling and original completion rules; conditional Critic calibration applied equally to all arms. No new demonstration or exploration modification.')
    save(BASE/'initialization.json',report);emit({k:v for k,v in report.items() if k!='configs'})

def verify(arm,stage):
    cfg=read(read(BASE/'initialization.json')['configs'][f'{arm}_{stage}'])
    before,_=prior.load(cfg['resume_from']);after,m=prior.load(ROOT/cfg['checkpoint'])
    actor_names=[k for k in after['model'] if k.startswith(('net.','actor.','angular_velocity.','flat_actor_'))]
    deltas={k:float((after['model'][k]-before['model'][k]).abs().max()) for k in actor_names}
    if stage=='warm':assert all(v==0 for v in deltas.values())
    else:assert any(v>0 for v in deltas.values())
    assert all(torch.isfinite(v).all() for v in after['model'].values())
    r=prior.native(ROOT/cfg['checkpoint']);save(BASE/f'{arm}_{stage}_validation.json',dict(deltas=deltas,native=r,finite=True))
    emit(dict(arm=arm,stage=stage,greedy_success=r['success'],frames=r['frames'],actor_changed=any(v>0 for v in deltas.values())))
    assert r['success'], 'Short validation failed native completion; adjust before main training'

def sample(checkpoint,episodes,seed,output,device='cpu',num_envs=128):
    """Record ALL first episodes and action hashes, with no best-only reporting."""
    data,model=prior.load(checkpoint,device);n=min(num_envs,episodes);torch.manual_seed(seed)
    _,reference=read_replay(ROOT/'replays/classic_shadow_bc_33p03.bin')
    aliases=[0,1,2,1,4,5,6,5,8,9,10,9,4,5,6,5]
    reference_effective=[aliases[a] for a in reference]
    env=VectorRunningGameEnv(n,num_threads=16 if device=='cuda' else 8,reward_profile='classic_100m_task',game_mode='classic_100m')
    records=[];best=None;novel={}
    while len(records)<episodes:
        raw,_=env.reset();frames=np.repeat(raw[:,None,:],data['config']['frame_stack'],axis=1)
        active=np.arange(n)<min(n,episodes-len(records));hist=[[] for _ in range(n)];low=np.zeros(n,int)
        for _ in range(3601):
            low[active]+=((429.8-raw[active,26]*40)/506.85<.8)
            with torch.no_grad():
                logits,values=model(torch.as_tensor(frames.reshape(n,-1),device=device))
                assert torch.isfinite(logits).all() and torch.isfinite(values).all()
                actions=torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
            for i in np.flatnonzero(active):hist[i].append(int(actions[i]))
            r=env.step_arrays(actions)
            for i in np.flatnonzero(active&r['has_final']):
                h=hist[i];qualified=bool(r['final_success'][i] and not r['truncated'][i] and r['reward'][i]>0)
                effective=[aliases[a] for a in h]
                row=dict(success=bool(r['final_success'][i]),qualified=qualified,frames=len(h),time_s=float(r['final_time'][i]),
                    distance_m=float(r['final_distance'][i]),low_torso_fraction=float(low[i]/len(h)),
                    exact_reference_actions=h==reference,action_sha256=prior.hashlib.sha256(bytes(h)).hexdigest(),
                    effective_action_sha256=prior.hashlib.sha256(bytes(effective)).hexdigest(),effective_reference_actions=effective==reference_effective,
                    first_different_action_frame=next((j+1 for j in range(min(len(h),len(reference))) if h[j]!=reference[j]),None))
                records.append(row);active[i]=False
                if qualified and (best is None or len(h)<best['frames']):best=dict(row,actions=h.copy())
                if qualified and not row['effective_reference_actions'] and row['effective_action_sha256'] not in novel:
                    novel[row['effective_action_sha256']]=dict(row,actions=h.copy())
            if not active.any():break
            raw=r['obs'];frames=np.roll(frames,-1,axis=1);frames[:,-1,:]=raw
            done=r['terminated']|r['truncated'];frames[done]=raw[done,None,:]
        assert not active.any()
        if device=='cuda':emit(dict(progress=len(records),episodes=episodes,qualified=sum(r['qualified'] for r in records)))
    wins=[r for r in records if r['qualified']]
    result=dict(checkpoint=str(checkpoint),checkpoint_sha256=prior.sha(checkpoint),episodes=len(records),seed=seed,device=device,
        successes=sum(r['success'] for r in records),qualified_finishes=len(wins),
        qualified_best_frames=min((r['frames'] for r in wins),default=None),
        qualified_mean_time_s=float(np.mean([r['time_s'] for r in wins])) if wins else None,
        unique_qualified_action_sequences=len({r['action_sha256'] for r in wins}),
        unique_qualified_effective_sequences=len({r['effective_action_sha256'] for r in wins}),
        unique_all_action_sequences=len({r['action_sha256'] for r in records}),
        qualified_reference_sequences=sum(r['exact_reference_actions'] for r in wins),
        qualified_under_991=sum(r['frames']<991 for r in wins),
        qualified_time_distribution={str(t):sum(r['frames']==t for r in wins) for t in sorted({r['frames'] for r in wins})},
        best=best,results=records)
    if best:
        replay=ROOT/'replays'/(Path(output).stem+'.bin');write_replay(replay,best['actions'],ReplayRuleset.CLASSIC_100M);result['replay']=str(replay)
        _,actions=read_replay(replay);single=RunningGameEnv('classic_100m_task','classic_100m');single.reset()
        for i,action in enumerate(actions):
            obs,reward,done,tr,info=single.step(action)
            assert bool(done or tr)==(i==len(actions)-1)
        assert info['success'] and not tr and reward>0 and abs(info['time']-best['time_s'])<1e-5
        result['best_replay_crc_and_native_qualification_verified']=True
    novel_replays=[]
    for index,row in enumerate(novel.values(),1):
        path=ROOT/'replays'/f'{Path(output).stem}_novel{index:03d}_{row["frames"]}f.bin'
        write_replay(path,row['actions'],ReplayRuleset.CLASSIC_100M);_,actions=read_replay(path)
        single=RunningGameEnv('classic_100m_task','classic_100m');single.reset()
        for i,action in enumerate(actions):
            obs,reward,done,tr,info=single.step(action);assert bool(done or tr)==(i==len(actions)-1)
        assert info['success'] and not tr and reward>0 and abs(info['time']-row['time_s'])<1e-5
        novel_replays.append(dict(path=str(path),frames=row['frames'],time_s=row['time_s'],
            effective_action_sha256=row['effective_action_sha256'],native_qualified_reexecution=True))
    result['novel_qualified_replays']=novel_replays
    save(output,result);emit({k:v for k,v in result.items() if k not in ('results','best')})

def evaluate(arm):
    cfg=read(read(BASE/'initialization.json')['configs'][arm+'_train']);final=ROOT/cfg['checkpoint']
    candidates=[SOURCE]+sorted(final.parent.glob(final.stem+'_update_*.pt'))+[final];rows=[]
    for checkpoint in dict.fromkeys(candidates):
        r=prior.native(checkpoint);save(BASE/(checkpoint.stem+'_native.json'),r)
        row=dict(checkpoint=str(checkpoint),native_success=r['success'],native_qualified=bool(r['success'] and r['task_reward']>0),
                 frames=r['frames'],time_s=r['time_s'],distance_m=r['distance_m'])
        if row['native_qualified']:
            path=BASE/(arm+'_'+checkpoint.stem+'_selection64.json');sample(checkpoint,64,2026101586,path);s=read(path)
            row.update(qualified_finishes=s['qualified_finishes'],unique_qualified_action_sequences=s['unique_qualified_action_sequences'],
                qualified_under_991=s['qualified_under_991'],qualified_mean_time_s=s['qualified_mean_time_s'])
        rows.append(row);emit(row)
    valid=[r for r in rows if r['native_qualified'] and r.get('qualified_finishes',0)>=48]
    assert valid
    valid.sort(key=lambda r:(r['frames'],-r['qualified_finishes'],r['qualified_mean_time_s']))
    save(BASE/(arm+'_selection.json'),dict(arm=arm,chosen=valid[0],candidates=rows,selection_seed=2026101586,
        rule='Native greedy success; >=48/64 qualified sampled finishes; shortest frames then qualified finish count then mean time'))
    emit(dict(selected=valid[0]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','verify','evaluate','sample','search'])
    p.add_argument('--arm',choices=list(ARMS));p.add_argument('--stage',choices=['smoke','warm','train'])
    p.add_argument('--checkpoint',type=Path);p.add_argument('--episodes',type=int,default=64);p.add_argument('--seed',type=int,default=2026101586);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='train':prior.train(read(BASE/'initialization.json')['configs'][a.arm+'_'+a.stage])
    elif a.mode=='verify':verify(a.arm,a.stage)
    elif a.mode=='evaluate':evaluate(a.arm)
    elif a.mode=='sample':sample(a.checkpoint,a.episodes,a.seed,a.output)
    else:sample(a.checkpoint,a.episodes,a.seed,a.output,device='cuda',num_envs=256)
