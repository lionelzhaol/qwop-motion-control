"""Isolated classic100m experiments using the existing PPO and native rules."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]/'QWOP_rl_training_public'
sys.path.insert(0,str(ROOT/'scripts'))
from train_ppo import ActorCritic,FrameStacker,PPOConfig
from running_env import RunningGameEnv,VectorRunningGameEnv
from replay_io import read_replay,write_replay,ReplayRuleset

torch.set_num_threads(1)
def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(checkpoint,device='cpu'):
    d=torch.load(checkpoint,map_location=device,weights_only=False);c=d['config']
    m=ActorCritic(68*c['frame_stack'],16,tuple(c['hidden_sizes'])).to(device)
    m.load_state_dict(d['model'],strict=True);m.eval();return d,m
def native(checkpoint):
    d,m=load(checkpoint);c=d['config'];env=RunningGameEnv('classic_100m_task','classic_100m')
    raw,_=env.reset();st=FrameStacker(1,68,c['frame_stack']);obs=st.reset(raw[None])
    actions=[];low=0;reward_sum=0.
    for step in range(3601):
        low+=int((429.8-raw[26]*40)/506.85<.8)
        with torch.no_grad():
            logits,v=m(torch.as_tensor(obs));assert torch.isfinite(logits).all() and torch.isfinite(v).all()
            action=int(logits.argmax(-1)[0])
        raw,r,done,tr,info=env.step(action);reward_sum+=r;actions.append(action)
        assert np.isfinite(raw).all() and math.isfinite(r)
        if done or tr:break
        obs=st.update(raw[None],np.array([False]))
    assert done or tr
    return dict(checkpoint=str(checkpoint),success=bool(info['success']),truncated=bool(tr),
        time_s=float(info['time']),distance_m=float(info['distance']),frames=len(actions),
        low_torso_fraction=low/len(actions),task_reward=reward_sum,actions=actions)
def prepare():
    source=ROOT/'checkpoints/bc_classic100m_shadow.pt';data,model=load(source)
    expert=BASE.parent/'shadow_audit/expert_dataset.npz'
    with np.load(expert,allow_pickle=False) as e:x=e['observations'];y=e['actions']
    # Default-off extension must leave legacy inference bit-identical.
    sys.path.insert(0,str(BASE));from train_ppo_before import ActorCritic as OldModel
    old=OldModel(x.shape[1],16,tuple(data['config']['hidden_sizes']))
    old.load_state_dict(data['model'],strict=True)
    with torch.no_grad():
        before=model(torch.as_tensor(x));prior=old(torch.as_tensor(x))
        assert all(torch.equal(a,b) for a,b in zip(before,prior))
        assert set(old.state_dict())==set(model.state_dict())
        model.actor.weight.mul_(5.);model.actor.bias.mul_(5.)
    model.enable_flat_residual()
    with torch.no_grad():model.critic.weight.zero_();model.critic.bias.zero_()
    initial=ROOT/'checkpoints/ppo_classic_shadow_local_initial_20261015.pt'
    out=copy.deepcopy(data);out.pop('optimizer',None);out['update']=0;out['global_step']=0
    out['model']=model.state_dict();out['config']=dict(data['config'],checkpoint=str(initial),
        reward_profile='classic_100m_task',separate_critic_features=True,
        enable_flat_actor_residual=True,freeze_backbone=True)
    out['local_improvement_initialization']=dict(source=str(source),source_sha256=sha(source),
        actor_logit_multiplier=5.,sampling_temperature_equivalent=.2,
        zero_local_residual=True,critic_head_reset=True,original_expert='XYF 991-frame public demonstration')
    torch.save(out,initial)
    baseline=native(source);initialized=native(initial)
    assert initialized['actions']==baseline['actions'] and baseline['frames']==991 and baseline['success']
    c=asdict(PPOConfig());c.update({k:v for k,v in data['config'].items() if k in c})
    c.update(game_mode='classic_100m',reward_profile='classic_100m_task',num_envs=128,num_threads=16,
        rollout_steps=1024,updates=128,epochs=4,minibatches=8,gamma=.9999,gae_lambda=.995,
        clip_coef=.2,target_kl=.006,ent_coef=.0005,ent_coef_final=.0001,ent_coef_decay_updates=96,
        learning_rate=.0001,critic_learning_rate=.0001,vf_coef=.1,seed=2026101501,
        enable_flat_actor_residual=True,freeze_backbone=True,separate_critic_features=True,
        reset_optimizer_on_resume=True,effective_action_entropy=True,
        imitation_dataset=str(expert),imitation_coef=.02,imitation_coef_final=.002,
        resume_from=str(initial),save_interval=8,rnd_beta=0.,run_dir=str(ROOT/'runs'))
    configs={}
    for tag in ('smoke','critic_warm','v1'):
        cc=copy.deepcopy(c);stem=f'ppo_classic_shadow_local_{tag}_20261015'
        cc.update(checkpoint=f'checkpoints/{stem}.pt',run_name=stem)
        if tag=='smoke':cc.update(num_envs=4,num_threads=2,updates=2,save_interval=1)
        elif tag=='critic_warm':cc.update(num_envs=64,updates=32,learning_rate=0.,critic_learning_rate=.0003,
            imitation_coef=0.,imitation_coef_final=0.,save_interval=16)
        else:cc['resume_from']=str(ROOT/'checkpoints/ppo_classic_shadow_local_critic_warm_20261015.pt')
        cc={k:str(v) if isinstance(v,Path) else v for k,v in cc.items()}
        p=ROOT/'configs'/f'{stem}.json';write_json(p,cc);configs[tag]=str(p)
    write_json(BASE/'initialization.json',dict(task_started_utc='2026-10-15T02:57:16+00:00',
        source=str(source),source_sha256=sha(source),physics_sha256=sha(ROOT/'bin/running_physics.dll'),
        initial=str(initial),baseline=baseline,initialized=initialized,configs=configs,
        default_extension_logits_values_identical=True,expert_samples=len(x)))
    print(json.dumps(dict(configs=configs,baseline_frames=baseline['frames'],initialized_frames=initialized['frames'],
        default_compatibility_verified=True),ensure_ascii=False),flush=True)
def train(config):
    config=Path(config).resolve();c=json.loads(config.read_text(encoding='utf-8'))
    started=datetime.now(timezone.utc);timer=time.perf_counter();log=BASE/(config.stem+'.log')
    env={k.upper():v for k,v in os.environ.items()}
    p=subprocess.Popen([sys.executable,'-u','-X','utf8','scripts/train_ppo.py','--config',str(config)],
        cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    count=0;last=''
    with log.open('w',encoding='utf-8') as target:
        for line in p.stdout:
            target.write(line);target.flush()
            if line.startswith('update='):
                count+=1;last=line.strip()
                for key in ('policy_loss','value_loss','entropy','kl','imitation_loss'):
                    match=re.search(key+r'=([^ ]+)',line)
                    if match and not math.isfinite(float(match[1])):
                        p.terminate();raise RuntimeError('Nonfinite '+key)
            if line.startswith('saved_checkpoint='):print(last,flush=True)
    code=p.wait();report=dict(config=str(config),returncode=code,elapsed_seconds=time.perf_counter()-timer,
        started_utc=started.isoformat(),ended_utc=datetime.now(timezone.utc).isoformat(),
        updates=count,new_transitions=count*c['num_envs']*c['rollout_steps'],finite_losses=code==0 and count>0)
    write_json(BASE/(config.stem+'_time.json'),report);print(json.dumps(report),flush=True)
    if code:raise RuntimeError('Training failed: '+str(log))
def evaluate(pattern):
    reports=[]
    for checkpoint in sorted((ROOT/'checkpoints').glob(pattern)):
        path=BASE/(checkpoint.stem+'_native.json')
        r=json.loads(path.read_text(encoding='utf-8')) if path.exists() else native(checkpoint)
        if not path.exists():write_json(path,r)
        reports.append(r);print(json.dumps({k:r[k] for k in ('checkpoint','success','time_s','distance_m','frames')},ensure_ascii=False),flush=True)
    reports.sort(key=lambda r:(not r['success'],r['time_s'] if r['success'] else -r['distance_m']))
    write_json(BASE/'native_candidates.json',reports)
def prepare_v2():
    source=ROOT/'checkpoints/ppo_classic_shadow_local_v1_20261015.pt';d,m=load(source)
    original=native(source)
    with torch.no_grad():
        # Within the existing flat gate only, halve the complete actor logits.
        # A positive scale keeps every greedy action unchanged at initialization.
        m.flat_actor_weight.mul_(.5).add_(m.actor.weight,alpha=-.5)
        m.flat_actor_bias.mul_(.5).add_(m.actor.bias,alpha=-.5)
    target=ROOT/'checkpoints/ppo_classic_shadow_explore_initial_20261015.pt'
    d.pop('optimizer',None);d['model']=m.state_dict();d['update']=0;d['global_step']=0
    d['config']=dict(d['config'],checkpoint=str(target))
    d['exploration_initialization']=dict(source=str(source),source_sha256=sha(source),
        flat_logit_multiplier=.5,greedy_preserved=True,phase='10-45m and65-100m only')
    torch.save(d,target);new=native(target)
    assert new['actions']==original['actions'] and new['success']
    c=json.loads((ROOT/'configs/ppo_classic_shadow_local_v1_20261015.json').read_text(encoding='utf-8'))
    configs={}
    for tag in ('warm','v2'):
        stem=f'ppo_classic_shadow_explore_{tag}_20261015'
        cc=dict(c,checkpoint=f'checkpoints/{stem}.pt',run_name=stem,seed=2026101502)
        if tag=='warm':
            cc.update(resume_from=str(target),num_envs=64,updates=8,save_interval=8,
                learning_rate=0.,critic_learning_rate=.0003,imitation_coef=0.,imitation_coef_final=0.)
        else:
            cc.update(resume_from=str(ROOT/'checkpoints/ppo_classic_shadow_explore_warm_20261015.pt'),
                updates=64,learning_rate=.0003,ent_coef=.002,ent_coef_final=.0003,ent_coef_decay_updates=48,
                imitation_coef=.005,imitation_coef_final=.0005)
        p=ROOT/'configs'/f'{stem}.json';write_json(p,cc);configs[tag]=str(p)
    write_json(BASE/'explore_initialization.json',dict(source=str(source),initial=str(target),
        frames=new['frames'],greedy_actions_preserved=True,configs=configs))
    print(json.dumps(dict(configs=configs,frames=new['frames'],greedy_actions_preserved=True)),flush=True)
def prepare_v3():
    from train_ppo import compute_gae
    source=ROOT/'checkpoints/ppo_classic_shadow_explore_v2_20261015.pt';d,m=load(source)
    original=native(source);assert original['success'] and original['frames']==991
    dataset=np.load(BASE.parent/'shadow_audit/expert_dataset.npz',allow_pickle=False)
    x=torch.as_tensor(dataset['observations'])
    e=RunningGameEnv('classic_100m_task','classic_100m');e.reset()
    actions=read_replay(ROOT/'replays/classic_shadow_bc_33p03.bin')[1]
    rewards=np.array([e.step(a)[1] for a in actions],dtype=np.float64)
    g=0.;targets=[]
    for r in rewards[::-1]:g=.01*r+.9999*g;targets.append(g)
    y=torch.as_tensor(np.array(targets[::-1]),dtype=torch.float64)
    # Restore informative Actor features to the independent Critic, then fit
    # scaled successful-demonstration MC returns. No old data enters PPO buffer.
    m.critic_net.load_state_dict(m.net.state_dict())
    m.critic_motion.load_state_dict(m.angular_velocity.state_dict())
    with torch.no_grad():h=m.critic_features(x).double()
    hm=h.mean(0);ym=y.mean();hc=h-hm;yc=y-ym
    w=torch.linalg.solve(hc.T@hc+.01*torch.eye(h.shape[1],dtype=torch.float64),hc.T@yc)
    b=ym-hm@w
    with torch.no_grad():m.critic.weight.copy_(w.float()[None]);m.critic.bias.copy_(b.float().reshape(1))
    with torch.no_grad():prediction=m(x)[1].double()
    rmse=float(((prediction-y).square().mean()).sqrt());assert rmse<.1 and torch.isfinite(prediction).all()
    # Scaling rewards and values together must scale GAE/returns linearly.
    torch.manual_seed(2026101503);rr=torch.randn(16,4);vv=torch.randn(16,4);dd=torch.zeros(16,4);dd[7,1]=1
    nv=torch.randn(4);cfg=PPOConfig(num_envs=4,rollout_steps=16,gamma=.9999,gae_lambda=.995)
    aa,ret=compute_gae(rr,dd,vv,nv,cfg);bb,scaledret=compute_gae(rr*.01,dd,vv*.01,nv*.01,cfg)
    assert torch.allclose(aa*.01,bb,atol=1e-6) and torch.allclose(ret*.01,scaledret,atol=1e-6)
    target=ROOT/'checkpoints/ppo_classic_shadow_calibrated_initial_20261015.pt'
    d.pop('optimizer',None);d['model']=m.state_dict();d['update']=0;d['global_step']=0
    d['config']=dict(d['config'],checkpoint=str(target),reward_scale=.01)
    d['critic_calibration']=dict(source=str(source),source_sha256=sha(source),reward_scale=.01,
        method='Ridge fit to successful expert Monte Carlo returns, then on-policy PPO',
        ridge=.01,expert_rmse_scaled=rmse,actor_unchanged=True,
        limitation='Demonstration-conditioned initialization, not an unbiased value estimate for sampled recovery states')
    torch.save(d,target);new=native(target);assert new['actions']==original['actions']
    c=json.loads((ROOT/'configs/ppo_classic_shadow_explore_v2_20261015.json').read_text(encoding='utf-8'))
    configs={}
    for tag in ('smoke','v3'):
        stem=f'ppo_classic_shadow_calibrated_{tag}_20261015'
        cc=dict(c,resume_from=str(target),checkpoint=f'checkpoints/{stem}.pt',run_name=stem,
            seed=2026101503,reward_scale=.01,learning_rate=.0001,critic_learning_rate=.0003,
            vf_coef=.5,ent_coef=.001,ent_coef_final=.0002,imitation_coef=.002,imitation_coef_final=.0002)
        if tag=='smoke':cc.update(num_envs=4,num_threads=2,updates=2,save_interval=1)
        p=ROOT/'configs'/f'{stem}.json';write_json(p,cc);configs[tag]=str(p)
    report=dict(configs=configs,initial=str(target),demonstration_scaled_value_rmse=rmse,
        target_quantiles=np.quantile(y.numpy(),[0,.5,1]).tolist(),greedy_preserved=True,
        gae_scaling_verified=True,source=str(source))
    write_json(BASE/'critic_calibrated_initialization.json',report);print(json.dumps(report),flush=True)
def sample(checkpoint,episodes,seed,output):
    started=time.perf_counter();torch.manual_seed(seed);d,m=load(checkpoint)
    n=min(128,episodes);env=VectorRunningGameEnv(n,num_threads=8,reward_profile='classic_100m_task',game_mode='classic_100m')
    records=[];best=None
    while len(records)<episodes:
        raw,_=env.reset();st=FrameStacker(n,68,d['config']['frame_stack']);obs=st.reset(raw)
        active=np.arange(n)<min(n,episodes-len(records));hist=[[] for _ in range(n)];low=np.zeros(n,dtype=int)
        for _ in range(3601):
            low[active]+=((429.8-raw[active,26]*40)/506.85<.8)
            with torch.no_grad():
                logits,v=m(torch.as_tensor(obs));assert torch.isfinite(logits).all() and torch.isfinite(v).all()
                actions=torch.distributions.Categorical(logits=logits).sample().numpy().astype(np.int64)
            for i in np.flatnonzero(active):hist[i].append(int(actions[i]))
            r=env.step_arrays(actions);done=r['terminated']|r['truncated']
            for i in np.flatnonzero(active&r['has_final']):
                entry=dict(success=bool(r['final_success'][i]),time_s=float(r['final_time'][i]),
                    distance_m=float(r['final_distance'][i]),frames=len(hist[i]),low_torso_fraction=float(low[i]/len(hist[i])),
                    truncated=bool(r['truncated'][i]))
                records.append(entry);active[i]=False
                if entry['success'] and entry['low_torso_fraction']<=.1 and (best is None or entry['frames']<best['frames']):
                    best=dict(entry,actions=hist[i].copy());print(json.dumps(dict(new_best={k:v for k,v in entry.items()})),flush=True)
            if not active.any():break
            raw=r['obs'];obs=st.update(raw,done)
        assert not active.any()
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),episodes=episodes,seed=seed,
        successes=sum(r['success'] for r in records),under33p03=sum(r['success'] and r['frames']<991 for r in records),
        best=best,elapsed_seconds=time.perf_counter()-started,results=records)
    if best is not None:
        replay=ROOT/'replays'/(Path(output).stem+'.bin');write_replay(replay,best['actions'],ReplayRuleset.CLASSIC_100M)
        report['replay']=str(replay)
    write_json(output,report);print(json.dumps({k:v for k,v in report.items() if k not in ('results','best')},ensure_ascii=False),flush=True)
def search(checkpoint,episodes,seed,output):
    checkpoint=Path(checkpoint).resolve();output=Path(output).resolve()
    timer=time.perf_counter();utc=datetime.now(timezone.utc);log=BASE/(output.stem+'_search.log')
    child="import sys,torch,runpy;sys.path.insert(0,'scripts');torch.set_num_threads(1);runpy.run_path('scripts/search_policy_best.py',run_name='__main__')"
    argv=[sys.executable,'-u','-X','utf8','-c',child,'--checkpoint',str(checkpoint),'--output',str(output),
        '--episodes',str(episodes),'--num-envs','256','--num-threads','16','--device','cuda',
        '--seed',str(seed),'--progress-interval','512']
    p=subprocess.Popen(argv,cwd=ROOT,env={k.upper():v for k,v in os.environ.items()},
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    lines=[]
    with log.open('w',encoding='utf-8') as target:
        for line in p.stdout:target.write(line);target.flush();lines.append(line);print(line.rstrip(),flush=True)
    code=p.wait()
    if code:raise RuntimeError('Search failed: '+str(log))
    rules,actions=read_replay(output);assert rules==ReplayRuleset.CLASSIC_100M
    env=RunningGameEnv('classic_100m_task','classic_100m');raw,_=env.reset();low=0
    for i,a in enumerate(actions):
        low+=int((429.8-raw[26]*40)/506.85<.8)
        raw,reward,done,tr,info=env.step(a)
        assert np.isfinite(raw).all() and math.isfinite(reward)
        assert bool(done or tr)==(i==len(actions)-1)
    assert info['success'] and not tr
    final=next(line for line in reversed(lines) if line.startswith('saved_replay='))
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),output=str(output),
        replay_sha256=sha(output),episodes=episodes,seed=seed,frames=len(actions),time_s=float(info['time']),
        native_success=True,low_torso_fraction=low/len(actions),faster_than_reference=len(actions)<991,
        replay_crc_and_native_reexecution=True,elapsed_seconds=time.perf_counter()-timer,
        started_utc=utc.isoformat(),ended_utc=datetime.now(timezone.utc).isoformat(),author_search_summary=final.strip(),
        protocol='Author asynchronous vector search; best-of-completed-episodes, separate from first-episode heldout performance')
    write_json(BASE/(output.stem+'_search.json'),report);print(json.dumps(report,ensure_ascii=False),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','prepare-v2','prepare-v3','train','evaluate','sample','search'])
    p.add_argument('--config',type=Path);p.add_argument('--pattern');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--episodes',type=int,default=128);p.add_argument('--seed',type=int,default=2026101511)
    p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='prepare-v2':prepare_v2()
    elif a.mode=='prepare-v3':prepare_v3()
    elif a.mode=='train':train(a.config)
    elif a.mode=='evaluate':evaluate(a.pattern)
    elif a.mode=='search':search(a.checkpoint,a.episodes,a.seed,a.output)
    else:sample(a.checkpoint,a.episodes,a.seed,a.output)
if __name__=='__main__':main()
