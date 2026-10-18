"""Reward validation, isolated experiment configs and native model evaluation.

Uses the project's original PPO entry point and environments. Evaluation always
uses original long-jump scoring independently of the training reward profile.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
WORK = BASE.parents[1]
ROOT = WORK / 'QWOP_rl_training_public'
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(WORK/'verification'))
from running_env import RunningGameEnv, VectorRunningGameEnv, infer_game_mode
from train_ppo import ActorCritic, FrameStacker
from replay_io import read_replay, write_replay, ReplayRuleset
from evaluate_long_jump import evaluate

torch.set_num_threads(1)
BASE.mkdir(exist_ok=True)

def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_reward():
    assert infer_game_mode('long_jump_task')=='long_jump_30m'
    paths=['long_jump_shadow_ppo_deterministic_2p59m.bin',
           'long_jump_shadow_ppo_sampled_best_3p16m.bin',
           'long_jump_ppo_sampled_best.bin']
    results=[]
    for filename in paths:
        rules,actions=read_replay(ROOT/'replays'/filename)
        assert rules==ReplayRuleset.LONG_JUMP_30M
        old=RunningGameEnv('long_jump_30m','long_jump_30m')
        new=RunningGameEnv('long_jump_task','auto')
        x,info=old.reset();y,other=new.reset();assert np.array_equal(x,y) and info==other
        # Reward progress uses view_target_x, not backend score, which can
        # change semantics even when the native jump stage first becomes active.
        progress=float(np.clip(math.floor(300-float(x[62])+.5)/10,0,30))
        potential=0.;shaping_discounted=0.;native_shaping_discounted=0.;reward_sum=0.;old_sum=0.
        score_drops=0
        for i,a in enumerate(actions):
            x,r,d,tr,info=old.step(a);y,rn,dn,trn,other=new.step(a)
            assert np.array_equal(x,y) and info==other and (d,tr)==(dn,trn)
            assert bool(d or tr)==(i==len(actions)-1)
            assert np.isfinite(y).all() and math.isfinite(rn)
            landed=abs(info['distance']-(30+info['score']))<1e-4
            # Jump distance is the final/public native mark, not runup score.
            current=8*float(np.clip(info['score'],0,9)) if landed and not(d or tr) else 0.
            shaping=.999*current-potential
            if current<potential and landed and not(d or tr):score_drops+=1
            runup=float(np.clip(math.floor(300-float(x[62])+.5)/10,0,30))
            base=.2*max(0.,runup-progress);progress=max(progress,runup)
            if d or tr:
                base+=10+40*float(np.clip(info['score'],0,9)) if info['success'] else -30
            assert abs(rn-base-shaping)<3e-4,(filename,i,rn,base,shaping)
            shaping_discounted+=(.999**i)*shaping
            native_shaping_discounted+=(.999**i)*(rn-base)
            potential=current;reward_sum+=rn;old_sum+=r
        assert info['success'] and abs(shaping_discounted)<1e-9
        assert abs(native_shaping_discounted)<.002
        # Reset must clear the progress and potential of the previous episode.
        y,_=new.reset();fresh=RunningGameEnv('long_jump_task');z,_=fresh.reset()
        assert np.array_equal(y,z)
        for a in actions[:30]:
            assert new.step(a)[1:]==fresh.step(a)[1:]
        results.append(dict(replay=filename,final_jump_m=info['score'],frames=len(actions),
                            old_reward=old_sum,new_reward=reward_sum,
                            discounted_shaping_sum=shaping_discounted,
                            native_discounted_shaping_sum=native_shaping_discounted,score_drop_frames=score_drops,
                            native_trajectory_identical=True,reset_state_verified=True))
    for a in (0,1,5,10):
        env=RunningGameEnv('long_jump_task');raw,info=env.reset();total=0.
        for i in range(1351):
            raw,r,d,tr,info=env.step(a);total+=r
            assert np.isfinite(raw).all() and math.isfinite(r)
            if d or tr:break
        assert d or tr
        if not info['success']:assert total<=-23.999
        results.append(dict(constant_action=a,success=info['success'],truncated=tr,total_reward=total))
    write_json(BASE/'reward_validation.json',dict(passed=True,results=results,
        formula='F=.999*Phi(next)-Phi(previous), Phi=8*official jump; terminal Phi=0',
        discount=.999,original_rules_unchanged=True))
    print(json.dumps(dict(reward_validation='passed',results=results),ensure_ascii=False),flush=True)

def prepare():
    source=ROOT/'checkpoints/ppo_long_jump_shadow_best.pt'
    data=torch.load(source,map_location='cpu',weights_only=False)
    source_hash=sha(source)
    data=copy.deepcopy(data);data.pop('optimizer',None)
    data['model']['critic.weight'].zero_();data['model']['critic.bias'].zero_()
    initial=ROOT/'checkpoints/ppo_long_jump_distance_initial_20261013.pt'
    data['update']=0;data['global_step']=0
    data['config'].update(reward_profile='long_jump_task',game_mode='long_jump_30m',checkpoint=str(initial),
        reset_optimizer_on_resume=True,rnd_beta=0.)
    data['jump_distance_reward_initialization']=dict(source=str(source),source_sha256=source_hash,
        source_update=256,source_global_step=9961472,actor_unchanged=True,
        critic_head_reset=True,experiment_counters_reset=True)
    torch.save(data,initial)
    cfg=json.loads((ROOT/'configs/ppo_long_jump_shadow_v4.json').read_text(encoding='utf-8'))
    cfg.update(reward_profile='long_jump_task',resume_from=str(initial),seed=2026101301,
               num_envs=128,updates=256,save_interval=16)
    configs={}
    for tag in ('smoke','critic_warm','v5'):
        c=copy.deepcopy(cfg);stem=f'ppo_long_jump_distance_{tag}_20261013'
        c.update(run_name=stem,checkpoint=f'checkpoints/{stem}.pt')
        if tag=='smoke':c.update(num_envs=4,num_threads=2,updates=4,save_interval=2)
        if tag=='critic_warm':c.update(num_envs=64,updates=16,save_interval=8,
                                     learning_rate=0.,critic_learning_rate=.0003,vf_coef=.5)
        if tag=='v5':c.update(resume_from=str(ROOT/'checkpoints/ppo_long_jump_distance_critic_warm_20261013.pt'))
        config=ROOT/'configs'/f'{stem}.json';write_json(config,c);configs[tag]=str(config)
    initial_eval=evaluate(initial)
    assert initial_eval['success'] and abs(initial_eval['long_jump_m']-2.588183641433716)<1e-7
    assert sha(source)==source_hash
    write_json(BASE/'initialization.json',dict(initial=str(initial),configs=configs,
        source_sha256=source_hash,physics_sha256=sha(ROOT/'bin/running_physics.dll'),
        initialized_deterministic_jump_m=initial_eval['long_jump_m']))
    print(json.dumps(dict(initial=str(initial),configs=configs,actor_trajectory_unchanged=True)),flush=True)

def train(config):
    config=Path(config).resolve();cfg=json.loads(config.read_text(encoding='utf-8'))
    log=BASE/(config.stem+'.log');started=datetime.now(timezone.utc);timer=time.perf_counter()
    env={k.upper():v for k,v in __import__('os').environ.items()}
    proc=subprocess.Popen([sys.executable,'-u','-X','utf8','scripts/train_ppo.py','--config',str(config)],
        cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    last=None;updates=0
    with log.open('w',encoding='utf-8') as target:
        for line in proc.stdout:
            target.write(line);target.flush()
            if line.startswith('update='):
                last=line.strip();updates+=1
                for field in ('policy_loss','value_loss','entropy','kl'):
                    found=re.search(field+r'=([^ ]+)',line)
                    if found and not math.isfinite(float(found.group(1))):
                        proc.terminate();raise RuntimeError('Non-finite '+field)
            if line.startswith('saved_checkpoint='):
                print(last or line.strip(),flush=True)
    code=proc.wait();elapsed=time.perf_counter()-timer
    report=dict(config=str(config),returncode=code,started_utc=started.isoformat(),
        ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=elapsed,
        updates_logged=updates,transitions=updates*cfg['num_envs']*cfg['rollout_steps'],
        all_logged_losses_finite=updates>0 and code==0,log=str(log))
    write_json(BASE/(config.stem+'_time.json'),report)
    print(json.dumps(report),flush=True)
    if code:raise RuntimeError('Training failed; see '+str(log))

def prepare_local_state():
    source=ROOT/'checkpoints/ppo_long_jump_distance_initial_20261013.pt'
    data=torch.load(source,map_location='cpu',weights_only=False)
    cfg=data['config'];torch.manual_seed(2026101303)
    model=ActorCritic(68*cfg['frame_stack'],16,tuple(cfg['hidden_sizes']))
    model.load_state_dict(data['model'],strict=True)
    model.enable_jump_state_residual(128)
    initial=ROOT/'checkpoints/ppo_long_jump_local_state_initial_20261013.pt'
    data['model']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    data['config']=dict(cfg,jump_residual_hidden_size=128,checkpoint=str(initial))
    data['local_state_initialization']=dict(source=str(source),source_sha256=sha(source),
        hidden_size=128,input='latest normalized 68 values + 12 angular velocities',
        zero_output_preserves_source=True,seed=2026101303)
    torch.save(data,initial)
    evaluation=evaluate(initial)
    assert evaluation['success'] and abs(evaluation['long_jump_m']-2.588183641433716)<1e-7
    c=json.loads((ROOT/'configs/ppo_long_jump_distance_v5_20261013.json').read_text(encoding='utf-8'))
    configs={}
    for tag in ('smoke','critic_warm','v6'):
        stem=f'ppo_long_jump_local_state_{tag}_20261013'
        cc=dict(c,jump_residual_hidden_size=128,run_name=stem,checkpoint=f'checkpoints/{stem}.pt',
            resume_from=str(ROOT/'checkpoints/ppo_long_jump_self_imitation_v2_20261013.pt'),seed=2026101304)
        if tag=='smoke':cc.update(num_envs=4,num_threads=2,updates=4,save_interval=2)
        elif tag=='critic_warm':cc.update(num_envs=64,updates=16,save_interval=8,
            learning_rate=0.,critic_learning_rate=.0003,vf_coef=.5)
        else:cc.update(resume_from=str(ROOT/'checkpoints/ppo_long_jump_local_state_critic_warm_20261013.pt'))
        p=ROOT/'configs'/f'{stem}.json';write_json(p,cc);configs[tag]=str(p)
    write_json(BASE/'local_state_initialization.json',dict(initial=str(initial),configs=configs,
        native_distance_m=evaluation['long_jump_m'],source_preserved=True))
    print(json.dumps(dict(initial=str(initial),configs=configs,native_distance_m=evaluation['long_jump_m'])),flush=True)

def search_replay(checkpoint,output,episodes,seed):
    checkpoint=Path(checkpoint).resolve();output=Path(output).resolve()
    log=BASE/(output.stem+'_search.log');started=datetime.now(timezone.utc);timer=time.perf_counter()
    child_code="import sys,torch,runpy;sys.path.insert(0,'scripts');torch.set_num_threads(1);runpy.run_path('scripts/search_policy_best.py',run_name='__main__')"
    argv=[sys.executable,'-u','-X','utf8','-c',child_code,'--checkpoint',str(checkpoint),
          '--output',str(output),'--episodes',str(episodes),'--num-envs','128',
          '--num-threads','8','--seed',str(seed),'--device','cpu','--progress-interval','256']
    env={k.upper():v for k,v in __import__('os').environ.items()}
    proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,encoding='utf-8')
    with log.open('w',encoding='utf-8') as target:
        for line in proc.stdout:
            target.write(line);target.flush();print(line.rstrip(),flush=True)
    code=proc.wait();elapsed=time.perf_counter()-timer
    if code:raise RuntimeError('Search failed; see '+str(log))
    rules,actions=read_replay(output);assert rules==ReplayRuleset.LONG_JUMP_30M
    native=RunningGameEnv('long_jump_30m','long_jump_30m');native.reset()
    for i,a in enumerate(actions):
        raw,reward,done,truncated,info=native.step(a)
        assert np.isfinite(raw).all() and math.isfinite(reward)
        assert bool(done or truncated)==(i==len(actions)-1)
    assert info['success'] and not truncated
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),output=str(output),
        replay_sha256=sha(output),seed=seed,episodes_searched=episodes,device='cpu',
        elapsed_seconds=elapsed,started_utc=started.isoformat(),ended_utc=datetime.now(timezone.utc).isoformat(),
        log=str(log),frames=len(actions),official_jump_m=float(info['score']),native_success=True,
        final_time_s=float(info['time']),native_rules='long_jump_30m',replay_reexecution_verified=True)
    write_json(BASE/(output.stem+'_search.json'),report);print(json.dumps(report,ensure_ascii=False),flush=True)

def prepare_v7():
    c=json.loads((ROOT/'configs/ppo_long_jump_local_state_v6_20261013.json').read_text(encoding='utf-8'))
    configs={}
    for tag in ('critic_warm','v7'):
        stem=f'ppo_long_jump_elite_{tag}_20261013'
        cc=dict(c,run_name=stem,checkpoint=f'checkpoints/{stem}.pt',seed=2026101306)
        if tag=='critic_warm':
            cc.update(resume_from=str(ROOT/'checkpoints/ppo_long_jump_self_imitation_v3_20261013.pt'),
                num_envs=64,updates=16,save_interval=8,learning_rate=0.,critic_learning_rate=.0003,vf_coef=.5)
        else:
            cc.update(resume_from=str(ROOT/'checkpoints/ppo_long_jump_elite_critic_warm_20261013.pt'),
                updates=128,ent_coef=.001,ent_coef_final=.0003,ent_coef_decay_updates=96)
        p=ROOT/'configs'/f'{stem}.json';write_json(p,cc);configs[tag]=str(p)
    print(json.dumps(dict(configs=configs)),flush=True)

def stochastic(checkpoint,episodes,seed):
    torch.manual_seed(seed)
    data=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=data['config']
    env=VectorRunningGameEnv(episodes,num_threads=8,reward_profile='long_jump_30m',game_mode='long_jump_30m')
    raw,_=env.reset();stack=FrameStacker(episodes,68,cfg['frame_stack']);obs=stack.reset(raw)
    model=ActorCritic(stack.obs_dim,16,tuple(cfg['hidden_sizes']));model.load_state_dict(data['model'],strict=True);model.eval()
    active=np.ones(episodes,dtype=bool);results=[]
    for _ in range(1351):
        with torch.no_grad():
            logits,v=model(torch.as_tensor(obs));assert torch.isfinite(logits).all() and torch.isfinite(v).all()
            actions=torch.distributions.Categorical(logits=logits).sample().numpy().astype(np.int64)
        r=env.step_arrays(actions);ended=r['terminated']|r['truncated']
        assert np.isfinite(r['obs']).all() and np.isfinite(r['reward']).all()
        for i in np.flatnonzero(active&r['has_final']):
            success=bool(r['final_success'][i]);score=float(r['final_score'][i])
            results.append(dict(env_index=int(i),success=success,jump_m=score if success else None,
                                time_s=float(r['final_time'][i]),truncated=bool(r['truncated'][i])))
            active[i]=False
        if not active.any():break
        obs=stack.update(r['obs'],ended)
    assert len(results)==episodes
    scores=[r['jump_m'] for r in results if r['success']]
    return dict(checkpoint=str(checkpoint),seed=seed,episodes=episodes,successes=len(scores),
        successful_mean_m=float(np.mean(scores)) if scores else None,
        success_weighted_mean_m=sum(scores)/episodes,median_successful_m=float(np.median(scores)) if scores else None,
        best_successful_m=max(scores) if scores else None,over3m=sum(x>=3 for x in scores),
        results=sorted(results,key=lambda x:x['env_index']))

def evaluate_candidates(pattern):
    reports=[]
    for checkpoint in sorted((ROOT/'checkpoints').glob(pattern)):
        out=BASE/(checkpoint.stem+'_native.json')
        report=json.loads(out.read_text(encoding='utf-8')) if out.exists() else evaluate(checkpoint)
        if not out.exists():write_json(out,report)
        reports.append(report)
        print(json.dumps({k:report[k] for k in ('checkpoint','success','long_jump_m','time_s')},ensure_ascii=False),flush=True)
    assert reports
    reports.sort(key=lambda r:(r['success'],r['long_jump_m'] or -1),reverse=True)
    write_json(BASE/('comparison_'+pattern.replace('*','all').replace('.pt','')+'.json'),dict(reports=reports,best=reports[0]))

def select_candidates(pattern):
    candidates=[ROOT/'checkpoints/ppo_long_jump_self_imitation_v3_20261013.pt']
    candidates+=sorted((ROOT/'checkpoints').glob(pattern))
    scores=[]
    for checkpoint in candidates:
        native_path=BASE/(checkpoint.stem+'_native.json')
        native=json.loads(native_path.read_text(encoding='utf-8')) if native_path.exists() else evaluate(checkpoint)
        if not native_path.exists():write_json(native_path,native)
        if not native['success'] or native['long_jump_m']<4.0:continue
        result_path=BASE/(checkpoint.stem+'_selection64.json')
        result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else stochastic(checkpoint,64,2026101311)
        if not result_path.exists():write_json(result_path,result)
        entry=dict(checkpoint=str(checkpoint),greedy_m=native['long_jump_m'],
            successes=result['successes'],successful_mean_m=result['successful_mean_m'],
            success_weighted_mean_m=result['success_weighted_mean_m'])
        scores.append(entry);print(json.dumps(entry,ensure_ascii=False),flush=True)
    assert scores
    scores.sort(key=lambda x:(x['success_weighted_mean_m'],x['greedy_m']),reverse=True)
    write_json(BASE/'elite_selection64_comparison.json',dict(criterion='success-weighted mean on selection seed11; native greedy>=4m',
        evaluations=scores,best=scores[0],heldout_seeds_reserved=[2026101321,2026101322,2026101323]))
    print(json.dumps(dict(selected=scores[0]),ensure_ascii=False),flush=True)

def finalize_model():
    selection=json.loads((BASE/'elite_selection64_comparison.json').read_text(encoding='utf-8'))
    source=Path(selection['best']['checkpoint'])
    data=torch.load(source,map_location='cpu',weights_only=False)
    baseline=ROOT/'checkpoints/ppo_long_jump_shadow_best.pt'
    old=torch.load(baseline,map_location='cpu',weights_only=False)
    initial=json.loads((BASE/'initialization.json').read_text(encoding='utf-8'))
    assert sha(baseline)==initial['source_sha256']
    assert sha(ROOT/'bin/running_physics.dll')==initial['physics_sha256']
    for name,value in old['model'].items():
        if name.startswith(('net.','actor.','angular_velocity.','flat_actor_')) or name=='observation_scale':
            assert torch.equal(value,data['model'][name]),name
    native=evaluate(source);assert native['success'] and native['exited_sand']
    old_native=evaluate(baseline)
    assert native['actions'][:334]==old_native['actions'][:334]
    target=ROOT/'checkpoints/long_jump_distance_elite_best_20261013.pt'
    data['config']=dict(data['config'],checkpoint=str(target))
    data['long_jump_improvement_provenance']=dict(selected_source=str(source),source_sha256=sha(source),
        selected_by=selection['criterion'],selection_seed=2026101311,selection=selection['best'],
        own_imitation_checkpoints=['ppo_long_jump_self_imitation_v2_20261013.pt','ppo_long_jump_self_imitation_v3_20261013.pt'],
        native_rules_preserved=True,first25m334greedy_actions_preserved=True,
        frozen_running_weights_preserved=True,original_baseline_sha256=sha(baseline))
    torch.save(data,target)
    reloaded=evaluate(target)
    assert reloaded['actions']==native['actions'] and reloaded['long_jump_m']==native['long_jump_m']
    score=f"{native['long_jump_m']:.2f}".replace('.','p')
    replay=ROOT/'replays'/f'long_jump_distance_elite_deterministic_{score}m_20261013.bin'
    write_replay(replay,reloaded['actions'],ReplayRuleset.LONG_JUMP_30M)
    rules,actions=read_replay(replay);assert actions==native['actions']
    env=RunningGameEnv('long_jump_30m','long_jump_30m');env.reset()
    for i,a in enumerate(actions):
        raw,reward,done,truncated,info=env.step(a)
        assert bool(done or truncated)==(i==len(actions)-1)
    assert info['success'] and not truncated and info['score']==native['long_jump_m']
    write_json(BASE/'final_model_native.json',reloaded)
    report=dict(model=str(target),model_sha256=sha(target),source=str(source),
        replay=str(replay),replay_sha256=sha(replay),greedy_jump_m=native['long_jump_m'],
        original_greedy_jump_m=old_native['long_jump_m'],native_success=True,strict_reload_reproduces_actions=True,
        replay_crc_and_full_native_execution_verified=True,old_physics_and_baseline_unchanged=True,
        frozen_runup_weights_unchanged=True,first25m334greedy_actions_unchanged=True,selection=selection['best'])
    write_json(BASE/'final_model_artifacts.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['validate','prepare','prepare-local-state','prepare-v7','train','evaluate','select','finalize','stochastic','search'])
    parser.add_argument('--config',type=Path);parser.add_argument('--pattern');parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--episodes',type=int,default=64);parser.add_argument('--seed',type=int,default=2026101311)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='validate':validate_reward()
    elif args.mode=='prepare':prepare()
    elif args.mode=='prepare-local-state':prepare_local_state()
    elif args.mode=='prepare-v7':prepare_v7()
    elif args.mode=='train':train(args.config)
    elif args.mode=='evaluate':evaluate_candidates(args.pattern)
    elif args.mode=='select':select_candidates(args.pattern)
    elif args.mode=='finalize':finalize_model()
    elif args.mode=='search':search_replay(args.checkpoint,args.output,args.episodes,args.seed)
    else:
        report=stochastic(args.checkpoint,args.episodes,args.seed);write_json(args.output,report)
        print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
