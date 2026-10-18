"""Bounded unchanged-policy search; exclude the two previously saved times."""
from pathlib import Path
import sys, json, hashlib, time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]/'QWOP_rl_training_public'
sys.path.insert(0,str(ROOT/'scripts'))
from running_env import VectorRunningGameEnv, RunningGameEnv
from train_ppo import ActorCritic, FrameStacker
from replay_io import write_replay, read_replay, ReplayRuleset

torch.set_num_threads(1)
started=time.perf_counter()
path=ROOT/'checkpoints/bc_classic100m_shadow.pt'
digest=hashlib.sha256(path.read_bytes()).hexdigest()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data=torch.load(path,map_location=device,weights_only=True); cfg=data['config']
model=ActorCritic(68*cfg['frame_stack'],16,tuple(cfg['hidden_sizes'])).to(device)
model.load_state_dict(data['model'],strict=True); model.eval()
audit=[]; best=None; chosen=None
temperatures=[.25,.35,.4,.45,.5,.6,.65,.3,.375,.425,.475,.55,.25,.35,.4,.45]
for batch,temperature in enumerate(temperatures):
    seed=202610200+batch; torch.manual_seed(seed); n=512
    env=VectorRunningGameEnv(n,num_threads=16,reward_profile='classic_100m_task',game_mode='classic_100m')
    raw,_=env.reset(); stack=FrameStacker(n,68,cfg['frame_stack']); obs=stack.reset(raw)
    active=np.ones(n,dtype=bool); histories=[[] for _ in range(n)]; low=np.zeros(n,dtype=np.int64)
    success_count=0; distinct_times=[]; qualified_count=0
    for step in range(1,3601):
        with torch.no_grad():
            logits,_=model(torch.as_tensor(obs,device=device))
            actions=torch.distributions.Categorical(logits=logits/temperature).sample().cpu().numpy()
        for i in np.flatnonzero(active): histories[i].append(int(actions[i]))
        result=env.step_arrays(actions)
        assert np.isfinite(result['obs']).all() and np.isfinite(result['reward']).all()
        done=result['terminated']|result['truncated']; ended=active&done
        # Measure the pre-action torso geometry to avoid the auto-reset observation at termination.
        low[active]+=((429.8-raw[active,26]*40)/506.85 < .8)
        for i in np.flatnonzero(ended):
            elapsed=float(result['final_time'][i]); distance=float(result['final_distance'][i])
            if not result['final_success'][i]: continue
            success_count+=1
            if step in (991,994) or f'{elapsed:.2f}' in ('33.03','33.13'): continue
            distinct_times.append(elapsed)
            if low[i]/step > .1: continue
            qualified_count+=1
            if best is None or elapsed<best['time_s']:
                best=dict(temperature=temperature,seed=seed,batch_episodes=n,episode_index=int(i),frames=step,
                    time_s=elapsed,distance_m=distance,low_torso_fraction=float(low[i]/step))
                chosen=histories[i].copy()
        active[ended]=False
        if not active.any(): break
        raw=result['obs']; obs=stack.update(raw,done)
    assert not active.any(), 'Original episode limit exceeded'
    entry=dict(batch=batch,temperature=temperature,seed=seed,episodes=n,successes=success_count,
        nonduplicate_successes=len(distinct_times),qualified_nonduplicate_successes=qualified_count,
        nonduplicate_times_s=sorted(set(distinct_times)),selected_best_so_far=best)
    audit.append(entry); print(json.dumps(entry),flush=True)
    Path(__file__).with_name('another_random_search_progress.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    if best is not None and best['time_s']<33.03: break
if best is None: raise RuntimeError('No nonduplicate successful upright-proxy rollout; no manufactured replay')
name=f"classic100m_shadow_bc_random_{best['time_s']:.2f}s".replace('.','p')
replay=ROOT/'replays'/f'{name}.bin'
assert not replay.exists(), 'Preserve earlier replays'
write_replay(replay,chosen,ReplayRuleset.CLASSIC_100M)
ruleset,reloaded=read_replay(replay); assert reloaded==chosen
single=RunningGameEnv('classic_100m_task','classic_100m'); single.reset()
for i,action in enumerate(reloaded):
    raw,reward,done,truncated,info=single.step(action)
    assert not(done or truncated) or i==len(reloaded)-1
assert done and not truncated and info['success'] and info['distance']>=100.
assert info['distance']==best['distance_m'] and info['time']==best['time_s']
assert len(reloaded) not in (991,994) and f"{info['time']:.2f}" not in ('33.03','33.13')
assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
report=dict(checkpoint=str(path),checkpoint_sha256=digest,model_weights_changed=False,
    method='Observation-conditioned unchanged policy, Categorical(logits / temperature); bounded search excluding 991 and 994 frame finishes',
    selected=best,search_batches=audit,episodes_tested=sum(b['episodes'] for b in audit),
    faster_than_33p03=best['time_s']<33.03,replay=str(replay),replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),
    replay_crc_valid=True,native_reexecution=info,elapsed_seconds=time.perf_counter()-started,
    provenance='Behavior-cloning model trained from public XYF 991-frame expert; selected stochastic rollout, not an independent PPO discovery',
    selection_note='Selected successful trial; not typical performance. Low torso is a proxy, not direct knee contact.')
Path(__file__).with_name('another_random_finish_selected.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='search_batches'},ensure_ascii=False),flush=True)
