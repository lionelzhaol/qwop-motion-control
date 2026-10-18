"""Search unchanged BC policy with explicitly recorded sampling temperatures."""
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
data=torch.load(path,map_location=device,weights_only=True)
cfg=data['config']
model=ActorCritic(68*cfg['frame_stack'],16,tuple(cfg['hidden_sizes'])).to(device)
model.load_state_dict(data['model'],strict=True); model.eval()
audit=[]
for batch,(temperature,seed) in enumerate([(t,202610110+i) for i,t in enumerate([.1,.15,.2,.25,.3,.4,.5,.075,.125,.175,.225,.275])]):
    torch.manual_seed(seed)
    n=256
    env=VectorRunningGameEnv(n,num_threads=16,reward_profile='classic_100m_task',game_mode='classic_100m')
    raw,_=env.reset(); stack=FrameStacker(n,68,cfg['frame_stack']); obs=stack.reset(raw)
    active=np.ones(n,dtype=bool); actions_history=[[] for _ in range(n)]
    success_count=0; candidate=None; longest=0.
    for step in range(1,3601):
        with torch.no_grad():
            logits,_=model(torch.as_tensor(obs,device=device))
            actions=torch.distributions.Categorical(logits=logits/temperature).sample().cpu().numpy()
        for i in np.flatnonzero(active): actions_history[i].append(int(actions[i]))
        result=env.step_arrays(actions)
        assert np.isfinite(result['obs']).all() and np.isfinite(result['reward']).all()
        done=result['terminated']|result['truncated']
        ended=active&done
        for i in np.flatnonzero(ended):
            elapsed=float(result['final_time'][i]); distance=float(result['final_distance'][i])
            success=bool(result['final_success'][i]); success_count+=int(success); longest=max(longest,distance)
            if success and 33.<=elapsed<=34. and step!=991:
                candidate=dict(temperature=temperature,seed=seed,batch_episodes=n,episode_index=int(i),frames=step,time_s=elapsed,distance_m=distance)
                chosen=actions_history[i]; break
        active[ended]=False
        if candidate is not None or not active.any(): break
        obs=stack.update(result['obs'],done)
    entry=dict(batch=batch,temperature=temperature,seed=seed,successes_observed=success_count,max_distance_m=longest,stopped_on_candidate=candidate is not None)
    audit.append(entry); print(json.dumps(entry),flush=True)
    if candidate is not None:
        name=f"classic100m_shadow_bc_random_{candidate['time_s']:.2f}s".replace('.','p')
        replay=ROOT/'replays'/f'{name}.bin'
        write_replay(replay,chosen,ReplayRuleset.CLASSIC_100M)
        ruleset,reloaded=read_replay(replay); assert reloaded==chosen
        single=RunningGameEnv('classic_100m_task','classic_100m'); single.reset()
        for i,action in enumerate(reloaded):
            raw,reward,done,truncated,info=single.step(action)
            assert not(done or truncated) or i==len(reloaded)-1
        assert done and not truncated and info['success'] and info['distance']>=100.
        assert info['distance']==candidate['distance_m']
        assert info['time']==candidate['time_s']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        report=dict(checkpoint=str(path),checkpoint_sha256=digest,model_weights_changed=False,
            method='Observation-conditioned unchanged policy, Categorical(logits / temperature); search first successful 33-34s finish differing from 991 frames',
            selected=candidate,search_batches=audit,replay=str(replay),replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),
            replay_crc_valid=True,native_reexecution=info,elapsed_seconds=time.perf_counter()-started,
            provenance='Behavior-cloning model trained from public XYF 991-frame expert; selected stochastic rollout, not an independent PPO discovery',
            selection_note='This is a selected successful trial; no claim of success rate or typical stochastic performance.')
        Path(__file__).with_name('random_finish_selected.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False),flush=True)
        break
else:
    Path(__file__).with_name('random_finish_failed.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    raise RuntimeError('No qualifying different-time finish found; no replay manufactured')
