"""Population search of autonomous observation-conditioned local policy biases.

Uses original physics and neural BC features, with a checkpointed distance gate.
This is parameter search, not PPO; any gain is reported with this provenance.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,sys,time
import numpy as np
import torch
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]/'QWOP_rl_training_public'
sys.path.insert(0,str(ROOT/'scripts'))
from train_ppo import ActorCritic,FrameStacker
from running_env import RunningGameEnv,VectorRunningGameEnv
from replay_io import write_replay,ReplayRuleset
torch.set_num_threads(1)

def dump(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--lower',type=float,default=65.);p.add_argument('--upper',type=float,default=100.)
    p.add_argument('--tag',default='sprint');p.add_argument('--generations',type=int,default=16)
    p.add_argument('--population',type=int,default=256);p.add_argument('--seed',type=int,default=2026101521)
    a=p.parse_args();started=time.perf_counter();utc=datetime.now(timezone.utc);rng=np.random.default_rng(a.seed)
    data=torch.load(a.source,map_location='cpu',weights_only=False);cfg=data['config'];n=a.population
    model=ActorCritic(68*cfg['frame_stack'],16,tuple(cfg['hidden_sizes']));model.load_state_dict(data['model'],strict=True);model.eval()
    assert model.flat_actor_bias is None and model.jump_sampling is None,'Search starts from unchanged BC teacher'
    env=VectorRunningGameEnv(n,num_threads=8,reward_profile='classic_100m_task',game_mode='classic_100m')
    canonical=np.array([0,1,2,4,5,6,8,9,10]);center=np.zeros(9);best_vector=np.zeros(9)
    best=None;history=[];scales=[.015,.03,.06,.12,.24,.48,.96,1.92]
    for generation in range(a.generations):
        scale=scales[generation%len(scales)]*(.7 if generation>=8 else 1.)
        values=center+rng.normal(size=(n,9))*scale
        values[0]=best_vector;values[1]=0.
        # Keep a broad exploratory subset alongside the local elite search.
        values[n//2:]=rng.normal(size=(n-n//2,9))*scale
        values-=values.mean(axis=1,keepdims=True)
        bias=np.zeros((n,16),dtype=np.float32);bias[:,canonical]=values
        bias_tensor=torch.as_tensor(bias)
        raw,_=env.reset();st=FrameStacker(n,68,cfg['frame_stack']);obs=st.reset(raw)
        active=np.ones(n,dtype=bool);low=np.zeros(n,dtype=int);plans=np.zeros((991,n),dtype=np.int64)
        entries=[None]*n;entered_gate=False
        for step in range(991):
            low[active]+=((429.8-raw[active,26]*40)/506.85<.8)
            with torch.no_grad():
                in_gate=(obs[:,-8]>=a.lower)&(obs[:,-8]<a.upper)
                entered_gate=entered_gate or bool(np.any(in_gate&active))
                # Before the first gate crossing all states share the exact
                # unperturbed prefix; thereafter infer every state separately.
                if not entered_gate:logits=model.actor(model.features(torch.as_tensor(obs[:1]))).expand(n,-1)
                else:logits=model.actor(model.features(torch.as_tensor(obs)))
                assert torch.isfinite(logits).all()
                actions=(logits+bias_tensor*torch.as_tensor(in_gate[:,None])).argmax(-1).numpy()
            plans[step]=actions
            r=env.step_arrays(actions);done=r['terminated']|r['truncated']
            assert np.isfinite(r['obs']).all() and np.isfinite(r['reward']).all()
            for i in np.flatnonzero(active&r['has_final']):
                success=bool(r['final_success'][i]);qual=bool(success and low[i]/(step+1)<=.1)
                entry=dict(success=success,qualified=qual,frames=step+1,time_s=float(r['final_time'][i]),
                    distance_m=float(r['final_distance'][i]),low_torso_fraction=float(low[i]/(step+1)),
                    generation=generation,individual=int(i))
                entries[i]=entry;active[i]=False
                rank=(int(qual),-(step+1) if qual else float(r['final_distance'][i]),float(r['final_distance'][i]))
                if qual and (best is None or rank>best['rank']):
                    best=dict(entry,rank=rank,actions=plans[:step+1,i].tolist(),bias=bias[i].tolist())
                    best_vector=values[i].copy()
                    checkpoint=copy.deepcopy(data);model.enable_flat_bias([a.lower,a.upper,110.,120.])
                    state={k:v.detach().clone() for k,v in model.state_dict().items()}
                    state['flat_actor_bias']=torch.as_tensor(bias[i]).clone()
                    checkpoint['model']=state;model.flat_actor_bias=None;model.flat_actor_bounds=None
                    path=ROOT/'checkpoints'/f'classic_shadow_bias_{a.tag}_best_20261015.pt'
                    checkpoint['config']=dict(cfg,checkpoint=str(path),enable_flat_actor_bias=True,freeze_backbone=True)
                    checkpoint['parameter_search']=dict(method='CEM-style local bias population',parent=str(a.source),seed=a.seed,
                        generation=generation,lower=a.lower,upper=a.upper,own_native_result={k:v for k,v in entry.items()})
                    torch.save(checkpoint,path)
                    replay=ROOT/'replays'/f'classic_shadow_bias_{a.tag}_best_20261015.bin'
                    write_replay(replay,best['actions'],ReplayRuleset.CLASSIC_100M)
                    print(json.dumps(dict(new_best=entry,checkpoint=str(path))),flush=True)
            if not active.any():break
            raw=r['obs'];obs=st.update(raw,done)
        for i in np.flatnonzero(active):
            entries[i]=dict(success=False,qualified=False,frames=991,time_s=float(r['time'][i]),
                distance_m=float(r['score'][i]),censored=True,generation=generation,individual=int(i))
        ranking=sorted(range(n),key=lambda i:(entries[i]['qualified'],
            -entries[i]['frames'] if entries[i]['qualified'] else entries[i]['distance_m'],entries[i]['distance_m']),reverse=True)
        elite=values[ranking[:max(4,n//16)]];center=.5*center+.5*elite.mean(axis=0)
        status=dict(generation=generation,population=n,std=scale,
            successes_within991=sum(e['success'] for e in entries),
            faster_than991=sum(e['qualified'] and e['frames']<991 for e in entries),
            best={k:v for k,v in best.items() if k not in ('actions','bias','rank')} if best else None)
        history.append(status);dump(BASE/f'bias_{a.tag}_progress.json',history);print(json.dumps(status),flush=True)
    assert best is not None
    report=dict(source=str(a.source),tag=a.tag,seed=a.seed,generations=a.generations,population=n,attempts=a.generations*n,
        method='Autonomous BC neural policy with a distance-gated, fixed learned-by-search action-logit bias',
        source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),best=best,checkpoint=str(path),replay=str(replay),
        history=history,elapsed_seconds=time.perf_counter()-started,started_utc=utc.isoformat(),
        note='Horizon991 truncates search evaluation; not a full-episode success-rate estimate. Model still requires independent strict-load/native verification.')
    dump(BASE/f'bias_{a.tag}_result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('best','history')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
