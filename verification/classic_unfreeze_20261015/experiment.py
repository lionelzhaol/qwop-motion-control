"""Staged Actor experiment; delegates PPO, native timing and sampling to existing code."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import importlib.util
import json
import sys
import torch
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1] / 'QWOP_rl_training_public'
sys.path.insert(0, str(BASE.parent / 'classic_improvement_20261015'))
prior_spec = importlib.util.spec_from_file_location('classic_previous_experiment', BASE.parent/'classic_improvement_20261015/experiment.py')
prior = importlib.util.module_from_spec(prior_spec)
sys.modules[prior_spec.name] = prior
prior_spec.loader.exec_module(prior)
prior.BASE = BASE
from train_ppo import ActorCritic, configure_actor_unfreeze
from replay_io import write_replay, read_replay, ReplayRuleset

SOURCE = ROOT / 'checkpoints/ppo_classic_shadow_finetuned_best_20261015.pt'
STAGES = {'head': 3e-5, 'last_hidden': 1e-5, 'full': 3e-7}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def emit(data):
    print(json.dumps(data, ensure_ascii=False), flush=True)

def prepare():
    data, model = prior.load(SOURCE)
    with np.load(BASE.parent/'shadow_audit/expert_dataset.npz') as expert:
        indices = np.linspace(0, len(expert['observations'])-1, 32, dtype=int)
        obs = torch.as_tensor(expert['observations'][indices].copy())
    spec = importlib.util.spec_from_file_location('before_unfreeze', BASE/'train_ppo_before.py')
    old_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old_module
    spec.loader.exec_module(old_module)
    old = old_module.ActorCritic(68*32, 16, (256,256))
    old.load_state_dict(data['model'], strict=True)
    with torch.no_grad():
        assert all(torch.equal(x,y) for x,y in zip(old(obs),model(obs)))
    tests = {}
    for scope in ('none', *STAGES):
        _, m = prior.load(SOURCE)
        m.net.requires_grad_(False)
        m.angular_velocity.requires_grad_(False)
        m.actor.requires_grad_(False)
        before = {k:v.clone() for k,v in m.state_dict().items()}
        configure_actor_unfreeze(m, scope)
        actor_names = [k for k,p in m.named_parameters() if p.requires_grad and not k.startswith(('critic.','critic_net.','critic_motion.'))]
        assert ('actor.weight' in actor_names) == (scope != 'none')
        assert ('net.2.weight' in actor_names) == (scope in ('last_hidden','full'))
        assert ('net.0.weight' in actor_names) == (scope == 'full')
        assert ('angular_velocity.weight' in actor_names) == (scope == 'full')
        _, value = m(obs)
        value.square().mean().backward()
        assert all(p.grad is None for k,p in m.named_parameters() if k in actor_names)
        m.zero_grad(set_to_none=True)
        logits, _ = m(obs)
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(obs)) % 16)
        loss.backward()
        opt = torch.optim.Adam(m.parameters(), lr=1e-5)
        opt.step()
        for name, p in m.named_parameters():
            if not p.requires_grad:
                assert torch.equal(before[name], p), (scope,name)
        for name in actor_names:
            assert not torch.equal(before[name],m.state_dict()[name]), (scope,name)
        tests[scope] = dict(trainable_actor_parameters=sum(p.numel() for k,p in m.named_parameters() if k in actor_names),
                            names=actor_names, frozen_unchanged=True, critic_gradients_isolated=True)
    configs = {}
    for scope,lr in STAGES.items():
        for smoke in (True, False):
            stem = f'ppo_classic_unfreeze_{scope}' + ('_smoke' if smoke else '') + '_20261015'
            cfg = dict(data['config'], actor_unfreeze=scope, learning_rate=lr,
                       resume_from=str(SOURCE), checkpoint=f'checkpoints/{stem}.pt', run_name=stem,
                       reset_optimizer_on_resume=True, updates=64, save_interval=16,
                       ent_coef_decay_updates=48, seed=2026101541,
                       run_dir=str(ROOT/'runs'))
            if smoke:
                cfg.update(num_envs=4,num_threads=2,updates=2,save_interval=1)
            path = ROOT/'configs'/f'{stem}.json'
            prior.write_json(path,cfg)
            configs[scope+('_smoke' if smoke else '')] = str(path)
    r = prior.native(SOURCE)
    assert r['success'] and r['frames']==991
    report=dict(started_utc=datetime.now(timezone.utc).isoformat(),source=str(SOURCE),
                source_sha256=prior.sha(SOURCE),physics_sha256=prior.sha(ROOT/'bin/running_physics.dll'),
                bc_sha256=prior.sha(ROOT/'checkpoints/bc_classic100m_shadow.pt'),configs=configs,
                scopes=tests,legacy_logits_values_identical=True,baseline_native=r)
    prior.write_json(BASE/'initialization.json',report)
    emit({k:v for k,v in report.items() if k not in ('baseline_native','configs')})

def config(scope, source, smoke):
    paths=read(BASE/'initialization.json')['configs']
    p=Path(paths[scope+('_smoke' if smoke else '')]);c=read(p)
    c['resume_from']=str(Path(source).resolve())
    prior.write_json(p,c)
    emit(dict(config=str(p),resume_from=c['resume_from'],scope=scope))

def verify_smoke(scope):
    c=read(read(BASE/'initialization.json')['configs'][scope+'_smoke'])
    before,_=prior.load(c['resume_from']);after,m=prior.load(ROOT/c['checkpoint'])
    deltas={k:float((v-before['model'][k]).abs().max()) for k,v in after['model'].items() if torch.is_floating_point(v)}
    for name in ('net.0.weight','net.2.weight','angular_velocity.weight','actor.weight'):
        allowed = name=='actor.weight' or scope=='full' or (scope=='last_hidden' and name=='net.2.weight')
        assert (deltas[name]>0)==allowed, (name,deltas[name],scope)
    assert all(torch.isfinite(v).all() for v in after['model'].values())
    native=prior.native(ROOT/c['checkpoint'])
    prior.write_json(BASE/f'{scope}_smoke_validation.json',dict(deltas=deltas,native=native,finite=True))
    emit(dict(scope=scope,deltas={k:deltas[k] for k in ('net.0.weight','net.2.weight','angular_velocity.weight','actor.weight')},
              greedy_success=native['success'],greedy_frames=native['frames']))
    assert native['success'], 'Smoke policy failed native completion; adjust before starting a long run'

def select(scope):
    cfg=read(read(BASE/'initialization.json')['configs'][scope])
    final=ROOT/cfg['checkpoint'];candidates=[Path(cfg['resume_from'])]+sorted(final.parent.glob(final.stem+'_update_*.pt'))+[final]
    initial, reference=prior.load(cfg['resume_from'])
    with np.load(BASE.parent/'shadow_audit/expert_dataset.npz') as expert:
        obs=torch.as_tensor(expert['observations'].copy())
    with torch.no_grad(): reference_logits=reference(obs)[0]
    rows=[]
    for checkpoint in dict.fromkeys(candidates):
        native=prior.native(checkpoint)
        prior.write_json(BASE/(checkpoint.stem+'_native.json'),native)
        row=dict(checkpoint=str(checkpoint),native_success=native['success'],frames=native['frames'],time_s=native['time_s'],
                 distance_m=native['distance_m'])
        d,m=prior.load(checkpoint)
        with torch.no_grad():
            logits=m(obs)[0]
            logp=reference_logits.log_softmax(-1)
            row['expert_state_mean_kl_from_stage_source']=float((logp.exp()*(logp-logits.log_softmax(-1))).sum(-1).mean())
            row['expert_state_argmax_changes']=int((reference_logits.argmax(-1)!=logits.argmax(-1)).sum())
        row['max_parameter_changes']={name:float((d['model'][name]-initial['model'][name]).abs().max())
                                     for name in ('net.0.weight','net.2.weight','angular_velocity.weight','actor.weight')}
        if native['success']:
            path=BASE/(scope+'_'+checkpoint.stem+'_selection64.json')
            if not path.exists():
                prior.sample(checkpoint,64,2026101551,path)
            r=read(path);times=[x['time_s'] for x in r['results'] if x['success']]
            assert r['checkpoint_sha256']==prior.sha(checkpoint) and r['seed']==2026101551 and r['episodes']==64
            row.update(successes=r['successes'],mean_success_time_s=float(np.mean(times)) if times else None,
                       sample_best_frames=r['best']['frames'] if r['best'] else None)
        rows.append(row)
        emit(row)
    valid=[r for r in rows if r['native_success'] and r.get('successes',0)>=32]
    if not valid:
        valid=[r for r in rows if r['native_success']]
    assert valid, 'No successful source or checkpoint'
    valid.sort(key=lambda r:(r['frames'],-r.get('successes',0),r.get('mean_success_time_s') or 1e9))
    report=dict(scope=scope,selection_seed=2026101551,selection_episodes=64,chosen=valid[0],candidates=rows,
                rule='Prefer native greedy finish then shortest frames, stochastic successes, mean successful time; >=32/64 gate when available')
    prior.write_json(BASE/f'{scope}_selection.json',report)
    emit(dict(selected=valid[0]))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','config','train','verify-smoke','select','sample','search'])
    p.add_argument('--scope',choices=list(STAGES));p.add_argument('--source',type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--episodes',type=int,default=64);p.add_argument('--seed',type=int,default=2026101551);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='config':config(a.scope,a.source,a.smoke)
    elif a.mode=='train':prior.train(read(BASE/'initialization.json')['configs'][a.scope+('_smoke' if a.smoke else '')])
    elif a.mode=='verify-smoke':verify_smoke(a.scope)
    elif a.mode=='select':select(a.scope)
    elif a.mode=='sample':prior.sample(a.source,a.episodes,a.seed,a.output)
    else:prior.search(a.source,a.episodes,a.seed,a.output)

if __name__=='__main__':main()
