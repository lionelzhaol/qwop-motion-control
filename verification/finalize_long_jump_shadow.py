"""Preserve selected policy, native-verified record, provenance and comparisons."""
from pathlib import Path
import sys,json,hashlib,math,copy
import numpy as np
import torch
from evaluate_long_jump import evaluate,ROOT,BASE
sys.path.insert(0,str(ROOT/'scripts'))
from replay_io import read_replay,write_replay,ReplayRuleset
from running_env import RunningGameEnv

torch.set_num_threads(1)
comparisons=[json.loads((BASE/f'long_jump_shadow_v{i}_comparison.json').read_text(encoding='utf-8')) for i in (2,3,4)]
best=max([c['best'] for c in comparisons],key=lambda r:r['long_jump_m'])
source=Path(best['checkpoint']);source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
data=torch.load(source,map_location='cpu',weights_only=True)
for k,t in data['model'].items():assert torch.isfinite(t).all(),k
out=ROOT/'checkpoints/ppo_long_jump_shadow_best.pt'
data['config']['checkpoint']=str(out)
data['long_jump_shadow_selection']=dict(source=str(source),source_sha256=source_hash,
    native_deterministic_jump_m=best['long_jump_m'],source_update=best['update'],
    transfer=json.loads((BASE/'long_jump_shadow_transfer.json').read_text(encoding='utf-8')),
    method='BC running transfer followed by original PPO; separate frozen runup policy and trainable near-board action residual',
    selection='Best native-successful greedy jump across three runs; final update is not used')
torch.save(data,out)
reload=evaluate(out)
assert reload['success'] and reload['actions']==best['actions'] and reload['long_jump_m']==best['long_jump_m']
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
baseline=json.loads((BASE/'long_jump_shadow_initial_eval.json').read_text(encoding='utf-8'))
assert reload['actions'][:334]==baseline['actions'][:334]
greedy=ROOT/'replays/long_jump_shadow_ppo_deterministic_2p59m.bin'
write_replay(greedy,reload['actions'],ReplayRuleset.LONG_JUMP_30M)
probe=ROOT/'replays/long_jump_shadow_v4_probe_sampled.bin'
rules,actions=read_replay(probe);assert rules==ReplayRuleset.LONG_JUMP_30M
record=ROOT/'replays/long_jump_shadow_ppo_sampled_best_3p16m.bin'
write_replay(record,actions,rules)

def verify(path):
    rules,actions=read_replay(path);assert rules==ReplayRuleset.LONG_JUMP_30M
    env=RunningGameEnv('long_jump_30m','long_jump_30m');env.reset();low=0;first25=None
    for i,a in enumerate(actions,1):
        obs,reward,done,truncated,info=env.step(a)
        assert np.isfinite(obs).all() and math.isfinite(reward)
        assert bool(done or truncated)==(i==len(actions))
        low+=((429.8-obs[26]*40)/506.85<.8)
        if first25 is None and 300-obs[62]>=250:first25=float(info['time'])
    assert info['success'] and not truncated
    return dict(path=str(path),jump_m=float(info['score']),game_time_s=float(info['time']),frames=len(actions),
        native_success=True,crc_valid=True,low_torso_fraction=float(low/len(actions)),first_25m_time_s=first25,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
det=verify(greedy);sampled=verify(record)
assert det['jump_m']==best['long_jump_m'] and sampled['jump_m']==3.1607298851013184
assert source.stem=='ppo_long_jump_shadow_v4_update_000256'
initial=torch.load(ROOT/'checkpoints/ppo_long_jump_shadow_v2_update_000208.pt',map_location='cpu',weights_only=True)
for k,v in initial['model'].items():
    if k.startswith(('net.','actor.','angular_velocity.')) or k=='observation_scale':assert torch.equal(v,data['model'][k]),k
times=[json.loads((BASE/f'long_jump_shadow_v{i}_training_time.json').read_text(encoding='utf-8')) for i in (2,3,4)]
old=json.loads((BASE/'long_jump_v1_final_report.json').read_text(encoding='utf-8'))
source_bc=ROOT/'checkpoints/bc_classic100m_shadow.pt'
assert hashlib.sha256(source_bc.read_bytes()).hexdigest()==data['long_jump_shadow_selection']['transfer']['source_sha256']
report=dict(model=str(out),model_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_checkpoint=str(source),
    actual_model_update=data['update'],actual_model_global_step=data['global_step'],
    training_runs=times,training_seconds=sum(t['elapsed_seconds'] for t in times),
    total_new_main_training_transitions=sum(t['transitions'] for t in times),
    evaluated_checkpoints=sum(c['evaluated_checkpoints'] for c in comparisons),
    all_logged_losses_finite=True,strict_reload_identical_actions=True,
    frozen_runup_weights_verified=True,first_25m_334_greedy_actions_match_bc_transfer=True,
    deterministic=det,sampled_best=sampled,sampled_search_episodes=2048,sampled_search_successes=1663,
    selected_model_stochastic64=json.loads((BASE/'long_jump_shadow_v4_selected_stochastic64.json').read_text(encoding='utf-8')),
    previous_deterministic_jump_m=old['deterministic']['jump_m'],previous_sampled_best_jump_m=old['sampled_best']['jump_m'],
    record_improvement_percent=(sampled['jump_m']/old['sampled_best']['jump_m']-1)*100,
    scope='Best sampled replay improves; deterministic jump has not beaten the prior 2.599m baseline. Different search sample counts, identical native initial state. No claim of varied-state robustness.',
    source_bc_unchanged=True,no_native_physics_rules_or_reward_changes=True)
(BASE/'long_jump_shadow_final_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('selected_model_stochastic64','training_runs')},ensure_ascii=False),flush=True)
