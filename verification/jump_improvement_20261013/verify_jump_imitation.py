"""CPU-only one-step and collector checks for optional jump imitation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
import torch.nn.functional as F

BASE=Path(__file__).resolve().parent; ROOT=BASE.parents[1]/'QWOP_rl_training_public'
sys.path.insert(0,str(ROOT/'scripts'))
from train_replay_imitation import collect_replay,evaluate_policy,configure_imitation
from train_ppo import ActorCritic

torch.set_num_threads(1)
source_path=ROOT/'checkpoints/ppo_long_jump_distance_initial_20261013.pt'
replay_path=ROOT/'replays/long_jump_shadow_ppo_sampled_best_3p16m.bin'
source_hash=hashlib.sha256(source_path.read_bytes()).hexdigest()
replay_hash=hashlib.sha256(replay_path.read_bytes()).hexdigest()
source=torch.load(source_path,map_location='cpu',weights_only=False); cfg=source['config']
data=np.load(BASE/'imitation_smoke.npz')
x=torch.as_tensor(data['observations']); y=torch.as_tensor(data['actions'])
assert x.shape==(529,2176) and y.shape==(529,)
model=ActorCritic(x.shape[1],16,tuple(cfg['hidden_sizes']));model.load_state_dict(source['model'],strict=True)
configure_imitation(model,True)
names=[name for name,p in model.named_parameters() if p.requires_grad]
assert names==['jump_actor_weight','jump_actor_bias']
assert all(torch.equal(v,model.state_dict()[k]) for k,v in source['model'].items())
before={k:v.clone() for k,v in model.state_dict().items()}
mask=x[:,-6]<=model.jump_sampling[1]
loss=F.cross_entropy(model(x[mask])[0],y[mask]);assert torch.isfinite(loss)
loss.backward()
gradient_norms={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.requires_grad}
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
assert max(gradient_norms.values())>0
optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
optimizer.step()
assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k not in names)
assert any(not torch.equal(before[k],model.state_dict()[k]) for k in names)
smoke=torch.load(BASE/'imitation_smoke.pt',map_location='cpu',weights_only=False)
reload=ActorCritic(x.shape[1],16,tuple(cfg['hidden_sizes']));reload.load_state_dict(smoke['model'],strict=True)
assert all(torch.equal(v,reload.state_dict()[k]) for k,v in smoke['model'].items())
assert all(torch.equal(v,smoke['model'][k]) for k,v in source['model'].items() if k not in names)
teacher=ActorCritic(x.shape[1],16,tuple(cfg['hidden_sizes']));teacher.load_state_dict(source['model'],strict=True)
with torch.no_grad(): assert torch.equal(teacher(x[~mask])[0],reload(x[~mask])[0])
initial_eval=evaluate_policy(teacher,cfg['frame_stack'],'long_jump_30m')
smoke_eval=evaluate_policy(reload,cfg['frame_stack'],'long_jump_30m')
assert initial_eval['actions'][:334]==smoke_eval['actions'][:334]
classic_x,classic_y,classic_info,_=collect_replay(ROOT/'replays/classic_shadow_bc_33p03.bin',cfg['frame_stack'])
assert classic_info['success']
try: collect_replay(replay_path,cfg['frame_stack'])
except ValueError: pass
else: raise AssertionError('Default classic collector accepted a jump ruleset')
assert hashlib.sha256(source_path.read_bytes()).hexdigest()==source_hash
assert hashlib.sha256(replay_path.read_bytes()).hexdigest()==replay_hash
report=dict(device='cpu',collected_jump_frames=len(x),training_phase_frames=int(mask.sum()),
    finite_loss=float(loss.detach()),finite_nonzero_gradient_norms=gradient_norms,
    only_trainable_parameters=names,all_runup_and_critic_weights_unchanged=True,
    prefix_logits_exact=True,prefix_greedy_actions_equal_frames=334,
    strict_checkpoint_reload=True,
    smoke_native_success=smoke_eval['success'],smoke_final_jump_m=smoke_eval['long_jump_m'],
    default_classic_collector_frames=len(classic_y),default_classic_native_time_s=classic_info['time'],
    replay_ruleset_mismatch_rejected=True,source_model_and_replay_sha256_unchanged=True)
(BASE/'imitation_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
