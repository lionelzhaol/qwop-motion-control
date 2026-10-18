from pathlib import Path
import ast
import copy
import json
import sys
from types import SimpleNamespace
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/"QWOP_rl_training_public"
sys.path.insert(0,str(ROOT/"scripts"))
from train_ppo import ActorCritic
torch.set_num_threads(1);torch.manual_seed(701)
data=torch.load(ROOT/"checkpoints"/"ppo_target43_v39_time_priority_initial.pt",map_location="cpu",weights_only=False)
state=data["model"];hidden=tuple(data["config"]["hidden_sizes"]);dim=state["net.0.weight"].shape[1]
def fresh():
    model=ActorCritic(dim,16,hidden);model.load_state_dict(state,strict=True);return model
model=fresh();named=list(model.named_parameters())
actor_names=[name for name,_ in named if not name.startswith(("critic.","critic_net.","critic_motion."))]
critic_names=[name for name,_ in named if name.startswith(("critic.","critic_net.","critic_motion."))]
assert len(actor_names)==7 and len(critic_names)==7 and not set(actor_names)&set(critic_names)
assert set(actor_names)|set(critic_names)==set(dict(named))
obs=torch.randn(16,dim)*model.observation_scale
frames=obs.reshape(16,-1,68)
for i in range(frames.shape[1]):frames[:,i,61]=2-(frames.shape[1]-1-i)/30
frames[:,-1,60]=torch.linspace(0,100,16)
lp,entropy,value=model.evaluate(obs,torch.arange(16),1,3,True)
loss=-lp.mean()-.005*entropy.mean()+.001*.5*(value-100).square().mean()
loss.backward()
gradients={name:param.grad.clone() for name,param in named}
assert all(torch.isfinite(gradient).all() for gradient in gradients.values())

# Execute the exact clipping branch extracted from the actual train function,
# rather than a separately reimplemented imitation of its condition.
tree=ast.parse((ROOT/"scripts"/"train_ppo.py").read_text(encoding="utf-8"))
train_fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="train")
blocks=[node for node in ast.walk(train_fn) if isinstance(node,ast.If)
    and isinstance(node.test,ast.Attribute) and node.test.attr=="separate_critic_features"
    and "clip_grad_norm_" in ast.dump(node)]
assert len(blocks)==1
clip_code=compile(ast.fix_missing_locations(ast.Module(body=blocks,type_ignores=[])),"actual_train_clipping_branch","exec")
def run(factor,separate):
    trial=fresh();params=dict(trial.named_parameters())
    actor=[params[name] for name in actor_names];critic=[params[name] for name in critic_names]
    for name,param in params.items():param.grad=gradients[name].clone()*(factor if name in critic_names else 1)
    exec(clip_code,{"cfg":SimpleNamespace(separate_critic_features=separate),"nn":torch.nn,
        "model":trial,"actor_params":actor,"critic_params":critic})
    actor_clipped=torch.cat([param.grad.flatten() for param in actor])
    critic_clipped=torch.cat([param.grad.flatten() for param in critic])
    before=copy.deepcopy(trial.state_dict())
    optimizer=torch.optim.Adam([{"params":actor,"lr":3e-6,"role":"actor"},{"params":critic,"lr":3e-5,"role":"critic"}],eps=1e-5)
    optimizer.step()
    changed=[name for name,param in trial.named_parameters() if not torch.equal(param,before[name])]
    if separate:
        assert any(name in actor_names for name in changed) and any(name in critic_names for name in changed)
    assert all(torch.isfinite(value).all() for value in trial.state_dict().values())
    return actor_clipped,critic_clipped,changed
a1,c1,changed=run(1,True);a_big,c_big,_=run(1e6,True)
assert torch.equal(a1,a_big)
# torch's float32 norm reduction over the large encoder is approximate; use
# float64 only to check its actual relative numerical error (<0.04%). The
# main isolation property above is exact equality, not a relaxed tolerance.
assert a1.double().norm()<=.5002 and c1.double().norm()<=.5002 and c_big.double().norm()<=.5002
g1,_,_=run(1,False);g_big,_,_=run(1e6,False)
assert not torch.equal(g1,g_big) and g_big.norm()<g1.norm()/100
# Default false must reproduce the original whole-model clip exactly.
control=fresh()
for name,param in control.named_parameters():param.grad=gradients[name].clone()
torch.nn.utils.clip_grad_norm_(control.parameters(),.5)
control_actor=torch.cat([dict(control.named_parameters())[name].grad.flatten() for name in actor_names])
assert torch.equal(g1,control_actor)
report={"status":"passed","actual_train_clipping_branch_executed":True,"actor_critic_parameter_sets_disjoint_complete":True,
    "parameter_count":14,"actor_count":7,"critic_count":7,
    "separate_actor_clipped_grad_bitwise_invariant_to_critic_1e6_scale":True,
    "separate_actor_clip_norm":float(a1.norm()),"separate_critic_clip_norm":float(c1.norm()),
    "global_actor_norm_original":float(g1.norm()),"global_actor_norm_huge_critic":float(g_big.norm()),
    "default_global_clip_exactly_legacy":True,"single_step_updates_each_role":changed,
    "all_parameters_gradients_finite":True,"source_checkpoint_unchanged":True,"verification_files_only":True}
(BASE/"role_gradient_clipping_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
