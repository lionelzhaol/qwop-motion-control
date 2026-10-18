from pathlib import Path
import sys
import json
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/"QWOP_rl_training_public"
sys.path.insert(0,str(ROOT/"scripts"))
from train_ppo import ActorCritic, PPOConfig

torch.set_num_threads(1)
source=torch.load(ROOT/"checkpoints"/"ppo_phase_v34_credit_calibrated.pt",map_location="cpu",weights_only=False)
smoke=torch.load(ROOT/"checkpoints"/"ppo_phase_v35_smoke.pt",map_location="cpu",weights_only=False)
assert PPOConfig().freeze_backbone is False
assert smoke["config"]["freeze_backbone"] is True
assert set(smoke["model"])==set(source["model"])
unchanged_backbone=[]
changed_heads=[]
for key,value in smoke["model"].items():
    assert torch.isfinite(value).all(),key
    if key.startswith("net.") or key in {"angular_velocity.weight","observation_scale"}:
        assert torch.equal(value,source["model"][key]),key
        unchanged_backbone.append(key)
    elif key.startswith(("actor.","critic.")):
        assert not torch.equal(value,source["model"][key]),key
        changed_heads.append(key)
    else:
        assert torch.equal(value,source["model"][key]),key

def load():
    model=ActorCritic(smoke["model"]["net.0.weight"].shape[1],16,tuple(smoke["config"]["hidden_sizes"]))
    result=model.load_state_dict(smoke["model"],strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    return model

model=load()
assert all(param.requires_grad for param in model.parameters())
optimizer=torch.optim.Adam(model.parameters(),lr=3e-6,eps=1e-5)
optimizer.load_state_dict(smoke["optimizer"])
assert len(optimizer.param_groups[0]["params"])==len(list(model.parameters()))
assert len(source["optimizer"]["param_groups"][0]["params"])==len(smoke["optimizer"]["param_groups"][0]["params"])
torch.manual_seed(953)
obs=torch.randn(8,model.net[0].in_features)
if model.observation_scale is not None:obs*=model.observation_scale
frames=obs.reshape(8,-1,68)
for i in range(frames.shape[1]):frames[:,i,61]=2-(frames.shape[1]-1-i)/30
frames[:,-1,60]=torch.linspace(0,5,8)

model.net.requires_grad_(False)
if model.angular_velocity is not None:model.angular_velocity.requires_grad_(False)
actions=torch.arange(8)%16
lp,entropy,value=model.evaluate(obs,actions,.125,3,True)
loss=-lp.mean()-.005*entropy.mean()+.001*.5*(value-3).square().mean()
loss.backward()
frozen_grad_names=[]
head_grad_names=[]
for name,param in model.named_parameters():
    if name.startswith(("net.","angular_velocity.")):
        assert not param.requires_grad and param.grad is None,name
        frozen_grad_names.append(name)
    else:
        assert param.requires_grad and param.grad is not None and torch.isfinite(param.grad).all(),name
        head_grad_names.append(name)

default=load()
lp,entropy,value=default.evaluate(obs,actions,.125,3,True)
(-lp.mean()-.005*entropy.mean()+.001*.5*(value-3).square().mean()).backward()
assert all(param.requires_grad and param.grad is not None and torch.isfinite(param.grad).all() for param in default.parameters())
report={"status":"passed","smoke_checkpoint":"checkpoints/ppo_phase_v35_smoke.pt",
    "source_checkpoint":"checkpoints/ppo_phase_v34_credit_calibrated.pt","freeze_backbone_config":True,
    "unchanged_frozen_tensors":unchanged_backbone,"updated_head_tensors":changed_heads,
    "strict_load_success":True,"default_all_parameters_trainable":True,
    "frozen_backbone_grad_none":frozen_grad_names,"head_gradients_finite":head_grad_names,
    "default_backbone_gradients_restored":True,"optimizer_parameter_count":len(list(model.parameters())),
    "old_optimizer_parameter_list_compatible":True,"smoke_optimizer_strict_load_success":True,
    "all_checkpoint_tensors_finite":True,
    "head_update_max_abs":{key:float((smoke["model"][key]-source["model"][key]).abs().max()) for key in changed_heads}}
(BASE/"freeze_backbone_smoke_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
