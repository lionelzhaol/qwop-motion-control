from pathlib import Path
import copy
import json
import sys
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/"QWOP_rl_training_public"
sys.path.insert(0,str(ROOT/"scripts"))
from train_ppo import ActorCritic,PPOConfig,save_checkpoint
from export_policy_replay import choose_action

torch.set_num_threads(1);torch.manual_seed(379)
source=torch.load(ROOT/"checkpoints"/"ppo_phase_v34_credit_calibrated.pt",map_location="cpu",weights_only=False)
state=source["model"];hidden=tuple(source["config"]["hidden_sizes"]);dim=state["net.0.weight"].shape[1]
def fresh():
    model=ActorCritic(dim,16,hidden);result=model.load_state_dict(state,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    return model
normal=fresh();detached=fresh();detached.detach_critic_features=True
assert not PPOConfig().detach_critic_features and not normal.detach_critic_features
assert set(normal.state_dict())==set(detached.state_dict())==set(state)
normal_names=[name for name,_ in normal.named_parameters()]
assert len(normal_names)==9 and normal_names==[name for name,_ in detached.named_parameters()]
assert all(parameter.requires_grad for parameter in detached.parameters())
obs=torch.randn(16,dim)*normal.observation_scale
frames=obs.reshape(16,-1,68)
for i in range(frames.shape[1]):frames[:,i,61]=2-(frames.shape[1]-1-i)/30
frames[:,-1,60]=torch.linspace(0,50,16)
actions=torch.arange(16)
logits,value=normal(obs);d_logits,d_value=detached(obs)
assert torch.equal(logits,d_logits) and torch.equal(value,d_value)

# Value-head supervision cannot reach shared features after detaching them.
detached.zero_grad(set_to_none=True)
(.5*(detached(obs)[1]-17).square().mean()).backward()
for name,parameter in detached.named_parameters():
    if name.startswith("critic."):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm()>0,name
    else:
        assert parameter.grad is None or torch.equal(parameter.grad,torch.zeros_like(parameter)),name

# Policy supervision still trains the original actor and shared motion model.
detached.zero_grad(set_to_none=True)
lp,entropy,_=detached.evaluate(obs,actions,.125,3,True)
(-lp.mean()-.005*entropy.mean()).backward()
policy_gradients={}
for name,parameter in detached.named_parameters():
    if name.startswith("critic."):
        assert parameter.grad is None,name
    else:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm()>0,name
        policy_gradients[name]=float(parameter.grad.norm())
normal.zero_grad(set_to_none=True)
(.5*(normal(obs)[1]-17).square().mean()).backward()
normal_shared_value_gradients={}
for name,parameter in normal.named_parameters():
    if name.startswith(("net.","angular_velocity.")):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm()>0,name
        normal_shared_value_gradients[name]=float(parameter.grad.norm())

torch.manual_seed(562);sampled,oldlp,oldentropy,oldvalue=detached.act(obs,.125,3,True)
newlp,newentropy,newvalue=detached.evaluate(obs,sampled,.125,3,True)
assert torch.equal((newlp-oldlp).exp(),torch.ones_like(oldlp))
assert torch.equal(oldentropy,newentropy) and torch.equal(oldvalue,newvalue)
optimizer=torch.optim.Adam(detached.parameters(),lr=3e-6,eps=1e-5)
assert len(optimizer.param_groups[0]["params"])==9
optimizer.load_state_dict(source["optimizer"])
assert len(optimizer.param_groups[0]["params"])==9
probe=BASE/"detach_critic_features_probe.pt"
cfg=PPOConfig(**dict(source["config"],detach_critic_features=True,checkpoint=probe,run_dir=BASE/"detach_critic_features_probe_logs"))
save_checkpoint(probe,detached,optimizer,cfg,source["update"],source["global_step"])
saved=torch.load(probe,map_location="cpu",weights_only=False)
assert saved["config"]["detach_critic_features"] is True
reloaded=ActorCritic(dim,16,hidden);reloaded.load_state_dict(saved["model"],strict=True)
assert not reloaded.detach_critic_features
r_logits,r_value=reloaded(obs)
assert torch.equal(r_logits,d_logits) and torch.equal(r_value,d_value)
for row in obs:
    assert choose_action(reloaded,row.numpy(),torch.device("cpu"),False)==choose_action(detached,row.numpy(),torch.device("cpu"),False)
report={"status":"passed","default_false":True,"forward_logits_values_exactly_unchanged":True,
    "detached_value_loss_only_updates_critic":True,"policy_gradients_shared_motion_actor_finite_nonzero":policy_gradients,
    "default_value_shared_gradients_restored":normal_shared_value_gradients,"same_policy_ratio_exactly_one":True,
    "state_dict_and_parameter_order_unchanged":True,"optimizer_parameters":9,
    "saved_config_true":True,"export_without_detach_config_forward_exact":True,"verification_files_only":True}
(BASE/"detach_critic_features_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
