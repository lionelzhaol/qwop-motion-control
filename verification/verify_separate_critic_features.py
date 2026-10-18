from pathlib import Path
import copy
import contextlib
import json
import sys
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/"QWOP_rl_training_public"
sys.path.insert(0,str(ROOT/"scripts"))
from train_ppo import ActorCritic,PPOConfig,load_model_compatible,save_checkpoint,train
from export_policy_replay import choose_action

torch.set_num_threads(1);torch.manual_seed(908)
data=torch.load(ROOT/"checkpoints"/"ppo_phase_v34_credit_calibrated.pt",map_location="cpu",weights_only=False)
legacy=data["model"];hidden=tuple(data["config"]["hidden_sizes"]);dim=legacy["net.0.weight"].shape[1]
def fresh():return ActorCritic(dim,16,hidden)
base=fresh();base.load_state_dict(legacy,strict=True)
assert not PPOConfig().separate_critic_features and PPOConfig().critic_learning_rate is None
assert len(list(base.parameters()))==9 and set(base.state_dict())==set(legacy)
obs=torch.randn(16,dim)*base.observation_scale
frames=obs.reshape(16,-1,68)
for i in range(frames.shape[1]):frames[:,i,61]=2-(frames.shape[1]-1-i)/30
frames[:,-1,60]=torch.linspace(0,50,16)
base_logits,base_values=base(obs)
model=fresh();model.load_state_dict(legacy,strict=True);model.enable_separate_critic()
assert torch.equal(model(obs)[0],base_logits) and torch.equal(model(obs)[1],base_values)
assert model.critic_motion is not None and len(list(model.parameters()))==14
actor_feature_ids={id(param) for name,param in model.named_parameters() if name.startswith(("net.","angular_velocity."))}
critic_feature_ids={id(param) for name,param in model.named_parameters() if name.startswith(("critic_net.","critic_motion."))}
assert not actor_feature_ids & critic_feature_ids
for name,param in model.net.named_parameters():
    assert torch.equal(param,dict(model.critic_net.named_parameters())[name])
    assert param.data_ptr()!=dict(model.critic_net.named_parameters())[name].data_ptr()
assert model.angular_velocity.weight.data_ptr()!=model.critic_motion.weight.data_ptr()
frozen_actor=fresh();frozen_actor.load_state_dict(legacy,strict=True)
frozen_actor.net.requires_grad_(False);frozen_actor.angular_velocity.requires_grad_(False);frozen_actor.actor.requires_grad_(False)
frozen_actor.enable_separate_critic()
assert all(param.requires_grad for param in frozen_actor.critic_net.parameters())
assert all(param.requires_grad for param in frozen_actor.critic_motion.parameters())
(.5*(frozen_actor(obs)[1]-11).square().mean()).backward()
for name,param in frozen_actor.named_parameters():
    if name.startswith(("critic.","critic_net.","critic_motion.")):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm()>0,name
    else:assert param.grad is None,name
model.zero_grad(set_to_none=True)
(.5*(model(obs)[1]-11).square().mean()).backward()
for name,param in model.named_parameters():
    if name.startswith(("critic.","critic_net.","critic_motion.")):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm()>0,name
    else:assert param.grad is None,name
model.zero_grad(set_to_none=True)
lp,ent,_=model.evaluate(obs,torch.arange(16),.125,3,True)
(-lp.mean()-.005*ent.mean()).backward()
for name,param in model.named_parameters():
    if name.startswith(("critic.","critic_net.","critic_motion.")):assert param.grad is None,name
    else:assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm()>0,name

# Distinguish trained Critic features from Actor features, then ensure loaders
# really preserve the independent state instead of just copying the Actor.
with torch.no_grad():
    model.critic_net[0].weight.add_(.005);model.critic_motion.weight.add_(.01)
newstate=copy.deepcopy(model.state_dict())
model.enable_separate_critic()
assert all(torch.equal(value,newstate[key]) for key,value in model.state_dict().items())
for loader in [lambda m,s:m.load_state_dict(s,strict=True),load_model_compatible]:
    restored=fresh();loader(restored,newstate)
    assert set(restored.state_dict())==set(newstate)
    assert all(torch.equal(value,newstate[key]) for key,value in restored.state_dict().items())
    assert torch.equal(restored(obs)[0],model(obs)[0]) and torch.equal(restored(obs)[1],model(obs)[1])
    # This ordering intentionally checks enabling before the legacy angular
    # module is restored, rather than only the safer production load order.
    upgraded=fresh();upgraded.enable_separate_critic();loader(upgraded,legacy)
    assert upgraded.critic_motion is not None
    assert torch.equal(upgraded(obs)[0],base_logits) and torch.equal(upgraded(obs)[1],base_values)
    assert torch.equal(upgraded.critic_motion.weight,upgraded.angular_velocity.weight)

named=list(model.named_parameters())
actor=[param for name,param in named if not name.startswith(("critic.","critic_net.","critic_motion."))]
critic=[param for name,param in named if name.startswith(("critic.","critic_net.","critic_motion."))]
assert not {id(param) for param in actor}&{id(param) for param in critic}
assert {id(param) for param in actor+critic}=={id(param) for param in model.parameters()}
optimizer=torch.optim.Adam([{"params":actor,"lr":3e-6,"role":"actor"},{"params":critic,"lr":1e-4,"role":"critic"}],eps=1e-5)
optimizer.zero_grad();lp,ent,value=model.evaluate(obs,torch.arange(16),1,3,True)
(-lp.mean()-.005*ent.mean()+.001*.5*(value-2).square().mean()).backward();optimizer.step()
assert len(optimizer.state_dict()["state"])==14
probe=BASE/"separate_critic_probe.pt"
cfg_values=dict(data["config"],checkpoint=probe,run_dir=BASE/"separate_critic_probe_logs",separate_critic_features=True,
    learning_rate=3e-6,critic_learning_rate=1e-4)
cfg=PPOConfig(**cfg_values)
save_checkpoint(probe,model,optimizer,cfg,data["update"],data["global_step"])
resume=PPOConfig(**dict(cfg_values,checkpoint=BASE/"separate_critic_role_resume.pt",resume_from=probe,
    separate_critic_features=False,learning_rate=7e-6,critic_learning_rate=2e-4,
    num_envs=1,num_threads=1,rollout_steps=4,updates=0,minibatches=1,reset_optimizer_on_resume=False,run_name="role_resume_zero_updates"))
with (BASE/"separate_critic_role_resume.log").open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):train(resume)
resumed=torch.load(resume.checkpoint,map_location="cpu",weights_only=False)
assert resumed["config"]["separate_critic_features"] is True
roles={group["role"]:group["lr"] for group in resumed["optimizer"]["param_groups"]}
assert roles=={"actor":7e-6,"critic":2e-4} and len(resumed["optimizer"]["state"])==14
for key,value in model.state_dict().items():assert torch.equal(value,resumed["model"][key])
export=fresh();export.load_state_dict(resumed["model"],strict=True)
assert torch.equal(export(obs)[0],model(obs)[0]) and torch.equal(export(obs)[1],model(obs)[1])
for row in obs:assert choose_action(export,row.numpy(),torch.device("cpu"),False)==choose_action(model,row.numpy(),torch.device("cpu"),False)
report={"status":"passed","legacy_default_parameter_count":9,"separate_parameter_count":14,
    "initial_copy_logits_values_exact":True,"no_shared_feature_parameters_or_storage":True,
    "critic_backward_only_independent_critic_branch":True,"policy_backward_only_original_actor_branch":True,
    "critic_features_trainable_when_copied_from_frozen_actor":True,
    "new_strict_compatible_state_exact":True,"legacy_pre_enabled_target_copy_exact":True,
    "optimizer_roles_disjoint_complete":True,"actor_group_parameter_count":len(actor),"critic_group_parameter_count":len(critic),
    "resumed_role_learning_rates":roles,"optimizer_states_preserved":14,"auto_config_separate_true":True,
    "export_logits_values_exact":True,"verification_files_only":True}
(BASE/"separate_critic_features_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
