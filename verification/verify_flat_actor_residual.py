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

torch.set_num_threads(1);torch.manual_seed(616)
source=torch.load(ROOT/"checkpoints"/"ppo_phase_v34_credit_calibrated.pt",map_location="cpu",weights_only=False)
legacy=source["model"];hidden=tuple(source["config"]["hidden_sizes"]);dim=legacy["net.0.weight"].shape[1]
def fresh():return ActorCritic(dim,16,hidden)
def must_fail(callback):
    try:callback()
    except (RuntimeError,ValueError):return
    raise AssertionError("incomplete/wrong residual checkpoint unexpectedly accepted")
assert not PPOConfig().enable_flat_actor_residual
baseline=fresh();baseline.load_state_dict(legacy,strict=True)
obs=torch.randn(12,dim)*baseline.observation_scale
frames=obs.reshape(12,-1,68)
for i in range(frames.shape[1]):frames[:,i,61]=2-(frames.shape[1]-1-i)/30
scores=torch.tensor([0.,9.99,10.,20.,44.9,45.,50.,64.99,65.,90.,100.,100.1])
frames[:,-1,60]=scores
outside=~(((scores>=10)&(scores<45))|((scores>=65)&(scores<=100)))
base_logits,base_value=baseline(obs)
model=fresh();model.load_state_dict(legacy,strict=True);model.enable_flat_residual()
assert model.flat_actor_weight.shape==(16,256) and torch.equal(model.flat_actor_weight,torch.zeros(16,256))
assert torch.equal(model(obs)[0],base_logits) and torch.equal(model(obs)[1],base_value)
with torch.no_grad():
    model.flat_actor_weight.normal_(0,.01);model.flat_actor_bias.copy_(torch.linspace(-.1,.2,16))
weight=model.flat_actor_weight.clone();bias=model.flat_actor_bias.clone();model.enable_flat_residual()
assert torch.equal(weight,model.flat_actor_weight) and torch.equal(bias,model.flat_actor_bias)
logits,value=model(obs)
expected=base_logits+torch.nn.functional.linear(model.features(obs),model.flat_actor_weight,model.flat_actor_bias)
assert torch.equal(logits[outside],base_logits[outside]) and torch.equal(logits[~outside],expected[~outside])
assert torch.equal(value,base_value)
state=copy.deepcopy(model.state_dict())
bias_only=dict(state);del bias_only["flat_actor_weight"]
for loader in [lambda m,s:m.load_state_dict(s,strict=True),load_model_compatible]:
    restored=fresh();loader(restored,state)
    assert torch.equal(restored.flat_actor_weight,weight) and torch.equal(restored.flat_actor_bias,bias)
    assert torch.equal(restored.flat_actor_bounds,model.flat_actor_bounds)
    assert torch.equal(restored(obs)[0],model(obs)[0])
    old=fresh();old.enable_flat_residual()
    with torch.no_grad():old.flat_actor_weight.fill_(5);old.flat_actor_bias.fill_(5)
    loader(old,legacy)
    assert torch.equal(old.flat_actor_weight,torch.zeros_like(weight)) and torch.equal(old.flat_actor_bias,torch.zeros_like(bias))
    assert torch.equal(old(obs)[0],base_logits)
    upgraded=fresh();upgraded.enable_flat_residual()
    with torch.no_grad():upgraded.flat_actor_weight.fill_(5)
    loader(upgraded,bias_only)
    assert torch.equal(upgraded.flat_actor_weight,torch.zeros_like(weight)) and torch.equal(upgraded.flat_actor_bias,bias)
    for key in ["flat_actor_bias","flat_actor_bounds"]:
        broken=dict(state);del broken[key];must_fail(lambda:loader(fresh(),broken))
    for shape in [(17,256),(16,255)]:
        wrong=dict(state);wrong["flat_actor_weight"]=torch.zeros(shape);must_fail(lambda:loader(fresh(),wrong))

model.net.requires_grad_(False);model.angular_velocity.requires_grad_(False);model.actor.requires_grad_(False)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,eps=1e-5)
assert len(optimizer.param_groups[0]["params"])==11
# Outside the flat ranges the residual gradients must be exact zeros.
lp,entropy,value=model.evaluate(obs[outside],torch.arange(int(outside.sum()))%16,1,3,True)
optimizer.zero_grad();(-lp.mean()-.005*entropy.mean()+.001*.5*(value-2).square().mean()).backward()
assert torch.equal(model.flat_actor_weight.grad,torch.zeros_like(weight))
assert torch.equal(model.flat_actor_bias.grad,torch.zeros_like(bias))
before=copy.deepcopy(model.state_dict())
for _ in range(3):
    lp,entropy,value=model.evaluate(obs[~outside],torch.arange(int((~outside).sum()))%16,1,3,True)
    optimizer.zero_grad();(-lp.mean()-.005*entropy.mean()+.001*.5*(value-2).square().mean()).backward()
    for name,param in model.named_parameters():
        if name.startswith(("net.","actor.","angular_velocity.")):assert param.grad is None,name
        else:assert param.grad is not None and torch.isfinite(param.grad).all(),name
    assert model.flat_actor_weight.grad.norm()>0 and model.flat_actor_bias.grad.norm()>0
    optimizer.step()
changed=[key for key,tensor in model.state_dict().items() if not torch.equal(tensor,before[key])]
assert set(changed)=={"flat_actor_weight","flat_actor_bias","critic.weight","critic.bias"}
probe=BASE/"flat_actor_residual_probe.pt"
cfg_values=dict(source["config"],enable_flat_actor_bias=True,enable_flat_actor_residual=True,checkpoint=probe,run_dir=BASE/"flat_residual_probe_logs")
cfg=PPOConfig(**cfg_values)
save_checkpoint(probe,model,optimizer,cfg,source["update"],source["global_step"])
auto_cfg=PPOConfig(**dict(cfg_values,enable_flat_actor_bias=False,enable_flat_actor_residual=False,resume_from=probe,
    checkpoint=BASE/"flat_actor_residual_auto_sync.pt",num_envs=1,num_threads=1,rollout_steps=4,updates=0,minibatches=1,reset_optimizer_on_resume=True,run_name="auto_sync_zero_updates"))
with (BASE/"flat_actor_residual_auto_sync.log").open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):train(auto_cfg)
saved=torch.load(auto_cfg.checkpoint,map_location="cpu",weights_only=False)
assert saved["config"]["enable_flat_actor_bias"] and saved["config"]["enable_flat_actor_residual"]
assert len(saved["optimizer"]["param_groups"][0]["params"])==11
report={"status":"passed","zero_enable_logits_values_exact":True,"outside_logits_exact":True,
    "inside_state_residual_matches_linear":True,"strict_compatible_full_restore":True,"legacy_and_bias_only_upgrade_zero_weights":True,
    "missing_bias_bounds_and_wrong_weight_shapes_rejected":True,"outside_residual_gradients_zero":True,
    "inside_residual_gradients_finite_nonzero":True,"only_updated_parameters":changed,
    "optimizer_parameters":11,"saved_config_auto_detect_both_flags":True,"verification_files_only":True}
(BASE/"flat_actor_residual_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
