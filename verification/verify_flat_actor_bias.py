from pathlib import Path
import copy
import contextlib
import json
import sys
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/"QWOP_rl_training_public"
sys.path.insert(0,str(ROOT/"scripts"))
from train_ppo import ActorCritic, PPOConfig, load_model_compatible, save_checkpoint, train
from export_policy_replay import choose_action

torch.set_num_threads(1)
torch.manual_seed(624)
source=torch.load(ROOT/"checkpoints"/"ppo_phase_v34_credit_calibrated.pt",map_location="cpu",weights_only=False)
legacy=source["model"]
hidden=tuple(source["config"]["hidden_sizes"])
dim=legacy["net.0.weight"].shape[1]

def fresh():return ActorCritic(dim,16,hidden)
def expect_failure(callback):
    try:callback()
    except (RuntimeError,ValueError):return
    raise AssertionError("invalid checkpoint should fail")

model=fresh();model.load_state_dict(legacy,strict=True)
assert model.flat_actor_bias is None and model.flat_actor_bounds is None
assert "flat_actor_bias" not in model.state_dict() and "flat_actor_bounds" not in model.state_dict()
assert PPOConfig().enable_flat_actor_bias is False
scores=torch.tensor([0.,9.99,10.,44.99,45.,64.99,65.,99.99,100.,100.01])
obs=torch.randn(len(scores),dim)
obs*=model.observation_scale
frames=obs.reshape(len(scores),-1,68)
for i in range(frames.shape[1]):frames[:,i,61]=2-(frames.shape[1]-1-i)/30
frames[:,:,60]=70
frames[:,-1,60]=scores
with torch.no_grad():base_logits,base_value=model(obs)
model.enable_flat_bias()
with torch.no_grad():zero_logits,zero_value=model(obs)
assert torch.equal(base_logits,zero_logits) and torch.equal(base_value,zero_value)
with torch.no_grad():model.flat_actor_bias.copy_(torch.linspace(-.8,.7,16))
bias_before=model.flat_actor_bias.detach().clone()
model.enable_flat_bias()
assert torch.equal(model.flat_actor_bias,bias_before)
flat=((scores>=10)&(scores<45))|((scores>=65)&(scores<=100))
with torch.no_grad():logits,values=model(obs)
assert torch.equal(logits[~flat],base_logits[~flat])
assert torch.equal(logits[flat],(base_logits+bias_before)[flat])
assert torch.equal(values,base_value)
assert flat.tolist()==[False,False,True,True,False,False,True,True,True,False]

for loader in [lambda m,s:m.load_state_dict(s,strict=True),load_model_compatible]:
    old_target=fresh();old_target.enable_flat_bias()
    with torch.no_grad():old_target.flat_actor_bias.fill_(2)
    loader(old_target,legacy)
    assert torch.equal(old_target.flat_actor_bias,torch.zeros(16))
    assert torch.equal(old_target(obs)[0],base_logits)

custom_bounds=torch.tensor([11.,44.,66.,99.])
model.enable_flat_bias(custom_bounds)
newstate=copy.deepcopy(model.state_dict())
for loader in [lambda m,s:m.load_state_dict(s,strict=True),load_model_compatible]:
    restored=fresh();loader(restored,newstate)
    assert torch.equal(restored.flat_actor_bias,newstate["flat_actor_bias"])
    assert torch.equal(restored.flat_actor_bounds,custom_bounds)
    assert torch.equal(restored(obs)[0],model(obs)[0])
    broken=dict(newstate);del broken["flat_actor_bounds"]
    expect_failure(lambda:loader(fresh(),broken))
    wrong=dict(newstate);wrong["flat_actor_bias"]=torch.zeros(17)
    expect_failure(lambda:loader(fresh(),wrong))
    invalid=dict(newstate);invalid["flat_actor_bounds"]=torch.tensor([10.,45.,45.,100.])
    expect_failure(lambda:loader(fresh(),invalid))
    nonfinite=dict(newstate);nonfinite["flat_actor_bounds"]=torch.tensor([10.,45.,65.,float("nan")])
    expect_failure(lambda:loader(fresh(),nonfinite))

# Exercise actual optimizer updates with exactly the production freeze scope.
model.enable_flat_bias([10.,45.,65.,100.])
model.net.requires_grad_(False)
model.angular_velocity.requires_grad_(False)
model.actor.requires_grad_(False)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,eps=1e-5)
assert len(optimizer.param_groups[0]["params"])==10
before=copy.deepcopy(model.state_dict())
for _ in range(3):
    lp,entropy,value=model.evaluate(obs,torch.arange(len(scores))%16,1,3,True)
    loss=-lp.mean()-.005*entropy.mean()+.001*.5*(value-2).square().mean()
    optimizer.zero_grad();loss.backward()
    for name,param in model.named_parameters():
        if name.startswith(("net.","angular_velocity.","actor.")):assert param.grad is None,name
        else:assert param.grad is not None and torch.isfinite(param.grad).all(),name
    optimizer.step()
changed=[key for key,tensor in model.state_dict().items() if not torch.equal(tensor,before[key])]
assert set(changed)=={"flat_actor_bias","critic.weight","critic.bias"}

# Save with the existing checkpoint function, then test actual trainer auto
# detection and config synchronization without performing another PPO update.
probe=BASE/"flat_actor_bias_probe.pt"
cfg_values=dict(source["config"],enable_flat_actor_bias=True,checkpoint=probe,run_dir=BASE/"flat_bias_probe_logs")
cfg=PPOConfig(**cfg_values)
save_checkpoint(probe,model,optimizer,cfg,source["update"],source["global_step"])
saved=torch.load(probe,map_location="cpu",weights_only=False)
saved_model=fresh();saved_model.load_state_dict(saved["model"],strict=True)
with torch.no_grad():assert torch.equal(model(obs)[0],saved_model(obs)[0])
for row in obs:
    action=choose_action(saved_model,row.numpy(),torch.device("cpu"),False)
    with torch.no_grad():assert action==int(saved_model(row[None])[0].argmax(-1)[0])
roundtrip=BASE/"flat_actor_bias_auto_sync.pt"
auto_values=dict(cfg_values,enable_flat_actor_bias=False,resume_from=probe,checkpoint=roundtrip,
    num_envs=1,num_threads=1,rollout_steps=4,updates=0,minibatches=1,reset_optimizer_on_resume=True,run_name="auto_sync_zero_updates")
auto_cfg=PPOConfig(**auto_values)
with (BASE/"flat_actor_bias_auto_sync.log").open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):
    train(auto_cfg)
auto_saved=torch.load(roundtrip,map_location="cpu",weights_only=False)
assert auto_saved["config"]["enable_flat_actor_bias"] is True and auto_cfg.enable_flat_actor_bias
assert len(auto_saved["optimizer"]["param_groups"][0]["params"])==10

report={"status":"passed","default_no_new_state_keys":True,"default_and_zero_enable_logits_exact":True,
    "gate_scores":scores.tolist(),"gate_results":flat.tolist(),"newest_raw_score_used":True,
    "outside_logits_exact":True,"inside_logits_add_bias_exact":True,"critic_values_unchanged_by_bias":True,
    "legacy_strict_and_compatible_zero_mapping":True,"new_strict_and_compatible_restore_bias_bounds":True,
    "missing_bounds_wrong_bias_shape_invalid_bounds_rejected":True,"optimizer_parameter_count":10,
    "only_updated_parameters":changed,"saved_config_auto_sync":True,
    "existing_export_choose_action_works":True,"verification_files_only":True}
(BASE/"flat_actor_bias_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
