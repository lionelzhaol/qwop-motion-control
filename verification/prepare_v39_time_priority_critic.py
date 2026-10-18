from pathlib import Path
import copy
import hashlib
import json
import sys
import time
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/"QWOP_rl_training_public"
sys.path.insert(0,str(ROOT/"scripts"))
from train_ppo import ActorCritic,FrameStacker
from running_env import VectorRunningGameEnv

torch.set_num_threads(1)
SOURCE=ROOT/"checkpoints"/"ppo_target43_v38_soft_actor_initial.pt"
OUTPUT=ROOT/"checkpoints"/"ppo_target43_v39_time_priority_initial.pt"
GAMMA=.9999
started=time.perf_counter()
source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
source=torch.load(SOURCE,map_location="cpu",weights_only=False)
cfg=source["config"]
def load(state):
    model=ActorCritic(state["net.0.weight"].shape[1],16,tuple(cfg["hidden_sizes"]))
    result=model.load_state_dict(state,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.eval();return model
original=load(source["model"])
model=load(source["model"]);model.enable_separate_critic()
assert model.critic_motion is not None
for name,tensor in model.critic_net.state_dict().items():assert torch.equal(tensor,model.net.state_dict()[name])
assert torch.equal(model.critic_motion.weight,model.angular_velocity.weight)
cloned_features={name:tensor.clone() for name,tensor in model.state_dict().items() if name.startswith(("critic_net.","critic_motion."))}

def collect(actor,record=False):
    env=VectorRunningGameEnv(1,num_threads=1,reward_profile="classic_100m",game_mode="classic_100m")
    raw,_=env.reset();stack=FrameStacker(1,68,cfg["frame_stack"]);obs=stack.reset(raw)
    features=[];observations=[];rewards=[];actions=[]
    for step in range(1,4001):
        x=torch.as_tensor(obs,dtype=torch.float32)
        with torch.no_grad():
            logits,value=actor(x);action=int(logits.argmax(-1)[0])
            old_logits,_=original(x)
            assert torch.equal(logits,old_logits)
            if record:
                hidden=actor.critic_features(x)
                assert torch.equal(hidden,actor.features(x))
                features.append(hidden[0].numpy().copy());observations.append(obs[0].copy())
        result=env.step_arrays(np.array([action],dtype=np.int64))
        assert np.isfinite(result["obs"]).all() and np.isfinite(result["reward"]).all()
        rewards.append(float(result["reward"][0]));actions.append(action)
        done=bool(result["terminated"][0] or result["truncated"][0])
        if done:
            assert result["has_final"][0] and result["final_success"][0] and not result["truncated"][0]
            terminal={"time_s":float(result["final_time"][0]),"distance_m":float(result["final_distance"][0]),"steps":step,"success":True}
            assert terminal["time_s"]==44.93333435058594
            quality=(65-terminal["time_s"])/22
            expected_finish=120-terminal["time_s"]+1200*quality
            assert rewards[-1]>=expected_finish-.1 and rewards[-1]<expected_finish+10
            # Production vector env has already reset the completed instance;
            # a subsequent step must belong to a fresh episode, not pay again.
            next_step=env.step_arrays(np.array([0],dtype=np.int64))
            assert not next_step["has_final"][0] and float(next_step["obs"][0,61])<.04
            assert float(next_step["reward"][0])<20
            return {"terminal":terminal,"features":np.array(features),"observations":np.array(observations),
                "rewards":np.array(rewards),"actions":np.array(actions),"terminal_success_component":expected_finish,
                "next_reset_step_reward":float(next_step["reward"][0]),"native_v39_finish_verified":True}
        raw=result["obs"];obs=stack.update(raw,np.array([False]))
    raise RuntimeError("Policy did not finish within original limit")

trajectory=collect(model,True)
returns=np.empty_like(trajectory["rewards"],dtype=np.float64);running=0.
for i in range(len(returns)-1,-1,-1):running=trajectory["rewards"][i]+GAMMA*running;returns[i]=running
x=trajectory["features"].astype(np.float64)
def fit(features,target,ridge):
    xm=features.mean(0);ym=target.mean();center=features-xm
    weight=np.linalg.solve(center.T@center+ridge*np.eye(features.shape[1]),center.T@(target-ym))
    return weight,ym-xm@weight
def rmse(pred,target):return float(np.sqrt(np.mean((pred-target)**2)))
validation=np.arange(len(returns))%5==0
trials=[]
for ridge in [1.,10.]:
    weight,bias=fit(x[~validation],returns[~validation],ridge)
    trials.append({"ridge":ridge,"train_rmse":rmse(x[~validation]@weight+bias,returns[~validation]),
        "heldout_rmse":rmse(x[validation]@weight+bias,returns[validation]),"weight_l2":float(np.linalg.norm(weight))})
minimum=min(trial["heldout_rmse"] for trial in trials)
selected=min((trial for trial in trials if trial["heldout_rmse"]<=1.5*minimum),key=lambda trial:trial["weight_l2"])
weight,bias=fit(x,returns,selected["ridge"])
with torch.no_grad():model.critic.weight.copy_(torch.tensor(weight[None],dtype=torch.float32));model.critic.bias.copy_(torch.tensor([bias],dtype=torch.float32))
for name,tensor in source["model"].items():
    if not name.startswith("critic."):assert torch.equal(tensor,model.state_dict()[name]),name
for name,tensor in cloned_features.items():assert torch.equal(tensor,model.state_dict()[name]),name
assert all(torch.isfinite(tensor).all() for tensor in model.state_dict().values())
obs_tensor=torch.tensor(trajectory["observations"],dtype=torch.float32)
with torch.no_grad():
    old_logits,old_values=original(obs_tensor);new_logits,new_values=model(obs_tensor)
assert torch.equal(old_logits,new_logits)
predictions=new_values.numpy().astype(np.float64)
actor_params=[];critic_params=[]
for name,parameter in model.named_parameters():
    (critic_params if name.startswith(("critic.","critic_net.","critic_motion.")) else actor_params).append(parameter)
assert len(actor_params)==7 and len(critic_params)==7
assert not {id(param) for param in actor_params}&{id(param) for param in critic_params}
optimizer=torch.optim.Adam([{"params":actor_params,"lr":3e-6,"role":"actor"},{"params":critic_params,"lr":3e-5,"role":"critic"}],eps=1e-5)
assert not optimizer.state_dict()["state"]
checkpoint=copy.deepcopy(source)
checkpoint["model"]={name:tensor.detach().clone() for name,tensor in model.state_dict().items()}
checkpoint["optimizer"]=optimizer.state_dict()
checkpoint["config"]=dict(cfg,checkpoint=str(OUTPUT),resume_from=str(SOURCE),gamma=GAMMA,gae_lambda=.995,
    learning_rate=3e-6,critic_learning_rate=3e-5,separate_critic_features=True,detach_critic_features=False,
    freeze_backbone=False,reset_optimizer_on_resume=True,enable_flat_actor_bias=False,enable_flat_actor_residual=False)
metadata={"source":str(SOURCE),"gamma":GAMMA,"goal_seconds":43.,"cadence_coefficient":.3,"fast_finish_bonus":1200.,
    "actor_soft_scale":.25,"samples":len(returns),"selected_ridge":selected["ridge"],"fit_rmse":rmse(predictions,returns),
    "source_value_rmse_on_new_reward_gamma":rmse(old_values.numpy(),returns),"critic_weight_l2":float(np.linalg.norm(weight)),
    "actor_and_backbone_all_tensors_unchanged":True,"critic_features_copied_exact_unchanged":True,
    "only_modified_existing_parameters":["critic.weight","critic.bias"],"optimizer":"empty Adam, actor3e-6/critic3e-5, seven parameters per role",
    "no_new_ppo_training":True,"scope":"One successful native v39 trajectory; copied Critic encoder with ridge value-head pre-calibration"}
checkpoint["time_priority_critic_precalibration"]=metadata
torch.save(checkpoint,OUTPUT)
saved=torch.load(OUTPUT,map_location="cpu",weights_only=False);reloaded=load(saved["model"])
assert len(list(reloaded.parameters()))==14 and not saved["optimizer"]["state"]
with torch.no_grad():assert torch.equal(reloaded(obs_tensor)[0],old_logits) and torch.equal(reloaded(obs_tensor)[1],new_values)
confirmed=collect(reloaded)
assert np.array_equal(confirmed["actions"],trajectory["actions"]) and np.array_equal(confirmed["rewards"],trajectory["rewards"])
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
report={"status":"passed","source":str(SOURCE),"output":str(OUTPUT),**metadata,"ridge_trials":trials,
    "terminal":confirmed["terminal"],"native_v39_finish_verified":True,"actual_terminal_step_reward":trajectory["rewards"][-1],
    "success_finish_component":trajectory["terminal_success_component"],"autoreset_next_step_reward":trajectory["next_reset_step_reward"],
    "sum_actual_v39_rewards":float(trajectory["rewards"].sum()),"discounted_return_gamma9999":float(returns[0]),
    "return_range":[float(returns.min()),float(returns.max())],"fitted_value_range":[float(predictions.min()),float(predictions.max())],
    "error_abs_max":float(np.abs(predictions-returns).max()),"actor_actions_rewards_bitwise_unchanged":True,
    "strict_reload_success":True,"all_model_tensors_finite":True,"optimizer_parameter_count":14,
    "checkpoint_sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),"elapsed_seconds":time.perf_counter()-started}
(BASE/"v39_time_priority_initialization.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
