from pathlib import Path
import sys
import json
import hashlib
import math
import time
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/"QWOP_rl_training_public"
sys.path.insert(0,str(ROOT/"scripts"))
from train_ppo import ActorCritic, FrameStacker, PPOConfig, compute_gae
from running_env import VectorRunningGameEnv

torch.set_num_threads(1)
started=time.perf_counter()
cfg_data=json.loads((ROOT/"configs"/"ppo_phase_v34_long_credit.json").read_text(encoding="utf-8"))
cfg=PPOConfig(**dict(cfg_data,num_envs=8,num_threads=1))
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
source=ROOT/"checkpoints"/"ppo_phase_v34_credit_calibrated.pt"
digest_before=hashlib.sha256(source.read_bytes()).hexdigest()
data=torch.load(source,map_location="cpu",weights_only=False)
env=VectorRunningGameEnv(8,num_threads=1,reward_profile="classic_100m",game_mode="classic_100m")
raw,_=env.reset()
stacker=FrameStacker(8,68,cfg.frame_stack)
obs=stacker.reset(raw)
model=ActorCritic(stacker.obs_dim,16,tuple(cfg.hidden_sizes))
model.load_state_dict(data["model"],strict=True)
original={key:tensor.clone() for key,tensor in model.state_dict().items()}
obs_buffer=[];action_buffer=[];lp_buffer=[];rewards=[];dones=[];values=[]
finished=[]
for step in range(cfg.rollout_steps):
    x=torch.as_tensor(obs,dtype=torch.float32)
    with torch.no_grad():
        acts,lp,_,value=model.act(x,cfg.start_logit_scale,cfg.start_explore_meters,cfg.effective_action_entropy)
    result=env.step_arrays(acts.numpy())
    assert np.isfinite(result["obs"]).all() and np.isfinite(result["reward"]).all()
    done=result["terminated"]|result["truncated"]
    obs_buffer.append(x.clone());action_buffer.append(acts);lp_buffer.append(lp)
    rewards.append(torch.as_tensor(result["reward"],dtype=torch.float32));dones.append(torch.as_tensor(done,dtype=torch.float32));values.append(value)
    for i in np.flatnonzero(done):
        finished.append({"step":step+1,"env":int(i),"success":bool(result["final_success"][i]),"distance":float(result["final_distance"][i]),"time":float(result["final_time"][i])})
    obs=stacker.update(result["obs"],done)
with torch.no_grad():
    _,next_value=model(torch.as_tensor(obs,dtype=torch.float32))
    advantages,returns=compute_gae(torch.stack(rewards),torch.stack(dones),torch.stack(values),next_value,cfg)
b_obs=torch.stack(obs_buffer).reshape(-1,stacker.obs_dim)
b_actions=torch.stack(action_buffer).flatten()
b_oldlp=torch.stack(lp_buffer).flatten()
b_adv=advantages.flatten()
b_adv=(b_adv-b_adv.mean())/(b_adv.std()+1e-8)
b_returns=returns.flatten()
named=list(model.named_parameters())
params=[parameter for _,parameter in named]

def gradient_vector(grads,prefixes):
    parts=[]
    for (name,param),gradient in zip(named,grads):
        if name.startswith(prefixes):parts.append((torch.zeros_like(param) if gradient is None else gradient).flatten())
    return torch.cat(parts)

def cosine(a,b):
    denom=a.norm()*b.norm()
    return float(torch.dot(a,b)/denom) if denom>0 else None

def diagnose(indices):
    newlp,entropy,value=model.evaluate(b_obs[indices],b_actions[indices],cfg.start_logit_scale,cfg.start_explore_meters,True)
    raw_ratio=(newlp-b_oldlp[indices]).exp()
    raw_ratio_error=float((raw_ratio.detach()-1).abs().max())
    assert raw_ratio_error<.01
    # Collection batches have eight samples while this diagnostic has up to
    # 4096; CPU GEMM rounding can slightly change logits. Recompute the frozen
    # old log probability on this same batch to anchor ratio exactly at one,
    # preserving the policy-loss gradient at the unchanged current policy.
    ratio=(newlp-newlp.detach()).exp()
    ratio_error=float((ratio.detach()-1).abs().max())
    policy=-torch.mean(b_adv[indices]*ratio)-cfg.ent_coef*entropy.mean()
    value_loss=cfg.vf_coef*.5*torch.mean((value-b_returns[indices])**2)
    pg=torch.autograd.grad(policy,params,retain_graph=True,allow_unused=True)
    vg=torch.autograd.grad(value_loss,params,allow_unused=True)
    assert all(gradient is None or torch.isfinite(gradient).all() for gradient in (*pg,*vg))
    shared_policy=gradient_vector(pg,("net.","angular_velocity."))
    shared_value=gradient_vector(vg,("net.","angular_velocity."))
    combined=shared_policy+shared_value
    actor_head=gradient_vector(pg,("actor.",))
    critic_head=gradient_vector(vg,("critic.",))
    cos=cosine(combined,shared_policy)
    result={"samples":len(indices),"policy_loss_including_entropy":float(policy.detach()),"weighted_value_loss":float(value_loss.detach()),
        "shared_policy_grad_l2":float(shared_policy.norm()),"shared_value_grad_l2":float(shared_value.norm()),
        "shared_value_to_policy_norm_ratio":float(shared_value.norm()/shared_policy.norm()),
        "shared_policy_value_cosine":cosine(shared_policy,shared_value),
        "shared_total_to_policy_cosine":cos,"shared_total_to_policy_angle_degrees":math.degrees(math.acos(max(-1,min(1,cos)))) if cos is not None else None,
        "shared_total_grad_l2":float(combined.norm()),"actor_head_policy_grad_l2":float(actor_head.norm()),
        "critic_head_weighted_value_grad_l2":float(critic_head.norm()),"ratio_max_absolute_error":ratio_error,
        "stored_rollout_ratio_rounding_max_error":raw_ratio_error,"all_gradients_finite":True}
    return result

full=diagnose(torch.arange(len(b_actions)))
permutation=torch.randperm(len(b_actions))
minibatches=[diagnose(permutation[i:i+512]) for i in [0,512,1024,1536]]
assert all(torch.equal(tensor,original[key]) for key,tensor in model.state_dict().items())
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest_before
report={"source":str(source),"checkpoint_sha256_unchanged":digest_before,"device":"cpu","seed":cfg.seed,
    "num_envs":8,"rollout_steps":cfg.rollout_steps,"gamma":cfg.gamma,"gae_lambda":cfg.gae_lambda,"start_logit_scale":cfg.start_logit_scale,
    "ent_coef":cfg.ent_coef,"vf_coef":cfg.vf_coef,"effective_action_entropy":True,"parameters_never_updated":True,
    "completed_episodes":finished,"raw_advantage_mean":float(advantages.mean()),"raw_advantage_std":float(advantages.std()),
    "gae_return_range":[float(returns.min()),float(returns.max())],"rollout_value_range":[float(torch.stack(values).min()),float(torch.stack(values).max())],
    "full_batch":full,"random_512_sample_minibatches":minibatches,"elapsed_seconds":time.perf_counter()-started,
    "interpretation_limit":"Gradient balance on one 8-env CPU rollout; not actual optimizer updates or proof of performance after freezing."}
(BASE/"v34_gradient_balance.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
