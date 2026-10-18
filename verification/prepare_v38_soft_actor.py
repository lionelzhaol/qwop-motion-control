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
started=time.perf_counter()
SOURCE=ROOT/"checkpoints"/"ppo_phase_v34_credit_calibrated.pt"
OUTPUT=ROOT/"checkpoints"/"ppo_target43_v38_soft_actor_initial.pt"
SCALE=.25
GROUPS=((0,),(1,3),(2,),(4,12),(5,7,13,15),(6,14),(8,),(9,11),(10,))
source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
data=torch.load(SOURCE,map_location="cpu",weights_only=False)
config=data["config"]
def load(state):
    model=ActorCritic(state["net.0.weight"].shape[1],16,tuple(config["hidden_sizes"]))
    result=model.load_state_dict(state,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.eval();return model
source=load(data["model"])
soft=load(data["model"])
with torch.no_grad():
    soft.actor.weight.mul_(SCALE);soft.actor.bias.mul_(SCALE)
changed=[key for key,tensor in soft.state_dict().items() if not torch.equal(tensor,data["model"][key])]
assert set(changed)=={"actor.weight","actor.bias"}
assert torch.equal(soft.actor.weight,source.actor.weight*SCALE) and torch.equal(soft.actor.bias,source.actor.bias*SCALE)
assert all(torch.isfinite(tensor).all() for tensor in soft.state_dict().values())
optimizer=torch.optim.Adam(soft.parameters(),lr=float(config["learning_rate"]),eps=1e-5)
assert not optimizer.state_dict()["state"]
checkpoint=copy.deepcopy(data)
checkpoint["model"]={key:tensor.clone() for key,tensor in soft.state_dict().items()}
checkpoint["optimizer"]=optimizer.state_dict()
checkpoint["config"]=dict(config,checkpoint=str(OUTPUT),resume_from=str(SOURCE),reset_optimizer_on_resume=True,
    detach_critic_features=True,freeze_backbone=False,enable_flat_actor_bias=False,enable_flat_actor_residual=False)
checkpoint["soft_actor_initialization"]={"source":str(SOURCE),"actor_scale":SCALE,"changed_parameters":changed,
    "method":"binary-exact positive global Actor linear weight/bias scaling; most likely action preserved",
    "no_new_training":True,"optimizer":"empty Adam constructed after full model restore",
    "critic_gamma":config["gamma"],"scope":"Exploration warmstart; does not claim a faster deterministic policy"}
torch.save(checkpoint,OUTPUT)
reloaded=torch.load(OUTPUT,map_location="cpu",weights_only=False)
verified=load(reloaded["model"])
assert not reloaded["optimizer"]["state"]
env=VectorRunningGameEnv(1,num_threads=1,reward_profile="classic_100m",game_mode="classic_100m")
raw,_=env.reset();stack=FrameStacker(1,68,int(config["frame_stack"]));obs=stack.reset(raw)
metrics=[];greedy_equal=True;logits_exact=True;value_exact=True;trajectory_actions=[]
for step in range(1,4001):
    x=torch.as_tensor(obs,dtype=torch.float32)
    with torch.no_grad():
        old_logits,old_value=source(x);new_logits,new_value=verified(x)
        logits_exact &= torch.equal(new_logits,old_logits*SCALE)
        value_exact &= torch.equal(old_value,new_value)
        old_action=int(old_logits.argmax(-1)[0]);action=int(new_logits.argmax(-1)[0])
        greedy_equal &= old_action==action
        assert logits_exact and value_exact and greedy_equal
        if float(raw[0,61])<3:
            old_dist=torch.distributions.Categorical(logits=old_logits)
            new_dist=torch.distributions.Categorical(logits=new_logits)
            old_groups=torch.stack([old_dist.probs[0,list(group)].sum() for group in GROUPS])
            new_groups=torch.stack([new_dist.probs[0,list(group)].sum() for group in GROUPS])
            metrics.append({"time":float(raw[0,61]),"score":float(raw[0,60]),
                "old_mask_entropy":float(old_dist.entropy()[0]),"new_mask_entropy":float(new_dist.entropy()[0]),
                "old_physical_entropy":float(source.action_entropy(old_dist,True)[0]),
                "new_physical_entropy":float(verified.action_entropy(new_dist,True)[0]),
                "old_max_mask_probability":float(old_dist.probs.max()),"new_max_mask_probability":float(new_dist.probs.max()),
                "old_max_physical_probability":float(old_groups.max()),"new_max_physical_probability":float(new_groups.max()),
                "old_physical_9_probabilities":old_groups.tolist(),"new_physical_9_probabilities":new_groups.tolist()})
    trajectory_actions.append(action)
    result=env.step_arrays(np.array([action],dtype=np.int64))
    assert np.isfinite(result["obs"]).all() and np.isfinite(result["reward"]).all()
    done=bool(result["terminated"][0] or result["truncated"][0])
    if done:
        terminal={"time_s":float(result["final_time"][0]),"distance_m":float(result["final_distance"][0]),
            "steps":step,"success":bool(result["final_success"][0]),"truncated":bool(result["truncated"][0])}
        assert terminal["success"] and not terminal["truncated"] and abs(terminal["time_s"]-44.93333435058594)<1e-6
        break
    raw=result["obs"];obs=stack.update(raw,np.array([False]))
else:raise RuntimeError("Scaled actor failed to terminate normally")
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
early_summary={key:float(np.mean([row[key] for row in metrics])) for key in metrics[0] if key not in {"time","score","old_physical_9_probabilities","new_physical_9_probabilities"}}
early_summary["old_physical_9_probabilities_mean"]=np.mean([row["old_physical_9_probabilities"] for row in metrics],axis=0).tolist()
early_summary["new_physical_9_probabilities_mean"]=np.mean([row["new_physical_9_probabilities"] for row in metrics],axis=0).tolist()
report={"status":"passed","source":str(SOURCE),"output":str(OUTPUT),"actor_scale":SCALE,
    "changed_parameters":changed,"all_other_tensors_bitwise_unchanged":True,"actor_logits_exactly_quarter":logits_exact,
    "critic_values_bitwise_unchanged":value_exact,"strict_reload_success":True,"empty_adam_parameters":len(list(verified.parameters())),
    "all_greedy_actions_unchanged":greedy_equal,"deterministic_terminal":terminal,
    "early_3s_state_count":len(metrics),"early_3s_entropy_probability_means":early_summary,
    "source_sha256_unchanged":source_hash,"output_sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    "elapsed_seconds":time.perf_counter()-started,"new_training_performed":False}
(BASE/"v38_soft_actor_initialization.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
(BASE/"v38_soft_actor_first3s_probability.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
