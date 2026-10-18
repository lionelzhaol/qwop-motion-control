from pathlib import Path
import sys
import json
import math
import time
import hashlib
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / "QWOP_rl_training_public"
sys.path.insert(0, str(ROOT / "scripts"))
from running_env import VectorRunningGameEnv
from train_ppo import ActorCritic, FrameStacker

torch.set_num_threads(1)
POLICIES = {
    "archived_fast_44p93": "ppo_phase_v32_hidden2_baseline.pt",
    "v36_best_12528": "ppo_target43_v36_flat_residual_update_012528.pt",
    "v37_best_12640": "ppo_target43_v37_actor_features_update_012640.pt",
    "v38_best_10912": "ppo_target43_v38_soft_actor_update_010912.pt",
}
gamma=.9999
results=[]
started=time.perf_counter()
for label, filename in POLICIES.items():
    path=ROOT/"checkpoints"/filename
    data=torch.load(path,map_location="cpu",weights_only=False)
    cfg=data["config"]
    env=VectorRunningGameEnv(1,num_threads=1,reward_profile="classic_100m",game_mode="classic_100m")
    raw,_=env.reset()
    stack=FrameStacker(1,68,int(cfg["frame_stack"]))
    obs=stack.reset(raw)
    model=ActorCritic(stack.obs_dim,16,tuple(cfg["hidden_sizes"]))
    model.load_state_dict(data["model"],strict=True)
    model.eval()
    rewards=[]
    segments=[0.0]*10
    crossing={}
    actions=[]
    low_steps=0
    for step in range(1,4001):
        segment=min(9,max(0,int(raw[0,60]//10)))
        low_steps+=int((429.8-float(raw[0,26])*40)/506.85 < .8)
        with torch.no_grad():
            logits,_=model(torch.as_tensor(obs,dtype=torch.float32))
            action=int(logits.argmax(-1)[0])
        actions.append(action)
        outcome=env.step_arrays(np.array([action],dtype=np.int64))
        reward=float(outcome["reward"][0])
        assert math.isfinite(reward) and np.isfinite(outcome["obs"]).all()
        rewards.append(reward)
        segments[segment]+=reward
        done=bool(outcome["terminated"][0] or outcome["truncated"][0])
        now=float(outcome["final_time"][0] if done else outcome["obs"][0,61])
        distance=float(outcome["final_distance"][0] if done else outcome["obs"][0,60])
        for milestone in [3,10,50,58,60,100]:
            if str(milestone) not in crossing and distance>=milestone:
                crossing[str(milestone)]=now
        if done:
            assert outcome["has_final"][0]
            success=bool(outcome["final_success"][0])
            terminal_component=(120-now+1200*np.clip((65-now)/(65-43),0,1)) if success and low_steps/step <=.1 else -30
            discounted=sum(gamma**i*r for i,r in enumerate(rewards))
            total=sum(rewards)
            report={"label":label,"checkpoint":str(path),"steps":step,"time_s":now,"distance_m":distance,
                "success":success,"truncated":bool(outcome["truncated"][0]),"return_sum":total,
                "discounted_return_gamma_0p9999":discounted,"terminal_step_reward":rewards[-1],
                "estimated_success_finish_component":float(terminal_component),
                "dense_return_excluding_success_finish":float(total-terminal_component),
                "finish_discount_factor":gamma**(step-1),
                "discounted_finish_component":float(gamma**(step-1)*terminal_component),
                "discounted_dense_excluding_finish":float(discounted-gamma**(step-1)*terminal_component),
                "reward_by_10m_segment":segments,"crossings":crossing,"low_posture_fraction":low_steps/step,
                "action_sha256":hashlib.sha256(bytes(actions)).hexdigest()}
            results.append(report)
            print(json.dumps({k:report[k] for k in ["label","time_s","success","return_sum","discounted_return_gamma_0p9999","discounted_finish_component"]}),flush=True)
            break
        raw=outcome["obs"]
        obs=stack.update(raw,np.array([False]))
    else:
        raise RuntimeError(f"{label} did not terminate")
speed_order=sorted(results,key=lambda r:r["time_s"])
conflicts=[]
for i,fast in enumerate(speed_order):
    for slow in speed_order[i+1:]:
        if not fast["success"] or not slow["success"]:continue
        for metric in ["return_sum","discounted_return_gamma_0p9999"]:
            if slow[metric]>fast[metric]:
                conflicts.append({"faster":fast["label"],"slower":slow["label"],"time_gap":slow["time_s"]-fast["time_s"],"metric":metric,"slower_reward_advantage":slow[metric]-fast[metric]})
output={"reward_code_sha256":hashlib.sha256((ROOT/"src"/"reward_profiles.cpp").read_bytes()).hexdigest(),
    "gamma":gamma,"deterministic":True,"device":"cpu","elapsed_seconds":time.perf_counter()-started,
    "policies":results,"speed_order":[r["label"] for r in speed_order],"ordering_conflicts":conflicts,
    "finish_time_reward_slope_points_per_second":1+1200/(65-43),
    "notes":"Actual unchanged native env rewards. Finish component formula applies to qualified success; discount exponents start at the first action. Reset reward is not counted, matching training."}
(BASE/"policy_reward_ordering_v39_gamma9999.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"conflicts":conflicts,"elapsed_seconds":output["elapsed_seconds"]}),flush=True)
