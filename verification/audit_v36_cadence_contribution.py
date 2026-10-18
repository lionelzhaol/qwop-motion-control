from pathlib import Path
import ctypes as C
import json
import sys
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/"QWOP_rl_training_public"
sys.path.insert(0,str(ROOT/"scripts"))
from running_env import RunningGameEnv
from train_ppo import ActorCritic,FrameStacker
from evaluate_gait_bottlenecks import Frame,Course
POLICIES={
    "archived_fast_44p93":"ppo_phase_v32_hidden2_baseline.pt",
    "v36_best_12528":"ppo_target43_v36_flat_residual_update_012528.pt",
    "v37_best_12640":"ppo_target43_v37_actor_features_update_012640.pt",
    "v38_best_10912":"ppo_target43_v38_soft_actor_update_010912.pt",
}

torch.set_num_threads(1)
gamma=.999
dll=C.CDLL(str(ROOT/"bin"/"running_physics.dll"))
dll.rg_create.argtypes=[C.POINTER(C.c_void_p)];dll.rg_create.restype=C.c_int
dll.rg_reset.argtypes=[C.c_void_p,C.POINTER(Course)];dll.rg_reset.restype=C.c_int
dll.rg_read_frame.argtypes=[C.c_void_p,C.POINTER(Frame)];dll.rg_read_frame.restype=C.c_int
dll.rg_step.argtypes=[C.c_void_p,C.c_uint8];dll.rg_step.restype=C.c_int
dll.rg_destroy.argtypes=[C.c_void_p]
audit=C.CDLL(str(BASE/"cadence_reward_audit.dll"))
audit.audit_create.restype=C.c_void_p
audit.audit_destroy.argtypes=[C.c_void_p]
audit.audit_reward.argtypes=[C.c_void_p,C.POINTER(Frame),C.c_uint8,C.POINTER(C.c_float)]
results=[]
for label,filename in POLICIES.items():
    data=torch.load(ROOT/"checkpoints"/filename,map_location="cpu",weights_only=False)
    cfg=data["config"]
    model=ActorCritic(data["model"]["net.0.weight"].shape[1],16,tuple(cfg["hidden_sizes"]))
    model.load_state_dict(data["model"],strict=True);model.eval()
    env=RunningGameEnv("classic_100m","classic_100m")
    obs,_=env.reset();stack=FrameStacker(1,68,cfg["frame_stack"]);stacked=stack.reset(obs[None])
    native=C.c_void_p();assert dll.rg_create(C.byref(native))==0
    course=Course(C.sizeof(Course),0);assert dll.rg_reset(native,C.byref(course))==0
    frame=Frame();frame.struct_size=C.sizeof(Frame);assert dll.rg_read_frame(native,C.byref(frame))==0
    pair=audit.audit_create();out=(C.c_float*2)();audit.audit_reward(pair,C.byref(frame),0,out)
    reward_sum=reward_discount=cadence_sum=cadence_discount=0.;max_reward_error=0.;paid=0
    try:
        for step in range(1,4001):
            with torch.no_grad():action=int(model(torch.as_tensor(stacked,dtype=torch.float32))[0].argmax(-1)[0])
            obs,reward,done,truncated,info=env.step(action)
            assert dll.rg_step(native,action)==0 and dll.rg_read_frame(native,C.byref(frame))==0
            assert abs(info["time"]-frame.sim_time)<1e-5 and abs(info["distance"]-frame.score)<1e-5
            audit.audit_reward(pair,C.byref(frame),int(truncated),out)
            error=abs(float(out[0])-float(reward));max_reward_error=max(max_reward_error,error)
            assert error<2e-4,(label,step,error)
            cadence=float(out[0])-float(out[1]);assert cadence>=-2e-4
            paid+=cadence>1e-4
            reward_sum+=float(reward);reward_discount+=gamma**(step-1)*float(reward)
            cadence_sum+=cadence;cadence_discount+=gamma**(step-1)*cadence
            if done or truncated:
                assert info["success"] and not truncated
                quality=float(np.clip((65-info["time"])/22,0,1))
                extra_finish=1120*quality
                assert reward>120-info["time"]+80*quality-1
                report={"label":label,"time_s":info["time"],"distance":info["distance"],"steps":step,
                    "success":info["success"],"current_return_sum":reward_sum,"current_return_gamma999":reward_discount,
                    "cadence_only_contribution_sum":cadence_sum,"cadence_only_contribution_gamma999":cadence_discount,
                    "cadence_reward_paid_events":paid,"return_without_cadence":reward_sum-cadence_sum,
                    "discount_return_without_cadence":reward_discount-cadence_discount,
                    "predicted_cadence0p3_finish1200_sum":reward_sum-.875*cadence_sum+extra_finish,
                    "predicted_cadence0p3_finish1200_gamma999":reward_discount-.875*cadence_discount+gamma**(step-1)*extra_finish,
                    "native_trajectory_and_current_reward_verified":True,"reward_error_max":max_reward_error}
                results.append(report);print(json.dumps(report),flush=True);break
            stacked=stack.update(obs[None],np.array([False]))
    finally:
        audit.audit_destroy(pair);dll.rg_destroy(native)
report={"gamma":gamma,"current_cadence_coefficient":2.4,"policies":results,
    "method":"Current compiled C++ reward profile compared on every actual duplicated native frame with an otherwise identical C++ copy having only cadence coefficient set to zero. Source C++ unchanged.",
    "predicted_reward_change_note":"Same fixed trajectories; linear counterfactual of proposed coefficients, not new policy training/performance."}
(BASE/"v36_cadence_component_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
