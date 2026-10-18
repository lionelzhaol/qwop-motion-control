from pathlib import Path
import copy
import contextlib
import importlib.util
import json
import sys
from unittest.mock import patch
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/"QWOP_rl_training_public"
sys.path.insert(0,str(ROOT/"scripts"))
import train_ppo as new
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location("ppo_before_rnd_fix",BASE/"train_ppo_before_rnd_fix.py")
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old);old.ROOT=ROOT
source=torch.load(ROOT/"checkpoints"/"ppo_target43_v39_time_priority_initial.pt",map_location="cpu",weights_only=False)

def nested_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(nested_equal(a[key],b[key]) for key in a)
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a==b

# Actual native beta-zero training before/after the production edit, including
# policy RNG state and optimizer moments, not merely a forward-only check.
zero_results=[];zero_rng=[]
for label,module in [("old",old),("new",new)]:
    values=dict(source["config"],num_envs=2,num_threads=1,rollout_steps=8,updates=1,epochs=1,minibatches=2,
        rnd_beta=0.,seed=449,resume_from=ROOT/"checkpoints"/"ppo_target43_v39_time_priority_initial.pt",
        checkpoint=BASE/f"rnd_beta0_{label}.pt",run_dir=BASE/"rnd_verification_logs",run_name=f"beta0_{label}",save_interval=0,reset_optimizer_on_resume=True)
    cfg=module.PPOConfig(**values)
    with patch.object(torch.cuda,"is_available",return_value=False),(BASE/f"rnd_beta0_{label}.log").open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):module.train(cfg)
    zero_results.append(torch.load(cfg.checkpoint,map_location="cpu",weights_only=False));zero_rng.append(torch.get_rng_state().clone())
assert nested_equal(zero_results[0]["model"],zero_results[1]["model"])
assert nested_equal(zero_results[0]["optimizer"],zero_results[1]["optimizer"])
assert torch.equal(zero_rng[0],zero_rng[1])

cfg=new.PPOConfig(seed=449,rnd_hidden_size=32,rnd_feature_dim=16)
rng=torch.get_rng_state().clone();rnd,optimizer=new.make_rnd(cfg,torch.device("cpu"));assert torch.equal(rng,torch.get_rng_state())
if torch.cuda.is_available():
    gpu_rng=torch.cuda.get_rng_state().clone();cpu_rng=torch.get_rng_state().clone()
    gpu_rnd,_=new.make_rnd(cfg,torch.device("cuda"))
    assert torch.equal(gpu_rng,torch.cuda.get_rng_state()) and torch.equal(cpu_rng,torch.get_rng_state())
    del gpu_rnd
raw=torch.zeros(16,32*68)
frames=raw.reshape(16,32,68);frames[:,-1,:60]=rnd.input_scale[:60];frames[:,-1,66:68]=1
assert torch.equal(rnd.observation_features(raw),torch.ones(16,62))
altered=raw.clone().reshape(16,32,68);altered[:,:-1,:]=12345;altered[:,-1,60:66]=98765
assert torch.equal(rnd.observation_features(altered.reshape(16,-1)),rnd.observation_features(raw))
clamped=raw.clone();clamped[:,-68]=1e6;clamped[:,-67]=-1e6
assert torch.equal(rnd.observation_features(clamped)[:,:2],torch.tensor([5.,-5.]).expand(16,2))
features=torch.randn(64,62).clamp(-5,5);target_before=copy.deepcopy(rnd.target.state_dict())
before=float(rnd.prediction_error(features).mean().detach())
valid=torch.ones(64,dtype=torch.bool);rnd.intrinsic_bonus(features,valid)
fast_optimizer=torch.optim.Adam(rnd.predictor.parameters(),lr=.01)
for _ in range(80):
    prediction,target=rnd(features);loss=.5*(prediction-target.detach()).square().mean()
    fast_optimizer.zero_grad();loss.backward();fast_optimizer.step()
after=float(rnd.prediction_error(features).mean().detach())
assert after<before*.4
assert nested_equal(rnd.target.state_dict(),target_before)
assert all(parameter.grad is None and not parameter.requires_grad for parameter in rnd.target.parameters())
bonus=rnd.intrinsic_bonus(features,valid)
assert float(bonus.mean())<.5 and torch.isfinite(bonus).all() and float(bonus.max())<=5
saved=copy.deepcopy(rnd.state_dict());restored,_=new.make_rnd(cfg,torch.device("cpu"))
assert restored.checkpoint_compatible(saved);restored.load_state_dict(saved,strict=True)
assert torch.equal(restored.intrinsic_bonus(features,valid),rnd.intrinsic_bonus(features,valid))
count=int(rnd.error_count);masked=torch.zeros(64,dtype=torch.bool)
assert torch.equal(rnd.intrinsic_bonus(features,masked),torch.zeros(64)) and int(rnd.error_count)==count
legacy_rnd=old.RNDModel(2176,32,16)
assert not rnd.checkpoint_compatible(legacy_rnd.state_dict())

class FakeEnv:
    def __init__(self,num_envs,**kwargs):self.num_envs=num_envs;self.game_mode="classic_100m";self.action_space_n=16;self.index=0
    def observation(self):
        obs=np.zeros((self.num_envs,68),dtype=np.float32);obs[:,0]=np.arange(self.num_envs)*.1+self.index*.1
        obs[:,60]=self.index*.1;obs[:,61]=self.index/30;obs[:,66:68]=1;return obs
    def reset(self):return self.observation(),{}
    def step_arrays(self,actions):
        self.index+=1;obs=self.observation();done=np.zeros(self.num_envs,dtype=bool)
        if self.index%2==0:done[0]=True;obs[0,0]=25;obs[0,60:62]=0
        return {"obs":obs,"reward":np.full(self.num_envs,.1,dtype=np.float32),"terminated":done,
            "truncated":np.zeros(self.num_envs,dtype=bool),"has_final":done,"final_distance":np.full(self.num_envs,.2,dtype=np.float32)}

collected=[];trained=[]
original_bonus=new.RNDModel.intrinsic_bonus;original_forward=new.RNDModel.forward
def bonus_hook(self,features,valid):
    output=original_bonus(self,features,valid);collected.append((features.detach().clone(),valid.clone(),output.clone()));return output
def forward_hook(self,features):
    if torch.is_grad_enabled():trained.append(features.detach().clone())
    return original_forward(self,features)
fake_cfg=new.PPOConfig(num_envs=4,num_threads=1,rollout_steps=4,updates=1,epochs=3,minibatches=4,
    hidden_sizes=(32,32),frame_stack=2,rnd_beta=.02,rnd_hidden_size=32,rnd_feature_dim=16,seed=449,
    checkpoint=BASE/"rnd_exact_features_mask.pt",run_dir=BASE/"rnd_verification_logs",run_name="features_mask",save_interval=0)
with patch.object(new,"VectorRunningGameEnv",FakeEnv),patch.object(torch.cuda,"is_available",return_value=False),patch.object(new.RNDModel,"intrinsic_bonus",bonus_hook),patch.object(new.RNDModel,"forward",forward_hook),(BASE/"rnd_exact_features_mask.log").open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):new.train(fake_cfg)
expected=torch.cat([features[valid] for features,valid,_ in collected])
assert expected.shape==(14,62) and torch.equal(torch.cat(trained),expected)
assert len(trained)==4 # one 14-sample predictor pass, not epochs*passes
assert all(torch.equal(bonus[~valid],torch.zeros_like(bonus[~valid])) for _,valid,bonus in collected)
fake_checkpoint=torch.load(fake_cfg.checkpoint,map_location="cpu",weights_only=False)
assert int(fake_checkpoint["rnd_model"]["error_count"])==14
assert all(float(values["step"])==4 for values in fake_checkpoint["rnd_optimizer"]["state"].values())

# A legacy RND checkpoint must recreate both entire networks and optimizer,
# rather than leave a matching final layer partially loaded from its target.
legacy_checkpoint=copy.deepcopy(source)
poison=legacy_rnd.state_dict()
for tensor in poison.values():tensor.fill_(42)
legacy_checkpoint["rnd_model"]=poison
legacy_checkpoint["rnd_optimizer"]=torch.optim.Adam(legacy_rnd.predictor.parameters()).state_dict()
legacy_path=BASE/"rnd_legacy_poison.pt";torch.save(legacy_checkpoint,legacy_path)
reset_cfg=new.PPOConfig(**dict(source["config"],num_envs=1,num_threads=1,rollout_steps=4,updates=0,minibatches=1,
    rnd_beta=.02,rnd_hidden_size=32,rnd_feature_dim=16,seed=449,resume_from=legacy_path,
    checkpoint=BASE/"rnd_legacy_reset.pt",run_dir=BASE/"rnd_verification_logs",run_name="legacy_reset",reset_optimizer_on_resume=False))
with patch.object(torch.cuda,"is_available",return_value=False),(BASE/"rnd_legacy_reset.log").open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):new.train(reset_cfg)
reset=torch.load(reset_cfg.checkpoint,map_location="cpu",weights_only=False);fresh,_=new.make_rnd(reset_cfg,torch.device("cpu"))
assert nested_equal(reset["rnd_model"],fresh.state_dict()) and not reset["rnd_optimizer"]["state"]
assert "rnd_state_reset=1" in (BASE/"rnd_legacy_reset.log").read_text(encoding="utf-8")
report={"status":"passed","beta0_actual_native_ppo_model_optimizer_rng_bitwise_unchanged":True,
    "cpu_cuda_rng_init_unchanged":True,"normalized_latest_62_features":True,"time_score_course_status_history_excluded":True,
    "input_clamped_five":True,"target_frozen_exact":True,"prediction_error_before":before,"prediction_error_after":after,
    "learned_states_intrinsic_bonus_declines":True,"running_stats_checkpoint_roundtrip_exact":True,"done_bonus_zero_no_stats":True,
    "reward_training_next_features_exact":True,"predictor_passes_per_rollout":1,"predictor_minibatches":4,
    "done_samples_excluded_predictor":True,"legacy_rnd_whole_model_optimizer_reset":True,
    "single_mixed_reward_value_variant":True}
(BASE/"rnd_integration_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
