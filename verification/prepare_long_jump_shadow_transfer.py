"""Align task-feature offsets, retaining the BC running features and controls."""
from pathlib import Path
import sys,copy,json,hashlib
import numpy as np
import torch
BASE=Path(__file__).resolve().parent; ROOT=BASE.parent/'QWOP_rl_training_public'
sys.path.insert(0,str(ROOT/'scripts'))
from train_ppo import ActorCritic,FrameStacker
from running_env import RunningGameEnv

torch.set_num_threads(1)
source=ROOT/'checkpoints/bc_classic100m_shadow.pt'
original=torch.load(source,map_location='cpu',weights_only=True); cfg=original['config']
model=ActorCritic(68*cfg['frame_stack'],16,tuple(cfg['hidden_sizes']))
model.load_state_dict(original['model'],strict=True)
teacher=copy.deepcopy(model)
classic=RunningGameEnv('classic_100m','classic_100m')
jump=RunningGameEnv('long_jump_30m','long_jump_30m')
c,_=classic.reset(); j,_=jump.reset()
delta=c-j
assert np.all(delta[np.r_[0:62,64:68]]==0)
assert np.allclose(delta[62:64],[200.,610.])
stacked_delta=torch.tensor(np.tile(delta,cfg['frame_stack']))
with torch.no_grad():
    # Affine change of input origin: W*(jump_input+delta)+b.
    # No observation, physics, score or course location is changed.
    model.net[0].bias.add_(model.net[0].weight@(stacked_delta/model.observation_scale))
    model.critic_net.load_state_dict(model.net.state_dict())
    if model.critic_motion is not None: model.critic_motion.load_state_dict(model.angular_velocity.state_dict())
    model.critic.weight.zero_();model.critic.bias.zero_()
cs=FrameStacker(1,68,cfg['frame_stack']);js=FrameStacker(1,68,cfg['frame_stack'])
co=cs.reset(c[None]);jo=js.reset(j[None]); verified=0
while j[60]<25.:
    with torch.no_grad():
        cl,_=teacher(torch.tensor(co));jl,_=model(torch.tensor(jo))
    assert torch.allclose(cl,jl,atol=3e-5,rtol=3e-5)
    ca=int(cl.argmax(-1)[0]);ja=int(jl.argmax(-1)[0]);assert ca==ja
    c,_,cd,ct,_=classic.step(ca);j,_,jd,jt,_=jump.step(ja)
    assert not(cd or ct or jd or jt)
    assert np.array_equal(c[:62],j[:62]) and np.array_equal(c[64:],j[64:])
    verified+=1;co=cs.update(c[None],np.array([False]));jo=js.update(j[None],np.array([False]))
with torch.no_grad():
    # Temperature .4 preserves greedy actions and reduces BC sampling errors.
    model.actor.weight.mul_(2.5);model.actor.bias.mul_(2.5)
output=ROOT/'checkpoints/ppo_long_jump_shadow_initial.pt'
out=copy.deepcopy(original);out.pop('optimizer',None)
out['model']=model.state_dict();out['update']=0;out['global_step']=0
out['config']=dict(cfg,game_mode='long_jump_30m',reward_profile='long_jump_30m',checkpoint=str(output),
    separate_critic_features=True,reset_optimizer_on_resume=True,rnd_beta=0.)
out['long_jump_shadow_transfer']=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    input_origin_delta_indices={'62':200.,'63':610.},actor_logit_multiplier=2.5,critic_head_reset=True,
    runup_verified_identical_frames=verified,runup_verified_distance_m=float(j[60]),
    note='Running body trajectory exactly matches teacher through 25m before jump-specific fine-tuning; native long-jump course unchanged')
torch.save(out,output)
reload=ActorCritic(model.net[0].in_features,16,tuple(cfg['hidden_sizes']))
reload.load_state_dict(torch.load(output,map_location='cpu',weights_only=True)['model'],strict=True)
assert all(torch.equal(v,reload.state_dict()[k]) for k,v in model.state_dict().items())
(BASE/'long_jump_shadow_transfer.json').write_text(json.dumps(out['long_jump_shadow_transfer'],indent=2),encoding='utf-8')
print(json.dumps(out['long_jump_shadow_transfer']),flush=True)
