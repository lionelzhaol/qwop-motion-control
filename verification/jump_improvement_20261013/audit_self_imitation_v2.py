"""CPU-only closed-loop and native-trajectory audit of the learned jump policy."""
from pathlib import Path
import json,sys,hashlib,inspect
import numpy as np
import torch

BASE=Path(__file__).resolve().parent; ROOT=BASE.parents[1]/'QWOP_rl_training_public'
sys.path.insert(0,str(ROOT/'scripts'))
from train_ppo import ActorCritic
from train_replay_imitation import evaluate_policy,canonical_action
from replay_io import read_replay
from audit_native_replays import audit

torch.set_num_threads(1)
new_path=ROOT/'checkpoints/ppo_long_jump_self_imitation_v2_20261013.pt'
old_path=ROOT/'checkpoints/ppo_long_jump_shadow_best.pt'
best_path=ROOT/'replays/long_jump_shadow_ppo_sampled_best_3p16m.bin'
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (new_path,old_path,best_path)}
def model_rollout(path):
    data=torch.load(path,map_location='cpu',weights_only=False);cfg=data['config']
    model=ActorCritic(68*cfg['frame_stack'],16,tuple(cfg['hidden_sizes']))
    model.load_state_dict(data['model'],strict=True);model.eval()
    calls=[]
    hook=model.register_forward_hook(lambda mod,inputs,outputs:calls.append(tuple(inputs[0].shape)))
    rollout=evaluate_policy(model,cfg['frame_stack'],'long_jump_30m')
    hook.remove()
    assert len(calls)==len(rollout['actions']) and all(s==(1,68*cfg['frame_stack']) for s in calls)
    return model,data,rollout,calls

model,new,new_eval,calls=model_rollout(new_path)
old_model,old,old_eval,_=model_rollout(old_path)
rules,best_actions=read_replay(best_path)
canonical=[canonical_action(a) for a in best_actions]
assert new_eval['actions']==canonical
assert new_eval['actions'][:334]==old_eval['actions'][:334]
new_audit=audit(new_path,new_eval['actions'])
old_audit=audit(old_path,old_eval['actions'])
best_audit=audit(best_path)
new_rows=json.loads((BASE/(new_path.stem+'_native_trace.json')).read_text(encoding='utf-8'))
old_rows=json.loads((BASE/(old_path.stem+'_native_trace.json')).read_text(encoding='utf-8'))
best_rows=json.loads((BASE/(best_path.stem+'_native_trace.json')).read_text(encoding='utf-8'))
body_fields=('body_x_m','body_y_m')
assert all(all(new_rows[i][f]==old_rows[i][f] for f in body_fields) for i in range(335))
assert all(all(new_rows[i][f]==best_rows[i][f] for f in body_fields) for i in range(530))
for name,value in old['model'].items():
    if name not in ('jump_actor_weight','jump_actor_bias') and not name.startswith(('critic.','critic_net.','critic_motion.')):
        assert torch.equal(value,new['model'][name]),name
dataset=np.load(BASE/'self_imitation_v2.npz')
samples=torch.as_tensor(dataset['observations'],dtype=torch.float32)
with torch.no_grad():
    repeated=model(samples[390:391].expand(2,-1))[0]
    assert torch.equal(repeated[0],repeated[1])
    reversed_logits=model(samples[390:395].flip(0))[0].flip(0)
    direct_logits=model(samples[390:395])[0]
    assert torch.allclose(direct_logits,reversed_logits,atol=1e-6)
def condensed(r):
    main=r['key_flights'][-1]
    return dict(official_jump_m=r['official_jump_m'],native_success=r['native_success'],
        time_s=r['time_s'],frames=r['frames'],first25_s=r['crossings']['25']['time_s'],
        first30_s=r['crossings']['30']['time_s'],native_jumped_s=r['first_jumped']['time_s'],
        actual_feet_off_step=main['start_step'],actual_feet_off_s=main['start']['time_s'],
        actual_feet_off_vx_mps=main['start']['vx_displacement_mps'],
        actual_feet_off_vy_up_mps=main['start']['vy_up_displacement_mps'],
        feet_no_contact_duration_s=main['duration_s'],
        peak_torso_height_ratio=main['peak_height_ratio'],
        torso_rise_after_feet_off_m=(main['peak_height_ratio']-main['start']['torso_height_ratio'])*506.85/400,
        airborne_torso_dx_m=main['dx_m'],first_landing_s=r['first_landing']['time_s'],
        first_landing_torso_x_m=r['first_landing']['torso_x_m'],
        first_landing_score_m=r['first_landing']['long_jump_m'],
        score_loss_after_first_landing_m=r['landing_score_decrease_m'],
        landing_to_exit_s=r['post_landing_seconds'],
        post_landing_low_height_fraction=r['post_landing_low_height_fraction'])
report=dict(model=str(new_path),cpu_only=True,model_sha256=hashes[str(new_path)],
    source_mode=new['config']['game_mode'],frame_stack=new['config']['frame_stack'],
    local_state_net_shape=[list(new['model'][k].shape) for k in ('jump_state_net.0.weight','jump_state_net.2.weight')],
    closed_loop_observation_forward_calls=len(calls),exact_original_replay_actions=False,
    exact_canonical_replay_actions=True,all529_frames_native_body_positions_match_demonstration=True,
    first25_334_actions_and_body_positions_match_old_policy=True,
    all_frozen_running_weights_equal_old_policy=True,
    duplicate_input_same_logits=True,input_reordering_only_reorders_logits=True,
    policy_inference='Every step obtains current stacked native observations and runs ActorCritic.forward; no replay file/controller/step-indexed action table is an inference input.',
    new_policy=condensed(new_audit),old_shadow_policy=condensed(old_audit),record_replay=condensed(best_audit),
    evaluation_scope='One deterministic original initial state; no varied-state robustness claim',
    further_room=dict(no_current_post_landing_score_loss=True,
        runup_already_fast=True,board_to_airborne_torso_m=new_audit['key_flights'][-1]['start']['torso_x_m']-30,
        landing_recovery_seconds=new_audit['post_landing_seconds'],
        previous_greedy_recovery_seconds=old_audit['post_landing_seconds'],
        longest_flight_was_learned_but_not_optimized_beyond_demo=True))
report['exact_original_replay_actions']=new_eval['actions']==best_actions
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
report['source_weights_and_replay_files_unchanged']=True
(BASE/'self_imitation_v2_native_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
