"""Native CPU audit: learned 4.10m closed-loop policy vs 3.16m and baseline."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch

BASE=Path(__file__).resolve().parent; ROOT=BASE.parents[1]/'QWOP_rl_training_public'
sys.path.insert(0,str(ROOT/'scripts'))
from train_ppo import ActorCritic
from train_replay_imitation import evaluate_policy,canonical_action
from replay_io import read_replay
from audit_native_replays import audit

torch.set_num_threads(1)
paths={name:ROOT/'checkpoints'/file for name,file in {
    'new_v3':'ppo_long_jump_self_imitation_v3_20261013.pt',
    'previous_v2':'ppo_long_jump_self_imitation_v2_20261013.pt',
    'baseline_shadow':'ppo_long_jump_shadow_best.pt'}.items()}
replay=ROOT/'replays/long_jump_self_imitation_v2_search_20261013.bin'
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths.values(),replay]}
audits={}; rows={}; rolls={}; models={}; data={}
for name,path in paths.items():
    data[name]=torch.load(path,map_location='cpu',weights_only=False);cfg=data[name]['config']
    model=ActorCritic(68*cfg['frame_stack'],16,tuple(cfg['hidden_sizes']))
    model.load_state_dict(data[name]['model'],strict=True);model.eval();models[name]=model
    calls=[];hook=model.register_forward_hook(lambda mod,inputs,out:calls.append(tuple(inputs[0].shape)))
    rolls[name]=evaluate_policy(model,cfg['frame_stack'],'long_jump_30m');hook.remove()
    assert len(calls)==len(rolls[name]['actions']) and all(shape==(1,2176) for shape in calls)
    audits[name]=audit(path,rolls[name]['actions'])
    rows[name]=json.loads((BASE/(path.stem+'_native_trace.json')).read_text(encoding='utf-8'))
rules,expert=read_replay(replay)
assert rolls['new_v3']['actions']==[canonical_action(a) for a in expert]
audits['sampled_expert']=audit(replay)
expert_rows=json.loads((BASE/(replay.stem+'_native_trace.json')).read_text(encoding='utf-8'))
assert all(rows['new_v3'][i]['body_x_m']==expert_rows[i]['body_x_m'] and rows['new_v3'][i]['body_y_m']==expert_rows[i]['body_y_m'] for i in range(len(expert_rows)))
assert rolls['new_v3']['actions'][:334]==rolls['baseline_shadow']['actions'][:334]
assert all(rows['new_v3'][i]['body_x_m']==rows['baseline_shadow'][i]['body_x_m'] and rows['new_v3'][i]['body_y_m']==rows['baseline_shadow'][i]['body_y_m'] for i in range(335))
for key,value in data['previous_v2']['model'].items():
    if key not in ('jump_actor_weight','jump_actor_bias') and not key.startswith('jump_state_net.'):
        assert torch.equal(value,data['new_v3']['model'][key]),key
dataset=np.load(BASE/'self_imitation_v3.npz');x=torch.as_tensor(dataset['observations'])
with torch.no_grad():
    assert torch.equal(models['new_v3'](x[391:392].expand(2,-1))[0][0],models['new_v3'](x[391:392].expand(2,-1))[0][1])
    reordered=models['new_v3'](x[390:398].flip(0))[0].flip(0)
    direct=models['new_v3'](x[390:398])[0]
    reorder_error=float((reordered-direct).abs().max())
    assert reorder_error<2e-5 and torch.equal(reordered.argmax(-1),direct.argmax(-1))
diff=[i for i,(a,b) in enumerate(zip(rolls['new_v3']['actions'],rolls['previous_v2']['actions'])) if a!=b]
def condensed(r):
    flight=r['key_flights'][-1]
    return dict(official_jump_m=r['official_jump_m'],success=r['native_success'],frames=r['frames'],
        game_time_s=r['time_s'],first25_s=r['crossings']['25']['time_s'],first30_s=r['crossings']['30']['time_s'],
        feet_off_step=flight['start_step'],feet_off_s=flight['start']['time_s'],
        feet_off_vx_mps=flight['start']['vx_displacement_mps'],feet_off_vy_up_mps=flight['start']['vy_up_displacement_mps'],
        feet_no_contact_s=flight['duration_s'],peak_torso_height_ratio=flight['peak_height_ratio'],
        torso_rise_from_feet_off_m=(flight['peak_height_ratio']-flight['start']['torso_height_ratio'])*506.85/400,
        torso_airborne_dx_m=flight['dx_m'],first_landing_s=r['first_landing']['time_s'],
        first_landing_torso_x_m=r['first_landing']['torso_x_m'],first_landing_score_m=r['first_landing']['long_jump_m'],
        landing_score_loss_m=r['landing_score_decrease_m'],landing_to_exit_s=r['post_landing_seconds'],
        post_landing_low_height_fraction=r['post_landing_low_height_fraction'])
comparison={name:condensed(r) for name,r in audits.items()}
new_landing=audits['new_v3']['first_landing']['step'];old_landing=audits['previous_v2']['first_landing']['step']
def pose(row):
    return dict(step=row['step'],game_time_s=row['time_s'],action=row['action'],torso_x_m=row['torso_x_m'],
        torso_height_ratio=row['torso_height_ratio'],left_foot=row['left_foot'],right_foot=row['right_foot'],
        left_foot_xy_m=[row['body_x_m'][3],row['body_y_m'][3]],
        right_foot_xy_m=[row['body_x_m'][11],row['body_y_m'][11]],long_jump_m=row['long_jump_m'])
report=dict(model=str(paths['new_v3']),model_sha256=hashes[str(paths['new_v3'])],cpu_only=True,
    exact_canonical_demonstration_actions=True,all_expert_frames_native_body_positions_equal=True,
    forward_observation_calls=len(rolls['new_v3']['actions']),strict_load=True,
    all_runup_and_critic_weights_unchanged_from_v2=True,first25_334_actions_and_positions_equal_old_baseline=True,
    closed_loop_model_inference=True,duplicate_inputs_equal_logits=True,permuted_batch_equal_actions=True,
    reordered_batch_max_fp32_logit_error=reorder_error,
    comparison=comparison,
    first_different_action_index=diff[0],first_different_native_step=diff[0]+1,
    first_difference=dict(v3=pose(rows['new_v3'][diff[0]+1]),v2=pose(rows['previous_v2'][diff[0]+1])),
    action_changes_until_new_landing=[dict(step=i+1,game_time_s=rows['new_v3'][i+1]['time_s'],v3_action=rolls['new_v3']['actions'][i],v2_action=rolls['previous_v2']['actions'][i]) for i in diff if i+1<=new_landing],
    landing_pose=dict(v3=pose(rows['new_v3'][new_landing]),v2=pose(rows['previous_v2'][old_landing])),
    pose_when_v2_lands=dict(v3=pose(rows['new_v3'][old_landing]),v2=pose(rows['previous_v2'][old_landing])),
    score_improvement_m=comparison['new_v3']['official_jump_m']-comparison['previous_v2']['official_jump_m'],
    score_improvement_pct=(comparison['new_v3']['official_jump_m']/comparison['previous_v2']['official_jump_m']-1)*100,
    evaluation_scope='One original initial state deterministic rollout, not varied-state robustness')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
report['source_checkpoints_and_replay_unchanged']=True
(BASE/'self_imitation_v3_native_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
