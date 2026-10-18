"""Behavior cloning from an action replay using the existing ActorCritic.

Unlike distill_policy.py, this accepts a recorded expert, not a teacher model.
The saved policy chooses its own actions from observations during evaluation.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

from running_env import RunningGameEnv
from replay_io import read_replay, ReplayRuleset
from train_ppo import ActorCritic, FrameStacker, TrainingLogger

ROOT = Path(__file__).resolve().parents[1]

def canonical_action(action):
    return (1 if action & 1 else 2 if action & 2 else 0) | (4 if action & 4 else 8 if action & 8 else 0)

def collect_replay(path, frame_stack, game_mode='classic_100m'):
    ruleset, actions = read_replay(path)
    expected = {'classic_100m': ReplayRuleset.CLASSIC_100M, 'long_jump_30m': ReplayRuleset.LONG_JUMP_30M}
    if game_mode not in expected or ruleset != expected[game_mode]:
        raise ValueError(f'Replay ruleset does not match requested game mode: {game_mode}')
    env = RunningGameEnv(game_mode, game_mode)
    equivalent = RunningGameEnv(game_mode, game_mode)
    raw, _ = env.reset(); other, _ = equivalent.reset()
    assert np.array_equal(raw, other)
    stack = FrameStacker(1, 68, frame_stack); obs = stack.reset(raw[None])
    samples, labels, rewards = [], [], []
    for i, action in enumerate(actions):
        samples.append(obs[0].copy()); labels.append(canonical_action(action))
        raw, reward, done, truncated, info = env.step(action)
        other, other_reward, other_done, other_truncated, other_info = equivalent.step(labels[-1])
        # Verify equivalent key aliases on every expert state before relabeling.
        assert np.array_equal(raw, other) and reward == other_reward and info == other_info
        assert done == other_done and truncated == other_truncated
        assert bool(done or truncated) == (i == len(actions)-1)
        assert np.isfinite(raw).all() and np.isfinite(reward)
        rewards.append(float(reward))
        obs = stack.update(raw[None], np.array([done or truncated]))
    assert info['success'] and not truncated
    return np.asarray(samples, dtype=np.float32), np.asarray(labels, dtype=np.int64), info, rewards

def evaluate_policy(model, frame_stack, game_mode='classic_100m'):
    env = RunningGameEnv(game_mode, game_mode)
    raw, _ = env.reset(); stack = FrameStacker(1,68,frame_stack); obs=stack.reset(raw[None])
    device = next(model.parameters()).device
    actions=[]; low=0
    max_steps = 1350 if game_mode == 'long_jump_30m' else 3600
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(obs,dtype=torch.float32,device=device))
            if not torch.isfinite(logits).all(): raise RuntimeError('Non-finite evaluation logits')
            action=int(logits.argmax(-1)[0])
        raw, reward, done, truncated, info = env.step(action)
        assert np.isfinite(raw).all() and np.isfinite(reward)
        low += (429.8-raw[26]*40)/506.85 < .8
        actions.append(action)
        if done or truncated: break
        obs=stack.update(raw[None],np.array([False]))
    assert done or truncated
    landed = game_mode == 'long_jump_30m' and abs(info['distance'] - (30 + info['score'])) < 1e-4
    return dict(success=bool(info['success']), truncated=bool(truncated), game_mode=game_mode,
        time_s=float(info['time']), distance_m=float(info['distance']), score=float(info['score']),
        long_jump_m=float(info['score']) if landed else None,
        low_torso_fraction=float(low/len(actions)), actions=actions)

def configure_imitation(model, freeze_runup=False):
    """Keep the existing clone unchanged outside the checkpoint's jump gate."""
    model.enable_separate_critic()
    if freeze_runup:
        if model.jump_sampling is None or model.jump_actor_weight is None or model.jump_actor_bias is None:
            raise ValueError('Frozen runup requires an existing jump sampling gate and residual checkpoint')
        if float(model.jump_sampling[1]) > 50.0:
            raise ValueError('Frozen runup requires the residual gate to begin at or after 25m')
        model.requires_grad_(False)
        model.jump_actor_weight.requires_grad_(True)
        model.jump_actor_bias.requires_grad_(True)
        if model.jump_state_net is not None:
            model.jump_state_net.requires_grad_(True)
    else:
        model.critic.requires_grad_(False); model.critic_net.requires_grad_(False)
        if model.critic_motion is not None: model.critic_motion.requires_grad_(False)
        with torch.no_grad(): model.actor.weight.mul_(.25); model.actor.bias.mul_(.25)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=1000)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--eval-every',type=int,default=100)
    parser.add_argument('--seed',type=int,default=20261010)
    parser.add_argument('--game-mode',choices=('classic_100m','long_jump_30m'),default='classic_100m')
    parser.add_argument('--freeze-runup',action='store_true')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--log-dir',type=Path)
    args=parser.parse_args()
    if min(args.epochs,args.batch_size,args.eval_every)<=0 or args.learning_rate<=0: raise ValueError('Positive training settings required')
    if args.freeze_runup and args.game_mode != 'long_jump_30m': raise ValueError('Frozen runup is only supported for long_jump_30m')
    if args.output.resolve() == args.source.resolve(): raise ValueError('Output must preserve the source checkpoint')
    if args.device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA was requested but is unavailable')
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    started=time.perf_counter(); device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    source=torch.load(args.source,map_location=device,weights_only=False); cfg=source['config']
    if cfg.get('game_mode','classic_100m') != args.game_mode: raise ValueError('Source checkpoint observation layout must match the requested game mode')
    frame_stack=int(cfg['frame_stack'])
    x_np,y_np,teacher_info,rewards=collect_replay(args.replay,frame_stack,args.game_mode)
    args.dataset.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.dataset,observations=x_np,actions=y_np,rewards=np.asarray(rewards),frame_stack=frame_stack,game_mode=args.game_mode)
    model=ActorCritic(x_np.shape[1],16,tuple(cfg['hidden_sizes'])).to(device)
    model.load_state_dict(source['model'],strict=True); configure_imitation(model,args.freeze_runup)
    frozen_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()
        if args.freeze_runup and k not in ('jump_actor_weight','jump_actor_bias') and not k.startswith('jump_state_net.')}
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(trainable,lr=args.learning_rate,eps=1e-5)
    x=torch.as_tensor(x_np,device=device); y=torch.as_tensor(y_np,device=device)
    train_mask=x[:,-6]<=model.jump_sampling[1] if args.freeze_runup else torch.ones(len(x),device=device,dtype=torch.bool)
    train_ids=train_mask.nonzero().flatten()
    if not len(train_ids): raise ValueError('Replay has no observations inside the trainable jump phase')
    provenance=dict(reference_replay=str(args.replay),reference_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),
        source_checkpoint=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        expert_frames=len(y_np),expert_native_time_s=teacher_info['time'],expert_native_score=float(teacher_info['score']),
        game_mode=args.game_mode,freeze_runup=args.freeze_runup,device=str(device),training_frames=len(train_ids),
        trainable_parameters=[name for name,p in model.named_parameters() if p.requires_grad],
        labels='Physically equivalent canonical Q/W/O/P masks, verified on every expert frame',
        training='Observation-conditioned behavioral cloning; no replay controller at inference',
        critic_note='Copied source critic kept separate; recalibrate before PPO fine-tuning')
    logger_path=args.log_dir or ROOT/'runs'/((args.output.stem+'_imitation') if args.game_mode=='long_jump_30m' else 'replay_imitation_pilot')
    logger=TrainingLogger(logger_path)
    best=None; evaluations=[]
    try:
        for epoch in range(1,args.epochs+1):
            order=train_ids[torch.randperm(len(train_ids),device=device)]
            for batch in order.split(args.batch_size):
                logits,_=model(x[batch]); loss=F.cross_entropy(logits,y[batch])
                if not torch.isfinite(loss): raise RuntimeError('Non-finite imitation loss')
                optimizer.zero_grad(); loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable): raise RuntimeError('Non-finite imitation gradient')
                torch.nn.utils.clip_grad_norm_(trainable,1.)
                optimizer.step()
            if epoch%args.eval_every==0 or epoch==args.epochs:
                with torch.no_grad():
                    logits,_=model(x); accuracy=float((logits.argmax(-1)==y).float().mean()); ce=float(F.cross_entropy(logits[train_mask],y[train_mask]))
                    train_accuracy=float((logits[train_mask].argmax(-1)==y[train_mask]).float().mean())
                rollout=evaluate_policy(model,frame_stack,args.game_mode)
                entry=dict(epoch=epoch,expert_action_accuracy=accuracy,train_phase_action_accuracy=train_accuracy,cross_entropy=ce,**{k:v for k,v in rollout.items() if k!='actions'})
                evaluations.append(entry); print(json.dumps(entry),flush=True)
                logger.log(epoch,{'imitation/action_accuracy':accuracy,'imitation/loss':ce,'evaluation/time_s':rollout['time_s'],
                    'evaluation/success':int(rollout['success']),'evaluation/final_score':rollout['score']})
                if args.game_mode=='long_jump_30m':
                    rank=(rollout['success'],rollout['long_jump_m'] if rollout['long_jump_m'] is not None else -1.,accuracy)
                else:
                    rank=(rollout['success'] and rollout['low_torso_fraction']<=.1, -rollout['time_s'] if rollout['success'] else rollout['distance_m'],accuracy)
                if best is None or rank>best[0]:
                    best=(rank,entry)
                    checkpoint=deepcopy(source); checkpoint.pop('optimizer',None)
                    checkpoint['model']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    checkpoint['config']=dict(cfg,checkpoint=str(args.output),game_mode=args.game_mode,separate_critic_features=True,reset_optimizer_on_resume=True)
                    checkpoint['update']=0; checkpoint['global_step']=0
                    checkpoint['replay_imitation']=dict(provenance,selected_epoch=epoch,own_policy_evaluation=entry)
                    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save(checkpoint,args.output)
                if accuracy==1. and rollout['success'] and len(rollout['actions'])==len(y_np) and rollout['actions']==y_np.tolist(): break
    finally: logger.close()
    assert all(torch.equal(value,model.state_dict()[name].detach().cpu()) for name,value in frozen_state.items())
    selected_checkpoint=torch.load(args.output,map_location=device,weights_only=False)
    reloaded=ActorCritic(x_np.shape[1],16,tuple(cfg['hidden_sizes'])).to(device)
    reloaded.load_state_dict(selected_checkpoint['model'],strict=True)
    assert all(torch.equal(value,reloaded.state_dict()[name].detach().cpu()) for name,value in selected_checkpoint['model'].items())
    final_evaluation=evaluate_policy(reloaded,frame_stack,args.game_mode)
    report=dict(**provenance,output=str(args.output),dataset_shape=list(x_np.shape),selected=best[1],evaluations=evaluations,
        final_evaluation={k:v for k,v in final_evaluation.items() if k!='actions'},
        frozen_runup_state_unchanged=bool(args.freeze_runup),strict_checkpoint_reload=True,
        logger_path=str(logger_path),elapsed_seconds=time.perf_counter()-started,seed=args.seed)
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'selected':best[1],'elapsed_seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__': main()
