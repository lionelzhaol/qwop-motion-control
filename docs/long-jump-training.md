# Long-jump PPO baseline

Task switched to the author's `long_jump_30m` game and reward. The course has
a 30m approach, a 9m sand pit and the original 45s limit. Native successful
completion requires walking out of the sand pit. The leaderboard objective
is measured jump distance, not approach distance or completion time.

## Changes for this task

* `scripts/train_ppo.py`: allow existing fixed observation scaling and the
  existing angular-velocity branch on the long-jump 68-value body layout.
  Body indices 0--59 and time index 61 match classic100m. Fresh long-jump
  scaling uses body scales `[5,15,pi,10,10]` and task scales
  `[30,45,300,400,1,1,1,1]`. Classic preprocessing retains its old scales.
* `configs/ppo_long_jump_original_smoke.json`: small author-style training
  check from scratch, using the original long-jump reward.
* `configs/ppo_long_jump_v1.json`: independent long-jump training run and
  output paths. Existing running checkpoints remain separate artifacts.

The original long-jump C++ reward, game mode, native physics, action masks,
PPO, rollout buffer/GAE and replay format are used. RND is off for this
baseline. No jump-specific body-angle, cadence or anti-crawling reward was
introduced. Knee contact is not directly available through this backend.

## Initialization and settings

The trained running policy can reach the long-jump board but its unmodified
greedy transfer produced a zero-distance landing and timed out in the pit.
`verification/prepare_long_jump_transfer.py` creates a separate initialization:

* Copy the original running features, angular branch and checkpointed input
  scales. Preserving those scales preserves feature units for transfer;
  they override fresh-task scales when the checkpoint is loaded.
* Multiply actor weights/bias by 0.5 to broaden action sampling. Greedy
  actions stay unchanged; this is initialization, not a trained jump result.
* Use the existing separate-Critic feature branch and reset only its output
  head to zero: running value targets do not measure jump performance.
* Reset PPO/step counters and optimizer state for the new task.

Run settings: 64 environments/16 native threads, 512-step rollouts, 1024 PPO
updates (33,554,432 new transitions), four epochs/eight minibatches, hidden
layers 256x256 and 32-frame history. Actor/Critic rates are 1e-5/1e-4, gamma
.999, GAE lambda .995, clip .2 and target KL .01. Entropy coefficient decays
from .01 to .002 over 768 updates. Checkpoints are saved every 32 updates.

The history and features reuse learned approach control. The high discount
and lambda retain credit for later landing and exit rewards. A conservative
actor rate protects approach control while the fresh Critic adapts faster.
A 3e-5 actor rate caused a .0846 KL excursion in the initial small test;
1e-5 reduced the tested excursions to approximately .011.

## Validation

The original four-environment, 4096-transition smoke run passed. The transfer
four-environment, 8192-transition run exercised complete episodes, autoreset,
finite losses, native 68-value observations/16 actions and checkpoint saving.
Transfer initialization strictly reloads with identical parameters and its
original greedy zero-distance trajectory was independently replayed.

New verification helpers are separate because long-jump scoring differs
from classic-race timing:

* `evaluate_long_jump.py`: load the policy, execute the original long-jump
  environment and replay every action through a second native physics handle
  in mode 2. Verify time, score, total distance and torso position. Report
  approach progress, measured landing distance, exit status and native success.
* `evaluate_long_jump_batch.py`: compare saved policies by success and jump
  distance; cache the actual action sequence and telemetry.
* `evaluate_long_jump_stochastic.py`: collect only the first sampled-action
  episode from each environment, avoiding selection bias toward fast resets.

`charts/final_distance_mean` includes 30m approach plus measured landing
distance once landed. A training value around 30 is not a 30m jump. Native
landing distance can decrease after first contact in the observed native
trajectories; use the final official score, not the first-contact snapshot.

## Results

Training completed: 1024 updates and 33,554,432 transitions. All 32 saved
policies were checked in the original native long-jump environment; 26 of
these distinct policies completed successfully. This is not a repeated-trial
success rate. All logged training losses were finite.

Deterministic saved models:

| Update | Official jump | Native success |
| --- | ---: | --- |
| initialization | 0m | timeout |
| 64 | .662m | timeout |
| 128 | 1.225m | timeout |
| 192 | .606m | yes |
| 256 | 1.626m | yes |
| 384 | 2.176m | yes |
| 576 | **2.599m** | yes |
| 1024 | 2.025m | yes |

Update256 had 57/64 native successes in a CPU sampled-action test, seed
20261010. Mean successful jump was .608m and best sampled successful jump
1.767m. This is one initial condition with action randomness, not a test of
varied courses or initial poses.

The best successful deterministic candidate was update576, not the final
update. Its full checkpoint, including optimizer state and its actual update
counter, is saved as `checkpoints/ppo_long_jump_best.pt`. Strict reload produced
all 631 original greedy actions again and the same 2.598794m score. Its run
completed in 21.033333 seconds; this time is not the leaderboard objective.

Final replay files, exported with the author's existing tools:

* `replays/long_jump_ppo_deterministic.bin`: 2.598794m, 631 actions.
* `replays/long_jump_ppo_sampled_best.bin`: 2.802300m, 611 actions; best of 1024
  sampled-action episodes using the selected policy. The search had 992 native
  successes. This record is not the policy's deterministic performance.

Both binary files have long-jump ruleset 3, correct action counts and valid
CRC. Re-executing their complete action sequences yielded native success at
the last frame, no early termination and the same official jump score.

The selected model's separate CPU first-episode evaluation (64 environments,
seed20261010) had 64/64 native successes, mean jump .974706m, median 1.011921m
and best sampled score 2.564097m. These tests use the same native initial state
and vary sampled actions; they do not establish performance across different
initial poses or courses. This is an initial usable long-jump baseline, not
a claim of an optimal jump policy.

Training took 1217.35 seconds (20m17s), from 12:48:39 to 13:08:56 on
2026-10-10 in Asia/Shanghai. Full preparation and verification took about
32 minutes. The machine-readable report is in the parent directory at
`verification/long_jump_v1_final_report.json`; logs are in
`verification/long_jump_v1_training.log`, and TensorBoard data in
`runs/ppo_long_jump_v1`.
