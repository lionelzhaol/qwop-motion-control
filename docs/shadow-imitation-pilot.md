# Public ghost replay imitation pilot

On 2026-10-10, the public running website's frontend `game.js` was inspected.
Its ghost loader requests `/api/rank/<ruleset>/top-replay` as an ArrayBuffer
and passes those replay bytes to the game. Thus the ghost exposes a recorded
Q/W/O/P action sequence rather than only a rendered video.

Sources:

* https://run.lyihub.com/
* https://imyrs.net/u/lyi/running-man/v0.4/js/game.js
* https://run.lyihub.com/api/rank/classic_100m
* https://run.lyihub.com/api/rank/classic_100m/top-replay

The retrieved classic100m top record was attributed to player **XYF**, metric
991 frames. It is an external expert demonstration, not our own PPO record.
Its snapshot, leaderboard metadata, public frontend and validation reports
are stored under the parent `verification/shadow_audit` directory.

## Native validation and data

The expert file has ruleset 1, valid CRC and 991 actions. Its SHA256 is
`a5b04f529e6c062e9899d5d8fd2740943a07f1f2a50ecec09c572a4d64f936f3`.
Playing it in the original bundled physics backend succeeded at precisely
the last action: 100m in 33.033333 game seconds. No physics, clock or game
success rules were changed. Torso height fell below 80% standing height on
about 1.1% of frames. This is a posture proxy, not a direct knee-contact test.

`scripts/train_replay_imitation.py` is the only new training entry point.
The existing `distill_policy.py` requires a teacher checkpoint and its logits;
it cannot directly train from an external action-only replay. The new entry
reuses `ActorCritic`, `FrameStacker`, `TrainingLogger`, existing checkpoint
loading and the original environment. It does not rewrite PPO or its buffer.

Two original engine instances execute expert masks and their canonical
equivalent masks. All 68 observations, rewards and episode flags are exactly
equal on all 991 expert frames before labels are canonicalized. This removes
redundant key combinations only after checking physical equivalence.

The dataset has 991 samples of 2176 values (32 x 68), each paired with the
action selected before that transition. It is stored as
`verification/shadow_audit/expert_dataset.npz`. Its saved reward sequence uses
the existing classic reward; it is not used in behavior-cloning loss and must
not be mistaken for task-only returns in future value calibration.

## Behavior-cloning result

Initialization uses the existing 44.933s running model, two 256-wide hidden
layers, checkpointed input normalization and angular-velocity branch. The
actor logits are initially multiplied by .25. An independent copied Critic
is frozen during the action-only training; its old value head is not claimed
to be calibrated for subsequent PPO.

Cross entropy trains the policy on canonical expert actions, Adam rate 1e-4,
batch size256, seed20261010, at most1000 epochs. The actual pilot took 12.47s
including dataset construction and periodic closed-loop evaluations.

At epoch900, expert-state action accuracy was 99.899%, yet the policy failed
after 52m. At epoch1000, accuracy was 100% and its own observation-conditioned
actions completed the original game in 33.033333s. This illustrates why
classification accuracy alone is insufficient: a wrong early action can
change subsequent inputs. No expert-action lookup is used during inference.

Files:

* `checkpoints/bc_classic100m_shadow.pt`: behavior-cloned model, with source
  attribution and training method in metadata; no PPO updates yet.
* `replays/classic_shadow_bc_33p03.bin`: exported using the author's existing
  model replay exporter. It contains the student's canonical actions.

The exported replay's CRC and original native terminal state were verified.
It has991 actions and reproduces the expert's physically equivalent path.
The checkpoint loads with `weights_only=True`; a NumPy scalar in an initial
evaluation metadata field was converted to an ordinary float to preserve
compatibility, without changing the model parameters or exporter.

## Scope and limitations

| Test | Native successes | Successful time |
| --- | ---: | ---: |
| Original PPO baseline, greedy | verified previously | 44.933s |
| BC student, fixed initial state / greedy | 16/16 | all33.033s |
| BC student, same initial state / sampled actions | 0/64 | no successful episode |

The student reproduces one expert trajectory under the game's fixed initial
conditions. It has not learned reliable recovery from off-demonstration
states. The16 greedy runs are repeatability checks, not a varied-condition
generalization test. Do not present this model as a robust PPO-trained policy
or the reference record as a newly discovered independent RL achievement.

## Reward conflict discovered using the expert

Actual native replay of both policies under both existing reward profiles:

| Policy | Native time | Current classic return | Task-only return |
| --- | ---: | ---: | ---: |
| Expert / matching BC trajectory | 33.033s | 1450.95 | 2300.67 |
| Old running baseline | 44.933s | 1877.42 | 2055.80 |

Under the old classic reward, the substantially slower policy earns more.
The expert's head-to-torso lean proxy averages -14.64 degrees and only18.97%
of frames qualify for the old speed-bonus pose gate (-10 to30 degrees with
upright torso), versus99.18% for the slower baseline. These measurements
demonstrate a conflict between the handcrafted posture-gated objective and
the native speed objective. They do not prove any specific pose is generally
optimal or that every reward component is harmful.

The existing `classic_100m_task` profile correctly ranks these two trajectories.
The canonical BC checkpoint selects that profile for future fine-tuning. The
change only selects a reward; it does not alter observations, physics or
the student's inference behavior. The copied Critic still needs calibration
before PPO fine-tuning.

Next work should preserve imitation constraints while adding task-directed
PPO and recovery/perturbation experience. A recorded action sequence is not
an expert policy that can answer arbitrary off-trajectory state queries;
simply labeling a deviated state with the same-time replay action is not a
valid implementation of DAgger. Related primary references:
https://proceedings.mlr.press/v15/ross11a.html and
https://arxiv.org/abs/1804.02717.

Machine-readable results: `reference_native_validation.json`,
`reward_ordering_with_expert.json`, `imitation_pilot.json`,
`student_deterministic16.json`, `student_stochastic64.json` and
`final_pilot_validation.json` in `verification/shadow_audit`.
