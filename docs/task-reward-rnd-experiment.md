# Task reward and RND experiment

This experiment follows Git baseline `062a79f`. Its validated fastest policy is
`checkpoints/ppo_classic100m_phase_fast.pt`: original native classic100m success
at 44.933334 seconds, 1348 actions. The target remains 43 seconds or less.

## Why revisit posture shaping

The existing classic reward targets 5 degrees of forward lean, penalizes torso
height variation, and rewards alternating foot contact cadence. These are
hypotheses about useful movement, rather than the task objective itself.
Torso height is only a proxy for avoiding sustained low crawling: the backend
does not expose knee contact flags or masses for a true centre of mass.

Replay auditing with the actual C++ reward implementation found:

| V39 reward component | 44.933s policy | 47.833s posture policy |
| --- | ---: | ---: |
| Exact 5-degree progress reward minus angle penalty | 6.36 | 19.08 |
| Vertical motion penalty | -8.23 | -6.16 |
| Cadence reward | 9.08 | 12.16 |

These three components favour the slower policy by 17.86 points, but the full
V39 return still favours the faster policy by 247.72 points. This is evidence
of competing shaping preferences, not proof that any individual term causes
the training plateau. Six replayed policies remained correctly ordered by
full V39 return. The older cadence coefficient 2.4 did reverse the order of
one 46.07s/46.90s pair; V39 already reduced that coefficient to 0.3.

## Independent reward profile

`classic_100m_task` uses the original `classic_100m` game, course, observations,
actions, physics, clock and native success condition. The existing
`classic_100m` reward profile remains selectable.

For each transition, task reward is:

* +2 per newly reached metre, capped at 100m, while upright;
* -1 per elapsed game second;
* -5 per second of sustained low torso posture beyond a 0.2s grace period;
* -200 for failure, timeout, or a finish with excessive cumulative low posture;
* +2800 - 20 times total game seconds for a qualified native successful finish.

Upright uses the existing 80% of standing torso-height threshold. Finishing
qualification retains the existing maximum 10% penalized low-posture fraction.
Standing up cannot erase previous low time or refund a penalty. Backtracking
cannot earn previously visited distance again. No angle, cadence, vertical
oscillation or permanently grounded-foot preference is added. Native failure
and termination still come from the original game.

For an entirely upright successful trajectory from the actual initial score
of 0.3m, undiscounted return is approximately `2999.4 - 21*T`. Thus a two-second
improvement yields 42 more points. Pure integrated forward speed alone would
largely reward distance, giving similar totals to every completed 100m run.

## RND correction

The pre-existing, disabled RND option used raw stacked observations, divided
each step's errors by their current batch mean, rewarded autoreset states,
and trained on current observations after rewarding next observations.

The corrected version in `scripts/train_ppo.py`:

* Uses the latest 60 body values plus two foot contact flags; fixed scaling
  and clipping to [-5, 5]; excludes score, time, course and terminal flags.
* Keeps the random target frozen and trains a predictor on exactly the
  next-state features that generated the intrinsic reward.
* Excludes done transitions from novelty reward, predictor training and
  normalization statistics because their observations are autoreset states.
* Uses checkpointed cumulative error second moment/count, with bonus capped
  at 5, instead of forcing the current batch mean to 1.
* Trains the predictor for one pass over each rollout, independent of PPO's
  number of epochs. Initialization and training add no policy RNG draws.
* Restores compatible RND state and optimizer together, or rebuilds the whole
  RND module for incompatible legacy state.

This is a single-value, episodic mixed-reward PPO variant, not a reproduction
of the original paper's separate extrinsic/intrinsic value streams. Novelty
is an aid to exploration; it is not an evaluation metric for running speed.
Reference: https://arxiv.org/abs/1810.12894

## Matched comparison

Both arms start from the identical original 44.933s actor; only the value head
is recalibrated to the new task reward (RMSE 2.08 on that fixed trajectory).
The original actor's 1348 greedy actions and strict reload were checked.

* V40: task reward, RND beta 0.
* V41: same task reward, RND beta 0.02 linearly decaying to 0 over 256 updates.
* Each: seed 449, 128 environments, 512-step rollouts, 256 updates
  (16,777,216 transitions), four PPO epochs/eight minibatches.
* Actor/Critic rates 3e-6/3e-5; independent Critic features and gradient
  clipping; gamma .9999, GAE lambda .995, target KL .006.
* Same entropy coefficient .003 to .001; checkpoints every eight updates.

This is one seed and one setting. Removing the shaping terms as a group
cannot identify the causal contribution of a single removed term. Selecting
the best saved checkpoint is not the same as showing repeatable improvement
across training seeds or varied initial conditions.
The retained torso-height proxy is another untested assumption: it cannot
distinguish genuine knee support from every form of deep leg flexion. This
experiment does not establish whether its 80% threshold is optimal.

## Validation and results

Production changes are limited to the existing reward and training modules:

| File | Change and training effect |
| --- | --- |
| `src/reward_profiles.cpp` | Adds the optional task-only calculation, leaving the prior classic profile available. |
| `src/course_rules.h` | Appends a reward-profile identifier without changing existing identifiers. |
| `src/course_rules.cpp` | Maps that reward identifier to the original classic game mode. |
| `src/python_module.cpp` | Exposes `classic_100m_task` through the existing configuration interface. |
| `scripts/train_ppo.py` | Repairs the optional RND implementation and accepts the new reward name; beta-zero behavior is preserved. |

The two new experiment configuration files isolate matched runs. The new
verification scripts and this report record replay calibration, reward
boundaries, RND regression checks and timings; they do not replace the
project's PPO, environment, rollout buffer or model export architecture.

Standalone C++ tests cover posture invariance, time preference, failures,
timeouts, low-posture grace/history, backtracking, reset and native endpoint.
The real environment replay and four-update smoke tests passed for both arms.
RND tests confirmed beta-zero actual PPO model/Adam/RNG bitwise equivalence,
fixed target, decreasing error on learned states, exact checkpoint statistics,
autoreset exclusion, one predictor pass and whole-module legacy reset.

Both runs completed with 16,777,216 new transitions each. All 32 saved
checkpoints per arm were evaluated from the same native initial state using
greedy actions. The final files were verified to contain exactly the same
policy tensors and inference configuration as their final saved snapshots.

| Policy | Fastest saved greedy policy | Best checkpoint update | Training time |
| --- | ---: | ---: | ---: |
| Retained baseline | 44.933334s | existing committed policy | prior run |
| V40 task reward without RND | 46.800000s | 10672 | 470.65s |
| V41 task reward with RND | 47.233334s | 10688 | 586.43s |

There were 31/32 successful deterministic checkpoint evaluations for V40 and
29/32 for V41. These counts compare different saved policies; they are not
repeated-trial success rates. Both selected policies had zero low-posture
fraction in their greedy trajectory. Neither beat the baseline or reached
43 seconds. The original fastest weights and replay are retained.

The three selected policies were separately evaluated with 64 stochastic
action episodes on CPU, seed 20261008:

| Policy | Native successful episodes | Mean successful time | Median successful time | Successful under 50s |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 58/64 | 48.984s | 48.750s | 43/64 |
| V40 best greedy candidate | 57/64 | 49.260s | 49.133s | 41/64 |
| V41 best greedy candidate | 61/64 | 48.836s | 48.800s | 55/64 |

All successful episodes passed the low-posture fraction check. No sampled
episode completed under 45s. Initial conditions were identical; only actions
were random. The V41 candidate had more successful and under-50s episodes in
this sample, but did not improve fastest greedy speed. Its successful mean
improved only 0.15s versus the baseline while its median was 0.05s slower.
One training seed and 64 evaluation episodes do not establish a general RND
benefit or a reliable success-rate difference. Conditional mean times exclude
failures; success counts are therefore reported alongside them. Returns from
the two different reward definitions are not compared directly.

Both training logs and final model tensors were finite. Value loss remained
large on some rollouts (about 4e4 at its logged peak), so passing NaN checks
does not imply an accurate value function or successful learning. RND loss
declined from about .0422 to .0002 and beta reached zero as configured.
The running processes retained fixed reward and PPO settings throughout each
arm; the RND initializer preserves the policy's random-number stream.

Total comparison training time: 1057.08 seconds (17m37s). Reports are in the
parent `verification` directory: `target43_v40_v41_comparison.json`,
`target43_stochastic_cpu_same_seed_comparison.json`,
`task_rnd_final_training_qa.json`, and `task_rnd_comparison_timing.json`.

The next investigation should test assumptions individually and measure
whether the torso-height proxy rejects useful leg flexion; it should also
inspect how well value estimates distinguish faster full episodes. Adding
more novelty reward alone is not supported as a remedy by this experiment.
