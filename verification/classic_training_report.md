# 经典 100 米 PPO 续训结果

日期：2026-10-08（Asia/Shanghai）
本轮按固定训练步数完成，没有达到 100 米通关目标。

## 实际用时

- 训练运行：164.962 秒，约 2 分 45 秒，包括两次训练进程启动、采样、更新及保存。
- 规模试运行：6.978 秒；主训练：157.984 秒。
- 本次检查开始至训练与评估验证完成：725 秒，即 12 分 05 秒。
- 统计起点：2026-10-08 14:11:22；验证完成：2026-10-08 14:23:27（北京时间）。
- 总用时包含配置检查、评估脚本准备、训练、评估和检查；游戏 episode 中的 120 秒是模拟时间，与上述实际运行时间不同。

## 训练规模和配置

从 checkpoints/baseline_resume_20261008.pt 的第 3 次 update、1536 条转移继续。
新增 320 次 PPO update：规模试运行 8 次，主训练 312 次。
新增转移 10,485,760；累计转移 10,487,296；最终 absolute_update=323。

配置文件：QWOP_rl_training_public/configs/ppo_classic_continue_20261008.json。

| 参数 | 值 | 选择原因 |
|---|---|---|
| num_envs / num_threads | 128 / 8 | 批量采样提高吞吐，使用部分 CPU 线程处理物理环境 |
| rollout_steps | 256 | 每个环境覆盖约 8.53 秒游戏时间 |
| 总 batch / minibatch | 32768 / 4096 | 提高批次覆盖；8 个小批次兼顾 GPU 批量计算与更新次数 |
| learning_rate | 0.0001 | 扩大批量并继续已有模型时采用较温和的更新 |
| gamma / GAE lambda | 0.995 / 0.95 | 稍加强后续前进奖励的权重，保留原 GAE 实现 |
| epochs / clip | 4 / 0.2 | 保留 PPO 多轮裁剪更新机制 |
| entropy / value coefficient | 0.01 / 0.5 | 保留原探索与价值损失权重 |
| 网络 / frame_stack | [128,128] / 1 | 与已有模型完全兼容，保留 Actor/Critic 结构 |
| checkpoint 间隔 | 20 次 update | 每新增 655360 条转移留存一次权重 |
| 评估间隔 | 80 次主训练 update 及最终模型 | 使用冻结的周期权重进行独立评估 |
| RND | 关闭 | 本轮先观察原外在奖励的学习效果 |

8 次规模试运行的平均 update 用时约 0.547 秒，所有数值有限后再扩大训练。

## 独立评估

每个 checkpoint 使用同一随机种子 20261008，64 个并行环境各运行第一个完整 episode。结束较快的环境不会贡献额外 episode，避免把短 episode 反复计入平均值。该评估采用随机动作采样。

| 累计转移 | update | 平均距离（米） | 最远距离（米） | 通关数 |
|---:|---:|---:|---:|---:|
| 263680 | 11 | 0.87 | 2.10 | 0/64 |
| 2885120 | 91 | 26.69 | 32.90 | 0/64 |
| 5506560 | 171 | 30.79 | 34.00 | 0/64 |
| 8128000 | 251 | 32.37 | 37.10 | 0/64 |
| 10487296 | 323 | 35.59 | 39.20 | 0/64 |

最终随机策略平均回报 25.289，距离中位数 35.45 米。
通过作者 export_policy_replay.py 执行一次确定性策略：3600 步、模拟时间 120.000 秒、距离 0.600 米、success=0。
因此随机策略的前进表现有所提高，但确定性策略仍很弱；当前模型未达到可靠行走或 100 米通关标准。这些成绩是有限样本评估，不代表理论最优成绩。

## 验证与修改范围

- 校验全部 312 次主训练 update 的 TensorBoard 标量，以及最终权重，均无 NaN/Inf。
- 严格模型加载和真实环境推理通过。
- Adam 累计更新次数为 10264，续训保留了优化器状态。
- 原仓库受版本控制的源码与作者示例配置没有改动。
- 新增一份实验配置，使用作者的配置系统；新增 verification/evaluate_classic_policy.py，复用原环境、FrameStacker 和 ActorCritic。现有回放脚本不能直接输出固定完整 episode 的均值和通关率，因此统计脚本独立保存在仓库外，不承担训练算法逻辑。

## 输出

- 最终权重：QWOP_rl_training_public/checkpoints/ppo_classic_continue_20261008.pt
- 训练日志：QWOP_rl_training_public/runs/ppo_classic_continue_20261008
- 确定性回放：QWOP_rl_training_public/replays/classic_continue_deterministic_20261008.bin
- 实测耗时：verification/classic_calibration_timing.json、classic_continue_timing.json
- 结构化结果：verification/classic_training_result.json
- 逐 episode 最终评估：verification/classic_evaluation_final.json

继续使用同一配置训练时，请通过命令行把 --resume-from 指向本轮最终权重，并使用新的 --checkpoint 和 --run-name。配置文件保留本轮初始 baseline 路径，直接原样运行会从该 baseline 开始新的实验。

