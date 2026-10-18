# QWOP 运动控制与策略优化研究

本仓库用于课程小组协作，基于 [LYiHub/QWOP_rl_training_public](https://github.com/LYiHub/QWOP_rl_training_public) 开展 PPO 自主训练、奖励优化、专家行为克隆（BC）与跑步策略向跳远迁移。沿用原项目的 C++ 物理环境、并行采样、PPO、日志、模型保存和回放架构；本地增加任务奖励、模仿训练、独立 Critic、局部策略分支等。原作者说明及许可证保留在内层工程中。

## 已验证成果

| 任务与方法 | 成绩与说明 |
| --- | --- |
| 百米，纯 PPO | 45.400 秒 |
| 百米，PPO 后参数搜索 | 44.933 秒 |
| 百米，专家 BC | 33.033 秒：复现外部 XYF 示范，尚未突破示范时间 |
| 同一 BC 模型随机采样 | 已保留 33.167 秒回放 `classic100m_shadow_bc_random_33p17s.bin` |
| 跳远，最终策略 | 确定性 4.099 米；随机搜索最好 4.130 米 |

成绩均以原生环境计时、计分为准。随机最好成绩不代表平均表现；详细过程与对照见工程 `docs/` 和 `verification/`。

## 目录

| 目录 | 内容 |
| --- | --- |
| `QWOP_rl_training_public/` | 源码、配置、环境组件、模型、回放及实验文档 |
| `verification/` | 实验编排、专家样本、评测结果及验证资料 |
| `presentation/` | 课堂 PPT、图表、演讲备注及制作资料 |
| `videos/` | 录像尝试的资料；录制工作未完成，暂无可交付成片 |
| `downloads/` | 依赖与开发工具安装包 |
| `.cache/` | 安装、编译和运行缓存 |
| `.tmp/` | 安装及实验临时文件 |
| `.tools/` | 本地工具相关资料 |

## 运行

需要 **64 位 Windows、Python 3.10**；GPU 训练使用匹配的 PyTorch/CUDA。重新构建需 CMake、MSVC C++ 工具链及 Windows SDK。以下命令均在内层工程目录执行：

```powershell
cd QWOP_rl_training_public
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\build.py
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe scripts\export_policy_replay.py --checkpoint checkpoints\bc_classic100m_shadow.pt --output replays\team_bc.bin
```

导出的 `.bin` 是动作回放，可在 [游戏网站](https://run.lyihub.com/) 导入，不是视频。继续训练前，需将配置中指向 `E:\Work\强化学习小项目` 的绝对路径调整为本机路径，并检查 `resume_from`、数据与模型文件是否齐全。更多用法见[原工程说明](QWOP_rl_training_public/README.md)。

## 完整资料与历史

Git 文件区提供代码和精选成果；完整工作区（包含全部模型、虚拟环境、缓存、安装包等）以 `.tar.zst` 分卷保存在[同仓库 Releases](https://github.com/lionelzhaol/qwop-motion-control/releases/tag/workspace-snapshot-20261018)，未纳入常规 Git 文件区的内容并非丢弃。快照约 **26.40 GiB**，压缩后 **22.25 GB、21 卷**，已通过 32,603 个文件的 SHA256 校验；恢复方法见 [RESTORE.md](handoff_20261018/RESTORE.md)。

跨电脑运行建议重建虚拟环境。三个指向外部软件的目录联接仅记录清单，不复制其外部目标。原工程 Git 历史通过 `training-history` 分支及 [Git bundle](handoff_20261018/training-history.bundle) 保留。最新训练归档提交为 `4e23f8d`；`main` 分支提供包含验证、汇报资料在内的小组工作区结构。
