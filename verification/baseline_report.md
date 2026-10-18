# 第二阶段：原始项目 baseline 验证记录

日期：2026-10-08（Asia/Shanghai）
项目：E:\Work\强化学习小项目\QWOP_rl_training_public
上游提交：a8b29747c03e297f115975f60b8ba11157a1634f
状态：原始项目构建、环境交互、CUDA PPO 训练、checkpoint 保存/续训/加载和完整推理均通过。

## 软件与数据位置

| 内容 | 版本 | 实际位置 |
|---|---|---|
| Python x64 | 3.10.11 | E:\Program Files\Python310\tools |
| 便携 Git | 2.56.0.2 | E:\Program Files\Git |
| 当前使用的 CMake | 4.4.4 | E:\Program Files\CMake |
| 保留的旧 CMake | 3.31.10 | E:\Program Files\CMake-3.31.10 |
| C++ Build Tools 2022 | 17.14.41 | E:\Program Files\Microsoft Visual Studio\2022\BuildTools |
| MSVC 工具集 | 14.44.35207 | BuildTools\VC\Tools\MSVC\14.44.35207 |
| 微软共享组件文件 | — | E:\Program Files\Microsoft Visual Studio\Shared |
| Windows 11 SDK | 10.0.26100 | C:\Program Files (x86)\Windows Kits\10 |
| 微软安装器 | — | C:\Program Files (x86)\Microsoft Visual Studio\Installer |
| 项目虚拟环境 | Python 3.10 | 项目目录\.venv |
| 模型、日志和回放 | — | 项目目录\checkpoints、runs、replays |

微软安装器保留了首次安装的共享组件注册路径。E:\Work\强化学习小项目\.tools\vs-shared 是指向 E:\Program Files\Microsoft Visual Studio\Shared 的目录链接，实际软件文件位于新目录。部分微软安装器和 SDK 组件仍使用系统规定的位置。
官方说明：https://learn.microsoft.com/en-us/visualstudio/install/change-installation-locations?view=vs-2022

训练依赖：PyTorch 2.11.0+cu128、NumPy 2.2.6、Gymnasium 1.4.0、TensorBoard 2.21.0。完整安装记录见 installed_requirements.txt。pip check 通过。
硬件：RTX 5070 Ti，15.92 GiB 可见显存，48 GiB 安装内存，24 个逻辑处理器。CUDA 前向与反向计算通过有限值检查。

## 根因与处理

1. 原 Git 缺少 HTTPS 组件：使用官方便携 Git，补齐真实 Git 历史及作者锁定的 pybind11 v3.0.4 子模块。
2. 缺少 Python 3.10、CMake、MSVC 与 Windows SDK：补齐对应工具，使用项目虚拟环境隔离训练依赖。
3. 编译器安装在中文路径时，自动检测输出中的路径出现编码损坏；CMake 3.31.10 在部分检测中异常退出。此前通过路径别名及明确的开发库路径完成构建。软件迁到 E:\Program Files 后，CMake 4.4.4 已能从干净缓存直接执行作者 scripts\build.py，无需源码修改或路径别名。
4. PyTorch 官方安装包单连接传输较慢：使用同一官方文件的分段下载，并核对官方 SHA-256。
   SHA-256：7c792fe95ad5edaf622cf9e4f5573f5aecf2bc0654c7e866eda6134088f95d72
5. 日志验证最初使用了错误的文本标签；按本地 PyTorch 实现改为 config/device/text_summary。仅调整临时验证代码，原训练代码没有改动。

修改范围：软件安装位置、虚拟环境的 pyvenv.cfg 和构建/训练产物。Git status --short 与 git diff --stat 均为空，原算法、网络、环境、奖励和示例配置未修改。本记录保存在仓库外，记录本机依赖与运行证据，不承担训练逻辑。

## 环境验证

每个模式使用 4 个环境、2 个工作线程，运行 256 个批量 step，即各 1024 条转移。

| 模式 | 观测 shape | 完成 episode 数 | 单步 reward 范围 | 自动 reset 最大观测误差 |
|---|---|---:|---|---:|
| classic_100m | (4, 68) | 8 | [-9.70, 0.10] | 0 |
| hurdle_110m | (4, 66) | 7 | [-9.70, 0.10] | 0 |
| long_jump_30m | (4, 68) | 9 | [-10.00, 0.10] | 0 |

动作 shape 为 (4,)，整数范围 0~15。观测、奖励、最终距离与时间均有限；done 与 has_final 一致。作者 check_environment.py 在重建前后均通过。

## PPO 短训练与 checkpoint

通过原 scripts\train_ppo.py 和 configs\ppo_example.json 启动，仅用命令行覆盖采样规模和输出路径：
- 4 个并行环境、2 个线程；rollout_steps=128。
- 每次 update 512 条转移；4 个 minibatch，每批 128 条；epochs=2。
- 学习率 0.0003、gamma=0.99、GAE lambda=0.95、clip=0.2。
- 原共享 Actor/Critic MLP [128,128]，frame_stack=1，RND 关闭。
- 首次运行 2 次 update，再恢复 checkpoint 运行 1 次；累计 1536 条转移。
- 每次 update 保存周期 checkpoint，同时保存最终 checkpoint。

| 累计 update | 累计转移 | policy loss | value loss | 已结束 episode 的平均回报 |
|---:|---:|---:|---:|---:|
| 1 | 512 | 0.0015 | 1.5375 | -9.650 |
| 2 | 1024 | 0.1265 | 3.1942 | -9.260 |
| 3 | 1536 | 0.1514 | 1.1602 | -10.000 |

loss 是作者日志记录的最后一个 minibatch 值，并非整个 update 的平均值。
所有 TensorBoard 标量和模型参数均无 NaN/Inf；日志确认设备为 cuda。严格 load_state_dict 通过。优化器累计更新次数从 16 增至 24，恢复训练后模型参数实际发生变化。

主要权重：checkpoints\baseline_resume_20261008.pt
首次短训练权重：checkpoints\baseline_smoke_20261008.pt
日志：runs\baseline_smoke_20261008、runs\baseline_resume_20261008

通过原 export_policy_replay.py 加载主要权重，在重建后的环境中运行完整 episode：3600 步、120.000 秒、距离 0.600、success=0。回放为 replays\baseline_full_episode_20261008.bin。短训练验证了完整链路，尚未得到具备通关能力的策略。

## 复现命令

在 PowerShell 中执行：

```powershell
Set-Location 'E:\Work\强化学习小项目\QWOP_rl_training_public'
$env:PATH = 'E:\Program Files\CMake\bin;' + $env:PATH
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe scripts\build.py
.\.venv\Scripts\python.exe scripts\check_environment.py
```

重新进行短测试时使用新的输出名称，保留本次权重：

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo.py --config configs\ppo_example.json --num-envs 4 --num-threads 2 --rollout-steps 128 --updates 2 --epochs 2 --minibatches 4 --save-interval 1 --checkpoint checkpoints\baseline_repeat.pt --run-name baseline_repeat
```

查看日志：

```powershell
.\.venv\Scripts\tensorboard.exe --logdir runs
```

加载本次模型执行完整推理：

```powershell
.\.venv\Scripts\python.exe scripts\export_policy_replay.py --checkpoint checkpoints\baseline_resume_20261008.pt --output replays\baseline_loaded.bin --max-steps 5000 --device cuda
```

下一阶段需要明确用户任务的环境、observation、action、reward 和成功条件，再逐模块迁移。当前没有修改原任务，也没有启动正式长训练。

## 验证证据

- gpu_validation.json：实际 GPU 前向/反向测试。
- environment_validation.json：三个模式的批量环境与 reset 检查。
- training_validation.json：权重、优化器进度、TensorBoard 数值及严格加载检查。
- build_after_software_relocation.log：新安装位置下的作者原始构建日志。
- baseline_train.log、baseline_resume.log：原训练入口与续训输出。
- baseline_full_episode.log：完整推理 episode 输出。
