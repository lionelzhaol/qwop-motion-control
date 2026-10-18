"""Freeze selection before heldout evaluation and record the actual experiment."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import re
import subprocess
import torch
import numpy as np
from experiment import BASE, ROOT, SOURCE, STAGES, prior, read, emit
from replay_io import write_replay, read_replay, ReplayRuleset

def diagnostics():
    with np.load(BASE.parent/'shadow_audit/expert_dataset.npz') as expert:
        obs=torch.as_tensor(expert['observations'].copy())
        actions=torch.as_tensor(expert['actions'].copy())
    rows={}
    with torch.no_grad():
        for label,path in [('baseline',SOURCE),('selected',ROOT/'checkpoints/ppo_classic_actor_unfreeze_best_20261015.pt')]:
            _,model=prior.load(path);logits=model(obs)[0]
            logp=logits.log_softmax(-1).gather(1,actions[:,None]).squeeze()
            rows[label]=dict(checkpoint=str(path),checkpoint_sha256=prior.sha(path),expert_frames=len(obs),
                probability_of_canonical_sequence_along_expert_states=float(logp.sum().exp()),
                expert_argmax_matches=int((logits.argmax(-1)==actions).sum()))
    prior.write_json(BASE/'policy_concentration.json',rows);emit(rows)

def finalize():
    selections=[read(BASE/f'{scope}_selection.json') for scope in STAGES]
    chosen=Path(selections[-1]['chosen']['checkpoint'])
    data,model=prior.load(chosen)
    initial=read(BASE/'initialization.json')
    assert prior.sha(SOURCE)==initial['source_sha256']
    assert prior.sha(ROOT/'bin/running_physics.dll')==initial['physics_sha256']
    assert prior.sha(ROOT/'checkpoints/bc_classic100m_shadow.pt')==initial['bc_sha256']
    target=ROOT/'checkpoints/ppo_classic_actor_unfreeze_best_20261015.pt'
    data['config']=dict(data['config'],checkpoint=str(target))
    data['actor_unfreeze_experiment']=dict(source=str(chosen),source_sha256=prior.sha(chosen),
        original_baseline=str(SOURCE),original_baseline_sha256=prior.sha(SOURCE),
        retained_previous_baseline=chosen.resolve()==SOURCE.resolve(),
        stages=selections,selection_seed=2026101551,heldout_not_used_for_selection=True,
        expert_credit='External XYF public 991-frame demonstration, then local BC and PPO',
        reward_and_physics_unchanged=True)
    torch.save(data,target)
    before=prior.native(chosen);after=prior.native(target)
    assert before['actions']==after['actions']
    replay=ROOT/'replays/classic_actor_unfreeze_greedy_20261015.bin'
    write_replay(replay,after['actions'],ReplayRuleset.CLASSIC_100M)
    rules,actions=read_replay(replay)
    assert rules==ReplayRuleset.CLASSIC_100M and actions==after['actions']
    reports={}
    for label,checkpoint in [('baseline',SOURCE),('selected',target)]:
        records=[]
        for seed in (2026101561,2026101562,2026101563):
            path=BASE/f'{label}_heldout64_seed{seed}.json'
            prior.sample(checkpoint,64,seed,path)
            r=read(path);records.extend(dict(x,seed=seed) for x in r['results'])
        wins=[x for x in records if x['success']]
        reports[label]=dict(checkpoint=str(checkpoint),checkpoint_sha256=prior.sha(checkpoint),
            episodes=len(records),successes=len(wins),success_rate=len(wins)/len(records),
            best_time_s=min((x['time_s'] for x in wins),default=None),
            mean_success_time_s=float(np.mean([x['time_s'] for x in wins])) if wins else None,
            faster_than_reference=sum(x['frames']<991 for x in wins),results=records)
    prior.write_json(BASE/'heldout_comparison.json',dict(results=reports,
        protocol='Three fixed seeds, 64 first episodes per seed, original identical initial state; action sampling varies',
        selection_fixed_before_evaluation=True))
    prior.write_json(BASE/'final_artifacts.json',dict(model=str(target),model_sha256=prior.sha(target),
        selected_source=str(chosen),retained_previous_baseline=chosen.resolve()==SOURCE.resolve(),
        retained_checkpoint_global_step=int(data['global_step']),retained_checkpoint_update=int(data['update']),
        greedy=after,replay=str(replay),replay_sha256=prior.sha(replay),strict_reload_actions_identical=True,
        old_artifacts_and_physics_unchanged=True))
    emit({k:{a:b for a,b in v.items() if a!='results'} for k,v in reports.items()})

def report():
    artifacts=read(BASE/'final_artifacts.json');heldout=read(BASE/'heldout_comparison.json')['results']
    artifacts['greedy_replay_identical_to_original_bc']=prior.sha(artifacts['replay'])==prior.sha(ROOT/'replays/classic_shadow_bc_33p03.bin')
    prior.write_json(BASE/'final_artifacts.json',artifacts)
    times=[read(p) for p in BASE.glob('ppo_*_time.json')]
    selections=[read(BASE/f'{scope}_selection.json') for scope in STAGES]
    searches=[read(p) for p in BASE.glob('*_search.json')]
    best_search=min(searches,key=lambda r:r['frames']) if searches else None
    best_frames=min(artifacts['greedy']['frames'],best_search['frames'] if best_search else 999999)
    summary=dict(best_frames=best_frames,best_time_s=best_frames/30.,broke_reference=best_frames<991,
        ppo_training_seconds=sum(x['elapsed_seconds'] for x in times),
        new_ppo_transitions=sum(x['new_transitions'] for x in times),
        sampled_search_episodes=sum(x['episodes'] for x in searches),
        elapsed_seconds=(datetime.now(timezone.utc).timestamp()-(BASE/'train_ppo_before.py').stat().st_ctime),
        model=artifacts['model'],greedy_replay=artifacts['replay'],best_search=best_search,
        retained_checkpoint_global_step=artifacts['retained_checkpoint_global_step'],
        heldout={k:{a:b for a,b in v.items() if a!='results'} for k,v in heldout.items()},
        stages=selections,retained_previous_baseline=artifacts['retained_previous_baseline'])
    prior.write_json(BASE/'summary.json',summary)
    colors=['#2468A0','#D17B24','#328C63'];runs=[]
    for color,scope in zip(colors,STAGES):
        stem=f'ppo_classic_unfreeze_{scope}_20261015';text=(BASE/f'{stem}.log').read_text(encoding='utf-8')
        origin=int(re.search(r'global_step=(\d+)',text)[1]);rows=[]
        for line in text.splitlines():
            if not line.startswith('update='):continue
            r={}
            for k,v in re.findall(r'(\w+)=([^\s]+)',line):
                try:r[k]=float(v)
                except ValueError:pass
            r['new_steps']=r['steps']-origin;rows.append(r)
        runs.append(dict(name=scope,color=color,rows=rows))
    panels=[]
    for key,title,label in [('episode_return_mean','原始回合回报','回合奖励'),('entropy','有效动作熵','熵 / (10⁻³ nat)'),
                            ('value_loss','独立 Critic 损失','价值损失'),('kl','PPO 近似 KL','KL / 10⁻⁵')]:
        scale=1000. if key=='entropy' else (100000. if key=='kl' else 1.)
        panels.append(dict(title=title,ylabel=label,series=[dict(name=r['name'],color=r['color'],
            points=[[x['new_steps']/1e6,x[key]*scale] for x in r['rows']]) for r in runs]))
    plan=[dict(file='training_curves.png',title='百米 Actor 逐步解冻：真实训练日志',
        subtitle='head → last_hidden → full；每阶段从此前选择集较优模型开始，学习率逐级降低',
        xlabel='本阶段新增交互步数 / 百万',panels=panels,
        footer='奖励、物理、网络层数和采样温度保持不变；回报变化不等于百米用时改善。熵和KL按纵轴单位展示，原始值见日志。')]
    prior.write_json(BASE/'chart_plan.json',plan);prior.write_json(BASE/'chart_sources.json',runs)
    subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
                    str(BASE.parent/'jump_improvement_20261013/render_jump_charts.ps1'),
                    '-PlanPath',str(BASE/'chart_plan.json'),'-OutputDirectory',str(BASE)],check=True)
    b=heldout['baseline'];s=heldout['selected'];rows=[]
    for entry in selections:
        scope=entry['scope'];c=entry['chosen'];t=read(BASE/f'ppo_classic_unfreeze_{scope}_20261015_time.json')
        rows.append(f"| {scope} | {STAGES[scope]:g} | {t['new_transitions']:,} | {t['elapsed_seconds']:.1f} | {c['frames']} | {c.get('successes',0)}/64 |")
    doc=f'''# 百米 Actor 逐步解冻实验（2026-10-15）

本轮最好原生成绩 **{best_frames/30.:.6f} 秒、{best_frames} 帧**，{'已突破' if best_frames<991 else '未突破'}外部 XYF 991 帧示范。最佳随机搜索与确定性模型表现分别记录，不能相互冒充。

## 修改与目的

生产代码仅修改 `scripts/train_ppo.py`：增加 `actor_unfreeze=none/head/last_hidden/full` 配置、CLI 和明确的分层冻结覆盖逻辑。`none` 保留旧冻结行为，模型结构和推理不变；有独立 Critic 才允许解冻，目前限定百米任务。解冻发生在读取旧 checkpoint 及旧局部残差冻结逻辑之后，避免旧残差重新锁住 Actor。

原 PPO、GAE、采样缓冲区、专家辅助损失和日志继续沿用。奖励函数、原生物理、计时、网络宽度/深度、logits 温度及示范数据均未改变。各阶段采用更小的 Actor 学习率，Critic 仍独立更新。此为渐进可行性试验，学习率也随阶段变化，不能把阶段差异全归因于解冻层数。

`head` 开放全程输出层，仍保留已有局部残差，共 8,224 个策略参数；`last_hidden` 再开放最后一层特征，共 74,016 个；`full` 再开放输入特征与角速度分支，共 634,400 个。各阶段独立训练64次更新，每次128环境×1024步。

新配置位于 `configs/ppo_classic_unfreeze_*_20261015.json`；验证目录的编排脚本用于分阶段续训、冻结检查、模型选择和报告，实际训练仍调用原训练入口。

## 验证与选择

已检查默认模式推理输出与修改前逐值一致；梯度隔离、冻结层不变和开放层实际更新均经过检查。每阶段先跑4环境×1024步×2更新的小测试，再严格加载 checkpoint 和完整原生运行。有限观测、奖励、损失及模型参数均检查通过。

整网解冻初始学习率3e-6的小测试在370帧失败，KL限制被触发；这属于训练行为退化，数值未溢出。保留其日志、配置及权重后，从安全的第二阶段模型重新开始，把学习率降为3e-7，小测试恢复991帧完赛，才启动正式整网训练。未通过删除核心功能规避失败。

阶段选择种子固定为2026101551，每个候选64个环境的首回合；优先原生确定性完赛、帧数、随机成功次数及成功平均用时。存在至少32/64成功候选时使用该门槛。若训练退化，允许保留进入该阶段前的模型，避免强制选用较差末次权重。此选择集样本较小，最终另做独立留出评测。

| 阶段 | Actor学习率 | 新增交互 | 正式训练秒 | 阶段所选帧数 | 选择集完赛 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

包含四次小测试（含整网学习率重试），新增 PPO 交互共 **{summary['new_ppo_transitions']:,}**，PPO进程共 **{summary['ppo_training_seconds']/60:.2f} 分钟**。从首次备份代码至报告约 **{summary['elapsed_seconds']/60:.2f} 分钟**。随机搜索 **{summary['sampled_search_episodes']:,}** 个完成回合，按原作者异步方法执行，其成功比例不作为独立评测成功率。实验各分支总交互数不能全部算到所选模型；其累计计数 global_step 为 **{artifacts['retained_checkpoint_global_step']:,}**。

## 独立留出评测

模型选择完成后固定权重，种子2026101561/62/63各64个首回合。初始状态固定，仅动作采样随机，因此不代表不同初态的鲁棒性。

| 指标 | 本轮开始前模型 | 本轮保留模型 |
| --- | ---: | ---: |
| 完赛 | {b['successes']}/192 | {s['successes']}/192 |
| 完赛率 | {b['success_rate']*100:.2f}% | {s['success_rate']*100:.2f}% |
| 成功平均时间 | {b['mean_success_time_s']:.6f}秒 | {s['mean_success_time_s']:.6f}秒 |
| 最好时间 | {b['best_time_s']:.6f}秒 | {s['best_time_s']:.6f}秒 |
| 快于991帧 | {b['faster_than_reference']} | {s['faster_than_reference']} |

本轮最终选择{'保留先前基线' if artifacts['retained_previous_baseline'] else '使用本轮训练候选'}。所选来源：`{artifacts['selected_source']}`。

## 保存与解释边界

- 保留模型：`{artifacts['model']}`，附原始示范来源、阶段选择与训练配置。
- 确定性回放：`{artifacts['replay']}`，严格重载动作一致，二进制CRC验证通过，模型已在完整原生规则下运行。
- 与原BC回放字节{'完全相同，991个动作一致，不能作为新纪录' if artifacts['greedy_replay_identical_to_original_bc'] else '不同；仍以完整原生计时判断成绩'}。
- 原百米模型、BC权重及原生物理DLL哈希与本轮开始前相同；未改动跳远模型，未自动提交Git。
- 单条示范、奖励目标和采样分布仍保留，因此若无突破，只能说明这组渐进解冻方案尚未找到更快策略，不能证明示范最优或解冻无效。
- 示范状态诊断显示，完整输出原991个规范动作的估计概率从89.40%升至94.62%；两模型在991个示范状态的最大概率动作都没有改变。这是沿示范状态的诊断值，真实完赛率见独立评测，二者不能混用。
- `*_selection.json` 包含逐模型结果、专家状态动作变化/KL及参数变化；`heldout_comparison.json` 是未用于模型选择的独立评测。

![实际训练曲线](../../verification/classic_unfreeze_20261015/training_curves.png)
'''
    path=ROOT/'docs/classic-actor-unfreeze-20261015.md';path.write_text(doc,encoding='utf-8')
    emit(dict(summary={k:v for k,v in summary.items() if k not in ('stages','best_search','heldout')},document=str(path)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['finalize','report','diagnostics']);a=p.parse_args()
    {'finalize':finalize,'report':report,'diagnostics':diagnostics}[a.mode]()
