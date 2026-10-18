"""Select before heldout evaluation; report speed and trajectory diversity."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse
import copy
import json
import re
import subprocess
import torch
import numpy as np
from experiment import BASE,ROOT,SOURCE,ARMS,REFERENCE,WIDTH,prior,read,save,emit,sample
from replay_io import read_replay,write_replay,ReplayRuleset

def finalize():
    initial=read(BASE/'initialization.json')
    assert prior.sha(SOURCE)==initial['source_sha256']
    assert prior.sha(ROOT/'bin/running_physics.dll')==initial['physics_sha256']
    assert prior.sha(ROOT/'checkpoints/bc_classic100m_shadow.pt')==initial['bc_sha256']
    selections={arm:read(BASE/f'{arm}_selection.json') for arm in ARMS}
    pool={}
    for arm,s in selections.items():
        for row in [s['candidates'][0],s['chosen']]:
            # Native success alone does not certify the existing posture rule.
            native=read(BASE/(Path(row['checkpoint']).stem+'_native.json'))
            if native['success'] and native['task_reward']>0 and row.get('qualified_finishes',0)>=48:
                pool[row['checkpoint']]=dict(row,arm='baseline' if Path(row['checkpoint'])==SOURCE else arm)
    candidates=sorted(pool.values(),key=lambda r:(r['frames'],-r['qualified_finishes'],r['qualified_mean_time_s']))
    assert candidates
    chosen=candidates[0];checkpoint=Path(chosen['checkpoint']);data,_=prior.load(checkpoint)
    target=ROOT/'checkpoints/ppo_classic_speed_signal_candidate_20261015.pt'
    data['config']=dict(data['config'],checkpoint=str(target))
    data['speed_signal_experiment']=dict(original_source=str(SOURCE),source_sha256=initial['source_sha256'],
        selected_source=str(checkpoint),selected_source_sha256=prior.sha(checkpoint),selected_arm=chosen['arm'],
        selection_seed=2026101586,selection=chosen,all_candidates=candidates,heldout_not_used_for_selection=True,
        native_physics_and_completion_unchanged=True,expert_credit='External XYF 991-frame public demonstration, then BC and PPO')
    torch.save(data,target)
    native=prior.native(target);original=prior.native(checkpoint)
    assert native['actions']==original['actions'] and native['success'] and native['task_reward']>0
    replay=ROOT/'replays/classic_speed_signal_greedy_20261015.bin';write_replay(replay,native['actions'],ReplayRuleset.CLASSIC_100M)
    _,actions=read_replay(replay);assert actions==native['actions']
    artifacts=dict(model=str(target),model_sha256=prior.sha(target),selected_source=str(checkpoint),selected_arm=chosen['arm'],
        replay=str(replay),replay_sha256=prior.sha(replay),native=native,strict_reload_actions_identical=True,
        replay_equals_original_bc=prior.sha(replay)==prior.sha(ROOT/'replays/classic_shadow_bc_33p03.bin'),
        retained_checkpoint_global_step=int(data['global_step']),retained_checkpoint_update=int(data['update']),
        old_model_and_physics_unchanged=True,selection=chosen)
    save(BASE/'final_artifacts.json',artifacts)
    models={'baseline':SOURCE,**{arm:ROOT/'checkpoints'/f'ppo_classic_speed_{arm}_train_20261015.pt' for arm in ARMS},'selected':target}
    results={}
    for label,path in models.items():
        records=[]
        for seed in (2026101601,2026101602,2026101603):
            output=BASE/f'{label}_heldout64_seed{seed}.json';sample(path,64,seed,output)
            records.extend(dict(r,seed=seed) for r in read(output)['results'])
        qualified=[r for r in records if r['qualified']]
        results[label]=dict(checkpoint=str(path),checkpoint_sha256=prior.sha(path),episodes=len(records),
            native_successes=sum(r['success'] for r in records),qualified_finishes=len(qualified),qualified_rate=len(qualified)/len(records),
            best_frames=min((r['frames'] for r in qualified),default=None),
            mean_qualified_time_s=float(np.mean([r['time_s'] for r in qualified])) if qualified else None,
            qualified_under_991=sum(r['frames']<991 for r in qualified),
            unique_qualified_action_sequences=len({r['action_sha256'] for r in qualified}),
            unique_qualified_effective_sequences=len({r['effective_action_sha256'] for r in qualified}),
            qualified_reference_sequences=sum(r['exact_reference_actions'] for r in qualified),
            qualified_frame_distribution=dict(Counter(r['frames'] for r in qualified)),results=records)
    save(BASE/'heldout_comparison.json',dict(results=results,seeds=[2026101601,2026101602,2026101603],
        protocol='First64 episodes per seed, CPU, fixed identical initial state; only action sampling random. Final arm checkpoints compare equal64-update training budgets; selected checkpoint chosen earlier.',
        selection_fixed_before_heldout=True))
    emit({k:{a:b for a,b in v.items() if a!='results'} for k,v in results.items()})

def report():
    initial=read(BASE/'initialization.json');artifacts=read(BASE/'final_artifacts.json')
    comparisons=read(BASE/'heldout_comparison.json')['results']
    if comparisons['selected']['best_frames']>=comparisons['baseline']['best_frames']:
        artifacts['usage_recommendation']='Keep the original baseline for use; archive the selected experimental candidate without replacing the baseline. No heldout speed improvement.'
    else:
        artifacts['usage_recommendation']='Selected candidate has a faster heldout finish; keep original baseline and assess repeatability separately.'
    save(BASE/'final_artifacts.json',artifacts)
    selections={arm:read(BASE/f'{arm}_selection.json') for arm in ARMS}
    searches=[read(p) for p in BASE.glob('*_search4096.json')]
    times=[read(p) for p in BASE.glob('ppo_*_time.json')]
    all_samples=[read(p) for p in BASE.glob('*_selection64.json')]+[read(p) for p in BASE.glob('*_heldout64_seed*.json')]+searches
    sampled_best=[(r['best']['frames'],r.get('replay'),r['checkpoint']) for r in all_samples if r.get('best')]
    best=min(sampled_best,key=lambda x:x[0])
    greedy_best=min((r for s in selections.values() for r in s['candidates'] if r['native_success']
        and read(BASE/(Path(r['checkpoint']).stem+'_native.json'))['task_reward']>0),key=lambda r:r['frames'])
    best_frames=min(best[0],greedy_best['frames'])
    novelty={}
    for r in all_samples:
        for replay in r.get('novel_qualified_replays',[]):novelty[replay['effective_action_sha256']]=replay
    summary=dict(best_frames=best_frames,best_time_s=best_frames/30.,broke_reference=best_frames<991,
        best_sampled_replay=best[1],best_sampled_model=best[2],greedy_best=greedy_best,
        ppo_new_transitions=sum(r['new_transitions'] for r in times),
        ppo_training_seconds=sum(r['elapsed_seconds'] for r in times),
        elapsed_seconds=datetime.now(timezone.utc).timestamp()-(BASE/'train_ppo_before.py').stat().st_ctime,
        random_search_episodes=sum(r['episodes'] for r in searches),selected_arm=artifacts['selected_arm'],
        model=artifacts['model'],greedy_replay=artifacts['replay'],retained_checkpoint_global_step=artifacts['retained_checkpoint_global_step'],
        novel_qualified_replays=list(novelty.values()),
        heldout={k:{a:b for a,b in v.items() if a!='results'} for k,v in comparisons.items()})
    save(BASE/'summary.json',summary)
    colors=['#2468A0','#D17B24','#328C63'];runs=[]
    for color,arm in zip(colors,ARMS):
        stem=f'ppo_classic_speed_{arm}_train_20261015';text=(BASE/f'{stem}.log').read_text(encoding='utf-8')
        origin=int(re.search(r'global_step=(\d+)',text)[1]);rows=[]
        for line in text.splitlines():
            if not line.startswith('update='):continue
            row={}
            for k,v in re.findall(r'(\w+)=([^\s]+)',line):
                try:row[k]=float(v)
                except ValueError:pass
            row['new_steps']=row['steps']-origin;rows.append(row)
        runs.append(dict(name=arm,color=color,rows=rows))
    training_outcomes={}
    for run in runs:
        rows=run['rows'];count=sum(int(r.get('qualified_finishes',0)) for r in rows)
        training_outcomes[run['name']]=dict(qualified_finishes=count,
            faster_than_reference_count=sum(int(r.get('faster_finishes',0)) for r in rows),
            best_time_s=min((r['best_finish_s'] for r in rows if 'best_finish_s' in r),default=None),
            updates_with_nonconstant_qualified_times=sum(r.get('mean_finish_s',0)>r.get('best_finish_s',0)+1e-5 for r in rows),
            weighted_qualified_mean_time_s=sum(r.get('mean_finish_s',0)*r.get('qualified_finishes',0) for r in rows)/max(1,count))
    summary['training_time_outcomes']=training_outcomes
    save(BASE/'summary.json',summary)
    panels=[]
    for key,title,label,scale in [('episode_return_mean','原生回报：未计入新增时间奖励','原生回合奖励',1.),
        ('entropy','有效动作熵','熵 / (10⁻³ nat)',1000.),('value_loss','Critic 损失：三组均使用奖励×0.01','缩放奖励单位的价值损失',1.),
        ('mean_finish_s','合格完赛的平均用时：包含较慢成功','游戏秒',1.)]:
        panels.append(dict(title=title,ylabel=label,series=[dict(name=r['name'],color=r['color'],
            points=[[x['new_steps']/1e6,x[key]*scale] for x in r['rows'] if key in x]) for r in runs]))
    plan=[dict(file='training_curves.png',title='保留完赛要求，提高速度奖励：三组匹配试验',
        subtitle='相同 Actor 起点、探索、训练预算及 Critic 校准流程；control / speed1200 / speed3600',
        xlabel='本组正式 PPO 新增交互步数 / 百万',panels=panels,
        footer='最短时间、完整分布与成功动作种类见报告；额外奖励变大不能当作成绩突破。')]
    save(BASE/'chart_plan.json',plan);save(BASE/'chart_sources.json',runs)
    subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
        str(BASE.parent/'jump_improvement_20261013/render_jump_charts.ps1'),'-PlanPath',str(BASE/'chart_plan.json'),
        '-OutputDirectory',str(BASE)],check=True)
    table=[]
    for label,r in comparisons.items():
        table.append(f"| {label} | {r['qualified_finishes']}/192 | {r['qualified_rate']*100:.2f}% | {r['best_frames']/30:.6f} | {r['mean_qualified_time_s']:.6f} | {r['unique_qualified_effective_sequences']} | {r['qualified_under_991']} |")
    training_rows=[]
    for arm in ARMS:
        r=read(BASE/f'ppo_classic_speed_{arm}_train_20261015_time.json')
        training_rows.append(f"| {arm} | {ARMS[arm]:.0f} | {r['new_transitions']:,} | {r['elapsed_seconds']:.2f} |")
    doc=f'''# 百米完赛速度信号增强实验（2026-10-15）

本轮最好 **{best_frames/30.:.6f} 秒、{best_frames} 帧**，{'已快于' if best_frames<991 else '尚未突破'}原 XYF 991 帧示范。所选模型来自 **{artifacts['selected_arm']}**；若来自对照组，其改善不能归因于新增速度奖励。

## 修改了什么

生产代码仅修改 `scripts/train_ppo.py`，增加默认关闭的完赛时间奖励、参数校验及原生完赛计时日志。它位于已有学习奖励处理处，与 reward_scale/RND 同层；没有改物理、课程、动作、环境终止或原生计时，不需要重编译环境。旧模型推理在991个示范状态上与修改前逐值一致。

训练奖励为 `0.01 × (原生奖励 + 合格完赛指示 × B / (1+exp((T-33.033333)/0.25)))`。实现用等价tanh表达式避免指数溢出。只有原生成功、未超时且原 `classic_100m_task` 的终止奖励为正才合格；该正负判据来自现有C++姿态约束。失败、姿态不合格、非终止步和超时均不加奖励。额外奖励范围为0至B，没有扣掉慢速完赛的原奖励。

| 奖励强度B | 示范时间的额外奖励 | 示范附近快一帧的总奖励增益（其他条件相同） |
| --- | ---: | ---: |
| 0 | 0 | 约0.70 |
| 1200 | 600 | 约40.64 |
| 3600 | 1800 | 约120.52 |

上表是原奖励单位的时间差值，不代表Actor梯度等倍增加；PPO仍然使用GAE和优势标准化。

该曲线针对33.033秒附近的小幅突破；它不会凭空生成更快轨迹。真实成绩始终由原生环境测量，不按训练回报换算。

## 公平对照与训练过程

三组相同的 Actor 初始权重来自 `ppo_classic_actor_unfreeze_best_20261015.pt`，整Actor开放，学习率3e-7，独立Critic学习率1e-4。每组128环境×1024步×64更新、4epochs、8minibatches、gamma=.9999、GAE=.995、PPOclip=.2、KL阈值.006；示范辅助系数.02→.002、熵系数.0005→.0001。保留原采样温度，RND关闭，未同时增加探索或改变网络结构。

各组超参数按原config系统存为独立JSON；新建的验证目录脚本只编排原训练入口、计算分布和保存回放，避免改动已留档的旧实验脚本，未另写一套PPO。

三组均用同一流程校准独立Critic：恢复有区分度的Actor特征到Critic副本，再对成功示范的折扣回报做正则线性拟合，内部奖励统一×0.01；根据各组奖励定义生成对应目标，Actor逐值不变。该初始化是成功条件回报拟合，不是随机策略真实价值的无偏估计。随后16次更新、64环境冻结Actor做在线价值适应，再正式训练。因采用了共同的Critic校准流程，与上一轮结果不能直接跨轮归因；本轮处理组和对照组可作匹配比较。

每组先4环境×1024步×2更新的小测试，检查奖励门控、有限数值、动作输出、权重变化、完整reset及保存加载。失败/不合格/超时/非终止的奖励为0均验证；关闭选项不改变推理；价值适应前后Actor逐值一致。原生回合回报继续单独记录，新增奖励记在 `finish_speed_bonus_sum`。

| 组别 | 奖励强度 | 正式新增交互 | 正式训练秒 |
| --- | ---: | ---: | ---: |
{chr(10).join(training_rows)}

包含所有小测试和价值适应，共新增 **{summary['ppo_new_transitions']:,}** 个PPO交互，训练进程 **{summary['ppo_training_seconds']/60:.2f}分钟**；从代码备份到本报告约 **{summary['elapsed_seconds']/60:.2f}分钟**。所选模型global_step为 **{artifacts['retained_checkpoint_global_step']:,}**，不能将所有实验分支的交互都计到该模型。

## 原生评测、动作变化和回放

阶段选择种子2026101586，每候选64个首回合；原生合格完赛、至少48/64合格为门槛，随后优先短用时及合格次数。所选模型先固定，再用2026101601/02/03三个独立种子各64个首回合。下面control/speed1200/speed3600均为各组相同64次更新后的末次模型；selected为事先选出的候选。初始状态固定，只有动作采样随机。

| 模型 | 合格完赛 | 合格率 | 最好秒 | 成功平均秒 | 有效动作序列种类 | 快于991帧 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

补充GPU随机搜索 **{summary['random_search_episodes']:,}** 个完整首回合，所有回合记录在 `*_search4096.json`。保存完整用时分布及每个回合动作哈希，合并Q/W、O/P优先级造成的等效按键，避免把别名算成不同有效动作序列；动作序列种类仍不等同于不同步态的生物力学解释。

正式训练日志合计快于参考时间的合格完赛计数为 **{sum(r['faster_than_reference_count'] for r in training_outcomes.values())}**，按组的最快时间和非恒定用时更新数保存在 `summary.json/training_time_outcomes`。训练日志中的候选时间与已保存回放的独立复核证据分开记录。

有效动作不同于原示范的合格回放单独保存并完整原生重放验证，本轮保存 **{len(novelty)}** 种。最快样本回放：`{best[1]}`，由模型 `{best[2]}` 生成；样本最好与确定性策略成绩分别记录。

保留模型：`{artifacts['model']}`。确定性回放：`{artifacts['replay']}`。{'该确定性回放与原BC的991个动作完全相同，不能作为新纪录。' if artifacts['replay_equals_original_bc'] else '该确定性回放与原BC动作不同，成绩由原生环境验证。'}所选组别为{artifacts['selected_arm']}，所有其他模型均保留；原BC、上一轮权重和物理DLL哈希未变，未自动提交Git。

使用建议：{'本轮没有独立留出提速证据，继续使用原模型，新增候选仅留作实验记录；没有根据留出结果回选其他训练权重。' if comparisons['selected']['best_frames']>=comparisons['baseline']['best_frames'] else '候选出现更快留出样本，仍需结合完赛率判断可复现性。'}不同奖励组的value loss大小不能直接解释为成绩优劣。

## 解释边界

本试验分别检验两个时间奖励强度，不能保证一定突破。若完整成功轨迹仍集中在示范附近，更大的奖励只会强化现有成功路径；后续应独立检验能产生有效新轨迹的探索方式，而不持续盲目放大奖励。若出现少量更快样本，还需验证能否稳定复现，不能把单次最好冒充稳定模型表现。

![真实训练日志](../../verification/classic_speed_signal_20261015/training_curves.png)

验证证据：本目录对应验证文件夹中的 initialization.json、各阶段日志、*_selection.json、heldout_comparison.json、*_search4096.json、final_artifacts.json、summary.json、chart_sources.json。
'''
    path=ROOT/'docs/classic-speed-signal-20261015.md';path.write_text(doc,encoding='utf-8')
    emit(dict(summary={k:v for k,v in summary.items() if k not in ('heldout','novel_qualified_replays','greedy_best')},document=str(path)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['finalize','report']);a=p.parse_args()
    finalize() if a.mode=='finalize' else report()
