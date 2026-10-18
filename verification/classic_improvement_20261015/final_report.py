"""Select before heldout evaluation, preserve artifacts and report actual results."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,json,time
import numpy as np
import torch
from experiment import BASE,ROOT,sha,load,native,sample,write_json
from replay_io import read_replay,write_replay,ReplayRuleset
def read(name):return json.loads((BASE/name).read_text(encoding='utf-8'))
def prepare():
    selection=[]
    for version,stem in [(1,'local_v1'),(2,'explore_v2'),(3,'calibrated_v3')]:
        c=ROOT/'checkpoints'/f'ppo_classic_shadow_{stem}_20261015.pt'
        r=read(f'ppo_v{version}_selection128.json');assert r['checkpoint_sha256']==sha(c)
        greedy=native(c);times=[e['time_s'] for e in r['results'] if e['success']]
        selection.append(dict(version=version,checkpoint=str(c),successes=r['successes'],episodes=r['episodes'],
            mean_success_time_s=float(np.mean(times)),greedy_frames=greedy['frames'],greedy_success=greedy['success']))
    valid=[r for r in selection if r['greedy_success']]
    valid.sort(key=lambda r:(r['greedy_frames'],-r['successes'],r['mean_success_time_s']))
    chosen=valid[0];source=Path(chosen['checkpoint']);d,m=load(source)
    baseline=ROOT/'checkpoints/bc_classic100m_shadow.pt';old,_=load(baseline);initial=read('initialization.json')
    assert sha(baseline)==initial['source_sha256'] and sha(ROOT/'bin/running_physics.dll')==initial['physics_sha256']
    for name,value in old['model'].items():
        if name.startswith(('net.','angular_velocity.')) or name=='observation_scale':assert torch.equal(value,d['model'][name]),name
        if name.startswith('actor.'):assert torch.equal(value*5.,d['model'][name]),name
    target=ROOT/'checkpoints/ppo_classic_shadow_finetuned_best_20261015.pt'
    d['config']=dict(d['config'],checkpoint=str(target))
    d['finetuning_provenance']=dict(source=str(source),source_sha256=sha(source),selection=chosen,
        selection_seed=2026101511,selection_candidates=selection,
        original_expert='External XYF 991-frame public demonstration, then local BC and PPO',
        original_bc_checkpoint=str(baseline),original_bc_sha256=sha(baseline),actor_logit_multiplier=5.,
        training_method='Frozen BC backbone and base Actor; PPO of gated linear residual with demonstration CE anchor')
    torch.save(d,target)
    before=native(source);after=native(target);assert before['actions']==after['actions']
    assert all(torch.isfinite(v).all() for v in d['model'].values())
    replay=ROOT/'replays'/f'classic_shadow_finetuned_{after["time_s"]:.2f}s_20261015.bin'.replace('.','p',1)
    write_replay(replay,after['actions'],ReplayRuleset.CLASSIC_100M)
    rules,actions=read_replay(replay);assert actions==after['actions']
    from running_env import RunningGameEnv
    env=RunningGameEnv('classic_100m_task','classic_100m');env.reset()
    for i,a in enumerate(actions):
        x,reward,done,tr,info=env.step(a);assert bool(done or tr)==(i==len(actions)-1)
    assert info['success'] and not tr and info['time']==after['time_s']
    write_json(BASE/'final_model_native.json',after)
    report=dict(model=str(target),model_sha256=sha(target),source=str(source),selection=chosen,all_selection=selection,
        deterministic_frames=after['frames'],deterministic_time_s=after['time_s'],replay=str(replay),
        replay_sha256=sha(replay),strict_reload_identical_actions=True,complete_native_reexecution=True,
        old_model_and_physics_unchanged=True,frozen_features_and_sharpened_actor_preserved=True)
    write_json(BASE/'final_model_artifacts.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
    all_reports={}
    for label,checkpoint in [('baseline',ROOT/'checkpoints/ppo_classic_shadow_local_critic_warm_20261015.pt'),('final',target)]:
        rows=[];runs=[]
        for seed in (2026101531,2026101532,2026101533):
            path=BASE/f'{label}_heldout64_seed{seed}.json'
            sample(checkpoint,64,seed,path);r=json.loads(path.read_text(encoding='utf-8'));runs.append(r)
            rows.extend(dict(e,seed=seed) for e in r['results'])
        successful=[e for e in rows if e['success']];times=[e['time_s'] for e in successful]
        all_reports[label]=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),episodes=len(rows),
            successes=len(successful),success_rate=len(successful)/len(rows),
            successful_mean_time_s=float(np.mean(times)),successful_median_time_s=float(np.median(times)),
            best_time_s=min(times),faster_than_reference=sum(e['frames']<991 for e in successful),
            results=rows)
    comparison=dict(seeds=[2026101531,2026101532,2026101533],per_model_episodes=192,
        selection_was_fixed_before_heldout=True,scope='Original identical initial state; only action sampling differs; first episode per environment',
        baseline_note='BC logits sharpened 5x before PPO; Critic-only warm leaves policy unchanged',
        results=all_reports)
    write_json(BASE/'heldout_comparison.json',comparison)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='results'} for k,v in all_reports.items()},ensure_ascii=False),flush=True)
def report():
    artifacts=read('final_model_artifacts.json');heldout=read('heldout_comparison.json')['results']
    searches=[read(f'classic_shadow_v{i}_search_best_20261015_search.json') for i in (1,2,3)]
    best=min(searches,key=lambda r:r['frames'])
    times=[json.loads(p.read_text(encoding='utf-8')) for p in BASE.glob('ppo_*_time.json')]
    elapsed=sum(r['elapsed_seconds'] for r in times);steps=sum(r['new_transitions'] for r in times)
    now=datetime.now(timezone.utc);started=datetime.fromisoformat(read('initialization.json')['task_started_utc'])
    wall=(now-started).total_seconds()
    retained_steps=int(torch.load(artifacts['model'],map_location='cpu',weights_only=False)['global_step'])
    trajectories=[read(f'trajectory_{tag}_result.json') for tag in ('single_frame','timing','closedloop','blocks')]
    beams=[read(f'beam_{tag}_result.json') for tag in ('finish950','finish900_lead')]
    native_candidates=read('native_candidates.json')
    summary=dict(best_native_time_s=min(artifacts['deterministic_time_s'],best['time_s']),
        best_frames=min(artifacts['deterministic_frames'],best['frames']),broke_reference=min(artifacts['deterministic_frames'],best['frames'])<991,
        retained_checkpoint_global_step=retained_steps,
        ppo_training_seconds=elapsed,ppo_new_transitions=steps,task_elapsed_seconds=wall,
        random_search_episodes=sum(r['episodes'] for r in searches),
        bounded_trajectory_trials=sum(r['candidates'] for r in trajectories),
        parameter_search_candidates=sum(read(f'bias_{tag}_result.json')['attempts'] for tag in ('sprint','flat')),
        beam_prefix_candidates=sum(r['candidate_prefixes_evaluated'] for r in beams),
        greedy_native_candidate_files=len(native_candidates),model=artifacts['model'],replay=artifacts['replay'],
        heldout={k:{a:b for a,b in v.items() if a!='results'} for k,v in heldout.items()},
        reference='XYF external demonstration, 991 frames',ended_utc=now.isoformat())
    write_json(BASE/'final_summary.json',summary)
    b=heldout['baseline'];f=heldout['final'];rows=[]
    for i,part in [(1,'local_v1'),(2,'explore_v2'),(3,'calibrated_v3')]:
        t=read(f'ppo_classic_shadow_{part}_20261015_time.json');s=read(f'ppo_v{i}_selection128.json')
        rows.append(f'| V{i} | {t["new_transitions"]:,} | {t["elapsed_seconds"]:.2f} | {s["successes"]}/128 | 33.033333 秒 |')
    text=f'''# 百米模仿后微调与有界搜索实验（2026-10-15）

本次最好原生成绩为 **{summary['best_native_time_s']:.6f} 秒、{summary['best_frames']} 帧**。{'已快于' if summary['broke_reference'] else '尚未突破'}当时下载的 XYF 991 帧、33.033333 秒示范。该结果不能证明示范全局最优，只反映本轮训练和有限搜索范围。

## 保存结果

- 模型：`checkpoints/{Path(artifacts['model']).name}`，选中 V{artifacts['selection']['version']}，根据独立选择种子上的原生成绩与成功次数保留。
- 自主回放：`replays/{Path(artifacts['replay']).name}`。严格加载与完整原生重放相同，CRC通过。该回放与原BC回放的SHA256相同，991个动作完全相同，不能作为新纪录。
- 每轮4096次、总计{summary['random_search_episodes']:,}次原作者策略随机搜索，三轮最好均记录在验证目录；搜索使用异步结束回合，其成功比例不当作独立评测成功率。
- 原 BC 模型、原示范、百米旧回放和跳远模型均保留。原物理 DLL 与 BC 模型哈希不变。

## 修改与初始化

生产代码只修改 scripts/train_ppo.py：增加默认关闭的示范交叉熵辅助损失及其配置，并增加默认1的 reward_scale。原 PPO、GAE、rollout buffer、模型主体、环境、动作、物理和原生计时没有重写。

PPO损失中加入小权重的专家状态动作分类损失。示范数据仅用于辅助监督，不混入 on-policy rollout，也不把同一时刻的示范动作错误标给偏离状态。系数逐步减弱；关闭时不采示范样本、不增加随机数调用。默认网络在991个示范状态上的输出与修改前逐值相同。

现有 flat residual 在10–45米及65–100米生效，零初始化。冻结原跑步特征、角速度主干和 Actor，只训练该调整层及独立 Critic。Actor logits先乘5（等效采样温度0.2），确定性动作保持991帧完整相同。随机基线采用同样锐化策略，避免将降低温度的效果误认为PPO改进。

新配置沿用 configs 系统；验证目录新增脚本仅编排原训练器、原生回放和有界搜索，以分开记录训练、自主策略评测和动作序列规划。

## 三轮训练

| 轮次 | 新PPO转移 | 正式训练耗时/秒 | 选择集原生成功 | 确定性最好 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

V1：保守局部微调，示范系数0.02→0.002。V2：通过已有线性残差，仅将平稳跑动区间的完整logits乘0.5，初始贪心动作不变；学习率3e-4，示范系数0.005→0.0005，扩大探索。V3：校准独立价值函数，训练目标缩放0.01，学习率1e-4，示范系数0.002→0.0002。

共用128环境、1024步rollout、4 epochs、8 minibatches、gamma=.9999、GAE lambda=.995、clip=.2、KL阈值.006，RND关闭。分阶段小规模验证均有有限观测、奖励、损失、完整reset及checkpoint重载；冻结跑步参数逐值一致。累计含校准、小测试的PPO进程 **{elapsed:.2f}秒，约{elapsed/60:.2f}分钟**，总计 **{steps:,}** 个新转移。本轮从首次检查到报告约 **{wall/60:.2f}分钟**，包含代码检查、训练、评测及搜索；并行运行时间不相加。保留模型当前阶段global_step为 **{retained_steps:,}**（Critic校准与V1）；其他实验分支总量不能算到该模型。

## Critic问题与边界

V1末次Critic在示范状态的预测约184，成功示范Monte Carlo折扣回报约2090–2139，提示初始价值校准明显不足。该对照是成功示范条件回报，并非随机策略无偏价值估计，不能据此断言这是停滞的唯一根因。

V3先恢复Actor的有区分度特征到独立Critic，再用成功示范MC回报做带正则的线性拟合。内部奖励×0.01，初始化示范价值RMSE为0.008236（缩放单位）；缩放reward和value后GAE/return等比例关系验证通过，Actor动作不变。在线训练仍需要学习偏离与失败状态。

日志继续报告原始环境回报。V3的原始value loss与前两轮单位不同，不能直接将数字降低解释成性能提升；下图已将价值损失换算回原奖励单位平方。完成时间仍由原生环境测得。

## 独立留出评测

在选择模型后固定权重，种子2026101531/32/33各64次、每环境只取首回合。每组192次、初始状态相同，仅动作采样随机；未据留出数据重新选择模型。

| 指标 | 同温度BC基线 | 保留微调模型 |
| --- | ---: | ---: |
| 原生成功 | {b['successes']}/192 | {f['successes']}/192 |
| 成功率 | {b['success_rate']*100:.2f}% | {f['success_rate']*100:.2f}% |
| 成功平均时间 | {b['successful_mean_time_s']:.6f}秒 | {f['successful_mean_time_s']:.6f}秒 |
| 最好时间 | {b['best_time_s']:.6f}秒 | {f['best_time_s']:.6f}秒 |
| 快于991帧 | {b['faster_than_reference']}次 | {f['faster_than_reference']}次 |

成功率结果是该采样协议下的经验值，不代表不同初态鲁棒性。确定性反复相同不是独立成功率样本。

## 额外搜索与解释

- 单帧替换、切换时机、偏离后BC闭环恢复、2/3/4/6帧块及邻接动作，共 **{summary['bounded_trajectory_trials']:,}** 个有界候选，未找到990帧以内完成。
- 冻结BC主干的局部动作偏置参数搜索：两段各4096组，共 **{summary['parameter_search_candidates']:,}** 组，未突破。
- 两轮冲线段beam搜索共检查 **{summary['beam_prefix_candidates']:,}** 个动作前缀；这些是共享前缀的规划节点，不能当作同数量完整独立比赛。
- 动作序列搜索是规划，不是自主网络表现；本轮没有把规划回放冒充PPO模型成果。搜索到更快轨迹才有必要重新自模仿，本轮未出现可学习的新纪录。

本轮结果支持：扩大随机探索会明显增加偏离与摔倒，单条高质量轨迹附近的局部调整不容易直接改进。当前奖励中，在其他条件相同时快一帧约多0.7分，而失去完赛会损失上千分，PPO更容易优先减少失败；这是机制解释，不能单凭本实验确定因果。冻结主干保护动作，也限制了全程协调改变。

后续可优先引入多条不同节奏的有效示范与偏离恢复数据，再尝试带约束的小学习率全Actor微调；本轮不据未突破结论宣称33.03秒不可超越。

## 图与证据

![真实训练日志及原生时间](../../verification/classic_improvement_20261015/training_curves.png)

验证目录：`../../verification/classic_improvement_20261015`。主要文件：final_summary.json、final_model_artifacts.json、heldout_comparison.json、native_candidates.json、critic_expert_calibration.json、critic_calibrated_initialization.json、各轮log/time、三份search报告和chart_sources.json。代码修改未自动提交Git。
'''
    doc=ROOT/'docs/classic-shadow-finetuning-20261015.md';doc.write_text(text,encoding='utf-8')
    print(json.dumps(dict(summary=summary,document=str(doc)),ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','report']);a=p.parse_args()
    prepare() if a.mode=='prepare' else report()
