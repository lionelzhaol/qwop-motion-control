from pathlib import Path
import json,re
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]/'QWOP_rl_training_public'
rows=json.loads((BASE/'experiment_registry.json').read_text(encoding='utf-8'))
inventory=json.loads((BASE/'runs_inventory.json').read_text(encoding='utf-8'))
by_name={r['name']:r for r in rows}

ideas={
'baseline_smoke_20261008':('原始PPO短测试','先用作者入口验证环境交互、更新和保存；这是链路验证，不是追求百米成绩。','baseline重复加载后仍未通关，符合短测试用途。'),
'baseline_resume_20261008':('原始checkpoint续训验证','恢复网络和Adam进度，验证续训确实更新权重并能重新保存。','累计只有1536条转移；不能把能加载模型视为会跑步。'),
'ppo_classic_calibration_20261008':('原始百米规模标定','在原算法、奖励和网络不变的条件下扩大并行采样，确认CUDA训练吞吐与数值有限。','它与后续主训练共同组成约1048.6万新转移；单独规模标定不说明已学会通关。'),
'ppo_classic_continue_20261008':('原奖励百米续训','保留作者奖励、网络和PPO，从标定模型继续学习向前移动。','最终64次随机测试平均35.59米、最好39.20米、0次通关；确定性仍在120秒只到0.6米。'),
'ppo_foot_running_v1':('脚支撑奖励第一轮','开始加入低姿态限制与脚支撑进度，使用2帧历史；检查原输入尺度是否支持姿态反馈。','一次64回合评测虽全部百米结束，但低姿态约98.89%，说明原生通关不足以满足跑姿目标。'),
'ppo_foot_running_v2':('脚支撑奖励第二轮','继续调试防持续低姿态规则和归一化/初始化路径，保留奖励状态reset验证。','当时32回合平均22.03米、低姿态98.42%，没有通关；该失败结果保留，不当作成功改进。'),
'ppo_foot_running_v3':('归一化与站立反馈形成可用步态','从短训练重新出发，固定输入缩放、躯干高度代理和脚支撑新进度，改善Tanh饱和问题。','固定运行75.27秒；随机38/64成功，低姿态约4.47%。不能证明严格膝盖零接触。'),
'ppo_foot_running_v4':('微前倾与明显后仰约束','新进度乘1.2；脚支撑下微前倾有软奖励，持续明显后仰按严重程度扣分。降低续训学习率保护步态。','确定性65.63秒，随机53/64通关，成功均值66.80秒；后来专家证据说明固定姿态并非速度必要条件。'),
'ppo_speed50_v5':('增加速度目标后的首轮训练','把速度质量、低速反馈和更早完赛偏好纳入同一阶段奖励，继续在2帧历史上训练。','已记录的最快成功保存点61.33秒；奖励版本变化后的回报不能直接和v4横比。'),
'ppo_speed50_v6':('继续速度奖励训练','保持较温和的学习率，继续同阶段速度/低速规则，观察更快候选与末次模型是否一致。','有60秒成功候选，但最终模型的一次确定性运行在58.3米失败，提醒保留中途模型。'),
'ppo_speed50_v7':('速度阶段扩大更新和探索','同阶段速度目标下提高学习率和熵，增加不同按键序列的采样；不修改原生赛道和时钟。','最快已记录成功57.20秒，仍未到50秒；不存在随机性越大必然更快的保证。'),
'ppo_speed50_v8':('扩展到8帧历史','使用8帧观测帮助判断动作趋势，降低学习率继续优化，保留已有输出层接口。','成功候选56.93秒；时序信息增加是条件变化，不是单因素因果证明。'),
'ppo_start50_v9':('前10米里程碑','针对起步慢提供首次10米限时反馈，继续8帧时序PPO。','最好记录53.60秒；需要防止倒退重复领奖和无效姿态补领。'),
'ppo_start50_v10_gamma':('提高gamma的部分试验','把gamma提高至0.999，试图延长远期奖励的影响。','现存日志未完成配置的全部上限，未找到成功确定性候选曲线；不把训练distance冒充通关成绩。'),
'ppo_start50_v11_value':('降低价值损失干扰的早期试验','在较高gamma设置下降低value loss系数，观察共享Actor/Critic表征是否更利于策略。','已找到保存点56.93秒；当时没有刷新此前53.50秒候选。后续v16采用了更稳定的组合。'),
'ppo_start50_v12_dense':('连续早期进度奖励','增加前10米连续新进度反馈，回到稳定gamma和价值权重组合，让起步反馈更近。','最好53秒，说明较近反馈有效候选仍未满足50秒目标。'),
'ppo_start50_v13_confidence':('从快候选集中动作再训练','从已有较快参数候选开始，增强动作分布集中程度后继续PPO，减少随机按键打断步态。','纯PPO候选51.03秒；这里不是仅采样挑出一个幸运回放。'),
'ppo_start50_v14_kl':('KL停止条件','新增/使用target KL0.015，过大分布变化时停止当前rollout的后续优化。','最好53.03秒，单独KL方向没有刷新v13速度；KL阈值是保护，不是性能保证。'),
'ppo_start50_v15_gae':('更长GAE','把lambda提高到0.98，尝试给早期动作更长信用分配，保持KL保护。','最好51.40秒，未超过v13；高lambda也会增加估计方差。'),
'ppo_start50_v16_policy':('较低价值权重的策略优化','保留起步反馈、lambda0.98和KL，将value系数降至0.05。','纯PPO最好50.17秒；之后512组偏置参数搜索才得到49.97秒，不能把二者混写。'),
'ppo_target45_v17_control':('45秒目标的探索对照','在已有快模型上保持熵0.02，作为提高探索强度的对照。','成功候选51.07秒；不优于49.97秒归档模型。'),
'ppo_target45_v17_explore':('45秒目标提高固定熵','与较低探索设置比较，将熵系数提高到0.04。','成功候选51.13秒，固定更大熵没有直接带来提速。'),
'ppo_target45_v18_schedule':('探索到巩固的熵衰减','熵0.06逐渐降至0.01，先探索再减少随机扰动，保持PPO更新匹配分布。','最好50.10秒；尚未超过此前含搜索的49.97模型。'),
'ppo_target45_v19_history':('32帧历史','扩至约1秒历史并降低学习率，让策略观察较完整动作周期。','候选49.07秒，纯PPO已低于50；仍不能据此声称32帧在所有任务最优。'),
'ppo_target45_v20_pose':('5°与高度波动软约束','调整姿态奖励、允许自然小幅起伏，在时序模型上尝试用户的稳定重心假设。','最好49.80秒；更严格姿态假设没有在该轮带来更快纪录。'),
'ppo_target45_v21_wide':('网络扩到256×256','复制可兼容参数、增加网络容量，在32帧输入上继续训练。','最好48.77秒；网络规模增大也增加优化/价值拟合压力。'),
'ppo_target45_v22_consolidate':('宽网络巩固','对已有候选降低探索并继续训练，观察好动作能否稳定保留。','最好48.33秒；中途候选可能好于最后权重。'),
'ppo_target45_v23_cadence':('有效交替脚落地奖励','只对确认离地、触地、左右交替并创造新进度的动作提供步频质量奖励，过滤原地踏步。','最好48.70秒；奖励步频并没有直接造就最快版本。'),
'ppo_target45_v24_actor':('降低价值回归系数','基于共享梯度诊断将value系数降到0.01，减小Critic影响。','最好48.13秒；loss数值本身不能替代共享梯度诊断。'),
'ppo_target45_v25_motion':('加入部件角速度投影','从相邻帧估计12个部件角速度，零初始化投影以保留起始旧策略，检查角度环绕与reset。','最好48.97秒；新特征单独没有刷新纪录，但后来巩固保留它。'),
'ppo_target45_v26_consolidate':('低熵巩固姿态候选','保留角速度与32帧，熵0.005降到0.001，巩固已学动作。','47.83秒姿态候选更符合部分假设，却比最终44.97秒速度候选慢。'),
'target45_v27_smoke':('冻结表征短测试','冻结共享网络与角速度，只让输出头学习，检验保护步态的可行性。','只留下短测试；没有对应正式主训练，不能据此判断速度收益。'),
'target45_v27_lr_smoke':('冻结表征的学习率短测试','在冻结表征方向继续校验适合小批次的学习率与输出更新。','属于验证而非正式成绩；该方向在后续v35和跳远局部调整中重新研究。'),
'ppo_target45_v28_confident':('集中动作后长期巩固','从可用候选提高动作集中程度后继续PPO，使用256×256、32帧、角速度与较低熵。','纯PPO最好45.70秒；2048组Actor偏置参数搜索另选出44.97秒最终模型。'),
'ppo_target45_v29_credit':('lambda到1的试验','延长GAE信用分配，比较短期步态和更远期完赛奖励。','最好48.23秒，没有刷新v28；不是lambda越接近1越好。'),
'ppo_phase_v30_hurdle':('越栏阶段识别和恢复','针对50米障碍的必要腾空局部放宽垂直/角度罚分，并给落地恢复反馈。','最好45.63秒；局部收益未超过保留的44.97秒模型。'),
'ppo_phase_v31_start':('前3米更近反馈','增加首次3米限时里程碑与连续新进度，状态在reset清零。','最好46.33秒；近反馈并未保证全程更快。'),
'ppo_phase_v32_start_explore':('仅起步弱探索','前3米logits乘0.5，采样与PPO重算使用同一规则，后续正常分布。','最好46.27秒；正缩放本身不改变当时argmax。'),
'ppo_phase_v33_effective_explore':('有效动作熵与更强起步探索','前3米起始logits缩放0.125，并按9类物理等价作用计算熵，随后收回缩放。','最好45.40秒，是这条独立RL路线保留的纯PPO候选；随机平均和成功数在一组样本里优于单次最快模型。'),
'ppo_phase_v34_long_credit':('长rollout、gamma和Critic校准','rollout512、gamma0.999、lambda0.995，并用成功轨迹预校准价值头，保持Actor。','最好47.63秒，没有刷新纪录；单轨迹价值校准不能保证偏离轨迹后估值正确。'),
'ppo_phase_v35_frozen_gait':('冻结共享表征只训练头','针对Critic梯度干扰，冻结网络和角速度，保留原步态表征。','最好46.63秒；冻结限制新特征学习，不能普遍替代完整策略更新。'),
'ppo_target43_v36_flat_residual':('平跑区域动作残差','在指定平跑区域使用可选残差输出，冻结基线，重点探索平跑速度。','最好46.07秒，未达到43秒，也未超过44.93归档。'),
'ppo_target43_v37_actor_features':('减少共享Critic干扰','将价值学习与Actor表征影响分开，允许Actor特征学习，观察梯度与动作分布。','最好46.00秒，结构保护没有自动变成速度提升。'),
'ppo_target43_v38_soft_actor':('放松过于集中的Actor','调整过于尖锐的策略初始化并继续训练，尝试恢复真实有效动作探索。','最好46.47秒；增大探索不意味着探索到了有效起步或过栏恢复。'),
'ppo_target43_v39_time_priority':('完赛时间优先与独立Critic','更高gamma、独立Critic特征、角色学习率/梯度裁剪，降低步频奖励的影响。','最好45.93秒，未达到43秒；后来外部专家进一步揭示旧姿态门槛的奖励排序冲突。'),
'ppo_target43_v40_task_control':('任务奖励对照','去掉5°、高度波动、步频目标，使用时间和正式成功导向的task奖励，RND关闭。','最好46.80秒；同64回合57次成功、均值49.26秒。这是配对对照，非43秒成功。'),
'ppo_target43_v41_task_rnd':('任务奖励加RND','相同初始Actor和训练设置，加入beta0.02到0的新颖性奖励，修正RND归一化与reset错误。','最好47.23秒；同64回合61次成功、50秒内55次，最快速度仍未刷新。'),
'ppo_long_jump_v1':('旧百米策略迁移原始跳远','保留身体特征/归一化，独立Critic重置任务价值头，PPO使用作者跳远奖励。','update576确定性2.60米；1024随机回合最好2.80米，末次模型约2.03米。'),
'replay_imitation_pilot':('公开影子示范行为克隆','从44.93秒模型初始化，Actor最初乘0.25；原生重放XYF的991帧，构建991×2176状态-动作样本，交叉熵训练现有Actor，冻结Critic。','正式epoch1000复现33.03秒；原始随机采样64回合全失败。推理不读取专家动作表，但纠错尚不足。'),
'ppo_long_jump_shadow_v2':('BC跑姿输入原点校准后PPO','将位置字段语义差折算进第一层偏置，复制归一化与身体特征；小Actor学习率，独立Critic适应跳远。','初始1.74米，最好2.576米；前25米动作仍和百米BC迁移一致。'),
'ppo_long_jump_shadow_v3':('起跳附近分阶段探索','用起跳线连续距离做门控，前5米及之后降低logits尺度，避免用落地后含义改变的score。','确定性仍2.576米；一轮1024次随机搜索最好2.776米，未超过旧2.80米。'),
'ppo_long_jump_shadow_v4':('冻结助跑、训练局部动作调整','冻结原身体网络、角速度分支和Actor；只训练起跳门控下的零初始化调整层与独立Critic。','确定性2.588米；2048次随机运行中最好3.161米。独立64次51次成功、成功平均1.507米，最佳不等于平均。'),
}
order=['baseline_smoke_20261008','baseline_resume_20261008','ppo_classic_calibration_20261008','ppo_classic_continue_20261008']
order += [f'ppo_foot_running_v{i}' for i in range(1,5)]
order += [f'ppo_speed50_v{i}' for i in range(5,9)]
order += ['ppo_start50_v9']+[f'ppo_start50_v{i}_{s}' for i,s in [(10,'gamma'),(11,'value'),(12,'dense'),(13,'confidence'),(14,'kl'),(15,'gae'),(16,'policy')]]
order += ['ppo_target45_v17_control','ppo_target45_v17_explore']+[f'ppo_target45_v{i}_{s}' for i,s in [(18,'schedule'),(19,'history'),(20,'pose'),(21,'wide'),(22,'consolidate'),(23,'cadence'),(24,'actor'),(25,'motion'),(26,'consolidate')]]
order += ['target45_v27_smoke','target45_v27_lr_smoke','ppo_target45_v28_confident','ppo_target45_v29_credit']
order += [f'ppo_phase_v{i}_{s}' for i,s in [(30,'hurdle'),(31,'start'),(32,'start_explore'),(33,'effective_explore'),(34,'long_credit'),(35,'frozen_gait')]]
order += [f'ppo_target43_v{i}_{s}' for i,s in [(36,'flat_residual'),(37,'actor_features'),(38,'soft_actor'),(39,'time_priority'),(40,'task_control'),(41,'task_rnd')]]
order += ['ppo_long_jump_v1','replay_imitation_pilot','ppo_long_jump_shadow_v2','ppo_long_jump_shadow_v3','ppo_long_jump_shadow_v4']
assert set(order)==set(by_name),(set(by_name)-set(order),set(order)-set(by_name))

text=(BASE/'report_front.md').read_text(encoding='utf-8')
text+='\n## 附录A：逐次训练参数、想法、效果与曲线\n\n'
text+='参数取对应TensorBoard配置标签；曲线来自同目录实际标量。正式BC图使用独立评测报告，避免混入短测试。这里的“最快成功保存点”只代表已记录独立评测中选出的候选，不暗示全部状态下性能或单因素因果关系。\n\n'
text+='| 序号 | 训练 / 验证 | 实际新增转移 | 已记录最好确定性原生成绩 |\n| --- | --- | ---: | ---: |\n'
for i,name in enumerate(order,1):
    r=by_name[name];title=ideas[name][0]
    value='无可用成功记录' if r['best_native'] is None else f"{r['best_native']:.3f}{'米' if r['mode']=='long_jump_30m' else '秒'}"
    if name=='replay_imitation_pilot':value='33.033秒（BC，非PPO）'
    amount='监督991样本' if name=='replay_imitation_pilot' else format(r['actual_new_steps'],',')
    text+=f"| A{i:02} | {title} / {name} | {amount} | {value} |\n"
for i,name in enumerate(order,1):
    r=by_name[name];cfg=r['config'];title,idea,conclusion=ideas[name]
    text+=f'\n### A{i:02}. {title} — `{name}`\n\n**想法与实现：**{idea}\n\n'
    if name=='replay_imitation_pilot':
        text+='**训练参数：**991条专家样本；32帧/2176维；256×256网络；Adam1e-4；batch256；正式最多1000epoch；原生每100epoch评估；Critic冻结。采集及正式训练/评测约12.47秒。\n\n'
    else:
        batch=int(cfg.get('num_envs',1))*int(cfg.get('rollout_steps',1))
        text+='| 参数 / 记录 | 实际设置 |\n| --- | --- |\n'
        pairs=[('环境 / 线程',f"{cfg.get('num_envs','未记录')} / {cfg.get('num_threads','未记录')}"),
            ('rollout / 总batch / minibatch',f"{cfg.get('rollout_steps','未记录')} / {batch} / {batch//int(cfg.get('minibatches','1'))}"),
            ('实际日志update / 配置计划update',f"{r['updates_logged']} / {cfg.get('updates','未记录')}"),
            ('本轮新增转移',f"{r['actual_new_steps']:,}"),
            ('学习率 Actor / Critic',f"{cfg.get('learning_rate','未记录')} / {cfg.get('critic_learning_rate','同Actor')}"),
            ('gamma / GAE lambda',f"{cfg.get('gamma','未记录')} / {cfg.get('gae_lambda','未记录')}"),
            ('clip / epochs / KL阈值',f"{cfg.get('clip_coef','未记录')} / {cfg.get('epochs','未记录')} / {cfg.get('target_kl','关闭')}"),
            ('熵系数 / 最终值 / value系数',f"{cfg.get('ent_coef','未记录')} / {cfg.get('ent_coef_final','未调度')} / {cfg.get('vf_coef','未记录')}"),
            ('网络 / 历史帧 / seed',f"{cfg.get('hidden_sizes','未记录')} / {cfg.get('frame_stack','未记录')} / {cfg.get('seed','未记录')}"),
            ('reward profile / game mode',f"{cfg.get('reward_profile','未记录')} / {cfg.get('game_mode','未记录')}")]
        for k,v in pairs:text+=f'| {k} | {v} |\n'
        special=[f'{k}={cfg[k]}' for k in ['normalize_observations','derive_angular_velocity','effective_action_entropy','start_logit_scale','freeze_backbone','detach_critic_features','separate_critic_features','enable_flat_actor_residual','rnd_beta','rnd_beta_final','jump_explore_logit_scale','jump_explore_meters','enable_jump_actor_residual'] if k in cfg and cfg[k] not in ('False','None','0.0','1.0')]
        if special:text+='\n**启用项：**'+ '；'.join(special)+'。\n'
        if r['reward_points']:
            text+=f"\n**奖励日志：**共{r['reward_points']}个有效记录点；首点{r['reward_first']:.3f}，末点{r['reward_last']:.3f}，最后最多20点均值{r['reward_last20_mean']:.3f}。这些是训练期间结束回合的平均回报，不是独立评测成绩。\n"
        else:text+='\n**奖励日志：**没有可用episode return标量，不补造曲线。\n'
        if r['best_native'] is not None:
            text+=f"\n**原生评测：**已索引{r['native_success_checkpoints']}个成功保存点/位置，最好{r['best_native']:.6f}{'米' if r['mode']=='long_jump_30m' else '秒'}。\n"
    text+=f'\n**效果与反思：**{conclusion}\n\n'
    text+=f"![{title}的实际训练和评测曲线](QWOP项目图表/{r['figure']})\n\n"
    logs=ROOT/'runs'/name
    text+=f'证据：[原始日志目录]({logs.as_posix()})'
    if r['evidence_files']:
        selected=[p for p in r['evidence_files'] if 'comparison' in p or 'summary' in p][:1] or r['evidence_files'][:1]
        for f in selected:text+=f"；[独立评测示例]({(ROOT.parent/'verification'/f).as_posix()})"
    if cfg.get('resume_from') not in (None,'None'):
        text+='。来源权重：`'+cfg['resume_from']+'`'
    text+='。完整参数和评测索引见本报告JSON材料。\n'

text+='\n## 附录B：其余最小测试与验证日志索引\n\n'
text+='下面是未作为主要训练图单列的记录，多用于构建后短训练、参数边界、episode/reset、结构加载或学习率检查。不能用几次update证明正式成绩收益。名字与原始目录保持一致，便于定位。\n\n'
text+='| 日志目录 | policy loss记录数 | 首末累计global_step |\n| --- | ---: | --- |\n'
for r in inventory:
    if r['name'] in by_name:continue
    text+=f"| [{r['name']}]({(ROOT/'runs'/r['name']).as_posix()}) | {r['points'].get('losses/policy_loss',0)} | {r['first_step']} → {r['last_step']} |\n"
text+='\n## 附录C：报告生成范围与审计说明\n\n'
text+=f"本次扫描{len(inventory)}个TensorBoard日志目录；单列{len(rows)}组训练/关键短测/BC图和3张结果对照图，共{len(rows)+3}张PNG。主要PPO与规模标定日志中的新转移加总约{sum(r['actual_new_steps'] for r in rows if r['name'].startswith('ppo_'))/1e9:.3f}十亿条；这是多个实验分支、控制组和失败尝试的总采样量，不是任何一个模型的累计训练步数。该总量不包含所有短测试或采样搜索。\n\n"
text+='图像用Windows内置图表组件渲染；无示意性虚构奖励。部分历史原生评测缺失时，图中展示诊断指标，正文不补填通关数。实际训练配置以对应事件文件/JSON为准；当前源码是累积修改后的版本，不能把当前奖励常数自动套到每一历史实验。\n\n'
text+='Markdown内图表使用相对附件路径，和同级`QWOP项目图表`文件夹一起移动即可保留图像；源码、权重和原始证据链接指向本机E盘项目，需要本地文件仍在原位置。\n\n'
text+='文档是截至2026-10-11的研究记录；后续计划未被写成已完成成果，外部专家身份与方法出处保留。\n'
out=BASE/'项目研究记录.md';out.write_text(text,encoding='utf-8')
image_refs=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)
expected=[BASE/'figures'/Path(s).name for s in image_refs]
assert all(p.exists() for p in expected),[str(p) for p in expected if not p.exists()]
assert len(image_refs)==len(rows)+3,(len(image_refs),len(rows)+3)
assert text.count('```')%2==0 and text.count('$$')%2==0
print(json.dumps(dict(markdown=str(out),characters=len(text),lines=len(text.splitlines()),images=len(image_refs),main_and_key_test_entries=len(rows)),ensure_ascii=False))
