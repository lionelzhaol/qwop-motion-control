from pathlib import Path
import json,re,statistics,torch
BASE=Path(__file__).resolve().parent
events=json.loads((BASE/'events_data.json').read_text(encoding='utf-8'))
records=json.loads((BASE/'deterministic_candidates.json').read_text(encoding='utf-8'))
plans=[];registry=[]
metadata={}
def source_update(path):
    if path not in metadata:
        p=Path(path)
        if not p.is_absolute():p=BASE.parents[1]/'QWOP_rl_training_public'/p
        if p.exists():
            data=torch.load(p,map_location='cpu',weights_only=False)
            metadata[path]=int(data.get('update',0));del data
        else:metadata[path]=None
    return metadata[path]

def moving(rows,window=20):
    return [[v['step'],sum(x['value'] for x in rows[max(0,i-window+1):i+1])/min(i+1,window)] for i,v in enumerate(rows)]

def series(name,rows,origin,color,secondary=False,smooth=False):
    vals=moving(rows) if smooth else [[p['step'],p['value']] for p in rows]
    return dict(name=name,color=color,secondary=secondary,points=[[(x-origin)/1e6,y] for x,y in vals])

for r in events:
    name=r['name'];cfg=r['config'];s=r['series']
    main=name.startswith('ppo_') and not any(t in name for t in ('smoke','minimal','load','resume'))
    extra=name in ('baseline_smoke_20261008','baseline_resume_20261008','replay_imitation_pilot','target45_v27_smoke','target45_v27_lr_smoke')
    if not(main or extra):continue
    batch=int(cfg.get('num_envs',1))*int(cfg.get('rollout_steps',1))
    anchor=s.get('losses/policy_loss') or next(iter(s.values()))
    origin=anchor[0]['step']-batch if name!='replay_imitation_pilot' else 0
    reward=s.get('charts/episode_return_mean',[]);distance=s.get('charts/final_distance_mean',[])
    mode=cfg.get('game_mode','classic_100m');long=mode=='long_jump_30m'
    stem=Path(cfg.get('checkpoint','missing')).stem
    matches=[p for p in records if (Path(p['checkpoint']).stem.startswith(stem+'_update_') or Path(p['checkpoint']).stem==stem) and p.get('stochastic_explicit') is not True]
    native={}
    for p in matches:
        update=p.get('update')
        if update is None:
            m=re.search(r'_update_(\d+)',Path(p['checkpoint']).stem);update=int(m[1]) if m else None
        step=p.get('global_step')
        if step is None and update is not None:
            first_abs=source_update(cfg.get('resume_from',''))
            if first_abs is None:
                # Correct inherited update origin from the first absolute logged step.
                first_abs=round(origin/batch)
            step=origin+(update-first_abs)*batch
        if step is None or step<origin or step>anchor[-1]['step']+batch:continue
        value=p['long_jump_m'] if long else p['time_s']
        if value is None:continue
        native[step]=value
    native_rows=[dict(step=k,value=v) for k,v in sorted(native.items())]
    panels=[dict(title='训练回合回报（不同奖励版本不可直接横比）',ylabel='Episode return',series=[series('原始日志',reward,origin,'#B4CBE6'),series('20点滚动均值',reward,origin,'#245B91',smooth=True)] if reward else []),
        dict(title='训练结束回合的平均 distance',ylabel='米（跳远包含助跑距离）' if long else '米',series=[series('distance mean',distance,origin,'#238471')]),
        dict(title='独立原生评估：成功保存点',ylabel='正式跳远长度（米）' if long else '百米完成用时（秒，越低越好）',series=[series('成功候选',native_rows,origin,'#B86A13')]),
        dict(title='Actor / Critic 损失（各自坐标轴）',ylabel='Policy loss',ylabel2='Value loss',series=[series('Actor',s.get('losses/policy_loss',[]),origin,'#245B91'),series('Critic 右轴',s.get('losses/value_loss',[]),origin,'#AA4B55',secondary=True)])]
    if name=='replay_imitation_pilot':
        origin=0
        report=json.loads((BASE.parent/'shadow_audit/imitation_pilot.json').read_text(encoding='utf-8'))
        cfg=dict(resume_from=str(BASE.parents[1]/'QWOP_rl_training_public'/report['source_checkpoint']),
            frame_stack='32',hidden_sizes='(256,256)',seed='20261010',game_mode='classic_100m',
            reward_profile='classic_100m',learning_rate='0.0001',method='BC',epochs='1000',batch_size='256')
        reward=[];distance=[];native={};matches=[]
        tag_fields={'imitation/loss':'cross_entropy','imitation/action_accuracy':'expert_action_accuracy','evaluation/time_s':'time_s','evaluation/success':'success'}
        def epochs(tag):return [{'step':p['epoch']*1e6,'value':float(p[tag_fields[tag]])} for p in report['evaluations']]
        panels=[dict(title='行为克隆交叉熵（不是奖励）',ylabel='Cross entropy',series=[series('分类损失',epochs('imitation/loss'),0,'#245B91')]),
            dict(title='专家状态动作分类准确率',ylabel='Accuracy',series=[series('准确率',epochs('imitation/action_accuracy'),0,'#238471')]),
            dict(title='闭环运行时间：与是否通关配合看',ylabel='游戏时间（秒）',series=[series('时间',epochs('evaluation/time_s'),0,'#B86A13')]),
            dict(title='闭环原生通关（0失败 / 1成功）',ylabel='Success',series=[series('通关',epochs('evaluation/success'),0,'#AA4B55')])]
    else:
        if not native_rows:
            panels[2]=dict(title='KL 与动作熵（没有成功成绩曲线）',ylabel='Approx KL',ylabel2='Entropy',series=[series('KL',s.get('losses/approx_kl',[]),origin,'#B86A13'),series('动作熵 右轴',s.get('losses/entropy',[]),origin,'#238471',secondary=True)])
    actual=anchor[-1]['step']-origin if name!='replay_imitation_pilot' else 0
    best_native=(max(native.values()) if long else min(native.values())) if native else None
    registry.append(dict(name=name,config=cfg,updates_logged=len(s.get('losses/policy_loss',[])),actual_new_steps=actual,
        reward_points=len(reward),reward_first=reward[0]['value'] if reward else None,reward_last=reward[-1]['value'] if reward else None,
        reward_last20_mean=statistics.mean([p['value'] for p in reward[-20:]]) if reward else None,
        best_native=best_native,native_success_checkpoints=len(native),mode=mode,figure=name+'.png',
        evidence_files=sorted(set(p['file'] for p in matches)),native_points=native_rows))
    plans.append(dict(name=name,title=name,subtitle='正式行为克隆评测 | 1000 epochs | 历史帧32 | 991条专家样本 | 不混入同目录短测试' if name=='replay_imitation_pilot' else f"实测日志 | 更新数 {len(s.get('losses/policy_loss',[]))} | 历史帧 {cfg.get('frame_stack','?')} | main training {actual/1e6:.2f}M steps",
        xlabel='训练 epoch' if name=='replay_imitation_pilot' else '本轮新增环境步（百万）',panels=panels,file=name+'.png'))

def categorical_panel(title,ylabel,labels,values,color='#245B91'):
    return dict(title=title,ylabel=ylabel,labels=labels,series=[dict(name='实测结果',color=color,secondary=False,chart_type='Column',points=[[i+1,v] for i,v in enumerate(values)])])
plans.append(dict(name='results_overview',title='项目成果概览：按方法区分成绩',subtitle='正式游戏计分；随机最佳值是筛选结果，不是平均表现',xlabel='实验阶段（非时间轴）',file='results_overview.png',panels=[
    categorical_panel('纯PPO已验证的阶段成绩','百米秒数，越低越好',['脚支撑v3','前倾v4','时序v8','起步v16','巩固v28','有效探索v33'],[75.2666702,65.6333313,56.9333344,50.1666679,45.7000008,45.4000015]),
    categorical_panel('PPO后参数搜索的确定性百米成绩','百米秒数',['50秒版','45秒版','瓶颈版'],[49.9666672,44.9666672,44.9333344],'#B86A13'),
    categorical_panel('外部示范行为克隆及低温随机回放','百米秒数',['BC确定性','温度0.4筛选','温度0.5筛选'],[33.0333328,33.1333351,33.1666679],'#238471'),
    categorical_panel('跳远：确定性与随机最佳分开记录','正式跳远长度（米）',['旧确定性','新确定性','旧随机最佳','新随机最佳'],[2.59879398,2.58818364,2.80229974,3.16072989],'#AA4B55')]))
plans.append(dict(name='reward_alignment',title='奖励审计：最快运动不一定得到最高旧奖励',subtitle='对同一回放在原生环境中重新累计奖励；两个奖励版本各自比较',xlabel='策略来源',file='reward_alignment.png',panels=[
    categorical_panel('旧classic奖励：排序与速度相反','累计回报',['专家33.03秒','旧策略44.93秒'],[1450.9540146,1877.4242071]),
    categorical_panel('任务奖励：快策略获得更高回报','累计回报',['专家33.03秒','旧策略44.93秒'],[2300.6668877,2055.7998947],'#238471'),
    categorical_panel('正式百米用时：原生时间','秒',['专家/BC','旧策略'],[33.0333328,44.9333344],'#B86A13'),
    categorical_panel('旧速度奖励的姿态门槛合格比例','百分比',['外部专家','旧策略'],[18.97073663,99.18397626],'#AA4B55')]))
plans.append(dict(name='rnd_comparison',title='RND配对实验：一次训练种子、相同64次评测种子',subtitle='成功回合均值不包含失败；这些差异不是统计显著性结论',xlabel='候选模型',file='rnd_comparison.png',panels=[
    categorical_panel('64次评估中的成功数','成功场次',['旧基线','仅任务奖励','任务奖励+RND'],[58,57,61]),
    categorical_panel('成功回合平均用时','秒',['旧基线','仅任务奖励','任务奖励+RND'],[48.9839,49.2596,48.8355],'#238471'),
    categorical_panel('全部64回合中50秒内成功数','场次',['旧基线','仅任务奖励','任务奖励+RND'],[43,41,55],'#B86A13'),
    categorical_panel('保存点中最快确定性成绩','秒',['旧基线','仅任务奖励','任务奖励+RND'],[44.933334,46.8,47.233334],'#AA4B55')]))
(BASE/'plot_plans.json').write_text(json.dumps(plans,ensure_ascii=False),encoding='utf-8')
(BASE/'experiment_registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'figures':len(plans),'main_experiments':len([p for p in registry if p['name'].startswith('ppo_')]),'experiments':[dict(name=p['name'],best_native=p['best_native'],actual_new_steps=p['actual_new_steps']) for p in registry]},ensure_ascii=False))
