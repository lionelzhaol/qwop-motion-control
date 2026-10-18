"""Plot actual native results and training logs with existing .NET Charting."""
from pathlib import Path
import json,re,subprocess
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]/'QWOP_rl_training_public'
colors=['#2468A0','#D17B24','#328C63'];runs=[]
for i,(part,label) in enumerate([('local_v1','V1：保守微调'),('explore_v2','V2：增加局部探索'),('calibrated_v3','V3：校准价值尺度')]):
    stem=f'ppo_classic_shadow_{part}_20261015'
    cfg=json.loads((ROOT/'configs'/f'{stem}.json').read_text(encoding='utf-8'))
    text=(BASE/f'{stem}.log').read_text(encoding='utf-8')
    resume=re.search(r'checkpoint_update=(\d+) global_step=(\d+)',text)
    update_origin,step_origin=(int(resume[1]),int(resume[2])) if resume else (0,0)
    rows=[]
    for line in text.splitlines():
        if not line.startswith('update='):continue
        r={}
        for k,v in re.findall(r'(\w+)=([^\s]+)',line):
            try:r[k]=float(v)
            except ValueError:pass
        r['new_steps']=r['steps']-step_origin
        rows.append(r)
    native=[]
    for path in sorted(BASE.glob(stem+'_update_*_native.json')):
        r=json.loads(path.read_text(encoding='utf-8'))
        update=int(re.search(r'_update_(\d+)_native$',path.stem)[1])
        native.append([(update-update_origin)*cfg['num_envs']*cfg['rollout_steps']/1e6,r['time_s'] if r['success'] else None])
    runs.append(dict(label=label,color=colors[i],rows=rows,native=native,reward_scale=cfg.get('reward_scale',1.)))
def series(run,key):
    return dict(name=run['label'],color=run['color'],points=[
        [row['new_steps']/1e6,row[key]/run['reward_scale']**2 if key=='value_loss' else row[key]] for row in run['rows']])
panels=[]
for key,title,ylabel in [('episode_return_mean','原始回合回报：同一 classic_100m_task 奖励','原生奖励 / 回合'),
    ('entropy','动作探索程度','有效动作熵 / nat'),
    ('value_loss','价值损失：统一换算到原奖励单位','价值损失 / 原奖励单位平方')]:
    panels.append(dict(title=title,ylabel=ylabel,series=[series(r,key) for r in runs]))
panels.append(dict(title='完整原生规则评测：确定性百米时间',ylabel='完赛时间 / 游戏秒',ymin=32,ymax=36,
    series=[dict(name=r['label'],color=r['color'],points=[p for p in r['native'] if p[1] is not None]) for r in runs]))
plan=[dict(file='training_curves.png',title='百米模仿后微调：训练过程与真实成绩',
    subtitle='全部点来自实际日志和 checkpoint 原生评测；不同轮次初始化与探索不同，不能单独归因于某个参数',
    xlabel='本轮 PPO 新增交互步数 / 百万',panels=panels,
    footer='V3 内部回报 ×0.01；图中回报为原始值，价值损失已换算回原单位。回报上升不能当作速度突破。')]
path=BASE/'chart_plan.json';path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
(BASE/'chart_sources.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2),encoding='utf-8')
renderer=BASE.parent/'jump_improvement_20261013/render_jump_charts.ps1'
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(renderer),
    '-PlanPath',str(path),'-OutputDirectory',str(BASE)],check=True)
print(str(BASE/'training_curves.png'))
