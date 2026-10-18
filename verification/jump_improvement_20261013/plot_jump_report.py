"""Refresh truthful jump charts with project Python and standard .NET Charting.

No dependencies are installed. Reads only complete logged updates and saved
native evaluations. All generated files stay alongside this script.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import json
import re
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1] / 'QWOP_rl_training_public'
COLORS = ['#2468A0', '#D17B24', '#328C63', '#A75493', '#7059AA']

def read_json(path):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError): return None

def series(name, points, color, **kwargs):
    return dict(name=name, points=points, color=color, **kwargs)

def collect():
    runs=[]
    for log in sorted(BASE.glob('ppo_long_jump_*_v*_*.log')):
        version = re.search(r'_v(\d+)_\d{8}$', log.stem)
        if not version: continue
        cfg = read_json(ROOT / 'configs' / (log.stem+'.json'))
        if not cfg or cfg.get('reward_profile') != 'long_jump_task': continue
        assert abs(cfg['gamma']-.999)<1e-10
        text = log.read_text(encoding='utf-8', errors='replace')
        start = re.search(r'checkpoint_update=(\d+) global_step=(\d+)', text)
        update_origin, step_origin = (int(start[1]),int(start[2])) if start else (0,0)
        rows=[]
        for line in text.splitlines():
            update=re.match(r'update=(\d+)/(\d+)\s',line)
            if not update: continue
            values={}
            for key, value in re.findall(r'(\w+)=([^\s]+)',line):
                try: values[key]=float(value)
                except ValueError: pass
            required=('steps','policy_loss','value_loss','entropy','episode_return_mean')
            if not all(key in values for key in required): continue
            values.update(local_update=int(update[1]),planned_updates=int(update[2]),
                          new_steps=values['steps']-step_origin)
            rows.append(values)
        if not rows: continue
        native=[]
        paths=sorted(BASE.glob(log.stem+'_update_*_native.json'))
        final_path=BASE/(log.stem+'_native.json')
        if final_path.exists(): paths.append(final_path)
        seen_updates=set()
        for path in paths:
            report=read_json(path)
            if not report: continue
            checkpoint_update=report.get('update')
            if checkpoint_update is None:
                match=re.search(r'_update_(\d+)_native$',path.stem)
                if not match: continue
                checkpoint_update=int(match[1])
            if checkpoint_update in seen_updates: continue
            seen_updates.add(checkpoint_update)
            native.append(dict(file=path.name,update=checkpoint_update,
                local_update=checkpoint_update-update_origin,success=bool(report['success']),
                official_jump_m=report.get('long_jump_m'),
                completed_jump_m=float(report['long_jump_m']) if report['success'] else 0.))
        native.sort(key=lambda row:row['local_update'])
        runs.append(dict(stem=log.stem,version=int(version[1]),config=cfg,
            reward_profile=cfg['reward_profile'],log=log.name,rows=rows,native=native))
    runs.sort(key=lambda run:run['version'])
    assert runs, 'No complete long_jump_task PPO log rows available'
    for i,run in enumerate(runs):
        run['color']=COLORS[i%len(COLORS)]
        run['label']=f"V{run['version']}" + ('：线性局部策略' if run['version']==5 else
            '：精英自模仿后 PPO' if run['version']==7 else '：局部状态 MLP')
    return runs

def make_plans(runs):
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M 北京时间')
    progress='；'.join(f"V{r['version']}已记录 {r['rows'][-1]['local_update']}/{r['rows'][-1]['planned_updates']} 次更新" for r in runs)
    panels=[]
    for key,title,ylabel in (
        ('episode_return_mean','同一奖励定义下的训练回合回报','外部奖励 / 回合'),
        ('policy_loss','Actor：PPO 策略损失','策略损失'),
        ('value_loss','Critic：价值拟合损失','价值损失'),
        ('entropy','策略探索程度：实际动作熵','Entropy / nat')):
        panels.append(dict(title=title,ylabel=ylabel,series=[
            series(r['label'],[[row['new_steps']/1e6,row[key]] for row in r['rows']],r['color']) for r in runs]))
    training=dict(file='training_curves.png',title='跳远策略训练曲线：正式距离奖励与局部状态策略',
        subtitle='奖励均为 long_jump_task；每个点来自一条实际 update 日志；'+progress,
        xlabel='本轮 PPO 新增交互步数 / 百万',panels=panels,
        footer='final_distance_mean 包含 30 m 助跑，不代表正式跳远成绩。下图使用单独原生规则评测。  '+stamp)

    imitation=read_json(BASE/'self_imitation_v2.json')
    assert imitation and imitation.get('evaluations'), 'Missing actual BC v2 native evaluation history'
    bc_evals=imitation['evaluations']
    imitation3=read_json(BASE/'self_imitation_v3.json')
    initial=read_json(BASE/'initialization.json')
    assert initial and 'initialized_deterministic_jump_m' in initial
    labels=['原跳远模型']
    scores=[initial['initialized_deterministic_jump_m']]
    names=['旧模型原生确定性']
    stage_sources=['initialization.json initialized_deterministic_jump_m']
    for r in runs:
        if r['version']!=5: continue
        if r['native']:
            best=max(r['native'],key=lambda row:row['completed_jump_m'])
            labels.append('V5 最佳候选');scores.append(best['completed_jump_m'])
            names.append('V5当前已评测最高');stage_sources.append(best['file'])
    labels.append('自模仿 v2');scores.append(imitation['selected']['long_jump_m'])
    names.append('BC v2已选策略');stage_sources.append('self_imitation_v2.json selected.long_jump_m')
    bc3_added=False
    for r in runs:
        if r['version']==5 or not r['native']: continue
        if r['version']>=7 and imitation3 and not bc3_added:
            labels.append('自模仿 v3');scores.append(imitation3['selected']['long_jump_m'])
            names.append('BC v3已选策略（100 epoch 单点评测）')
            stage_sources.append('self_imitation_v3.json selected.long_jump_m')
            bc3_added=True
        best=max(r['native'],key=lambda row:row['completed_jump_m'])
        labels.append(f"V{r['version']} 最佳候选");scores.append(best['completed_jump_m'])
        names.append(f"V{r['version']}当前已评测最高");stage_sources.append(best['file'])
    if imitation3 and not bc3_added:
        labels.append('自模仿 v3');scores.append(imitation3['selected']['long_jump_m'])
        names.append('BC v3已选策略（100 epoch 单点评测）')
        stage_sources.append('self_imitation_v3.json selected.long_jump_m')
    bc_series=[series('BC v2：完整原生评测',[[row['epoch'],row['long_jump_m'] if row['success'] else 0.] for row in bc_evals],'#328C63')]
    if imitation3:
        bc_series.append(series('BC v3：100 epoch 单点评测',
            [[row['epoch'],row['long_jump_m'] if row['success'] else 0.] for row in imitation3['evaluations']],
            '#A75493',chart_type='Point',marker_size=9))
    greedy_panels=[
        dict(title='PPO 候选：原生完整通关后的正式跳远',ylabel='正式距离 / m',xlabel='本轮 PPO 更新次数',ymin=0,
            series=[series(r['label'],[[row['local_update'],row['completed_jump_m']] for row in r['native']],r['color']) for r in runs]),
        dict(title='两轮自模仿：实际闭环确定性成绩',ylabel='正式距离 / m',xlabel='各轮独立训练 epoch',ymin=0,
            series=bc_series),
        dict(title='自模仿 v2：动作标签准确率',ylabel='准确率 / %',xlabel='自模仿训练 epoch',ymin=0,ymax=100,
            series=[series('195 帧起跳阶段',[[row['epoch'],100*row['train_phase_action_accuracy']] for row in bc_evals],'#328C63'),
                    series('529 帧完整示范',[[row['epoch'],100*row['expert_action_accuracy']] for row in bc_evals],'#7059AA')]),
        dict(title='阶段对比：各阶段已保留的确定性成绩',ylabel='正式距离 / m',xlabel='阶段；PPO 为当前已评测候选最高',labels=labels,
            series=[series('正式距离',[[i+1,float(score)] for i,score in enumerate(scores)],'#2468A0',chart_type='Column')])]
    greedy=dict(file='greedy_performance.png',title='原生跳远成绩：局部状态自模仿与 PPO 微调',
        subtitle='实际 checkpoint / BC epoch 评测；每次完整执行原始规则；成功距离不含 30 m 助跑',
        xlabel='实际评测位置',panels=greedy_panels,
        footer='失败候选按 0 m 显示；阶段最高值受候选数量影响，不代表独立平均性能。曲线连接已测点，不推断未测模型。  '+stamp)
    stages=[dict(stage=name,label=label,jump_m=float(score),source=source) for name,label,score,source in zip(names,labels,scores,stage_sources)]
    return [training,greedy],stages

def renderer():
    # Reuse the project's standard Windows plotting backend, copying it locally.
    source=(BASE.parent/'project_report/render_charts.ps1').read_text(encoding='utf-8-sig')
    source=source.replace('$chart.Width=1440','$chart.Width=1920').replace('$chart.Height=920','$chart.Height=1280')
    source=source.replace('$area.AxisX.Title=$plan.xlabel',
        '$area.AxisX.Title=$plan.xlabel\n        if($panel.xlabel){$area.AxisX.Title=$panel.xlabel}')
    source=source.replace('$area.AxisY.IsStartedFromZero=$false',
        '$area.AxisY.IsStartedFromZero=$false\n        if($null -ne $panel.ymin){$area.AxisY.Minimum=[double]$panel.ymin}\n        if($null -ne $panel.ymax){$area.AxisY.Maximum=[double]$panel.ymax}')
    source=source.replace("$line.LabelFormat='0.##'","$line.LabelFormat='0.000'")
    source=source.replace('foreach($p in $row.points)',
        'if($row.marker_size){$line.MarkerSize=[int]$row.marker_size}\n            foreach($p in $row.points)')
    source=source.replace('$destination=Join-Path $OutputDirectory $plan.file',
        '''if($plan.footer) {
        $footer=New-Object System.Windows.Forms.DataVisualization.Charting.Title($plan.footer)
        $footer.Font=New-Object Drawing.Font('Microsoft YaHei',9)
        $footer.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(3,95,94,3)
        $chart.Titles.Add($footer)
    }
    $destination=Join-Path $OutputDirectory $plan.file''')
    path=BASE/'render_jump_charts.ps1'
    path.write_text(source,encoding='utf-8-sig')
    return path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-render',action='store_true')
    args=parser.parse_args()
    runs=collect();plans,stages=make_plans(runs)
    plan_path=BASE/'jump_chart_plans.json'
    plan_path.write_text(json.dumps(plans,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=dict(backend='Standard Windows .NET Charting; project Python log parsing',
        runs=[{key:value for key,value in r.items() if key!='config'} for r in runs],stages=stages,
        reward_comparison='Only long_jump_task profiles compared. Native score plots never use training return or final_distance_mean.',
        smoothing='None. Every plotted training scalar is an actual logged update value.',
        native_count=sum(len(r['native']) for r in runs),figure_files=[p['file'] for p in plans])
    (BASE/'jump_chart_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.no_render:
        subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(renderer()),
            '-PlanPath',str(plan_path),'-OutputDirectory',str(BASE)],check=True)
    print(json.dumps(dict(runs=[dict(version=r['version'],updates=len(r['rows']),native_candidates=len(r['native'])) for r in runs],
                         stages=stages,figures=[str(BASE/p['file']) for p in plans]),ensure_ascii=False))

if __name__=='__main__':main()
