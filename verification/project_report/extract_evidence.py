from pathlib import Path
import json,math,hashlib,re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE=Path(__file__).resolve().parent; WORK=BASE.parent.parent; ROOT=WORK/'QWOP_rl_training_public'
events=[];runs=[]
for folder in sorted((ROOT/'runs').iterdir()):
    if not folder.is_dir():continue
    acc=EventAccumulator(str(folder),size_guidance={'scalars':0,'tensors':0});acc.Reload()
    tags=acc.Tags();cfg={}
    for tag in tags.get('tensors',[]):
        if tag.startswith('config/'):
            vals=acc.Tensors(tag)
            if vals and vals[0].tensor_proto.string_val:
                cfg[tag.split('/')[1]]=vals[0].tensor_proto.string_val[0].decode('utf-8')
    series={}
    for tag in tags.get('scalars',[]):
        series[tag]=[{'step':int(v.step),'value':float(v.value),'wall_time':float(v.wall_time)} for v in acc.Scalars(tag)]
        assert all(math.isfinite(v['value']) for v in series[tag]),folder
    row={'name':folder.name,'config':cfg,'series':series,'event_files':[str(p) for p in folder.glob('events*')]}
    events.append(row)
    runs.append({'name':folder.name,'config':cfg,'tags':list(series),'points':{k:len(v) for k,v in series.items()},
        'first_step':min([v[0]['step'] for v in series.values() if v] or [0]),
        'last_step':max([v[-1]['step'] for v in series.values() if v] or [0])})
(BASE/'events_data.json').write_text(json.dumps(events,ensure_ascii=False),encoding='utf-8')
(BASE/'runs_inventory.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2),encoding='utf-8')
configs={p.stem:json.loads(p.read_text(encoding='utf-8-sig')) for p in (ROOT/'configs').glob('*.json')}
(BASE/'configs_data.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2),encoding='utf-8')

def compact(obj):
    if isinstance(obj,dict):return {k:compact(v) for k,v in obj.items() if k not in ('actions','results','episodes_detail','evaluations','candidates','frames','trace','body_frames','telemetry','per_frame','source_actions','model','optimizer')}
    if isinstance(obj,list):return [compact(v) for v in obj[:20]]
    return obj

summary={};best_records=[]
def walk(obj,filename,inherited=None):
    if not isinstance(obj,(dict,list)):return
    if isinstance(obj,list):
        for x in obj:walk(x,filename,inherited)
        return
    cp=obj.get('checkpoint',inherited)
    if obj.get('deterministic') is False or obj.get('stochastic') is True:
        return
    if cp and obj.get('deterministic') is True and obj.get('successes',0)>0 and obj.get('success_time_min') is not None:
        best_records.append({'file':filename,'checkpoint':str(cp),'time_s':obj['success_time_min'],
            'long_jump_m':None,'distance_m':obj.get('distance_max'),'deterministic_explicit':True,
            'stochastic_explicit':False,'global_step':obj.get('global_step'),'update':obj.get('update')})
    elif cp and obj.get('deterministic') is True and obj.get('successes',0)>0:
        times=[r['time'] for r in obj.get('episodes_detail',[]) if r.get('success') and 'time' in r]
        value=min(times) if times else obj.get('success_time_mean')
        if value is not None:
            best_records.append({'file':filename,'checkpoint':str(cp),'time_s':value,'long_jump_m':None,
                'distance_m':obj.get('distance_max'),'deterministic_explicit':True,'stochastic_explicit':False,
                'global_step':obj.get('global_step'),'update':obj.get('update')})
    if cp and obj.get('success') is True and ('time_s' in obj or 'time' in obj):
        best_records.append({'file':filename,'checkpoint':str(cp),'time_s':obj.get('time_s',obj.get('time')),
            'long_jump_m':obj.get('long_jump_m'),'distance_m':obj.get('distance_m',obj.get('distance')),
            'deterministic_explicit':obj.get('deterministic'),'stochastic_explicit':obj.get('stochastic'),
            'global_step':obj.get('global_step'),'update':obj.get('update')})
    for k,v in obj.items():
        if k not in ('actions','episodes_detail','results','frames','candidates','evaluations','telemetry','trace'):walk(v,filename,cp)

for p in sorted((WORK/'verification').glob('*.json')):
    try:obj=json.loads(p.read_text(encoding='utf-8-sig'))
    except Exception:continue
    if any(s in p.stem for s in ('summary','comparison','report','training_time','training_result','timing','validation')):summary[p.name]=compact(obj)
    if not any(s in p.stem.lower() for s in ('stoch','sampled','random','search','refin','perturb')):walk(obj,p.name)
(BASE/'report_summaries.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(BASE/'deterministic_candidates.json').write_text(json.dumps(best_records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'event_runs':len(events),'configs':len(configs),'summaries':len(summary),'candidate_records':len(best_records),'main_runs':[{k:v for k,v in r.items() if k in ('name','first_step','last_step')} for r in runs if r['name'].startswith('ppo_') and 'smoke' not in r['name']]},ensure_ascii=False))
