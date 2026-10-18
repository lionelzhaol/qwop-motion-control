"""Read every training update, so progress never relies on the last line alone."""
from pathlib import Path
import json
import re

base=Path(__file__).resolve().parent
for arm in ('control','speed1200','speed3600'):
    for stage in ('warm','train'):
        path=base/f'ppo_classic_speed_{arm}_{stage}_20261015.log'
        if not path.exists():continue
        rows=[];updates=[]
        for line in path.read_text(encoding='utf-8').splitlines():
            match=re.match(r'update=(\d+)/(\d+)',line)
            if not match:continue
            row={}
            for key,value in re.findall(r'(\w+)=([^\s]+)',line):
                try:row[key]=float(value)
                except ValueError:pass
            rows.append(row);updates.append(tuple(map(int,match.groups())))
        if not rows:continue
        print(json.dumps(dict(arm=arm,stage=stage,progress=f'{updates[-1][0]}/{updates[-1][1]}',
            qualified_finishes=sum(int(r.get('qualified_finishes',0)) for r in rows),
            faster_finishes=sum(int(r.get('faster_finishes',0)) for r in rows),
            best_time_s=min((r['best_finish_s'] for r in rows if 'best_finish_s' in r),default=None),
            updates_with_varied_success_times=sum(r.get('mean_finish_s',0)>r.get('best_finish_s',0)+1e-5 for r in rows)),ensure_ascii=False))
