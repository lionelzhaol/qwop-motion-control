"""Run author PPO with periodic independent native scoring and a durable log."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys,os,time,json,re,math,argparse
from evaluate_long_jump import evaluate,ROOT,BASE

parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--name',required=True)
args=parser.parse_args();started=time.perf_counter();started_utc=datetime.now(timezone.utc).isoformat()
cfg=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
env=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
process=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/train_ppo.py','--config',args.config],cwd=ROOT,env=env,
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
reports=[];best=None;rows=0
with (BASE/(args.name+'_training.log')).open('w',encoding='utf-8') as log:
    for line in process.stdout:
        log.write(line);log.flush()
        if line.startswith('update='):
            rows+=1;fields=dict(re.findall(r'(\w+)=([^\s]+)',line))
            for key in ['policy_loss','value_loss','entropy','kl','clipfrac']:
                assert math.isfinite(float(fields[key])),line
            if rows%16==0: print(line.strip(),flush=True)
        elif line.startswith('saved_checkpoint='):
            path=Path(line.strip().split('=',1)[1])
            if '_update_' in path.stem:
                report=evaluate(path)
                (BASE/(path.stem+'_jump_eval.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
                reports.append(report)
                if report['success'] and (best is None or report['long_jump_m']>best['long_jump_m']): best=report
                summary=dict(evaluated_checkpoints=len(reports),best=best,
                    successful_checkpoints=sum(r['success'] for r in reports),all_logged_losses_finite=True,
                    scope='One native greedy episode per saved policy; not repeated-trial success rate')
                (BASE/(args.name+'_comparison.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
                print(json.dumps({k:report[k] for k in ['update','success','long_jump_m','time_s','approach_progress_m']}|
                    {'best_successful_jump_m':best['long_jump_m'] if best else None}),flush=True)
        elif 'Traceback' in line or 'Error' in line: print(line.strip(),flush=True)
returncode=process.wait()
result=dict(started_utc=started_utc,ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
    config=args.config,returncode=returncode,updates_logged=rows,transitions=rows*cfg['num_envs']*cfg['rollout_steps'])
(BASE/(args.name+'_training_time.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
assert returncode==0 and rows==cfg['updates']
