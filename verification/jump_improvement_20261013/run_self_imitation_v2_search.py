"""Run the author's policy sampler on CPU with a reproducible equal budget."""
from pathlib import Path
from datetime import datetime,timezone
import contextlib,json,sys,time,runpy
import torch

BASE=Path(__file__).resolve().parent; ROOT=BASE.parents[1]/'QWOP_rl_training_public'
sys.path.insert(0,str(ROOT/'scripts'))
torch.set_num_threads(1)
args=['--checkpoint',str(ROOT/'checkpoints/ppo_long_jump_self_imitation_v2_20261013.pt'),
    '--output',str(ROOT/'replays/long_jump_self_imitation_v2_search_20261013.bin'),
    '--episodes','2048','--num-envs','128','--num-threads','8','--device','cpu',
    '--seed','2026101331','--progress-interval','256']
sys.argv=['search_policy_best.py',*args]
started=datetime.now(timezone.utc); timer=time.perf_counter(); code=None
try:
    with (BASE/'self_imitation_v2_search2048.log').open('w',encoding='utf-8') as logfile:
        with contextlib.redirect_stdout(logfile),contextlib.redirect_stderr(logfile):
            try:
                runpy.run_path(str(ROOT/'scripts/search_policy_best.py'),run_name='__main__')
            except SystemExit as outcome:
                code=outcome.code or 0
                if code: raise
    code=0
finally:
    elapsed=time.perf_counter()-timer
    report=dict(started_utc=started.isoformat(),ended_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=elapsed,returncode=code,args=args,torch_threads=1,device='cpu')
    (BASE/'self_imitation_v2_search2048_time.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
