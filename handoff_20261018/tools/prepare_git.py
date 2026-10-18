"""Stage the browsable workspace without converting the inner repository to a gitlink."""
from pathlib import Path
import json
import os
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'handoff_20261018'
SELECTED = {'ppo_classic100m_phase_ppo.pt', 'ppo_classic100m_phase_fast.pt', 'bc_classic100m_shadow.pt',
    'ppo_classic_actor_unfreeze_best_20261015.pt', 'long_jump_distance_elite_best_20261013.pt',
    'ppo_foot_running_v1_update_000580.pt', 'ppo_foot_running_v3.pt'}
STATUS = {'pack_progress.json', 'upload_progress.json', 'remote_release.json', 'git_publish_report.json', 'git_main_files.json'}
ENV = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
PREFIX = ['git', '-c', 'safe.directory=' + ROOT.as_posix(), '-c', 'core.autocrlf=false']

def git(*args, data=None):
    result = subprocess.run(PREFIX + list(args), cwd=ROOT, env=ENV, input=data, capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).decode('utf-8', 'replace')[:4000])
    return result.stdout

def keep(rel):
    p = rel.parts
    if '.git' in p or 'node_modules' in p or '__pycache__' in p or rel.suffix == '.pyc':
        return False
    if p[0] in ('.cache', '.tmp', '.tools', 'downloads'):
        return False
    if len(p) > 1 and p[0] == 'QWOP_rl_training_public':
        if p[1] in ('.venv', 'build'):
            return False
        if p[1] == 'checkpoints' and len(p) > 2 and rel.name not in SELECTED:
            return False
    if p[0] == 'handoff_20261018' and len(p) > 1 and (p[1] == 'archive' or p[1] in STATUS):
        return False
    return True

files = []
for current, dirs, names in os.walk(ROOT, followlinks=False):
    base = Path(current)
    dirs[:] = [name for name in dirs if keep((base/name).relative_to(ROOT)) and
        not ((base/name).lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)]
    for name in names:
        p = base / name
        if keep(p.relative_to(ROOT)):
            if p.stat().st_size >= 100 * 1024 * 1024:
                raise RuntimeError('Oversized file belongs in the complete Release snapshot: ' + str(p))
            files.append(p.relative_to(ROOT).as_posix())
files.sort()
parent_check = subprocess.run(PREFIX + ['rev-parse', '--verify', 'HEAD'], cwd=ROOT, capture_output=True)
if parent_check.returncode:
    git('update-ref', 'refs/heads/main', 'refs/remotes/origin/main')
git('read-tree', 'HEAD')
hashes = git('hash-object', '-w', '--no-filters', '--stdin-paths', data=('\n'.join(files)+'\n').encode('utf-8')).decode('ascii').splitlines()
if len(hashes) != len(files):
    raise RuntimeError('File/blob count differs')
entries = b''.join(('100644 '+sha+'\t'+name+'\0').encode('utf-8') for name,sha in zip(files,hashes))
git('update-index', '-z', '--index-info', data=entries)
git('-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check', '--',
    'README.md', '.gitignore', '.gitattributes', 'handoff_20261018/tools')
result = git('commit', '-m', 'chore:整理小组工程、精选成果与完整工作区交接说明')
commit = git('rev-parse', 'HEAD').decode().strip()
report = {'commit': commit, 'browsable_files': len(files), 'browsable_source_bytes': sum((ROOT/p).stat().st_size for p in files),
    'latest_training_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/'QWOP_rl_training_public', text=True).strip()}
(OUT/'git_main_files.json').write_text(json.dumps({'files':files},ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'git_publish_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
