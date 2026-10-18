"""Upload all complete snapshot parts concurrently to the requested private release."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import threading
import time
import urllib.parse
from github_publish import API, OUT, REPO, TAG, sha256

lock = threading.Lock()

class ProgressReader:
    def __init__(self, stream, name, size):
        self.stream, self.name, self.size = stream, name, size
        self.sent = self.last = 0
        self.start = time.monotonic()
    def read(self, n=-1):
        data = self.stream.read(n)
        self.sent += len(data)
        if self.sent-self.last >= 64*1024*1024:
            self.last = self.sent
            with lock:
                print(f'PROGRESS {self.name} {self.sent/2**20:.0f}/{self.size/2**20:.0f} MiB elapsed={time.monotonic()-self.start:.0f}s', flush=True)
        return data
    def close(self):
        self.stream.close()

class ProgressPath:
    def __init__(self, path): self.path = path
    def stat(self): return self.path.stat()
    def open(self, mode): return ProgressReader(self.path.open(mode), self.path.name, self.path.stat().st_size)

api = API()
api.verify()
if not (OUT/'SHA256SUMS.txt').is_file():
    raise RuntimeError('Archive has not finished')
if not json.loads((OUT/'archive_verify.json').read_text())['passed']:
    raise RuntimeError('Full snapshot verification is required before upload')
releases = api.call('GET', f'/repos/{REPO}/releases?per_page=100')
release = next((x for x in releases if x['tag_name']==TAG), None)
if not release:
    release = api.call('POST', f'/repos/{REPO}/releases', {'tag_name':TAG,'target_commitish':'main',
        'name':'完整工作区快照（2026-10-18）','draft':True,'prerelease':False,
        'body':'完整工作区快照，保留全部模型、虚拟环境、安装包、缓存与实验资料。下载全部分卷并按 RESTORE.md 恢复；SHA256SUMS.txt 和 snapshot_manifest.json 提供逐卷、逐文件校验。外部目录联接仅记录清单。原训练历史位于 training-history 分支及 Git bundle。'})
files = sorted((OUT/'archive').glob('qwop-workspace-20261018.tar.zst.part*'))
files += [OUT/n for n in ('SHA256SUMS.txt','RESTORE.md','snapshot_manifest.json','directory_links.json','archive_verify.json')]
existing = {x['name']:x for x in release.get('assets',[])}
uploaded = {x['name']:x for x in release.get('assets',[])}

def upload(file):
    local_hash = sha256(file)
    old = existing.get(file.name)
    if old:
        if old.get('state')=='uploaded' and old['size']==file.stat().st_size and (not old.get('digest') or old['digest']=='sha256:'+local_hash):
            print('SKIP '+file.name,flush=True)
            return old
        raise RuntimeError('Existing asset is incomplete or different: '+file.name)
    url = release['upload_url'].split('{',1)[0]+'?name='+urllib.parse.quote(file.name)
    print('UPLOAD '+file.name+' '+str(file.stat().st_size),flush=True)
    result = api.call('POST',url,file=ProgressPath(file))
    if result['state']!='uploaded' or result['size']!=file.stat().st_size or (result.get('digest') and result['digest']!='sha256:'+local_hash):
        raise RuntimeError('Remote asset validation failed: '+file.name)
    print('DONE '+file.name,flush=True)
    return result

errors = []
with ThreadPoolExecutor(max_workers=3) as pool:
    future_files = {pool.submit(upload,p):p for p in files}
    for future in as_completed(future_files):
        try:
            result = future.result()
            uploaded[result['name']] = result
            state={'release_id':release['id'],'tag':TAG,'completed_assets':len(uploaded),'expected_assets':len(files),
                'uploaded':[{'name':a['name'],'size':a['size'],'digest':a.get('digest'),'state':a['state']} for a in uploaded.values()]}
            (OUT/'upload_progress.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        except Exception as error:
            name = future_files[future].name
            print('FAILED '+name+' '+str(error)[:800],flush=True)
            errors.append({'name':name,'error':str(error)[:1500]})
if errors:
    (OUT/'upload_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    raise RuntimeError('Some snapshot assets did not upload; successful assets are retained for resuming')
print('ALL_ASSETS_UPLOADED '+str(len(files)),flush=True)
