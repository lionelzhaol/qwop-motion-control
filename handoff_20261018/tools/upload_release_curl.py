"""Resumable, concurrent curl upload of the complete private workspace snapshot."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import time
from github_publish import API, OUT, REPO, TAG, sha256
from curl_upload_asset import curl_upload

api=API();api.verify()
release=next(x for x in api.call('GET',f'/repos/{REPO}/releases?per_page=100') if x['tag_name']==TAG)
expected_files=sorted((OUT/'archive').glob('qwop-workspace-20261018.tar.zst.part*'))
expected_files += [OUT/n for n in ('SHA256SUMS.txt','RESTORE.md','snapshot_manifest.json','directory_links.json','archive_verify.json')]
if not json.loads((OUT/'archive_verify.json').read_text())['passed']:
    raise RuntimeError('Archive verification must pass first')

def check_asset(asset,file,digest):
    return asset.get('state')=='uploaded' and asset['size']==file.stat().st_size and (not asset.get('digest') or asset['digest']=='sha256:'+digest)

def upload(file):
    digest=sha256(file)
    for attempt in range(1,4):
        assets=api.call('GET',f'/repos/{REPO}/releases/{release["id"]}/assets?per_page=100')
        old=next((a for a in assets if a['name']==file.name),None)
        if old:
            if check_asset(old,file,digest):
                print('SKIP '+file.name,flush=True);return old
            # Starter assets reserve the declared size but have no completed content digest.
            # Only remove this task's unfinished assets after its previous uploader has stopped.
            if release['draft'] and old['state']=='starter' and old.get('digest') is None:
                api.call('DELETE',f'/repos/{REPO}/releases/assets/{old["id"]}')
            else:
                raise RuntimeError('Existing uploaded asset differs: '+file.name)
        try:
            print(f'UPLOAD {file.name} attempt={attempt}',flush=True)
            result=curl_upload(api,release,file)
            if not check_asset(result,file,digest):raise RuntimeError('Remote digest or size mismatch')
            print('DONE '+file.name,flush=True);return result
        except Exception as e:
            print('RETRY '+file.name+' '+str(e)[-700:],flush=True)
            if attempt==3:raise
            time.sleep(3*attempt)

uploaded={}
errors=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    jobs={pool.submit(upload,p):p for p in expected_files}
    for job in as_completed(jobs):
        try:
            asset=job.result();uploaded[asset['name']]=asset
            state={'release_id':release['id'],'tag':TAG,'completed_assets':len(uploaded),'expected_assets':len(expected_files),
                'uploaded':[{'name':a['name'],'size':a['size'],'digest':a.get('digest'),'state':a['state']} for a in uploaded.values()]}
            (OUT/'upload_progress.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        except Exception as e:
            errors.append({'name':jobs[job].name,'error':str(e)[-1500:]})
            print('FAILED '+jobs[job].name+' '+str(e)[-600:],flush=True)
if errors:
    (OUT/'upload_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    raise RuntimeError('Incomplete snapshot assets; uploaded parts retained for resuming')
print('ALL_ASSETS_UPLOADED '+str(len(expected_files)),flush=True)
