"""Use Windows curl's streaming upload; keep authentication out of arguments/files."""
from pathlib import Path
import json
import subprocess
import sys
import threading
import time
import urllib.parse
from github_publish import API, OUT, REPO, TAG, PROXY

def curl_upload(api, release, file):
    url = release['upload_url'].split('{',1)[0]+'?name='+urllib.parse.quote(file.name)
    if '\n' in api.token or '\r' in api.token or '"' in api.token:
        raise RuntimeError('Invalid credential format')
    config = 'header = "Authorization: Bearer '+api.token+'"\n'
    command = ['curl.exe','--config','-','--noproxy','*','--http1.1','--request','POST',
        '--upload-file',str(file),'--header','Content-Type: application/octet-stream',
        '--header','Accept: application/vnd.github+json','--header','Expect:',
        '--connect-timeout','20','--speed-time','90','--speed-limit','1024','--max-time','3600',
        '--write-out','\n%{http_code}',url]
    process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        text=True,encoding='utf-8',errors='replace')
    process.stdin.write(config);process.stdin.close()
    errors=[]
    def progress():
        last=0
        for line in process.stderr:
            errors.append(line)
            if time.monotonic()-last>12 and line.strip():
                last=time.monotonic()
                print('CURL '+file.name+' '+line.strip()[:150],flush=True)
    t=threading.Thread(target=progress,daemon=True);t.start()
    output=process.stdout.read();code=process.wait();t.join()
    text,status=output.rsplit('\n',1) if '\n' in output else (output,'0')
    if code or status!='201':
        detail=(''.join(errors)[-1200:]+' '+text[:1200]).replace(api.token,'[redacted]')
        raise RuntimeError('curl upload failed '+str(code)+' HTTP '+status+': '+detail)
    return json.loads(text)

if __name__=='__main__':
    api=API();api.verify()
    release=next(x for x in api.call('GET',f'/repos/{REPO}/releases?per_page=100') if x['tag_name']==TAG)
    file=Path(sys.argv[1])
    result=curl_upload(api,release,file)
    print(json.dumps({k:result.get(k) for k in ('name','size','state','digest')},ensure_ascii=False),flush=True)
