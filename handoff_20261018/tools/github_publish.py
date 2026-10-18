"""Publish only to the user-requested private repository; credentials stay in memory."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'handoff_20261018'
REPO = 'lionelzhaol/qwop-motion-control'
TAG = 'workspace-snapshot-20261018'
PROXY = 'http://127.0.0.1:7897'

def credential():
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
    result = subprocess.run(['git', '-c', 'credential.interactive=never', 'credential', 'fill'],
        input='protocol=https\nhost=github.com\n\n', text=True, encoding='utf-8',
        capture_output=True, cwd=ROOT / 'QWOP_rl_training_public', env=env, timeout=25)
    if result.returncode:
        raise RuntimeError('Git credential manager has no usable non-interactive GitHub credential')
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not fields.get('password'):
        raise RuntimeError('Git credential manager returned no GitHub credential')
    return fields['password']

class API:
    def __init__(self):
        self.token = credential()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY}))

    def call(self, method, route, payload=None, file=None):
        url = route if route.startswith('https://') else 'https://api.github.com' + route
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname not in ('api.github.com', 'uploads.github.com'):
            raise ValueError('Unexpected GitHub API host')
        if file is None:
            # Windows curl uses the system TLS backend, which is reliable with the
            # existing local proxy; Python/OpenSSL handshakes were intermittent.
            def quoted(value):
                return value.replace('\\', '\\\\').replace('"', '\\"')
            config = 'header = "Authorization: Bearer ' + quoted(self.token) + '"\n'
            config += 'header = "Accept: application/vnd.github+json"\n'
            config += 'header = "X-GitHub-Api-Version: 2022-11-28"\n'
            if payload is not None:
                config += 'header = "Content-Type: application/json"\n'
                config += 'data-binary = "' + quoted(json.dumps(payload, ensure_ascii=True)) + '"\n'
            attempts = 3 if method == 'GET' else 1
            for attempt in range(attempts):
                result = subprocess.run(['curl.exe', '--config', '-', '--proxy', PROXY,
                    '--http1.1', '--request', method, '--connect-timeout', '15', '--max-time', '90',
                    '--silent', '--show-error', '--write-out', '\n%{http_code}', url],
                    input=config, text=True, encoding='utf-8', capture_output=True)
                raw, status = result.stdout.rsplit('\n', 1) if '\n' in result.stdout else (result.stdout, '0')
                if result.returncode == 0 and status.isdigit() and 200 <= int(status) < 300:
                    return json.loads(raw) if raw else None
                if attempt + 1 < attempts:
                    time.sleep(2)
                    continue
                detail = (result.stderr[-800:] + ' ' + raw[:1600]).replace(self.token, '[redacted]')
                raise RuntimeError('GitHub API curl HTTP ' + status + ': ' + detail)
        headers = {'Authorization': 'Bearer ' + self.token, 'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'QWOP-private-workspace-handoff'}
        if file:
            headers.update({'Content-Type': 'application/octet-stream', 'Content-Length': str(file.stat().st_size)})
            data = file.open('rb')
        else:
            data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
            if data is not None:
                headers['Content-Type'] = 'application/json'
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with self.opener.open(req, timeout=180) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            detail = error.read(1600).decode('utf-8', 'replace')
            raise RuntimeError(f'GitHub API {error.code}: {detail}') from None
        finally:
            if file:
                data.close()

    def verify(self):
        repo = self.call('GET', '/repos/' + REPO)
        if repo.get('full_name', '').lower() != REPO or not repo.get('private'):
            raise RuntimeError('Destination must be exactly the requested private repository')
        return repo

def sha256(file):
    h = hashlib.sha256()
    with file.open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['info', 'release-upload', 'release-info', 'publish'])
    args = parser.parse_args()
    api = API()
    repo = api.verify()
    if args.command == 'info':
        print(json.dumps({k: repo.get(k) for k in ('full_name', 'private', 'default_branch', 'size', 'permissions', 'html_url')}, ensure_ascii=False), flush=True)
        (OUT / 'remote_repository.json').write_text(json.dumps({k: repo.get(k) for k in ('full_name', 'private', 'default_branch', 'size', 'html_url')}, indent=2), encoding='utf-8')
        return
    releases = api.call('GET', f'/repos/{REPO}/releases?per_page=100')
    release = next((x for x in releases if x['tag_name'] == TAG), None)
    if args.command == 'release-upload':
        if release is None:
            release = api.call('POST', f'/repos/{REPO}/releases', {'tag_name': TAG, 'target_commitish': 'main',
                'name': '完整工作区快照（2026-10-18）', 'draft': True, 'prerelease': False,
                'body': '完整工作区快照，包含所有中间模型、虚拟环境、安装包及缓存。按 RESTORE.md 合并分卷后解压，SHA256SUMS.txt 提供校验。外部目录联接不递归复制；原训练代码历史保留于 training-history 分支及 Git bundle。'})
        files = sorted((OUT / 'archive').glob('qwop-workspace-20261018.tar.zst.part*'))
        files += [OUT / n for n in ('SHA256SUMS.txt', 'RESTORE.md', 'snapshot_manifest.json', 'directory_links.json')]
        if not files or any(not x.is_file() for x in files):
            raise RuntimeError('Complete local archive and manifests must exist before upload')
        for index, file in enumerate(files, 1):
            expected_hash = sha256(file)
            existing = next((x for x in release.get('assets', []) if x['name'] == file.name), None)
            if existing:
                if existing['size'] != file.stat().st_size or (existing.get('digest') and existing['digest'] != 'sha256:' + expected_hash):
                    raise RuntimeError(f'Existing release asset differs: {file.name}')
                print(f'SKIP {index}/{len(files)} {file.name}', flush=True)
                continue
            url = release['upload_url'].split('{', 1)[0] + '?name=' + urllib.parse.quote(file.name)
            print(f'UPLOAD {index}/{len(files)} {file.name} {file.stat().st_size}', flush=True)
            result = api.call('POST', url, file=file)
            if result.get('state') != 'uploaded' or result.get('size') != file.stat().st_size:
                raise RuntimeError('Asset upload did not finalize: ' + file.name)
            if result.get('digest') and result['digest'] != 'sha256:' + expected_hash:
                raise RuntimeError('Remote asset SHA256 mismatch: ' + file.name)
            release.setdefault('assets', []).append(result)
            (OUT / 'upload_progress.json').write_text(json.dumps({'release_id': release['id'], 'tag': TAG,
                'uploaded': [{'name': x['name'], 'size': x['size'], 'digest': x.get('digest'), 'state': x['state']} for x in release['assets']]}, indent=2), encoding='utf-8')
            print('DONE ' + file.name, flush=True)
    elif args.command == 'publish':
        if not release:
            raise RuntimeError('Snapshot release does not exist')
        files = sorted((OUT / 'archive').glob('qwop-workspace-20261018.tar.zst.part*'))
        files += [OUT / n for n in ('SHA256SUMS.txt', 'RESTORE.md', 'snapshot_manifest.json',
            'directory_links.json', 'archive_verify.json')]
        if len(files) != 26 or not json.loads((OUT / 'archive_verify.json').read_text())['passed']:
            raise RuntimeError('Expected 21 verified archive parts and five metadata files')
        assets = api.call('GET', f'/repos/{REPO}/releases/{release["id"]}/assets?per_page=100')
        for file in files:
            asset = next((a for a in assets if a['name'] == file.name), None)
            if not asset or asset['state'] != 'uploaded' or asset['size'] != file.stat().st_size:
                raise RuntimeError('Cannot publish an incomplete snapshot: ' + file.name)
            if asset.get('digest') != 'sha256:' + sha256(file):
                raise RuntimeError('Remote asset SHA256 mismatch: ' + file.name)
        release = api.call('PATCH', f'/repos/{REPO}/releases/{release["id"]}', {'draft': False})
    if release:
        summary = {k: release.get(k) for k in ('id', 'html_url', 'tag_name', 'draft', 'published_at')}
        summary['assets'] = [{'name': x['name'], 'size': x['size'], 'digest': x.get('digest')} for x in release.get('assets', [])]
        (OUT / 'remote_release.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    else:
        print('No snapshot release yet', flush=True)

if __name__ == '__main__':
    main()
