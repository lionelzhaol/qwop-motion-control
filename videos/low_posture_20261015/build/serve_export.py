"""Local-only original QWOP replay renderer and video receiver."""
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
SOURCE = Path(r'E:\Work\强化学习小项目\QWOP_rl_training_public\replays\foot_running_v1_eval_000580.bin')
shutil.copyfile(SOURCE, ROOT / 'replay.bin')
STATUS = {'stage': 'waiting', 'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
NAME = 'QWOP_早期膝盖蠕动_113秒83'

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == '/api/status':
            b = json.dumps(STATUS, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            super().do_GET()

    def do_POST(self):
        n = int(self.headers.get('Content-Length', '0'))
        if n > 256 * 1024 * 1024 or self.path not in ('/api/video', '/api/report'):
            self.send_error(400)
            return
        b = self.rfile.read(n)
        if self.path == '/api/report':
            report = json.loads(b)
            STATUS.update(report)
            (OUT / 'export_report.json').write_text(json.dumps(STATUS, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            ext = '.mp4' if 'video/mp4' in self.headers.get('Content-Type', '') else '.webm'
            target = OUT / (NAME + ext)
            target.write_bytes(b)
            STATUS.update({'file': str(target), 'bytes': len(b), 'video_sha256': hashlib.sha256(b).hexdigest()})
            (OUT / 'export_report.json').write_text(json.dumps(STATUS, ensure_ascii=False, indent=2), encoding='utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
url = f'http://127.0.0.1:{server.server_port}/'
(OUT / 'server.json').write_text(json.dumps({'url': url}), encoding='utf-8')
print(url, flush=True)
server.serve_forever()
