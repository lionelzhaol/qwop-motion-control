"""Stream-verify every archived file without restoring 26 GiB to disk."""
from pathlib import Path
import hashlib
import json
import sys
import tarfile

OUT = Path(__file__).resolve().parents[1]
manifest = json.loads((OUT/'snapshot_manifest.json').read_text(encoding='utf-8'))
expected = {'qwop-workspace/'+x['path']:x for x in manifest['files']}
verified = {}
total = 0
with tarfile.open(fileobj=sys.stdin.buffer, mode='r|') as archive:
    for member in archive:
        if member.isdir():
            continue
        if member.name not in expected:
            raise RuntimeError('Unexpected file in archive: '+member.name)
        if member.name in verified:
            raise RuntimeError('Duplicate archive entry: '+member.name)
        record = expected[member.name]
        if member.islnk():
            if member.linkname not in verified or verified[member.linkname] != record['sha256']:
                raise RuntimeError('Invalid content deduplication link: '+member.name)
            actual = verified[member.linkname]
        elif member.isfile():
            if member.size != record['size']:
                raise RuntimeError('Archive size mismatch: '+member.name)
            h = hashlib.sha256()
            stream = archive.extractfile(member)
            for b in iter(lambda: stream.read(8*1024*1024), b''):
                h.update(b)
            actual = h.hexdigest()
        else:
            raise RuntimeError('Unexpected archive entry type: '+member.name)
        if actual != record['sha256']:
            raise RuntimeError('Archive SHA256 mismatch: '+member.name)
        verified[member.name] = actual
        total += record['size']
        if len(verified) % 4000 == 0:
            print('VERIFY',len(verified),'/',len(expected),flush=True)
if set(verified) != set(expected):
    raise RuntimeError('Archive is missing manifest files')
report = {'passed':True,'files_verified':len(verified),'logical_bytes_verified':total,'method':'full tar entry and SHA256 verification, including deduplicated content links'}
(OUT/'archive_verify.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
