"""Inventory every workspace file without following external directory junctions."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
import stat
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'handoff_20261018'
MANIFEST = OUT / 'snapshot_manifest.json'
VOLATILE = {'snapshot_manifest.json', 'pack_progress.json', 'upload_progress.json', 'remote_release.json',
    'SHA256SUMS.txt', 'git_publish_report.json', 'git_main_files.json', 'archive_verify.json'}

def excluded(relative):
    parts = relative.parts
    return '.git' in parts or (parts and parts[0] == 'handoff_20261018' and
        (len(parts) > 1 and (parts[1] == 'archive' or parts[1] in VOLATILE)))

def enumerate_files():
    files, links, dirs = [], [], []
    for current, subdirs, names in os.walk(ROOT, followlinks=False):
        base = Path(current)
        keep = []
        for name in sorted(subdirs):
            p = base / name
            rel = p.relative_to(ROOT)
            if excluded(rel):
                continue
            st = p.lstat()
            if st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                links.append({'path': rel.as_posix(), 'target': os.readlink(p), 'action': 'recorded-only; external target not traversed'})
            else:
                keep.append(name)
                dirs.append(rel.as_posix())
        subdirs[:] = keep
        for name in sorted(names):
            p = base / name
            rel = p.relative_to(ROOT)
            if not excluded(rel):
                if p.is_symlink():
                    links.append({'path': rel.as_posix(), 'target': os.readlink(p), 'action': 'recorded-only'})
                else:
                    files.append(rel.as_posix())
    return sorted(files), links, dirs

def digest(relative):
    p = ROOT / relative
    st = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    if p.stat().st_size != st.st_size or p.stat().st_mtime_ns != st.st_mtime_ns:
        raise RuntimeError('File changed while inventorying: ' + relative)
    return {'path': relative, 'size': st.st_size, 'sha256': h.hexdigest(), 'mtime_ns': st.st_mtime_ns}

def inventory():
    files, links, dirs = enumerate_files()
    (OUT / 'directory_links.json').write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding='utf-8')
    files, _, dirs = enumerate_files()
    started = time.monotonic()
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, record in enumerate(pool.map(digest, files), 1):
            records.append(record)
            if index % 4000 == 0:
                print(f'HASH {index}/{len(files)}', flush=True)
    result = {'snapshot_date': '2026-10-18', 'workspace': str(ROOT), 'archive_root': 'qwop-workspace',
        'file_count': len(records), 'total_bytes': sum(x['size'] for x in records), 'files': records, 'directories': dirs,
        'directory_links': links, 'exclusions': ['Git administrative .git paths (history preserved separately in bundle and branch)',
        'Generated snapshot archive parts and volatile transfer reports (not original project contents)'],
        'hash_elapsed_seconds': time.monotonic() - started}
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('file_count', 'total_bytes', 'hash_elapsed_seconds')}, ensure_ascii=False), flush=True)

def emit_tar():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    seen = {}
    emitted = duplicate_bytes = 0
    started = time.monotonic()
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|', format=tarfile.PAX_FORMAT) as archive:
        for directory in manifest['directories']:
            item = tarfile.TarInfo('qwop-workspace/' + directory)
            item.type = tarfile.DIRTYPE
            item.mode = 0o755
            archive.addfile(item)
        for i, record in enumerate(manifest['files'], 1):
            p = ROOT / record['path']
            st = p.stat()
            if st.st_size != record['size'] or st.st_mtime_ns != record['mtime_ns']:
                raise RuntimeError('Source changed after snapshot manifest: ' + record['path'])
            name = 'qwop-workspace/' + record['path']
            info = archive.gettarinfo(str(p), arcname=name)
            key = (record['sha256'], record['size'])
            if key in seen and record['size']:
                info.type = tarfile.LNKTYPE
                info.linkname = seen[key]
                info.size = 0
                archive.addfile(info)
                duplicate_bytes += record['size']
            else:
                with p.open('rb') as stream:
                    archive.addfile(info, stream)
                seen[key] = name
            emitted += record['size']
            if i % 2000 == 0 or i == manifest['file_count']:
                print(f'TAR {i}/{manifest["file_count"]} sourceGiB={emitted/2**30:.2f} duplicateGiB={duplicate_bytes/2**30:.2f} seconds={time.monotonic()-started:.1f}', file=sys.stderr, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['inventory', 'tar'])
    args = parser.parse_args()
    inventory() if args.command == 'inventory' else emit_tar()
