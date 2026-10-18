"""Verify and restore the complete split zstd tar using Windows 11 built-in tar."""
from pathlib import Path
import argparse
import hashlib
import shutil
import subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', type=Path, required=True, help='Folder containing all parts and SHA256SUMS.txt')
    ap.add_argument('--destination', type=Path, required=True, help='An empty destination directory')
    args = ap.parse_args()
    source = args.parts.resolve()
    destination = args.destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Destination must be empty')
    tar = shutil.which('tar')
    if not tar:
        raise RuntimeError('Install a tar implementation with zstd support, or use Windows 11 built-in tar')
    records = []
    for line in (source / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        if Path(name).name != name:
            raise ValueError('Invalid part filename')
        p = source / name
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(b)
        if h.hexdigest() != expected:
            raise ValueError('SHA256 mismatch: ' + name)
        records.append(p)
        print('Verified', name, flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([tar, '-x', '-f', '-', '-C', str(destination)], stdin=subprocess.PIPE)
    try:
        for p in records:
            with p.open('rb') as stream:
                shutil.copyfileobj(stream, process.stdin, 8 * 1024 * 1024)
        process.stdin.close()
        if process.wait():
            raise RuntimeError('tar extraction failed')
    finally:
        if process.poll() is None:
            process.terminate()
    print('Restored to', destination / 'qwop-workspace')

if __name__ == '__main__':
    main()
