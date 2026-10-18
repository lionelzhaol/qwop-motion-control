"""Verify and restore the complete split zstd tar using Windows 11 built-in tar."""
from pathlib import Path
import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import tempfile

def detach_hardlinks(root):
    """Restore independent files without traversing symlinks or junctions."""
    count = 0
    for current, directories, files in os.walk(root, followlinks=False):
        base = Path(current)
        directories[:] = [name for name in directories if not (
            getattr((base / name).lstat(), 'st_file_attributes', 0)
            & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ) and not (base / name).is_symlink()]
        for name in files:
            path = base / name
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink <= 1 or
                    getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                continue
            fd, temporary = tempfile.mkstemp(prefix='.qwop-restore-', dir=base)
            os.close(fd)
            try:
                shutil.copy2(path, temporary)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            count += 1
    return count

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
    print('Detached', detach_hardlinks(destination / 'qwop-workspace'),
          'hard links into independent files', flush=True)
    print('Restored to', destination / 'qwop-workspace')

if __name__ == '__main__':
    main()
