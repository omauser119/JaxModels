"""Bounded, hash-checked training stage cache. Complete shards survive interruption."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


class ChunkCache:
    def __init__(self, root, cwd, binary_hashes):
        self.root = Path(root)
        self.cwd = Path(cwd)
        self.binary_hashes = binary_hashes
        self.state_path = self.root / 'stages.json'
        self.stages = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def checked(self, name):
        if name not in self.stages: return False
        path = self.root / name
        if not path.is_file() or sha(path) != self.stages[name]:
            raise ValueError('cached stage integrity failure: ' + name)
        return True

    def register(self, name):
        self.stages[name] = sha(self.root / name)
        atomic_json(self.state_path, self.stages)

    def run(self, name, command, resume_partial=False):
        if self.checked(name): return
        for path, expected in self.binary_hashes.items():
            if sha(path) != expected: raise ValueError('binary changed during training')
        destination = self.root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + '.partial')
        atomic_json(self.root / 'progress.json', {'phase': name, 'updated_unix': time.time(), 'complete': False})
        print('Stage: ' + name, flush=True)
        if resume_partial:command=[*command,'--resume-labels',str(partial)]
        with partial.open('ab' if resume_partial else 'wb') as stream:
            subprocess.run(command, stdout=stream, check=True, cwd=self.cwd)
        partial.replace(destination)
        self.register(name)

    def join(self, name, names):
        if self.checked(name): return
        destination = self.root / name
        partial = destination.with_name(destination.name + '.partial')
        with partial.open('wb') as output:
            for part in names:
                if not self.checked(part): raise ValueError('missing shard: ' + part)
                with (self.root / part).open('rb') as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        partial.replace(destination)
        self.register(name)
