#!/usr/bin/env python3
"""Download a pinned GGUF and verify its published SHA-256 before activation."""
import hashlib
import json
import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
meta = json.loads((root / 'models/manifest.json').read_text())
target = root / 'models/Jax-1-Abel/backbone.gguf'
target.parent.mkdir(parents=True, exist_ok=True)
partial = target.with_suffix(target.suffix + '.part')

def verified(path):
    if not path.is_file() or path.stat().st_size != meta['size']:
        return False
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest() == meta['sha256']

if target.exists():
    if not verified(target):
        raise SystemExit('Existing model has wrong checksum; move it aside before retrying.')
else:
    if not verified(partial):
        url = "https://github.com/omauser119/JaxModels/releases/download/v0.2.0-pilot.1/backbone.gguf"
        subprocess.run(['curl', '-fL', '--retry', '3', '--continue-at', '-', url, '-o', str(partial)], check=True)
    if not verified(partial):
        raise SystemExit('Downloaded model has wrong size/checksum; not activated.')
    partial.rename(target)
print(f'Verified: {target}')
