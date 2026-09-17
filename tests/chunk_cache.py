import importlib.util
import json
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('chunk_module', ROOT/'scripts/chunk_cache.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);cache=m.ChunkCache(root, ROOT, {})
    cache.run('parts/a', [sys.executable, '-c', 'print("a")'])
    # A completed shard must never execute a replacement command on resume.
    resumed=m.ChunkCache(root, ROOT, {})
    resumed.run('parts/a', [sys.executable,'-c','raise RuntimeError("must not execute")'])
    try: resumed.run('parts/b',[sys.executable,'-c','print("partial");raise SystemExit(2)']); raise AssertionError()
    except m.subprocess.CalledProcessError: pass
    assert 'parts/b' not in resumed.stages
    resumed.run('parts/b',[sys.executable,'-c','print("b")'])
    resumed.join('all',['parts/a','parts/b'])
    assert (root/'all').read_text()=='a\nb\n'
    (root/'parts/a').write_text('corrupted\n')
    try:resumed.checked('parts/a');raise AssertionError()
    except ValueError:pass
print('Chunk cache: resume, failed partial rerun, bounded join and corruption detection OK')
