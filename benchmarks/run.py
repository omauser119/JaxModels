#!/usr/bin/env python3
"""Label-blind, resumable runner. Does not import or open reference answers."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from service.app import Worker


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def request_hash(request):
    return hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def select_rows(rows, split, suite):
    rows = [row for row in rows if row['split'] == split]
    if suite == 'smoke':
        # First base case per family/language. Fixed selection; never claim this is a full run.
        seen, subset = set(), []
        for row in rows:
            if not row['id'].endswith('-base'):
                continue
            key = row['id'].rsplit('-', 2)[0]
            if key not in seen:
                subset.append(row); seen.add(key)
        rows = subset
    random.Random(19092026).shuffle(rows)
    return rows

async def run(args):
    dataset = Path(args.dataset)
    manifest = json.loads((dataset / 'manifest.json').read_text())
    fingerprint = sha(dataset / 'requests.jsonl')
    if fingerprint != manifest['sha256']['requests.jsonl']:
        raise ValueError('request dataset hash mismatch')
    rows = select_rows([json.loads(s) for s in (dataset / 'requests.jsonl').read_text().splitlines()], args.split, args.suite)
    ids = [row['id'] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty suite or duplicate IDs')
    model = args.model or ('Jax-1-Abel' if args.backend == 'abel' else None)
    if not model:
        raise ValueError('--model is required for Jev/HTTP; use the actual provider model ID')
    metadata = {'format': 'jax.decisionbench.run.v1', 'benchmark_version': manifest['version'],
        'requests_sha256': fingerprint, 'backend': args.backend, 'model': model, 'split': args.split,
        'suite': args.suite, 'expected_ids': ids, 'timeout_seconds': args.timeout, 'retries': 0,
        'runner_sha256': sha(Path(__file__)), 'python': platform.python_version(),
        'platform': platform.platform(), 'machine': platform.machine(), 'cpu_count': os.cpu_count()}
    if args.backend == 'abel':
        metadata['model_config'] = json.loads((ROOT / 'models/Jax-1-Abel/config.json').read_text())
        metadata['binary_sha256'] = sha(ROOT / 'build/jax-native')
    elif args.backend == 'jev':
        metadata['sdk_version'] = importlib.metadata.version('typesafe-sdk')
    else:
        metadata['endpoint'] = args.endpoint
    out = Path(args.output)
    if out.exists() and not args.resume:
        raise ValueError('output exists; use a fresh path or --resume')
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / 'run.json'
    if args.resume:
        old = json.loads(meta_path.read_text())
        for k, v in metadata.items():
            if old.get(k) != v:
                raise ValueError('resume metadata differs: ' + k)
    else:
        meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
    results_path = out / 'predictions.jsonl'
    done = set()
    expected = {r['id']: request_hash(r['request']) for r in rows}
    if results_path.exists():
        # A truncated last line fails closed; explicitly repair it before resuming.
        for line in results_path.read_text().splitlines():
            row = json.loads(line)
            if row['id'] not in expected or row['id'] in done or row['request_sha256'] != expected[row['id']]:
                raise ValueError('invalid existing prediction record')
            done.add(row['id'])
    worker = None
    client = None
    startup = time.monotonic()
    if args.backend == 'abel':
        worker = Worker()
        await worker.start()
    elif args.backend == 'jev':
        from typesafe_sdk import TypeSafeClient, RetryPolicy
        client = TypeSafeClient(model=model, timeout=args.timeout, retry=RetryPolicy(max_retries=0))
    startup = time.monotonic() - startup
    def remote(req):
        if args.backend == 'http':
            headers = {'Content-Type': 'application/json'}
            token = os.environ.get('JAX_SERVICE_TOKEN')
            if token: headers['Authorization'] = 'Bearer ' + token
            call = Request(args.endpoint, data=json.dumps(req).encode(), headers=headers, method='POST')
            with urlopen(call, timeout=args.timeout) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024: raise ValueError('response too large')
                return json.loads(raw)
        from typesafe_sdk import Choice, Noul, Score
        kinds = {'choice': Choice, 'noul': Noul, 'score': Score}
        qs = {qid: kinds[q['type']](**{k: v for k, v in q.items() if k != 'type'}) for qid, q in req['questions'].items()}
        result = client.system_one(state=req['state'], questions=qs)
        answers = {}
        for qid, q in req['questions'].items():
            kind = q['type']; answer = result.answers[qid]
            if kind == 'noul':
                answers[qid] = {'type': kind, 'noul': answer.noul}
            else:
                answers[qid] = {'type': kind, 'probabilities': {str(k): v for k, v in answer.probabilities.items()},
                                'confidence': answer.confidence}
                answers[qid][kind] = getattr(answer, kind)
        return {'model': result.model, 'answers': answers, 'provider_request_id': result.request_id,
                'usage': {'input_tokens': result.usage.input_tokens, 'output_tokens': result.usage.output_tokens} if result.usage is not None else None}
    try:
        with results_path.open('a') as stream:
            for row in rows:
                if row['id'] in done: continue
                start = time.monotonic()
                record = {'id': row['id'], 'request_sha256': expected[row['id']]}
                try:
                    if worker:
                        result = await asyncio.wait_for(worker.infer(row['request']), args.timeout)
                    else:
                        result = await asyncio.to_thread(remote, row['request'])
                    record.update(status='ok', response=result)
                except Exception as exc:
                    # Do not serialize provider exceptions: they can contain headers or prompts.
                    record.update(status='error', error_type=type(exc).__name__, latency_seconds=time.monotonic() - start)
                    if worker:
                        await worker.stop(); await worker.start()
                record.setdefault('latency_seconds', time.monotonic() - start)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush(); os.fsync(stream.fileno())
                done.add(row['id'])
                print(f'{len(done)}/{len(rows)} {row["id"]} {record["status"]} {record["latency_seconds"]:.2f}s', flush=True)
    finally:
        if worker: await worker.stop()
        if client: client.close()
    (out / 'completion.json').write_text(json.dumps({'complete': len(done) == len(rows), 'completed': len(done),
        'expected': len(rows), 'startup_seconds_this_session': startup, 'predictions_sha256': sha(results_path)}, indent=2) + '\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['abel', 'jev', 'http'], required=True)
    p.add_argument('--model')
    p.add_argument('--endpoint', default='http://127.0.0.1:8091/v1/decide')
    p.add_argument('--dataset', default=str(ROOT / 'benchmarks/decisionbench-v1'))
    p.add_argument('--split', choices=['dev', 'test'], default='test')
    p.add_argument('--suite', choices=['smoke', 'full'], default='full')
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not 1 <= a.timeout <= 3600: p.error('timeout must be 1..3600 seconds')
    asyncio.run(run(a))
