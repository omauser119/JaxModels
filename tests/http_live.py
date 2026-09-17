"""Real localhost HTTP framing, overload, usage reset and state isolation."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get('JAX_TEST_URL', 'http://127.0.0.1:8091')
HEADERS = {'Content-Type': 'application/json'}
if os.environ.get('JAX_SERVICE_TOKEN'): HEADERS['Authorization'] = 'Bearer ' + os.environ['JAX_SERVICE_TOKEN']
def call(path, data=None):
    data = json.dumps(data).encode() if isinstance(data, dict) else data
    req = Request(BASE + path, data=data, headers=HEADERS)
    try:
        with urlopen(req, timeout=130) as r: return r.status, json.load(r), dict(r.headers)
    except HTTPError as e: return e.code, json.load(e), dict(e.headers)

req = {'state': {'message': 'Please send the invoice.'}, 'questions': {'q': {'type': 'noul', 'instructions': 'Does the message request an invoice?'}}}
assert call('/readyz')[0] == 200
for body in [b'{"state":1,"state":2,"questions":{}}', b'{"state":NaN,"questions":{}}', b'\xff']:
    assert call('/v1/decide', body)[0] == 400
assert call('/v1/decide', b'x' * (256 * 1024 + 1))[0] == 413
assert call('/v1/systemone', dict(req, model='nonexistent'))[0] == 404
# Oversized token context is rejected without truncation or poisoning the following inference.
overlong = dict(req, state='invoice ' * 12000)
assert call('/v1/decide', overlong)[0] == 400
with ThreadPoolExecutor(max_workers=1) as pool:
    pending = pool.submit(call, '/v1/decide', req)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if call('/metrics')[1]['worker_busy']: break
        time.sleep(.02)
    else: raise AssertionError('did not observe occupied slot')
    assert call('/healthz')[0] == 200
    assert call('/v1/decide', req)[0] == 429
    status, first, headers = pending.result()
assert status == 200 and headers.get('x-typesafe-request-id')
status, second, _ = call('/v1/systemone', dict(req, model='Jax-1-Abel'))
assert status == 200
assert first['answers'] == second['answers']
assert first['usage'] == second['usage'] and first['usage']['input_tokens'] > 0
assert first['request_id'] != second['request_id']
report = {'passed': True, 'checks': ['ready', 'strict JSON over HTTP', 'body limit', 'unknown model',
    'context overflow then recovery', 'health while busy', '429 admission', 'SDK/native path parity',
    'token counter reset', 'unique request IDs'], 'response': second}
(ROOT / 'runtime/http-live.json').write_text(json.dumps(report, indent=2) + '\n')
print('Live HTTP: JSON, limits, context recovery, overload, probes, parity, token reset and request IDs OK')
