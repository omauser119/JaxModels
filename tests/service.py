"""Operational gateway tests: admission, auth, deadlines, isolation and recovery."""
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.app import Application, MAX_BODY

REQUEST = {'state': 'hello', 'questions': {'q': {'type': 'noul', 'instructions': 'Is this a greeting?'}}}
class FakeWorker:
    def __init__(self):
        self.ready = True
        self.starts = self.stops = 0
        self.mode = 'ok'
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
    async def start(self): self.starts += 1; self.ready = True
    async def stop(self): self.stops += 1; self.ready = False
    async def infer(self, req):
        self.entered.set()
        if self.mode == 'wait': await self.release.wait()
        if self.mode == 'crash': raise RuntimeError('secret internal error')
        if self.mode == 'invalid': raise ValueError('judgment exceeds JAX_CTX')
        return {'model': 'Jax-1-Abel', 'answers': {'q': {'type': 'noul', 'noul': .75}}}

async def request(app, data=None, path='/v1/decide', method='POST', headers=None):
    body = json.dumps(REQUEST).encode() if data is None else data
    events = [{'type': 'http.request', 'body': body, 'more_body': False}]
    output = []
    async def receive(): return events.pop(0)
    async def send(event): output.append(event)
    await app({'type': 'http', 'path': path, 'method': method,
               'headers': headers if headers is not None else [(b'content-type', b'application/json')]}, receive, send)
    return output[0]['status'], json.loads(output[1]['body'])

async def test():
    w = FakeWorker(); app = Application(w)
    assert (await request(app))[0] == 200
    for data in [b'{', b'{"state":1,"state":2,"questions":{}}', b'{"state":NaN,"questions":{}}',
                 b'{"state":1e999,"questions":{}}', b'{"state":"\\ud800","questions":{}}',
                 json.dumps({'state': 1, 'questions': {}}).encode(),
                 json.dumps(dict(REQUEST, unexpected=True)).encode()]:
        assert (await request(app, data))[0] == 400
    assert (await request(app, b'x' * (MAX_BODY + 1)))[0] == 413
    assert (await request(app, headers=[]))[0] == 415
    assert (await request(app, method='GET'))[0] == 405
    app.token = 'secret'
    assert (await request(app))[0] == 401
    assert (await request(app, path='/healthz', method='GET'))[0] == 200
    assert (await request(app, path='/metrics', method='GET'))[0] == 401
    app.token = ''
    w.mode = 'wait'
    running = asyncio.create_task(request(app)); await w.entered.wait(); await asyncio.sleep(.01)
    assert (await request(app))[0] == 429
    assert (await request(app, path='/healthz', method='GET'))[0] == 200
    w.release.set(); assert (await running)[0] == 200
    w.mode = 'invalid'; assert (await request(app))[0] == 400; assert w.starts == 0
    w.mode = 'crash'; status, body = await request(app)
    assert status == 503 and 'secret internal' not in json.dumps(body)
    await app.recovery; assert w.starts == w.stops == 1
    w.mode = 'wait'; w.release.clear(); app.timeout = .01
    assert (await request(app))[0] == 504
    await app.recovery; assert w.starts == w.stops == 2
    w.mode = 'ok'; assert (await request(app))[0] == 200
    w.ready = False; assert (await request(app))[0] == 503
    assert (await request(app, path='/readyz', method='GET'))[0] == 503

with contextlib.redirect_stdout(io.StringIO()) as logs:
    asyncio.run(test())
assert 'Is this a greeting' not in logs.getvalue() and 'secret' not in logs.getvalue()
print('Service: validation, auth, admission, health, crash/timeout recovery and log privacy OK')
