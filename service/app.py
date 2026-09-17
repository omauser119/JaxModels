"""ASGI gateway for the persistent C++ Abel worker. One process / one CPU context."""
import asyncio
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.2.0-pilot.1'
MAX_BODY = 256 * 1024


def strict_json(data):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out or '\0' in key:
                raise ValueError('duplicate or NUL object key')
            out[key] = value
        return out
    def number(s):
        value = float(s)
        if not math.isfinite(value):
            raise ValueError('nonfinite number')
        return value
    def constant(_):
        raise ValueError('nonfinite number')
    return json.loads(data, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)


def validate_request(req):
    if not isinstance(req, dict) or set(req) != {'state', 'questions'}:
        raise ValueError('expected exactly state and questions')
    questions = req['questions']
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 16:
        raise ValueError('expected 1..16 questions')
    jobs = 0
    for name, q in questions.items():
        if not name or len(name) > 128 or not isinstance(q, dict):
            raise ValueError('invalid question')
        if set(q) - {'type', 'instructions', 'criteria'} or not q.get('instructions'):
            raise ValueError('invalid question fields')
        kind, criteria = q.get('type'), q.get('criteria')
        if kind == 'noul':
            if criteria is not None and (not isinstance(criteria, dict) or set(criteria) != {'true', 'false'}):
                raise ValueError('Noul criteria must define true and false')
            jobs += 1
        elif kind == 'choice':
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255 or any(not k for k in criteria):
                raise ValueError('Choice requires 2..255 named criteria')
            jobs += len(criteria)
        elif kind == 'score':
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10 or any(c == '' for c in criteria):
                raise ValueError('Score requires 2..10 levels')
            jobs += len(criteria)
        else:
            raise ValueError('unknown question type')
    if jobs > 32:
        raise ValueError('service request exceeds 32 judgments')


def normalize_sdk(req):
    if not isinstance(req, dict) or set(req) - {'state', 'questions', 'model'}:
        raise ValueError('invalid SDK request fields')
    if req.get('model', 'Jax-1-Abel') != 'Jax-1-Abel':
        raise LookupError('unknown model; set defaultModel to Jax-1-Abel')
    out = {'state': req['state'], 'questions': {}}
    if not isinstance(req['questions'], dict):
        raise ValueError('questions must be an object')
    for name, original in req['questions'].items():
        if not isinstance(original, dict): raise ValueError('invalid question')
        q = dict(original)
        if q.get('instructions') is None:
            defaults = {
                'choice': 'Select the best matching option for the supplied state using the option descriptions.',
                'score': 'Evaluate the supplied state using the descriptions of the score levels.',
                'noul': 'Does the supplied state satisfy the YES criterion?',
            }
            if q.get('type') == 'noul' and not q.get('criteria'):
                raise ValueError('Noul needs instructions or criteria')
            q['instructions'] = defaults.get(q.get('type'), '')
        if q.get('type') == 'noul':
            if q.get('criteria') is None:
                q.pop('criteria', None)
            elif isinstance(q['criteria'], dict) and not set(q['criteria']) - {'true', 'false'}:
                c = q['criteria']
                q['criteria'] = {'true': c.get('true') if c.get('true') is not None else 'The question is true; the NO criterion does not hold.',
                                 'false': c.get('false') if c.get('false') is not None else 'The question is false; the YES criterion does not hold.'}
        out['questions'][name] = q
    return out


class Worker:
    def __init__(self):
        self.process = None
        self.ready = False

    @property
    def ready(self):
        return self._ready and self.process is not None and self.process.returncode is None

    @ready.setter
    def ready(self, value):
        self._ready = value

    async def start(self):
        # The launcher hashes both model components on every start/restart.
        env = dict(os.environ, JAX_TRACE='0')
        self.process = await asyncio.create_subprocess_exec(
            str(ROOT / 'scripts/jax-abel'), 'stream', stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=None, env=env, limit=2 * 1024 * 1024)
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), 90)
            if json.loads(line) != {'ready': True, 'protocol': 'jax.worker.v1'}:
                raise RuntimeError('invalid worker handshake')
        except BaseException:
            await self.stop()
            raise
        self.ready = True

    async def stop(self):
        self.ready = False
        p, self.process = self.process, None
        if p is not None and p.returncode is None:
            try:
                p.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(p.wait(), 5)
            except TimeoutError:
                p.kill()
                await p.wait()

    async def infer(self, request):
        p = self.process
        if p is None or p.returncode is not None:
            raise RuntimeError('worker unavailable')
        payload = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode() + b'\n'
        p.stdin.write(payload)
        await p.stdin.drain()
        line = await p.stdout.readline()
        if not line:
            raise RuntimeError('worker exited')
        result = strict_json(line)
        if 'error' in result:
            error = result['error']
            if error.get('code') == 'invalid_request':
                raise ValueError(error['message'])
            raise RuntimeError('worker inference failed')
        return result['result']


class Application:
    def __init__(self, worker=None):
        self.worker = worker or Worker()
        self.busy = False
        self.recovery = None
        self.stopping = False
        self.token = os.environ.get('JAX_SERVICE_TOKEN', '')
        self.timeout = float(os.environ.get('JAX_REQUEST_TIMEOUT', '120'))
        if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 3600:
            raise ValueError('JAX_REQUEST_TIMEOUT must be 1..3600 seconds')
        self.counts = {}
        self.total_seconds = 0.0
        self.config = json.loads((ROOT / 'models/Jax-1-Abel/config.json').read_text())

    async def recover(self):
        try:
            await self.worker.stop()
            if not self.stopping:
                await self.worker.start()
        except Exception:
            # Failed readiness stays visible; process supervisor can restart service.
            print(json.dumps({'event': 'worker_recovery_failed'}), flush=True)
        finally:
            self.busy = False

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'lifespan':
            while True:
                event = await receive()
                if event['type'] == 'lifespan.startup':
                    try:
                        await self.worker.start()
                        await send({'type': 'lifespan.startup.complete'})
                    except Exception:
                        await send({'type': 'lifespan.startup.failed', 'message': 'verified worker startup failed'})
                        return
                else:
                    self.stopping = True
                    if self.recovery:
                        await self.recovery
                    await self.worker.stop()
                    await send({'type': 'lifespan.shutdown.complete'})
                    return
        if scope['type'] != 'http':
            return
        request_id = uuid.uuid4().hex
        start = time.monotonic()
        status = 500
        async def reply(code, body, extra=()):
            nonlocal status
            status = code
            encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            await send({'type': 'http.response.start', 'status': code, 'headers': [
                (b'content-type', b'application/json; charset=utf-8'),
                (b'content-length', str(len(encoded)).encode()),
                (b'x-request-id', request_id.encode()), (b'x-typesafe-request-id', request_id.encode()), (b'cache-control', b'no-store'), *extra]})
            await send({'type': 'http.response.body', 'body': encoded})
        async def error(code, name, message, extra=()):
            await reply(code, {'error': {'code': name, 'message': message}, 'request_id': request_id}, extra)
        try:
            path, method = scope['path'], scope['method']
            if path == '/healthz' and method == 'GET':
                return await reply(200, {'status': 'alive'})
            if path == '/readyz' and method == 'GET':
                ready = self.worker.ready and not self.stopping
                return await reply(200 if ready else 503, {'ready': ready})
            headers = {}
            for k, v in scope['headers']:
                if k in headers and k in (b'authorization', b'content-length', b'content-type'):
                    return await error(400, 'invalid_headers', 'duplicate header')
                headers[k] = v
            if self.token and not hmac.compare_digest(headers.get(b'authorization', b''), ('Bearer ' + self.token).encode()):
                return await error(401, 'unauthorized', 'valid bearer token required')
            if path == '/v1/models' and method == 'GET':
                return await reply(200, {'models': [{'name': 'Jax-1-Abel',
                    'description': 'Universal Choice/Score/Noul; Qwen3.5-0.8B plus learned Abel decision head; experimental weights.',
                    'release_date': '2026-09-16'}]})
            if path == '/v1/model' and method == 'GET':
                return await reply(200, {'model': self.config['model_id'], 'model_version': self.config['version'],
                    'runtime_version': VERSION, 'model_status': self.config['status'], 'sha256': self.config['sha256'],
                    'limits': {'body_bytes': MAX_BODY, 'questions': 16, 'judgments': 32, 'concurrency': 1,
                               'timeout_seconds': self.timeout}, 'ready': self.worker.ready})
            if path == '/metrics' and method == 'GET':
                return await reply(200, {'http_status_counts': self.counts, 'http_seconds_total': self.total_seconds,
                                         'worker_ready': self.worker.ready, 'worker_busy': self.busy})
            if path not in ('/v1/decide', '/v1/systemone'):
                return await error(404, 'not_found', 'unknown endpoint')
            if method != 'POST':
                return await error(405, 'method_not_allowed', 'use POST', [(b'allow', b'POST')])
            if headers.get(b'content-type', b'').split(b';')[0].strip().lower() != b'application/json':
                return await error(415, 'content_type', 'expected application/json')
            if b'content-encoding' in headers:
                return await error(415, 'content_encoding', 'compressed requests are unsupported')
            try:
                length = int(headers.get(b'content-length', b'0'))
            except ValueError:
                return await error(400, 'content_length', 'invalid Content-Length')
            if length < 0 or length > MAX_BODY:
                return await error(413, 'request_too_large', 'body exceeds service limit')
            if not self.worker.ready or self.stopping:
                return await error(503, 'not_ready', 'model unavailable', [(b'retry-after', b'5')])
            if self.busy:
                return await error(429, 'busy', 'single inference slot is occupied', [(b'retry-after', b'1')])
            # No await between admission check and reservation. No unbounded inference queue.
            self.busy = True
            restart = False
            try:
                body = bytearray()
                async with asyncio.timeout(10):
                    while True:
                        event = await receive()
                        if event['type'] == 'http.disconnect':
                            return
                        body.extend(event.get('body', b''))
                        if len(body) > MAX_BODY:
                            return await error(413, 'request_too_large', 'body exceeds service limit')
                        if not event.get('more_body', False):
                            break
                try:
                    req = strict_json(body)
                    if path == '/v1/systemone':
                        req = normalize_sdk(req)
                    validate_request(req)
                    # Reject lone surrogate code points before handing data to the UTF-8 worker.
                    json.dumps(req, ensure_ascii=False).encode('utf-8')
                except LookupError as exc:
                    return await error(404 if isinstance(exc, LookupError) and not isinstance(exc, KeyError) else 400, 'invalid_request', str(exc))
                except (ValueError, TypeError, RecursionError, UnicodeError):
                    return await error(400, 'invalid_request', 'invalid JSON or request schema')
                try:
                    output = await asyncio.wait_for(self.worker.infer(req), self.timeout)
                    # Preserve original JSON rubric values in the SDK Score legend.
                    for qid, q in req['questions'].items():
                        if q['type'] == 'score':
                            output['answers'][qid]['legend'] = {str(i): value for i, value in enumerate(q['criteria'])}
                    output['request_id'] = request_id
                    return await reply(200, output)
                except ValueError as exc:
                    return await error(400, 'invalid_request', str(exc))
                except TimeoutError:
                    restart = True
                    return await error(504, 'inference_timeout', 'inference deadline exceeded; worker restarting')
                except asyncio.CancelledError:
                    restart = True
                    raise
                except Exception:
                    restart = True
                    return await error(503, 'worker_failed', 'worker failed; restarting')
            except TimeoutError:
                return await error(408, 'body_timeout', 'request body deadline exceeded')
            finally:
                if restart:
                    self.worker.ready = False
                    self.recovery = asyncio.create_task(self.recover())
                else:
                    self.busy = False
        finally:
            elapsed = time.monotonic() - start
            key = str(status)
            self.counts[key] = self.counts.get(key, 0) + 1
            self.total_seconds += elapsed
            # Never log state, instructions, credentials, raw answers or arbitrary client IDs.
            print(json.dumps({'event': 'http', 'request_id': request_id, 'status': status,
                              'seconds': round(elapsed, 6)}), flush=True)


def create_app():
    return Application()
