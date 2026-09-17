"""Loopback API contract tests; never contacts a paid provider."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('api_teacher_impl',ROOT/'scripts/api_teacher.py')
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
requests=[]
mode='ok'
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_POST(self):
        assert self.path=='/v1/chat/completions'
        assert self.headers['Authorization']=='Bearer fixture-secret'
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        requests.append(body)
        payload=json.loads(body['messages'][1]['content'])
        assert set(payload)=={'state','question','labels_in_order'}
        assert body['model']=='fixture-model'
        if mode=='unauthorized':
            self.send_response(401);self.end_headers();self.wfile.write(b'fixture-secret');return
        if mode=='redirect':
            self.send_response(307);self.send_header('Location','https://example.invalid/steal');self.end_headers();return
        n=len(payload['labels_in_order']);probs=[0.]*(n-1)+[1.]
        content=json.dumps({'probabilities':probs if mode!='invalid' else [False, True]})
        response={'choices':[{'finish_reason':'length' if mode=='truncated' else 'stop',
                             'message':{'content':content}}]}
        self.send_response(200);self.end_headers();self.wfile.write(json.dumps(response).encode())

server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
url=f'http://127.0.0.1:{server.server_port}/v1'
row={'id':'private-id','split':'test','family':'private-family','group':'private-group',
     'state':{'text':'hello'},'question':{'type':'noul','instructions':'A greeting?'}}
try:
    for kind,criteria,expected in [('noul',None,1),('choice',{'z':'last','a':'first'},'z'),('score',['low','medium','high'],2)]:
        q={'type':kind,'instructions':'Evaluate'}
        if criteria is not None:q['criteria']=criteria
        result=api.label({**row,'question':q},url,'fixture-secret','fixture-model')
        assert result['target']==expected and result['teacher_probabilities'][-1]==1
    for mode in ['unauthorized','redirect','invalid','truncated']:
        try:api.label(row,url,'fixture-secret','fixture-model');raise AssertionError('bad response accepted')
        except ValueError as exc:assert 'fixture-secret' not in str(exc)
    mode='ok'
    api.label(row,url,'fixture-secret','fixture-model',minimal=True)
    assert 'temperature' not in requests[-1] and 'max_tokens' not in requests[-1]
    original_label,original_sleep=api.label,api.time.sleep
    calls=[]
    def flaky(*args,**kwargs):
        calls.append(1)
        if len(calls)<3:raise api.RetryableTeacherError('temporary fixture failure')
        return {'ok':True}
    api.label=flaky;api.time.sleep=lambda _:None
    assert api.label_with_retry()=={'ok':True} and len(calls)==3
    def permanent(*args,**kwargs):
        calls.append(1);raise api.TeacherError('HTTP 401')
    calls.clear();api.label=permanent
    try:api.label_with_retry();raise AssertionError('permanent failure swallowed')
    except api.TeacherError:assert len(calls)==1
    def exhausted(*args,**kwargs):
        calls.append(1);raise api.RetryableTeacherError('still failing')
    calls.clear();api.label=exhausted
    try:api.label_with_retry();raise AssertionError('unbounded retries')
    except api.RetryableTeacherError:assert len(calls)==4
    api.label,api.time.sleep=original_label,original_sleep
    original_clock=api.time.monotonic
    clock=[0.];delays=[];calls=[]
    api.time.monotonic=lambda:clock[0]
    def advance(delay):delays.append(delay);clock[0]+=delay
    api.time.sleep=advance
    def outage(*args,**kwargs):
        calls.append(1)
        if len(calls)<4:raise api.UnavailableTeacherError('HTTP 502')
        return {'ok':True}
    api.label=outage
    assert api.label_with_retry(outage_seconds=100)=={'ok':True}
    assert delays==[5,10,20]
    clock[0]=0;calls.clear();delays.clear()
    try:api.label_with_retry(outage_seconds=6);raise AssertionError('outage budget ignored')
    except api.UnavailableTeacherError:assert delays==[5,1]
    api.label,api.time.sleep,api.time.monotonic=original_label,original_sleep,original_clock
    with tempfile.TemporaryDirectory() as folder:
        saved=Path(folder)/'partial'
        answer={**row,'target':1,'teacher_probabilities':[.1,.9]}
        saved.write_text(json.dumps(answer)+'\n')
        assert api.saved_prefix(saved,[row])==1
        try:api.saved_prefix(saved,[{**row,'state':{'changed':True}}]);raise AssertionError('mismatched cache accepted')
        except api.TeacherError:pass
        tasks=Path(folder)/'tasks.jsonl'
        second={**row,'id':'second'}
        tasks.write_text(json.dumps(row)+'\n'+json.dumps(second)+'\n')
        saved.write_text(json.dumps({**row,'target':1,'teacher_probabilities':[.1,.9]})+'\n')
        before=len(requests)
        with saved.open('ab') as output:
            subprocess.run([sys.executable,str(ROOT/'scripts/api_teacher.py'),str(tasks),
                '--base-url',url,'--model','fixture-model','--resume-labels',str(saved)],
                env={**os.environ,'JAX_TEACHER_API_KEY':'fixture-secret'},stdout=output,check=True)
        assert len(requests)==before+1 and api.saved_prefix(saved,[row,second])==2
        answer['teacher_probabilities']=[.1,.1];saved.write_text(json.dumps(answer)+'\n')
        try:api.saved_prefix(saved,[row]);raise AssertionError('invalid probabilities accepted')
        except api.TeacherError:pass
    for invalid in ['https://user:secret@example.com/v1','https://example.com/?key=secret','http://example.com/v1']:
        try:api.base_url(invalid);raise AssertionError('bad endpoint accepted')
        except ValueError:pass
    if '--pipeline' in sys.argv:
        with tempfile.TemporaryDirectory(prefix='jax-api-pipeline-') as folder:
            out=Path(folder)/'run'
            command=[str(ROOT/'scripts/train-jax'),'--teacher-backend','openai','--baseurl',url,
                '--model','fixture-model','--data',str(ROOT/'data/distillation-demo/tasks.jsonl'),
                '--output',str(out),'--chunk-size','2']
            env={**os.environ,'JAX_TEACHER_API_KEY':'fixture-secret'}
            subprocess.run(command,cwd=ROOT,env=env,check=True)
            n=len(requests)
            subprocess.run(command+['--resume'],cwd=ROOT,env=env,check=True)
            assert len(requests)==n,'resume repeated API calls'
            manifest=(out/'provenance.json').read_text()
            assert 'fixture-secret' not in manifest
            assert json.loads(manifest)['teacher_backend']=='openai-compatible'
            assert json.loads((out/'result.json').read_text())['complete']
    print('API teacher: all primitives, ordering, isolation, credential protection, errors and redirects OK')
finally:server.shutdown();server.server_close()
