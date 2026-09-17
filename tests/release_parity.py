"""Compare released head applied to cached features with live native inference."""
import json, math, subprocess, tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
fixture=json.loads((root/'tests/fixtures/release-parity.json').read_text())
f=fixture['feature'];t=fixture['task']
h=json.loads((root/'models/Jax-1-Abel/decision_head.json').read_text())
z=h['base_scale']*f['base_logits'][0]+h['bias']+sum(w*v for w,v in zip(h['weights'],f['hidden'][0]))
p=1/(1+math.exp(-z/h['temperature']))
req={'state':t['state'],'questions':{'first':t['question'],'renamed':t['question']}}
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'request.json';path.write_text(json.dumps(req))
 result=subprocess.run([str(root/'scripts/jax-abel'),'infer',str(path)],capture_output=True,text=True,check=True,timeout=180)
 out=json.loads(result.stdout)
 assert abs(out['answers']['first']['noul']-p)<1e-4
 assert out['answers']['first']==out['answers']['renamed']
print('Released candidate: cached-feature/live parity and question-ID isolation OK')
