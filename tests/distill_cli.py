import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[1]
loader=importlib.machinery.SourceFileLoader('train_jax',str(ROOT/'scripts/train-jax'))
spec=importlib.util.spec_from_loader(loader.name,loader); module=importlib.util.module_from_spec(spec);loader.exec_module(module)
for value in ['https://api.example/model','http://127.0.0.1/model','ollama://teacher','hf://repo/file.gguf']:
    try: module.local_gguf(value);raise AssertionError('URL accepted')
    except ValueError: pass
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    fake=tmp/'not.gguf';fake.write_bytes(b'NOT A MODEL')
    try:module.local_gguf(str(fake));raise AssertionError('bad magic accepted')
    except ValueError:pass
    data=[json.loads(s) for s in (ROOT/'data/distillation-demo/tasks.jsonl').read_text().splitlines()]
    data[1]['group']=data[2]['group']
    path=tmp/'bad.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in data))
    try:module.preflight(path);raise AssertionError('cross-split group accepted')
    except ValueError:pass
    rows=[]
    for split in ['train','validation','calibration','test']:
        for i in range(2):
            rows.append({'id':f'{split}-{i}','split':split,'type':'noul','family':'toy',
                         'hidden':[[1.,0.] if i==0 else [0.,1.]],'base_logits':[0.],
                         'target':i,'teacher_probabilities':[.8,.2] if i==0 else [.1,.9]})
    path=tmp/'fit.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['split']!='test'))
    binary=str(ROOT/'build/distill-train')
    result=subprocess.run([binary,'fit',str(path),str(tmp/'model')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads((tmp/'model/training_report.json').read_text())
    assert report['train']['teacher_kl']<.1
    assert report['teacher_metrics_are_ground_truth_accuracy'] is False
    assert subprocess.run([binary,'fit',str(path),str(tmp/'model')],capture_output=True).returncode!=0
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    bad=subprocess.run([binary,'fit',str(path),str(tmp/'leaked')],capture_output=True,text=True)
    assert bad.returncode!=0 and 'physically exclude test' in bad.stderr
print('Distillation CLI: URL/magic rejection, split leakage rejection, learned soft targets, test exclusion, no overwrite OK')
