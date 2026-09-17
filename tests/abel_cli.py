"""Exercise real C++ training and enforce non-use of test labels during selection."""
import json,math,pathlib,subprocess,tempfile
root=pathlib.Path(__file__).resolve().parents[1]
binary=root/'build/abel-train'
rows=[]
for split in ('train','validation','calibration','test'):
 for i in range(6):
  positive=i%2==0
  vector=[1.,0.] if positive else [-1.,0.]
  rows.append(dict(id=f'{split}-{i}',split=split,type='noul',family='fixture',hidden=[vector],base_logits=[-.5 if positive else .5],target=int(positive)))
with tempfile.TemporaryDirectory() as tmp:
 tmp=pathlib.Path(tmp);data=tmp/'data.jsonl'
 def write():data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 def run(*args):return subprocess.run([str(binary),*map(str,args)],capture_output=True,text=True,timeout=30)
 write();r=run('fit',data,tmp/'first');assert r.returncode==0,r.stderr
 head1=json.loads((tmp/'first/decision_head.json').read_text())
 assert sum(w*w for w in head1['weights'])>1e-6
 for row in rows:
  if row['split']=='test':row['target']=1-row['target']
 write();r=run('fit',data,tmp/'second');assert r.returncode==0,r.stderr
 head2=json.loads((tmp/'second/decision_head.json').read_text())
 assert head1==head2,'test labels changed model selection/calibration'
 assert run('fit',data,tmp/'first').returncode!=0,'existing checkpoint overwritten'
 report=json.loads(run('evaluate',data,tmp/'first/decision_head.json','validation').stdout)
 assert report['abel']['nll']<report['baseline']['nll']
 data.write_text(json.dumps(dict(rows[0],hidden=[[0,0]]))+'\n')
 assert run('fit',data,tmp/'bad').returncode!=0
print('Abel CLI: trained nonzero weights; test-label isolation; metrics; invalid features OK')
