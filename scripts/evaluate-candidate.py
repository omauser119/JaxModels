#!/usr/bin/env python3
"""Ground-truth scoring from held-out cached features; independent from teacher agreement."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def probabilities(feature, head):
 logits=[0.] if feature['type']=='noul' else []
 for base,hidden in zip(feature['base_logits'],feature['hidden']):
  if head is None:logits.append(base)
  else:
   if len(hidden)!=len(head['weights']):raise ValueError('head dimension mismatch')
   logits.append(head['base_scale']*base+head['bias']+sum(w*v for w,v in zip(head['weights'],hidden)))
 temp=head['temperature'] if head else 1
 if not math.isfinite(temp) or temp<=0:raise ValueError('invalid temperature')
 peak=max(logits);p=[math.exp((v-peak)/temp) for v in logits];total=sum(p)
 return [v/total for v in p]

def metrics(rows):
 n=len(rows)
 if not n:return None
 correct=sum(r['correct'] for r in rows);accuracy=correct/n;z=1.959963984540054
 center=(accuracy+z*z/(2*n))/(1+z*z/n);radius=z*math.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/(1+z*z/n)
 bins=collections.defaultdict(list)
 for r in rows:bins[min(9,int(r['confidence']*10))].append(r)
 return {'tasks':n,'correct':correct,'accuracy':accuracy,'wilson_95':[center-radius,center+radius],
         'nll':sum(r['nll'] for r in rows)/n,'brier':sum(r['brier'] for r in rows)/n,
         'ece10':sum(abs(sum(r['confidence']-r['correct'] for r in bucket)) for bucket in bins.values())/n}

def evaluate(run, corpus):
 manifest=json.loads((corpus/'manifest.json').read_text())
 if digest(corpus/'references.jsonl')!=manifest['sha256']['references.jsonl']:raise ValueError('reference checksum mismatch')
 refs={}
 for line in (corpus/'references.jsonl').read_text().splitlines():
  row=json.loads(line)
  if row['id'] in refs:raise ValueError('duplicate reference ID')
  if row['split']=='test':refs[row['id']]=row
 provenance=json.loads((run/'provenance.json').read_text())
 if provenance['dataset_sha256']!=manifest['sha256']['tasks.jsonl']:raise ValueError('different training corpus')
 checkpoint=run/'checkpoint/decision_head.json'
 heads={'candidate':json.loads(checkpoint.read_text()),'deployed_abel':json.loads((ROOT/'models/Jax-1-Abel/decision_head.json').read_text()),'base_qwen':None}
 observations={name:[] for name in [*heads,'teacher']};seen=set()
 with (run/'test-features.jsonl').open() as stream:
  for line in stream:
   feature=json.loads(line);ident=feature['id']
   if ident in seen or ident not in refs or feature['split']!='test':raise ValueError('invalid test features')
   seen.add(ident);reference=refs[ident];labels=feature['labels'];target=labels.index(str(reference['target']))
   for name in observations:
    p=feature['teacher_probabilities'] if name=='teacher' else probabilities(feature,heads[name])
    if len(p)!=len(labels) or any(not math.isfinite(v) or not 0<=v<=1 for v in p) or abs(sum(p)-1)>1e-6:raise ValueError('invalid probabilities')
    best=max(range(len(p)),key=p.__getitem__)
    observations[name].append({'id':ident,'family':reference['family'],'language':reference['language'],'kind':feature['type'],'source_type':reference.get('source_type','synthetic'),
      'correct':int(best==target),'confidence':max(p),'nll':-math.log(max(p[target],1e-12)),
      'brier':sum((v-int(i==target))**2 for i,v in enumerate(p))})
 if seen!=refs.keys():raise ValueError('missing test cases')
 report={'checkpoint_sha256':digest(checkpoint),'references_sha256':manifest['sha256']['references.jsonl'],
         'ground_truth_scored':True,'test_used_for_selection':False,'source':manifest.get('source'),
         'production_qualified':False,'metrics':{},'note':'Human-annotated and synthetic slices are reported separately. MASSIVE translations share source IDs; Wilson intervals ignore this dependence. Intent accuracy does not establish arbitrary-domain universality.'}
 for name,rows in observations.items():
  sections={}
  for field in ['family','language','kind','source_type']:
   buckets=collections.defaultdict(list)
   for r in rows:buckets[r[field]].append(r)
   sections[field]={k:metrics(v) for k,v in buckets.items()}
  report['metrics'][name]={'overall':metrics(rows),'by':sections}
 candidate=report['metrics']['candidate'];baseline=report['metrics']['deployed_abel']
 report['diagnostic_thresholds']={'min_tasks_1600':len(seen)>=1600,'accuracy_lower95_at_least_090':candidate['overall']['wilson_95'][0]>=.9,
  'worst_family_accuracy_at_least_085':min(x['accuracy'] for x in candidate['by']['family'].values())>=.85,
  'ece_at_most_005':candidate['overall']['ece10']<=.05,
  'no_accuracy_regression_over_001':candidate['overall']['accuracy']>=baseline['overall']['accuracy']-.01}
 report['diagnostic_gate_passed']=all(report['diagnostic_thresholds'].values())
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True)
 a=p.parse_args();result=evaluate(a.run,a.corpus);out=a.run/'ground-truth-evaluation.json'
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'report':str(out),'candidate':result['metrics']['candidate']['overall'],'diagnostic_gate_passed':result['diagnostic_gate_passed'],'production_qualified':False},indent=2))
