#!/usr/bin/env python3
"""Numerical checks for evaluation; these fixtures are not model-quality evidence."""
import importlib.util
import math
from pathlib import Path

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('candidate',root/'scripts/evaluate-candidate.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
head={'weights':[2.,-1.],'base_scale':.5,'bias':.25,'temperature':2.}
feature={'type':'noul','base_logits':[1.],'hidden':[[.5,.25]]}
p=module.probabilities(feature,head)
assert abs(p[1]-1/(1+math.exp(-.75)))<1e-12
assert abs(sum(p)-1)<1e-12
feature={'type':'choice','base_logits':[1000.,999.],'hidden':[[0.,0.],[0.,0.]]}
p=module.probabilities(feature,None)
assert abs(p[0]-1/(1+math.exp(-1)))<1e-12
rows=[{'correct':1,'confidence':.8,'nll':-math.log(.8),'brier':.08},
      {'correct':0,'confidence':.8,'nll':-math.log(.2),'brier':1.28}]
m=module.metrics(rows)
assert m['accuracy']==.5 and abs(m['ece10']-.3)<1e-12
assert m['wilson_95'][0]<.5<m['wilson_95'][1]
assert abs(m['brier']-.68)<1e-12
assert module.metrics([]) is None
print('Candidate evaluation: binary temperature, stable softmax, Brier/ECE/Wilson OK')
