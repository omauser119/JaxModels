"""Oracle scoring fixtures, malformed outputs, paired comparison and leakage checks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
ROOT = Path(__file__).resolve().parents[1]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj
E = module('evaluator', ROOT / 'benchmarks/evaluate.py')
D = ROOT / 'benchmarks/decisionbench-v1'
requests = E.load_rows(D / 'requests.jsonl'); refs = E.load_rows(D / 'references.jsonl')
assert len(requests) == 2560 and len({r['group'] for r in refs.values()}) == 640
assert {refs[r['id']]['split'] for r in requests.values()} == {'dev', 'test'}
for r in requests.values():
    assert set(r) == {'id', 'split', 'request'}
    assert set(r['request']) == {'state', 'questions'}
# Choose all variants of one EN case per family, retaining all three primitives.
ids = [i for i in requests if '-en-000-' in i]
assert len(ids) == 32
manifest = json.loads((D / 'manifest.json').read_text())
with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp)
    meta = {'requests_sha256': manifest['sha256']['requests.jsonl'], 'expected_ids': ids, 'model': 'oracle-fixture'}
    (run / 'run.json').write_text(json.dumps(meta))
    outputs = []
    for ident in ids:
        ref, req = refs[ident], requests[ident]['request']
        answers = {}
        for qid, q in req['questions'].items():
            target = ref['target'] if qid == ref['question_id'] else 1
            kind = q['type']
            if kind == 'noul': answer = {'type': kind, 'noul': float(target)}
            else:
                labels = sorted(q['criteria']) if kind == 'choice' else list(range(len(q['criteria'])))
                answer = {'type': kind, 'probabilities': {str(k): float(k == target) for k in labels}, kind: target}
            answers[qid] = answer
        h = hashlib.sha256(json.dumps(req, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        outputs.append({'id': ident, 'status': 'ok', 'request_sha256': h, 'latency_seconds': .1,
                        'response': {'model': 'oracle-fixture', 'answers': answers}})
    path = run / 'predictions.jsonl'
    def save(): path.write_text(''.join(json.dumps(r) + '\n' for r in outputs))
    save(); meta, rows = E.observations(D, run); report = E.report(meta, rows)
    assert report['overall']['accuracy_all'] == 1 and report['overall']['nll_valid'] == 0
    assert report['overall']['brier_valid'] == 0 and report['overall']['ece10_valid'] == 0
    assert report['accuracy_cluster_bootstrap_95'] == [1, 1]
    assert all(v['decision_change_rate'] == 0 and v['max_total_variation'] == 0 for v in report['invariance'].values())
    assert E.compare((meta, rows), (meta, rows))['accuracy_delta_cluster_bootstrap_95'] == [0, 0]
    original = copy.deepcopy(outputs)
    outputs[0]['response']['answers'] = {}; outputs.pop(); save()
    _, bad = E.observations(D, run)
    assert E.metrics(bad)['accuracy_all'] == 30 / 32
    assert E.metrics(bad)['failures'] == {'invalid_response': 1, 'missing': 1}
    outputs = copy.deepcopy(original); outputs[0]['request_sha256'] = 'wrong'; save()
    try: E.observations(D, run); raise AssertionError('hash mismatch accepted')
    except ValueError: pass
    outputs = copy.deepcopy(original); outputs.append(outputs[0]); save()
    try: E.observations(D, run); raise AssertionError('duplicate accepted')
    except ValueError: pass
for invalid in [float('nan'), float('inf'), -1, 2, True]:
    try: E.distribution({'type': 'noul'}, {'type': 'noul', 'noul': invalid}); raise AssertionError('bad probability accepted')
    except ValueError: pass
print('DecisionBench: dataset contract, oracle metrics, correlated pairs, failures, hashes and duplicates OK')
