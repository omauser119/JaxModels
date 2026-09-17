#!/usr/bin/env python3
"""Strict scoring, operational failures, calibration, paired cluster bootstrap."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

EPSILON = 1e-12

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load_rows(path):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    result = {}
    for row in rows:
        if row['id'] in result: raise ValueError('duplicate ID in ' + str(path))
        result[row['id']] = row
    return result

def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('expected finite number')
    return value

def distribution(question, answer):
    kind = question['type']
    if answer.get('type') != kind: raise ValueError('wrong answer type')
    if kind == 'noul':
        p = finite(answer['noul']); probs, labels = [1 - p, p], [0, 1]
    else:
        labels = sorted(question['criteria']) if kind == 'choice' else list(range(len(question['criteria'])))
        actual = answer['probabilities']
        if set(actual) != {str(k) for k in labels}: raise ValueError('probability labels differ')
        probs = [finite(actual[str(k)]) for k in labels]
    if any(p < 0 or p > 1 for p in probs) or abs(sum(probs) - 1) > 1e-5:
        raise ValueError('invalid distribution')
    if kind == 'choice':
        choice = answer['choice']
        if choice not in labels or probs[labels.index(choice)] < max(probs) - 1e-8:
            raise ValueError('choice disagrees with distribution')
    if kind == 'score':
        if abs(finite(answer['score']) - sum(i * p for i, p in enumerate(probs))) > 1e-4:
            raise ValueError('score disagrees with expectation')
    return labels, probs

def observations(dataset, run_dir):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if digest(dataset / name) != expected: raise ValueError('dataset hash mismatch')
    requests = load_rows(dataset / 'requests.jsonl')
    references = load_rows(dataset / 'references.jsonl')
    meta = json.loads((run_dir / 'run.json').read_text())
    if meta['requests_sha256'] != manifest['sha256']['requests.jsonl']: raise ValueError('run uses different dataset')
    expected_ids = meta['expected_ids']
    if len(set(expected_ids)) != len(expected_ids) or not set(expected_ids) <= requests.keys():
        raise ValueError('invalid expected IDs')
    predictions = load_rows(run_dir / 'predictions.jsonl')
    if set(predictions) - set(expected_ids): raise ValueError('unexpected prediction ID')
    output = []
    for ident in expected_ids:
        ref = references[ident]
        request = requests[ident]['request']
        r = dict(ref, valid=False, correct=0, error='missing', latency_seconds=None)
        prediction = predictions.get(ident)
        if prediction:
            expected_hash = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if prediction['request_sha256'] != expected_hash: raise ValueError('prediction request hash differs')
            r['latency_seconds'] = finite(prediction['latency_seconds'])
            if r['latency_seconds'] < 0: raise ValueError('negative latency')
            if prediction['status'] == 'ok':
                try:
                    response = prediction['response']
                    if response['model'] != meta['model']: raise ValueError('wrong model identity')
                    answers = response['answers']
                    if set(answers) != set(request['questions']): raise ValueError('missing or extra answers')
                    for qid, q in request['questions'].items(): distribution(q, answers[qid])
                    q = request['questions'][ref['question_id']]
                    answer = answers[ref['question_id']]
                    labels, probs = distribution(q, answer)
                    target = labels.index(ref['target'])
                    # Exact ties: Choice respects returned argmax; Noul uses p>=.5; Score lowest index.
                    best = (labels.index(answer['choice']) if ref['kind'] == 'choice' else
                            (int(probs[1] >= .5) if ref['kind'] == 'noul' else max(range(len(probs)), key=probs.__getitem__)))
                    r.update(valid=True, correct=int(best == target), error=None, predicted=labels[best],
                             probabilities=probs, labels=labels, confidence=max(probs),
                             nll=-math.log(max(EPSILON, probs[target])),
                             brier=sum((p - int(i == target)) ** 2 for i, p in enumerate(probs)))
                    if ref['kind'] == 'score':
                        delta = answer['score'] - ref['target']
                        r.update(score_abs_error=abs(delta), score_squared_error=delta ** 2,
                                 score_normalized_abs_error=abs(delta) / (len(labels) - 1))
                except (ValueError, KeyError, TypeError, IndexError):
                    r['error'] = 'invalid_response'
            else:
                r['error'] = prediction.get('error_type', 'backend_error')
        output.append(r)
    return meta, output

def quantile(values, p):
    if not values: return None
    values = sorted(values); pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)

def average(rows, key):
    values = [r[key] for r in rows if key in r]
    return sum(values) / len(values) if values else None

def metrics(rows):
    good = [r for r in rows if r['valid']]
    bins = [[] for _ in range(10)]
    for r in good: bins[min(9, int(r['confidence'] * 10))].append(r)
    ece = sum(abs(sum(r['confidence'] - r['correct'] for r in b)) for b in bins) / len(good) if good else None
    latencies = [r['latency_seconds'] for r in rows if r['latency_seconds'] is not None]
    failures = defaultdict(int)
    for r in rows:
        if not r['valid']: failures[r['error']] += 1
    selective = {}
    for threshold in [.5, .7, .9, .95, .99]:
        accepted = [r for r in good if r['confidence'] >= threshold]
        selective[str(threshold)] = {'coverage_all': len(accepted) / len(rows), 'accepted': len(accepted),
                                    'accuracy_accepted': average(accepted, 'correct')}
    return {'count': len(rows), 'valid': len(good), 'failures': dict(failures),
            'accuracy_all': average(rows, 'correct'), 'accuracy_valid': average(good, 'correct'),
            'nll_valid': average(good, 'nll'), 'brier_valid': average(good, 'brier'), 'ece10_valid': ece,
            'score_mae_valid': average(good, 'score_abs_error'),
            'score_normalized_mae_valid': average(good, 'score_normalized_abs_error'),
            'score_rmse_valid': math.sqrt(average(good, 'score_squared_error')) if any('score_squared_error' in r for r in good) else None,
            'latency_seconds_all': {'p50': quantile(latencies, .5), 'p95': quantile(latencies, .95), 'p99': quantile(latencies, .99)},
            'selective': selective}

def invariance(rows):
    groups = defaultdict(dict)
    for r in rows: groups[r['group']][r['variant']] = r
    result = {}
    for variant in ['renamed', 'distractor_order', 'batched']:
        changes, distances = [], []
        expected = 0
        for group in groups.values():
            if 'base' not in group or variant not in group: continue
            expected += 1
            a, b = group['base'], group[variant]
            if not a['valid'] or not b['valid']: continue
            changes.append(a['predicted'] != b['predicted'])
            distances.append(.5 * sum(abs(x - y) for x, y in zip(a['probabilities'], b['probabilities'])))
        result[variant] = {'expected_pairs': expected, 'valid_pairs': len(changes),
                           'decision_change_rate': sum(changes) / len(changes) if changes else None,
                           'mean_total_variation': sum(distances) / len(distances) if distances else None,
                           'max_total_variation': max(distances) if distances else None}
    return result

def bootstrap(values_by_group, samples=2000):
    values = list(values_by_group.values())
    if len(values) < 2: return None
    # Resample source scenarios, carrying all their perturbations together.
    rng = random.Random(19092026)
    trials = []
    sums_counts = [(sum(v), len(v)) for v in values]
    for _ in range(samples):
        chosen = rng.choices(sums_counts, k=len(values))
        trials.append(sum(s for s, n in chosen) / sum(n for s, n in chosen))
    return [quantile(trials, .025), quantile(trials, .975)]

def report(meta, rows):
    groups = defaultdict(list)
    for r in rows: groups[r['group']].append(r['correct'])
    by = {}
    for field in ['family', 'language', 'kind', 'variant']:
        buckets = defaultdict(list)
        for r in rows: buckets[r[field]].append(r)
        by[field] = {k: metrics(v) for k, v in buckets.items()}
    return {'run': meta, 'overall': metrics(rows), 'by': by,
            'macro_family_accuracy_all': statistics.mean(x['accuracy_all'] for x in by['family'].values()),
            'accuracy_cluster_bootstrap_95': bootstrap(groups), 'invariance': invariance(rows),
            'interpretation': 'Synthetic diagnostic benchmark. Scenario-cluster intervals do not account for shared templates or prove real-world generalization. Probability metrics exclude invalid/missing responses; accuracy_all counts them as wrong. NLL clips probabilities at 1e-12. Smoke runs are not full benchmark results.'}

def compare(left, right):
    lm, lr = left; rm, rr = right
    if lm['requests_sha256'] != rm['requests_sha256'] or set(lm['expected_ids']) != set(rm['expected_ids']):
        raise ValueError('paired comparison requires identical dataset and case IDs')
    right_by_id = {r['id']: r for r in rr}
    wins = losses = 0
    accuracy, nll = defaultdict(list), defaultdict(list)
    for a in lr:
        b = right_by_id[a['id']]
        d = a['correct'] - b['correct']
        wins += d > 0; losses += d < 0
        accuracy[a['group']].append(d)
        if a['valid'] and b['valid']: nll[a['group']].append(a['nll'] - b['nll'])
    def mean(values):
        values = [v for group in values.values() for v in group]
        return statistics.mean(values) if values else None
    return {'left': lm['model'], 'right': rm['model'], 'left_only_correct': wins, 'right_only_correct': losses,
            'accuracy_delta_left_minus_right': mean(accuracy), 'accuracy_delta_cluster_bootstrap_95': bootstrap(accuracy),
            'nll_delta_on_both_valid': mean(nll), 'nll_delta_cluster_bootstrap_95': bootstrap(nll),
            'nll_paired_cases': sum(map(len, nll.values())),
            'note': 'No unpaired significance test: perturbations are correlated. Intervals are diagnostic and depend on synthetic scenario sampling.'}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--compare', type=Path)
    p.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parent / 'decisionbench-v1')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    left = observations(args.dataset, args.run)
    result = report(*left)
    if args.compare: result['comparison'] = compare(left, observations(args.dataset, args.compare))
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output: args.output.write_text(text)
    else: print(text, end='')
