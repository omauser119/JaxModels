#!/usr/bin/env python3
"""Versioned synthetic task corpus. References are for independent scoring, not teacher prompts."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random

FAMILIES = ['access_policy', 'appointment_conflict', 'delivery_window', 'document_fields',
            'catalog_filter', 'expense_policy', 'language_routing', 'measurement_band',
            'message_intent', 'notification_preference', 'priority_policy', 'record_consistency',
            'reservation_capacity', 'return_policy', 'task_dependency', 'workflow_transition']
SPLITS = ['train', 'validation', 'calibration', 'test']


def case(family, language, i, split):
    rng = random.Random(f'abel-corpus-v2/{family}/{language}/{i}/{split}')
    ru = language == 'ru'
    text = lambda en, rus: rus if ru else en
    n = rng.randint(1, 9999)
    # Split-specific field names and phrasings; related symbolic families remain shared.
    marker = {'train': 'record', 'validation': 'entry', 'calibration': 'item', 'test': 'submission'}[split]
    if family == 'access_policy':
        state = {'role': rng.choice(['reader', 'editor', 'owner']), 'blocked': rng.choice([True, False])}
        yes = state['role'] in ('editor', 'owner') and not state['blocked']
        rule = text('Editing is allowed only for editor or owner when blocked is false.', 'Редактирование разрешено только editor или owner, если blocked равно false.')
    elif family == 'appointment_conflict':
        start = rng.randint(8, 16); end = start + rng.randint(1, 3); wanted = rng.randint(8, 19)
        state = {'booked_start': start, 'booked_end': end, 'requested_hour': wanted}
        yes = start <= wanted < end
        rule = text('A conflict occurs when requested_hour is at least booked_start and strictly before booked_end.', 'Конфликт есть, если requested_hour не меньше booked_start и строго меньше booked_end.')
    elif family == 'delivery_window':
        low = rng.randint(1, 12); high = low + rng.randint(1, 7); actual = rng.randint(1, 20)
        state = {'first_day': low, 'last_day': high, 'arrival_day': actual}
        yes = low <= actual <= high
        rule = text('Arrival is on time when arrival_day is within first_day and last_day inclusive.', 'Доставка своевременна, если arrival_day находится между first_day и last_day включительно.')
    elif family == 'document_fields':
        state = {'required': ['title', 'owner', 'date'], 'present': rng.sample(['title', 'owner', 'date', 'note'], rng.randint(1, 4))}
        yes = set(state['required']) <= set(state['present'])
        rule = text('The document is complete only if every required field is in present.', 'Документ заполнен полностью, только если каждое поле required есть в present.')
    elif family == 'catalog_filter':
        cost = rng.randint(1, 300); budget = rng.randint(1, 300)
        state = {'price': cost, 'budget': budget, 'in_stock': rng.choice([True, False])}
        yes = cost <= budget and state['in_stock']
        rule = text('An item qualifies only when in_stock is true and price is at most budget.', 'Товар подходит, только если in_stock равно true и price не превышает budget.')
    elif family == 'expense_policy':
        state = {'amount': rng.randint(1, 150), 'limit': rng.randint(1, 150), 'receipt': rng.choice([True, False])}
        yes = state['amount'] <= state['limit'] and state['receipt']
        rule = text('Reimburse only when receipt is true and amount does not exceed limit.', 'Возмещать расход можно только при receipt=true и amount не больше limit.')
    elif family == 'language_routing':
        en = ['Please arrange a meeting.', 'The order is ready.', 'I need more information.', 'Thank you for your reply.']
        rus = ['Пожалуйста, назначьте встречу.', 'Заказ готов.', 'Мне нужна дополнительная информация.', 'Спасибо за ответ.']
        positive = rng.choice([True, False]); message = rng.choice(rus if positive else en)
        state = {'message': message}; yes = positive
        rule = text('The condition holds when message is written in Russian.', 'Условие выполнено, если message написано на русском языке.')
    elif family == 'measurement_band':
        low = rng.randint(0, 40); high = low + rng.randint(1, 20)
        state = {'lower': low, 'upper': high, 'value': rng.randint(0, 65)}
        yes = low <= state['value'] <= high
        rule = text('The reading is in range when lower <= value <= upper.', 'Показание в диапазоне при lower <= value <= upper.')
    elif family == 'message_intent':
        positive = rng.choice([True, False])
        good = ['Please cancel my reservation.', 'I would like to cancel the booking.', 'Отмените моё бронирование.', 'Я прошу отменить бронь.']
        bad = ['Do not cancel my reservation.', 'If I needed cancellation I would ask; keep my booking.', 'Не отменяйте моё бронирование.', 'Сохраните мою бронь, отмена не нужна.']
        state = {'message': (good if positive else bad)[rng.randrange(2) + (2 if ru else 0)]}; yes = positive
        rule = text('The condition holds only if the author currently requests cancellation.', 'Условие выполнено, только если автор сейчас просит отмену бронирования.')
    elif family == 'notification_preference':
        state = {'opt_in': rng.choice([True, False]), 'muted': rng.choice([True, False]), 'channel': rng.choice(['email', 'sms'])}
        yes = state['opt_in'] and not state['muted'] and state['channel'] == 'email'
        rule = text('Send email only for opt_in=true, muted=false and channel=email.', 'Отправлять письмо можно только при opt_in=true, muted=false и channel=email.')
    elif family == 'priority_policy':
        state = {'blocked_users': rng.randint(0, 15), 'workaround': rng.choice([True, False]), 'threshold': rng.randint(1, 15)}
        yes = state['blocked_users'] >= state['threshold'] and not state['workaround']
        rule = text('Escalate when blocked_users reaches threshold and no workaround exists.', 'Эскалация нужна, если blocked_users не меньше threshold и workaround=false.')
    elif family == 'record_consistency':
        a = rng.randint(1, 999); b = a if rng.choice([True, False]) else a + 1
        state = {'expected_id': f'R{a}', 'observed_id': f'R{b}'}; yes = a == b
        rule = text('The record matches only if observed_id exactly equals expected_id.', 'Запись совпадает, только если observed_id точно равен expected_id.')
    elif family == 'reservation_capacity':
        state = {'capacity': rng.randint(1, 100), 'reserved': rng.randint(0, 100), 'new_seats': rng.randint(1, 10)}
        yes = state['reserved'] + state['new_seats'] <= state['capacity']
        rule = text('Accept only if reserved plus new_seats is at most capacity.', 'Принять заявку можно, только если reserved плюс new_seats не превышает capacity.')
    elif family == 'return_policy':
        state = {'elapsed_days': rng.randint(0, 45), 'window_days': rng.randint(1, 35), 'sealed': rng.choice([True, False])}
        yes = state['elapsed_days'] <= state['window_days'] and state['sealed']
        rule = text('A return is allowed only within window_days inclusive and with sealed=true.', 'Возврат разрешён только в пределах window_days включительно при sealed=true.')
    elif family == 'task_dependency':
        needed = rng.sample(['A', 'B', 'C', 'D', 'E'], 2); finished = rng.sample(['A', 'B', 'C', 'D', 'E'], rng.randint(0, 5))
        state = {'dependencies': needed, 'finished': finished}; yes = set(needed) <= set(finished)
        rule = text('A task can start only if every dependency is in finished.', 'Задачу можно начать, только если каждая зависимость из dependencies есть в finished.')
    else:
        state = {'current': rng.choice(['draft', 'review', 'done']), 'approved': rng.choice([True, False])}
        yes = state['current'] == 'review' and state['approved']
        rule = text('Move to done only from review and when approved is true.', 'Переход в done разрешён только из review при approved=true.')
    # Neutral unique identifier; no target or split in the model-visible state.
    state['reference'] = hashlib.sha256(f'{family}/{language}/{i}/{split}/{n}'.encode()).hexdigest()[:16]
    state = {marker: state}
    wording = {
        'train': text('Evaluate this rule against the supplied record. ', 'Проверь это правило по записи. '),
        'validation': text('Using only the entry, determine whether the condition holds. ', 'По данным entry определи выполнение условия. '),
        'calibration': text('Check the following requirement for item. ', 'Проверь следующее требование для item. '),
        'test': text('Decide whether submission satisfies the stated condition. ', 'Определи, соответствует ли submission указанному условию. '),
    }[split] + rule
    kind = ['noul', 'choice', 'score'][i % 3]
    if kind == 'noul':
        q = {'type': kind, 'instructions': wording}; target = int(yes)
    elif kind == 'choice':
        names = rng.sample(['amber', 'jade', 'cobalt', 'silver', 'indigo', 'coral'], 2)
        # Random mapping avoids making named labels fixed classifiers.
        q = {'type': kind, 'instructions': wording, 'criteria': {
            names[0]: text('The stated condition is satisfied.', 'Указанное условие выполнено.'),
            names[1]: text('The stated condition is not satisfied.', 'Указанное условие не выполнено.')}}
        target = names[0] if yes else names[1]
    else:
        levels = [text('The condition is not satisfied.', 'Условие не выполнено.'), text('The condition is satisfied.', 'Условие выполнено.')]
        if rng.choice([True, False]): levels.reverse(); target = int(not yes)
        else: target = int(yes)
        q = {'type': kind, 'instructions': wording, 'criteria': levels}
    return state, q, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/abel-corpus-v2'))
    args = parser.parse_args(); out = args.output
    if out.exists(): raise SystemExit('Output exists; version the corpus instead of overwriting')
    out.mkdir(parents=True)
    rows, references = [], []
    counts = {'train': 500, 'validation': 50, 'calibration': 25, 'test': 50}
    for family in FAMILIES:
        for language in ['en', 'ru']:
            for split, count in counts.items():
                for i in range(count):
                    ident = f'{split}-{family}-{language}-{i:04d}'
                    state, q, target = case(family, language, i, split)
                    # Families are shared, source instances are not. This is not an external-source holdout.
                    rows.append({'id': ident, 'split': split, 'family': family, 'group': ident, 'state': state, 'question': q})
                    references.append({'id': ident, 'split': split, 'family': family, 'language': language, 'target': target})
    # Interleave domains deterministically so early diagnostic shards are not single-domain.
    random.Random(19092027).shuffle(rows)
    for name, content in [('tasks.jsonl', rows), ('references.jsonl', references)]:
        (out / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in content))
    manifest = {'format': 'jax.synthetic.corpus.v2', 'tasks': len(rows), 'split_counts': dict(collections.Counter(r['split'] for r in rows)),
                'families': FAMILIES, 'languages': ['en', 'ru'], 'source': 'authored synthetic rule scenarios', 'license': 'MIT',
                'production_evidence': False, 'limitations': 'Related rule templates, two-class primitives; no external real-world data. Size is not independence.',
                'sha256': {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in ['tasks.jsonl', 'references.jsonl']}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__': main()
