"""Audit local VQA diagnostics against every completed raw answer and CSV export."""
import csv
import json
import sqlite3
from pathlib import Path
from atomic_io import atomic_json
from grade_native_vqa import score
from run_pope import ROOT, digest, now


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def same_csv(path, expected):
    with path.open(encoding='utf-8-sig', newline='') as f:
        actual = list(csv.DictReader(f))
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert set(a) == set(e)
        assert all(a[k] == ('' if v is None else str(v)) for k, v in e.items())


def main():
    folder = Path(read(ROOT/'logs/vqa-progress.json')['results'])
    validation = folder/'raw-validation.json'
    assert read(validation)['passed'] and read(validation)['raw_complete']
    out = folder/'native-evaluation'
    protocol = read(out/'protocol.json')
    assert protocol['complete'] and not protocol['external_examiner_called']
    for key, file in [('raw_config_sha256', folder/'config.json'),
                      ('raw_jsonl_sha256', folder/'raw-answers.jsonl'),
                      ('raw_validation_sha256', validation), ('grader_sha256', out/'runner-executed.py')]:
        assert protocol[key] == digest(file)
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro', uri=True) as db:
        rows = [json.loads(r[0]) for r in db.execute('SELECT payload FROM answers ORDER BY split,qid')]
    metrics = read(out/'metrics.json')
    same_csv(out/'metrics.csv', metrics)
    examples = []
    for item in metrics:
        dataset = item['dataset']
        selected = [r for r in rows if r['dataset'] == dataset]
        expected, per_question = score(dataset, selected)
        assert item == expected
        assert read(out/f'{dataset}-per-question.json') == per_question
        same_csv(out/f'{dataset}-per-question.csv', per_question)
        examples.extend({'dataset': dataset, 'question_id': r['question_id'], 'question': r['text'],
                         'raw_answer': r['answer'], 'references': r['reference_answers']} for r in selected[:3])
    atomic_json(out/'format-examples.json', {'selection': 'First three IDs in SQLite split,qid order per dataset; not a semantic correctness evaluation',
                                          'examples': examples})
    result = {'at': now(), 'passed': True, 'raw_answers_checked': len(rows),
              'all_per_question_scores_and_eligible_denominators_recomputed': True,
              'json_csv_and_raw_answers_identical': True, 'no_external_grading': True,
              'metrics': metrics, 'script_sha256': digest(__file__)}
    atomic_json(out/'independent-validation.json', result)
    (out/'verify-executed.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
