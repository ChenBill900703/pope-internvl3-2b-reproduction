"""Audit the completed 4000 Table 1 records while extra captions keep running."""
import csv
import json
import sqlite3
import textwrap
from functools import partial
from pathlib import Path

from atomic_io import atomic_json
from chair_eval import aggregate, make_parser
from run_pope import ROOT, digest, now


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    folder = Path(read(ROOT/'logs/captions-progress.json')['results'])
    config = read(folder/'config.json')
    ids = set(config['table1_image_ids'])
    assert len(ids) == 2000
    assert digest(ROOT/'third_party/POPE/segmentation/coco_ground_truth_segmentation.json') == config['ground_truth_sha256']
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro', uri=True) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        rows = [json.loads(x[0]) for x in db.execute('SELECT payload FROM answers ORDER BY split,qid')]
    rows = [r for r in rows if r['image_id'] in ids]
    keyed = {(r['prompt_id'], r['image_id']): r for r in rows}
    assert len(rows) == len(keyed) == 4000
    # The worker may append extra-image rows. All Table 1 rows are already immutable.
    raw = []
    with (folder/'raw-captions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if not line.endswith('\n'):
                break
            r = json.loads(line)
            if r['image_id'] in ids:
                raw.append(r)
    assert len(raw) == 4000
    assert {(r['prompt_id'], r['image_id']): r for r in raw} == keyed
    for name, key in [('run_captions.py', 'runner_sha256'), ('model_runtime.py', 'runtime_sha256'),
                      ('chair_eval.py', 'chair_adapter_sha256'), ('caption_quality.py', 'quality_adapter_sha256')]:
        assert digest(folder/name) == config[key], name
    image_hashes = {}
    for r in rows:
        assert r['prompt'] == config['prompts'][r['prompt_id']]
        assert r['image'] == f'COCO_val2014_{r["image_id"]:012d}.jpg'
        if r['image_id'] not in image_hashes:
            image_hashes[r['image_id']] = digest(ROOT/'data/coco/val2014'/r['image'])
        assert r['image_sha256'] == image_hashes[r['image_id']]
        assert r['answer'].strip() and 0 < len(r['generated_token_ids']) <= 1024
        assert all(isinstance(t, int) and t >= 0 for t in r['generated_token_ids'])
        assert r['generation_finish_reason'] == ('length' if r['possible_truncation'] else 'eos')

    source = (ROOT/'third_party/CHAIR/utils/chair.py').read_text(encoding='utf-8')
    compute = textwrap.dedent(source[source.index('    def compute_chair('):source.index('def load_generated_captions(')])
    loader = source[source.index('def load_generated_captions('):source.index('def save_hallucinated_words(')]
    audit = folder/'analysis/table1-audit'
    audit.mkdir(exist_ok=True)
    analysis = {r['prompt_id']: r for r in read(folder/'analysis/table1.json')}
    checks = []
    for prompt in ['I1', 'I2']:
        selected = [r for r in rows if r['prompt_id'] == prompt]
        assert len(selected) == 2000 and {r['image_id'] for r in selected} == ids
        exported = [r for r in read(folder/f'{prompt}-captions.json') if r['image_id'] in ids]
        assert {(r['prompt_id'], r['image_id']): r for r in exported} == {(r['prompt_id'], r['image_id']): r for r in selected}
        with (folder/f'{prompt}-captions.csv').open(encoding='utf-8-sig', newline='') as f:
            csvrows = [r for r in csv.DictReader(f) if int(r['image_id']) in ids]
        assert len(csvrows) == 2000
        assert {int(r['image_id']): r['answer'] for r in csvrows} == {r['image_id']: r['answer'] for r in selected}
        parser, gt = make_parser()
        ns = type(parser).caption_to_words.__globals__
        ns['json'] = json
        ns['open'] = partial(open, encoding='utf-8')
        exec(compile(loader+'\n'+compute, 'original_chair_compute', 'exec'), ns)
        type(parser).compute_chair = ns['compute_chair']
        parser.imid_to_objects = {r['image_id']: gt[r['image_id']] for r in selected}
        keys = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'CIDEr', 'SPICE', 'ROUGE_L']
        placeholders = {k: None for k in keys}
        inputs = {'overall': placeholders, 'imgToEval': {
            str(r['image_id']): {'image_id': r['image_id'], 'caption': r['answer'], **placeholders} for r in selected}}
        input_path = audit/f'{prompt}-official-input.json'
        atomic_json(input_path, inputs)
        official = parser.compute_chair(str(input_path))
        expected = aggregate(selected)
        for key in ['CHAIRi', 'CHAIRs']:
            assert abs(official['overall_metrics'][key]-expected[key]) < 1e-12
        for actual, r in zip(official['sentences'], selected):
            assert actual['mscoco_generated_words'] == r['chair']['mentioned_objects']
            assert [x[1] for x in actual['mscoco_hallucinated_words']] == [h['object'] for h in r['chair']['hallucinated_objects']]
        for key, value in expected.items():
            assert analysis[prompt][key] == value
        atomic_json(audit/f'{prompt}-official-output.json', official)
        checks.append({'prompt_id': prompt, **expected, 'official_per_caption_objects_identical': True})
    result = {'at': now(), 'passed': True, 'records': 4000, 'images_per_prompt': 2000,
              'sqlite_jsonl_json_csv_identical': True, 'images_prompts_tokens_source_hashes_checked': True,
              'original_compute_chair_crosscheck': True, 'checks': checks,
              'scope': 'Table 1 only; additional POPE captions still running are excluded',
              'script_sha256': digest(__file__)}
    atomic_json(folder/'analysis/table1-validation.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
