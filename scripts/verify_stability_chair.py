"""Original CHAIR cross-check and independent Table 4 mean/std audit."""
import json
import sqlite3
import statistics
import textwrap
from functools import partial
from pathlib import Path
from atomic_io import atomic_json
from chair_eval import aggregate, make_parser
from run_pope import ROOT, digest, now


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    folder = Path(read(ROOT/'logs/stability-progress.json')['results'])
    assert read(folder/'validation.json')['passed']
    result = read(folder/'table4.json')
    assert result['complete']
    metrics = read(folder/'metrics.json')
    for prefix, key, mean_key, std_key in [
        ('P', 'f1', 'POPE_F1_mean', 'POPE_F1_population_std'),
        ('C', 'CHAIRi', 'CHAIRi_mean', 'CHAIRi_population_std')]:
        values = [r[key] for r in metrics if r['prompt_id'].startswith(prefix)]
        assert len(values) == 4
        assert abs(statistics.mean(values)-result[mean_key]) < 1e-14
        assert abs(statistics.pstdev(values)-result[std_key]) < 1e-14
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro', uri=True) as db:
        rows = [json.loads(r[0]) for r in db.execute('SELECT payload FROM answers ORDER BY split,qid')]
    source = (ROOT/'third_party/CHAIR/utils/chair.py').read_text(encoding='utf-8')
    compute = textwrap.dedent(source[source.index('    def compute_chair('):source.index('def load_generated_captions(')])
    loader = source[source.index('def load_generated_captions('):source.index('def save_hallucinated_words(')]
    out = folder/'official-chair-crosscheck'
    out.mkdir(exist_ok=True)
    checks = []
    for prompt in ['C1', 'C2', 'C3', 'C4']:
        selected = [r for r in rows if r['prompt_id'] == prompt]
        assert len(selected) == 2000
        parser, gt = make_parser()
        ns = type(parser).caption_to_words.__globals__
        ns.update(json=json, open=partial(open, encoding='utf-8'))
        exec(compile(loader+'\n'+compute, 'original_chair_compute', 'exec'), ns)
        type(parser).compute_chair = ns['compute_chair']
        parser.imid_to_objects = {r['image_id']: gt[r['image_id']] for r in selected}
        placeholders = {k: None for k in ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'CIDEr', 'SPICE', 'ROUGE_L']}
        inputs = {'overall': placeholders, 'imgToEval': {str(r['image_id']): {
            'image_id': r['image_id'], 'caption': r['answer'], **placeholders} for r in selected}}
        input_path = out/f'{prompt}-input.json'
        atomic_json(input_path, inputs)
        official = parser.compute_chair(str(input_path))
        expected = aggregate(selected)
        for k in ['CHAIRi', 'CHAIRs']:
            assert abs(official['overall_metrics'][k]-expected[k]) < 1e-12
        for actual, r in zip(official['sentences'], selected):
            assert actual['mscoco_generated_words'] == r['chair']['mentioned_objects']
            assert [h[1] for h in actual['mscoco_hallucinated_words']] == [h['object'] for h in r['chair']['hallucinated_objects']]
        atomic_json(out/f'{prompt}-output.json', official)
        checks.append({'prompt_id': prompt, 'n': 2000, 'CHAIRi': expected['CHAIRi'], 'CHAIRs': expected['CHAIRs']})
    validation = {'at': now(), 'passed': True, 'caption_records_checked': 8000,
                  'original_compute_chair_and_all_objects_identical': True,
                  'mean_and_population_std_recomputed_from_four_unrounded_scores': True,
                  'checks': checks, 'script_sha256': digest(__file__)}
    atomic_json(folder/'chair-summary-validation.json', validation)
    (out/'verify-executed.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
