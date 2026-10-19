"""Audit completed consistency and Appendix F artifacts without rerunning inference."""
import collections
import csv
import json
import math
import sqlite3
from pathlib import Path
from atomic_io import atomic_json
from run_pope import ROOT, digest, now


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(folder):
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro', uri=True) as db:
        return [json.loads(r[0]) for r in db.execute('SELECT payload FROM answers')]


def csv_identical(path, expected):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        actual = list(csv.DictReader(stream))
    assert len(actual) == len(expected), str(path)
    for a, e in zip(actual, expected):
        assert set(a) == set(e), str(path)
        assert all(a[k] == ('' if v is None else str(v)) for k, v in e.items()), str(path)


def main():
    folder = Path(read(ROOT/'logs/captions-progress.json')['results'])
    table3 = Path(read(ROOT/'logs/progress.json')['results'])
    assert read(folder/'captions-validation.json')['passed']
    caps = {(r['prompt_id'], r['image_id']): r for r in rows(folder)}
    pope = {(r['split'], r['question_id']): r for r in rows(table3)}
    out = folder/'consistency'
    protocol = read(out/'protocol.json')
    assert protocol['complete'] and not protocol['partial_audit_only']
    assert protocol['source_config_sha256'] == {'captions': digest(folder/'config.json'), 'table3': digest(table3/'config.json')}
    assert protocol['script_sha256'] == digest(ROOT/'scripts/analyze_consistency.py')
    pairs = read(out/'all-pairs.json')
    assert len(pairs) == 18000
    csv_identical(out/'all-pairs.csv', pairs)
    seen = set()
    for r in pairs:
        key = (r['caption_prompt_id'], r['pope_split'], r['question_id'])
        assert key not in seen
        seen.add(key)
        q = pope[r['pope_split'], r['question_id']]
        c = caps[r['caption_prompt_id'], r['image_id']]
        assert r['image'] == c['image'] == q['image']
        assert c['image_sha256'] == q['image_sha256']
        assert r['caption'] == c['answer'] and r['pope_answer'] == q['answer']
        assert r['label'] == q['label'] and r['prediction'] == q['prediction']
        assert r['caption_finish_reason'] == c['generation_finish_reason']
        assert q['text'] in [f'Is there a {r["object"]} in the image?', f'Is there an {r["object"]} in the image?']
        count = collections.Counter(c['chair']['mentioned_objects'])[r['canonical_object']]
        assert count == r['caption_mention_instances'] and (count > 0) == r['caption_mentioned']
    summary = read(out/'summary.json')
    assert len(summary) == 6
    csv_identical(out/'summary.csv', summary)
    for s in summary:
        selected = [r for r in pairs if (r['caption_prompt_id'], r['pope_split']) == (s['caption_prompt_id'], s['pope_split'])]
        assert len(selected) == s['probed_image_object_pairs'] == 3000
        assert len({r['image_id'] for r in selected}) == s['images'] == 500
        no = [r for r in selected if r['prediction'] == 'no']
        mentioned = [r for r in selected if r['caption_mentioned']]
        assert len(no) == s['no_pairs'] and len(mentioned) == s['caption_mentioned_probed_pairs']
        assert sum(r['caption_mentioned'] for r in no) == s['no_pairs_mentioned_in_caption']
        assert sum(r['prediction'] == 'yes' for r in mentioned) == s['caption_mentioned_pairs_answered_yes']
        assert s['mentioned_given_no'] == s['no_pairs_mentioned_in_caption']/len(no)
        assert s['yes_given_caption_mentioned'] == s['caption_mentioned_pairs_answered_yes']/len(mentioned)
        assert s['no_caption_mention_instances'] == sum(r['caption_mention_instances'] for r in no)
        assert s['caption_mentioned_probed_instances'] == sum(r['caption_mention_instances'] for r in mentioned)
    refs_path = ROOT/'data/coco/pope-500-caption-references.json'
    refs = read(refs_path)
    appendix = folder/'appendixF'
    protocol = read(appendix/'protocol.json')
    assert protocol['complete'] and protocol['references_sha256'] == digest(refs_path)
    assert protocol['config_sha256'] == digest(folder/'config.json')
    assert digest(appendix/'runner-executed.py') == protocol['provenance']['script_sha256']
    for file, sha in protocol['provenance']['files'].items():
        assert digest(ROOT/file) == sha
    metrics = read(appendix/'metrics.json')
    assert len(metrics) == 2
    csv_identical(appendix/'metrics.csv', metrics)
    for s in metrics:
        prompt = s['prompt_id']
        assert s['n'] == 500 and s['complete']
        raw = read(appendix/f'{prompt}-raw-evaluation-inputs.json')
        assert raw['references'] == {i: [{'caption': text} for text in texts] for i, texts in refs.items()}
        assert set(raw['hypotheses']) == set(refs)
        for image_id, hypotheses in raw['hypotheses'].items():
            assert hypotheses == [{'caption': caps[prompt, int(image_id)]['answer']}]
        per_image = read(appendix/f'{prompt}-per-image.json')
        assert len(per_image) == 500 and {r['image_id'] for r in per_image} == {int(i) for i in refs}
        csv_identical(appendix/f'{prompt}-per-image.csv', per_image)
        for r in [s, *per_image]:
            for metric in ['BLEU-1', 'BLEU-2', 'METEOR', 'ROUGE-L']:
                assert math.isfinite(r[metric]) and 0 <= r[metric] <= 1.000001
        assert abs(s['ROUGE-L'] - sum(r['ROUGE-L'] for r in per_image)/500) < 1e-12
        assert s['length_limit_count'] == sum(caps[prompt, int(i)]['possible_truncation'] for i in refs)
        tokenized = read(appendix/f'{prompt}-tokenized.json')
        assert set(tokenized['references']) == set(tokenized['hypotheses']) == set(refs)
        assert all(len(tokenized['hypotheses'][i]) == 1 for i in refs)
    result = {'at': now(), 'passed': True, 'consistency_pairs': 18000, 'conditions': 6,
              'appendixF_images_per_prompt': 500, 'caption_and_reference_inputs_identical': True,
              'json_csv_identical': True, 'consistency_counts_and_rates_recomputed': True,
              'metric_outputs_finite_and_provenance_checked': True,
              'scope': 'Artifact integrity and source alignment; uses already executed original metric engines, no model or scorer rerun',
              'script_sha256': digest(__file__)}
    atomic_json(folder/'caption-analyses-validation.json', result)
    for script in ['analyze_consistency.py', 'verify_caption_analyses.py']:
        (out/script).write_bytes((ROOT/'scripts'/script).read_bytes())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
