"""Verify official VQA subsets and raw output provenance, without grading answers."""
import argparse,csv,json,sqlite3
from pathlib import Path
from run_pope import ROOT,digest,now
from atomic_io import atomic_json

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
    state=read(ROOT/'logs/vqa-progress.json');folder=Path(state['results']);config=read(folder/'config.json')
    assert not config['examiner_called'] and not config['references_in_model_input'] and not config['choices_in_model_input']
    assert config['generation']=={'do_sample':False,'num_beams':1,'max_new_tokens':1024}
    for name,value in config['source_sha256'].items():assert digest(folder/name)==value
    questions=read(folder/'original-questions.json');expected={}
    for dataset in ['aokvqa','gqa']:
        path=ROOT/f'data/{dataset}/pope-image-vqa-questions.json';assert digest(path)==config['question_sha256'][dataset]
        actual=read(path)
        expected[dataset]={q['question_id']:q for q in questions[dataset]}
        assert len(expected[dataset])==config['expected_counts'][dataset]
        originals={str(q['question_id']):q for q in actual} if dataset=='aokvqa' else actual
        assert set(originals)==set(expected[dataset])
        for qid,q in expected[dataset].items():
            assert isinstance(qid,str) and q['question_metadata']==originals[qid]
            assert q['text']==originals[qid]['question']
            refs=originals[qid]['direct_answers'] if dataset=='aokvqa' else [originals[qid]['answer']]
            assert q['reference_answers']==refs
        # Independently verify the prepared subset against the full official
        # validation annotations, rather than trusting a derived JSON alone.
        if dataset=='aokvqa':
            source=ROOT/'data/aokvqa/aokvqa_v1p0_val.json'
            assert digest(source)==read(ROOT/'data/aokvqa/audit.json')['annotation_sha256']
            full={str(q['question_id']):q for q in read(source)}
            image_ids={q['image_id'] for q in originals.values()}
            selected={qid:q for qid,q in full.items() if q['image_id'] in image_ids}
        else:
            source=ROOT/'data/gqa/val_balanced_questions.json'
            assert digest(source)==read(ROOT/'data/gqa/vqa-questions-audit.json')['member_sha256']
            full=read(source);image_ids={str(q['imageId']) for q in originals.values()}
            selected={qid:q for qid,q in full.items() if str(q['imageId']) in image_ids}
        assert len(image_ids)==500 and selected==originals
        del full,selected
        for image,value in config['image_sha256'][dataset].items():
            path=ROOT/('data/coco/val2014' if dataset=='aokvqa' else 'data/gqa/images')/image
            assert digest(path)==value
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert {r[1]:r[2] for r in db.execute('PRAGMA table_info(answers)')}['qid']=='TEXT'
        rows=[json.loads(x[0]) for x in db.execute('SELECT payload FROM answers ORDER BY split,qid')]
    keyed=lambda rr:{(r['dataset'],r['question_id']):r for r in rr}
    rawpath=folder/'raw-answers.jsonl'
    raw=[json.loads(line) for line in rawpath.read_text(encoding='utf-8').splitlines()] if rawpath.exists() else []
    assert len(rows)==len(raw)==len(keyed(rows)) and keyed(rows)==keyed(raw)
    for r in rows:
        assert isinstance(r['question_id'],str)
        dataset=r['dataset'];q=expected[dataset][r['question_id']]
        for key in ['image_id','image','text','reference_answers','question_metadata']:assert r[key]==q[key]
        assert r['image_sha256']==config['image_sha256'][dataset][r['image']]
        assert r['answer'].strip() and 0<len(r['generated_token_ids'])<=1024
        if 'reused_from' in r:
            origin=r['reused_from'];source=Path(origin['folder']).resolve();assert source.is_relative_to(ROOT/'results')
            with sqlite3.connect(f'file:{source/"answers.sqlite"}?mode=ro',uri=True) as db:
                prior=db.execute('SELECT payload FROM answers WHERE split=? AND qid=?',
                    (origin['split'],origin['question_id'])).fetchone()
            assert prior;prior=json.loads(prior[0])
            for key in ['answer','generated_token_ids','image_sha256']:assert r[key]==prior[key]
            assert r['text']==prior.get('text',prior.get('prompt'))
    summaries=read(folder/'raw-generation-summary.json');counts={}
    for item in summaries:
        dataset=item['dataset'];selected=[r for r in rows if r['dataset']==dataset];counts[dataset]=len(selected)
        assert item['saved']==len(selected) and item['expected']==config['expected_counts'][dataset]
        assert not item['graded'] and not item['external_examiner_called']
        assert keyed(read(folder/f'{dataset}-answers.json'))==keyed(selected)
        with (folder/f'{dataset}-answers.csv').open(encoding='utf-8-sig',newline='') as stream:csvrows=list(csv.DictReader(stream))
        assert len(csvrows)==len(selected)
        assert {r['question_id']:r['answer'] for r in csvrows}=={r['question_id']:r['answer'] for r in selected}
    complete=counts==config['expected_counts']
    if not args.allow_partial:assert complete
    atomic_json(folder/('raw-validation.json' if complete else 'partial-validation.json'),{'at':now(),'passed':True,
        'raw_complete':complete,'counts':counts,'expected':config['expected_counts'],
        'official_full_annotations_and_exact_subsets_checked':True,'question_ids_preserved_as_text':True,
        'original_question_images_references_and_raw_exports_checked':True,'raw_reuse_tokens_unchanged':True,
        'examiner_called':False,'graded':False,'scope':'Disclosed extension; Table6 exact subset unknown',
        'script_sha256':digest(__file__)})
    print(json.dumps({'passed':True,'raw_complete':complete,'counts':counts,'graded':False}))

if __name__=='__main__':main()
