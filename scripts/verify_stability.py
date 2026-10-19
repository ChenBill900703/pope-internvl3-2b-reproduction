"""Audit all Table4 records, including copied baseline provenance."""
import argparse,csv,json,sqlite3
from pathlib import Path
import run_pope
from run_pope import ROOT,digest,now,parse_answer,metrics
from atomic_io import atomic_json
from chair_eval import aggregate

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def load_db(folder):
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        return [json.loads(x[0]) for x in db.execute('SELECT payload FROM answers')]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
    state=read(ROOT/'logs/stability-progress.json');folder=Path(state['results']);config=read(folder/'config.json')
    rows=load_db(folder);keyed={(r['prompt_id'],r['question_id']):r for r in rows};assert len(keyed)==len(rows)
    raw=[json.loads(line) for line in (folder/'raw-answers.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(raw)==len(rows) and {(r['prompt_id'],r['question_id']):r for r in raw}==keyed
    for name,value in config['source_sha256'].items():assert digest(folder/name)==value
    table3={(r['split'],r['question_id']):r for r in load_db(Path(config['source_table3']))}
    captions={(r['prompt_id'],r['image_id']):r for r in load_db(Path(config['source_captions']))}
    questions={p:{r['question_id']:r for r in read(ROOT/f'data/table4/{p}.json')} for p in ['P1','P2','P3','P4']}
    image_ids=set(config['table1_image_ids']);checks=[]
    for r in rows:
        prompt=r['prompt_id'];assert r['image_sha256']==digest(ROOT/'data/coco/val2014'/r['image'])
        assert r['answer'].strip() and 0<len(r['generated_token_ids'])<=1024
        if prompt.startswith('P'):
            q=questions[prompt][r['question_id']]
            for key in ['question_id','image','text','label']:assert q[key]==r[key]
            assert r['prediction']==parse_answer(r['answer'])
        else:
            assert r['question_id']==r['image_id'] and r['image_id'] in image_ids
            assert r['prompt']==r['text']==config['caption_prompts'][prompt]
        if prompt in ['P1','C1']:
            source=table3[('random',r['question_id'])] if prompt=='P1' else captions[('I1',r['image_id'])]
            for key in ['answer','generated_token_ids','image_sha256']:assert r[key]==source[key]
            if prompt=='C1':assert r['chair']==source['chair']
    counts={p:sum(r['prompt_id']==p for r in rows) for p in ['P1','P2','P3','P4','C1','C2','C3','C4']}
    complete=all(counts[p]==(3000 if p.startswith('P') else 2000) for p in counts)
    if not args.allow_partial:assert complete,counts
    for entry in read(folder/'metrics.json'):
        selected=[r for r in rows if r['prompt_id']==entry['prompt_id']]
        expected=metrics(selected) if entry['prompt_id'].startswith('P') else aggregate(selected)
        for k,v in expected.items():assert entry[k]==v,(entry['prompt_id'],k)
        exported=read(folder/f'{entry["prompt_id"]}-answers.json')
        assert {r['question_id']:r for r in exported}=={r['question_id']:r for r in selected}
        if selected:
            with (folder/f'{entry["prompt_id"]}-answers.csv').open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
            assert len(csvrows)==len(selected)
            assert {int(r['question_id']):r['answer'] for r in csvrows}=={r['question_id']:r['answer'] for r in selected}
        if entry['prompt_id'].startswith('P') and len(selected)==3000:
            dest=folder/f'{entry["prompt_id"]}-official-validation';dest.mkdir(exist_ok=True)
            run_pope.atomic_json=atomic_json;run_pope.official_crosscheck(selected,dest);checks.append(entry['prompt_id'])
    atomic_json(folder/('validation.json' if complete else 'partial-validation.json'),{'at':now(),'passed':True,
        'complete':complete,'counts':counts,'raw_json_csv_sqlite_identical':True,'copied_outputs_unchanged':True,
        'official_evaluator_and_sklearn_checked':checks,'script_sha256':digest(__file__)})
    print(json.dumps({'passed':True,'complete':complete,'counts':counts}))

if __name__=='__main__':main()
