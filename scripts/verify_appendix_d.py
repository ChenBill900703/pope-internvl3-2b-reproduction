"""Read-only complete or partial audit of AppendixD durable outputs."""
import argparse,csv,json,sqlite3
from pathlib import Path
from run_pope import ROOT,digest,now,metrics,parse_answer
from atomic_io import atomic_json

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
    progress=read(ROOT/'logs/appendixD-progress.json');folder=Path(progress['results']);config=read(folder/'config.json')
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM answers ORDER BY split,qid')]
    raw=[json.loads(line) for line in (folder/'raw-answers.jsonl').read_text(encoding='utf-8').splitlines()] if (folder/'raw-answers.jsonl').exists() else []
    keyed=lambda rr:{(r['prompt_id'],r['question_id']):r for r in rr}
    assert len(rows)==len(raw)==len(keyed(rows)) and keyed(rows)==keyed(raw)
    for name,value in config['source_sha256'].items():assert digest(folder/name)==value
    baseline=Path(config['baseline'])
    with sqlite3.connect(f'file:{baseline/"answers.sqlite"}?mode=ro',uri=True) as db:
        prior={tuple(r[:2]):json.loads(r[2]) for r in db.execute('SELECT split,qid,payload FROM answers')}
    summaries=read(folder/'table11.json');counts={}
    for summary in summaries:
        split=summary['split'];dataset,strategy=split.split('_',1)
        qpath=ROOT/f'third_party/POPE/output/seem/{dataset}/{dataset}_pope_seem_{strategy}.json'
        assert digest(qpath)==config['question_sha256'][split]
        source={q['question_id']:q for q in read(qpath)}
        selected=[r for r in rows if r['prompt_id']==split];counts[split]=len(selected)
        for r in selected:
            for k in ['question_id','image','text','label']:assert r[k]==source[r['question_id']][k]
            assert r['prediction']==parse_answer(r['answer']) and r['answer'].strip()
            path=ROOT/('data/coco/val2014' if dataset=='aokvqa' else 'data/gqa/images')/r['image']
            assert r['image_sha256']==digest(path)==config['image_sha256'][f'{dataset}/{r["image"]}']
            if 'reused_from' in r:
                provenance=r['reused_from']
                if Path(provenance['folder'])==baseline:previous=prior[provenance['split'],provenance['question_id']]
                else:previous=keyed(rows)[provenance['split'],provenance['question_id']]
                for k in ['image_sha256','text','answer','generated_token_ids']:assert r[k]==previous[k]
        for k,v in metrics(selected).items():assert summary[k]==v
        assert keyed(read(folder/f'{split}-answers.json'))==keyed(selected)
        with (folder/f'{split}-answers.csv').open(encoding='utf-8-sig',newline='') as stream:csvrows=list(csv.DictReader(stream))
        assert len(csvrows)==len(selected)
        assert {int(r['question_id']):r['answer'] for r in csvrows}=={r['question_id']:r['answer'] for r in selected}
    complete=all(v==3000 for v in counts.values()) and len(counts)==6
    if not args.allow_partial:assert complete
    atomic_json(folder/('independent-validation.json' if complete else 'partial-validation.json'),{'at':now(),
        'passed':True,'complete':complete,'counts':counts,'original_questions_images_labels_checked':True,
        'sqlite_jsonl_json_csv_identical':True,'reused_outputs_unchanged':True,'script_sha256':digest(__file__)})
    print(json.dumps({'passed':True,'complete':complete,'counts':counts}))

if __name__=='__main__':main()
