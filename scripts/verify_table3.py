"""Verify completed splits against source questions, durable outputs and official scores."""
import argparse,csv,hashlib,io,json,sqlite3
from contextlib import redirect_stdout
from pathlib import Path
from run_pope import ROOT,atomic_json,digest,load_rows,metrics,now,parse_answer

def verify(folder,split):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
        saved=[json.loads(x[0]) for x in db.execute('SELECT payload FROM answers WHERE split=? ORDER BY qid',(split,))]
    assert len(saved)==3000,(split,len(saved))
    original=load_rows(split);images={}
    for r,q in zip(saved,original):
        assert all(r[k]==v for k,v in q.items()),q['question_id']
        assert r['answer'].strip() and r['prediction']==parse_answer(r['answer'])
        assert r['generated_token_ids'][-1]==151645 and not r['possible_truncation']
        if r['image'] not in images:images[r['image']]=digest(ROOT/'data/coco/val2014'/r['image'])
        assert images[r['image']]==r['image_sha256']
    raw={}
    for line in (folder/'raw-answers.jsonl').read_text(encoding='utf-8').splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue # another split may be appending its last line
        if r['split']==split:
            assert r['question_id'] not in raw
            raw[r['question_id']]=r
    assert len(raw)==3000
    assert all(raw[r['question_id']]==r for r in saved)
    exported=json.loads((folder/f'{split}-answers.json').read_text(encoding='utf-8'))
    assert exported==saved
    with (folder/f'{split}-answers.csv').open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==3000
    for r,c in zip(saved,csvrows):
        for key in ['image','text','label','answer','prediction']:assert r[key]==c[key]
        assert r['question_id']==int(c['question_id'])
    m=metrics(saved)
    published=json.loads((folder/f'{split}-metrics.json').read_text(encoding='utf-8'))
    assert all(published[k]==v for k,v in m.items())
    y=[r['label']=='yes' for r in saved];pred=[r['prediction']=='yes' for r in saved]
    independent={'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),
                 'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0)}
    assert all(abs(m[k]-v)<1e-12 for k,v in independent.items())
    af=folder/f'{split}-official-answers.jsonl';lf=folder/f'{split}-official-labels.jsonl'
    af.write_text(''.join(json.dumps({'question':r['text'],'answer':r['answer']})+'\n' for r in saved),encoding='utf-8')
    lf.write_text(''.join(json.dumps({'label':r['label']})+'\n' for r in saved),encoding='utf-8')
    source=(ROOT/'third_party/POPE/evaluate.py').read_text()
    source=source.replace("ans_file = ''",'ans_file = '+repr(str(af))).replace("label_file = ''",'label_file = '+repr(str(lf)))
    scope={};output=io.StringIO()
    with redirect_stdout(output):exec(compile(source,'official_evaluate.py','exec'),scope)
    for k,official in [('accuracy','acc'),('precision','precision'),('recall','recall'),('f1','f1'),('yes_ratio','yes_ratio')]:
        assert abs(m[k]-scope[official])<1e-12
    (folder/f'{split}-official-evaluator.txt').write_text(output.getvalue(),encoding='utf-8')
    validation={'at':now(),'split':split,'passed':True,'questions':3000,'images':len(images),
                'original_questions_and_labels':True,'image_sha256':True,'all_generated_eos':True,
                'sqlite_raw_jsonl_json_csv_agree':True,'official_evaluator_matches':True,'sklearn_matches':True,
                'metrics':m}
    atomic_json(folder/f'{split}-validation.json',validation)
    print(json.dumps(validation))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['random','popular','adversarial','all'],default='all');args=ap.parse_args()
    p=json.loads((ROOT/'logs/progress.json').read_text(encoding='utf-8'));folder=Path(p['results'])
    source=ROOT/'scripts/run_pope.py';config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    assert digest(source)==config['runner_sha256']
    snapshot=folder/'runner-executed.py'
    if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    assert digest(snapshot)==config['runner_sha256']
    for split in ['random','popular','adversarial'] if args.split=='all' else [args.split]:verify(folder,split)
