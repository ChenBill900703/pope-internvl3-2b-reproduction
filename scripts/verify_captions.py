"""Final input, checkpoint, raw-output and export audit for both caption selections."""
import csv,json,sqlite3
from pathlib import Path
from run_pope import ROOT,digest,now
from atomic_io import atomic_json
from chair_eval import aggregate

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    state=read(ROOT/'logs/captions-progress.json');assert state['stage']=='captions_complete'
    folder=Path(state['results']);config=read(folder/'config.json')
    assert config['ground_truth_sha256']==digest(ROOT/'third_party/POPE/segmentation/coco_ground_truth_segmentation.json')
    table1=set(config['table1_image_ids']);pope=set(config['pope_image_ids']);ids=table1|pope
    assert len(table1)==2000 and len(pope)==500 and len(ids)==2483
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert db.execute('SELECT COUNT(*) FROM failed_questions').fetchone()[0]==0
        rows=[json.loads(x[0]) for x in db.execute('SELECT payload FROM answers ORDER BY split,qid')]
    assert len(rows)==2*len(ids)
    keyed={(r['prompt_id'],r['image_id']):r for r in rows};assert len(keyed)==len(rows)
    raw=[json.loads(line) for line in (folder/'raw-captions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(raw)==len(rows) and {(r['prompt_id'],r['image_id']):r for r in raw}==keyed
    for name,key in [('run_captions.py','runner_sha256'),('model_runtime.py','runtime_sha256'),
                     ('chair_eval.py','chair_adapter_sha256'),('caption_quality.py','quality_adapter_sha256')]:
        assert digest(folder/name)==config[key],name
    official=read(folder/'official-chair-validation.json');assert official['passed']
    assert all(r['captions_checked']==2483 for r in official['checks']) and len(official['checks'])==2
    for r in rows:
        assert r['image_id'] in ids and r['prompt']==config['prompts'][r['prompt_id']]
        assert r['image']==f'COCO_val2014_{r["image_id"]:012d}.jpg'
        assert r['image_sha256']==digest(ROOT/'data/coco/val2014'/r['image'])
        assert r['answer'].strip() and 0<len(r['generated_token_ids'])<=1024
        assert all(isinstance(t,int) and t>=0 for t in r['generated_token_ids'])
        assert r['generation_finish_reason']==('length' if r['possible_truncation'] else 'eos')
    for prompt in ['I1','I2']:
        selected=[r for r in rows if r['prompt_id']==prompt]
        assert {r['image_id'] for r in selected}==ids
        exported=read(folder/f'{prompt}-captions.json')
        assert {(r['prompt_id'],r['image_id']):r for r in exported}=={(r['prompt_id'],r['image_id']):r for r in selected}
        with (folder/f'{prompt}-captions.csv').open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
        assert len(csvrows)==len(selected)
        assert {int(r['image_id']):r['answer'] for r in csvrows}=={r['image_id']:r['answer'] for r in selected}
    metrics=read(folder/'chair-metrics.json')
    for entry in metrics:
        wanted=table1 if entry['selection']=='table1_2000' else pope
        expected=aggregate([r for r in rows if r['prompt_id']==entry['prompt_id'] and r['image_id'] in wanted])
        assert entry['complete'] and entry['expected']==len(wanted)
        for k,v in expected.items():assert entry[k]==v,(k,entry[k],v)
    assert len(metrics)==4
    analysis=read(folder/'analysis/table1.json');assert len(analysis)==2
    for entry in analysis:
        expected=aggregate([r for r in rows if r['prompt_id']==entry['prompt_id'] and r['image_id'] in table1])
        for k,v in expected.items():assert entry[k]==v
    atomic_json(folder/'captions-validation.json',{'at':now(),'passed':True,'records':len(rows),
        'table1_images_per_prompt':2000,'pope_images_per_prompt':500,'unique_images':2483,
        'all_images_prompts_sha256_and_tokens_checked':True,'sqlite_jsonl_json_csv_identical':True,
        'official_chair_compute_checked_all_records':True,'metric_exports_recomputed':True,
        'length_limited':sum(bool(r['possible_truncation']) for r in rows),
        'script_sha256':digest(__file__)})
    print(json.dumps({'passed':True,'records':len(rows)}))

if __name__=='__main__':main()
