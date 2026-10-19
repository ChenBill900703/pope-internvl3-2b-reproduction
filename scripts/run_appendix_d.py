"""Run the six author-published SEEM POPE sets with durable per-answer writes."""
import argparse,csv,hashlib,importlib.metadata,json,msvcrt,os,sqlite3,time,traceback
from pathlib import Path
import run_pope
from run_pope import ROOT,digest,now,parse_answer,metrics
from atomic_io import atomic_json
from caption_quality import annotate
from run_stability import repair_journal,save

run_pope.atomic_json=atomic_json
SPLITS=[f'{dataset}_{split}' for dataset in ['aokvqa','gqa'] for split in ['random','popular','adversarial']]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(db,split=None):
    query='SELECT payload FROM answers'+(' WHERE split=?' if split else '')+' ORDER BY split,qid'
    return [json.loads(x[0]) for x in db.execute(query,(split,) if split else ())]
def counts(db):return {s:db.execute('SELECT COUNT(*) FROM answers WHERE split=?',(s,)).fetchone()[0] for s in SPLITS}
def status(db,folder,stage,error=None):
    atomic_json(ROOT/'logs/appendixD-progress.json',{'updated_at':now(),'stage':stage,'counts':counts(db),
                'results':str(folder),'error':error})

def export(db,folder):
    summary=[]
    for split in SPLITS:
        selected=rows(db,split);summary.append({'split':split,'complete':len(selected)==3000,**metrics(selected)})
        atomic_json(folder/f'{split}-answers.json',selected)
        with (folder/f'{split}-answers.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            keys=['dataset','pope_split','question_id','image','text','label','answer','prediction',
                  'generation_finish_reason','repetitive_generation']
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
            writer.writerows({k:r[k] for k in keys} for r in selected)
    atomic_json(folder/'table11.json',summary)
    with (folder/'table11.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    return all(r['complete'] for r in summary)

def prepare():
    baseline=Path(read(ROOT/'logs/progress.json')['results']);original=read(baseline/'config.json')
    captions=Path(read(ROOT/'logs/captions-progress.json')['results']);caption_config=read(captions/'config.json')
    assert original['generation']==caption_config['generation']
    assert caption_config['runtime_sha256']==digest(ROOT/'scripts/model_runtime.py')
    aok_audit=read(ROOT/'data/appendixD-questions-audit.json')
    gqa_audit=read(ROOT/'data/gqa/images-validation.json');assert gqa_audit['passed'] and gqa_audit['images']==500
    image_manifests={'aokvqa':{r['image']:r for r in aok_audit['aokvqa_images']},
                     'gqa':{r['image']:r for r in gqa_audit['manifest']}}
    questions={};hashes={};image_hashes={}
    for split in SPLITS:
        dataset,strategy=split.split('_',1)
        path=ROOT/f'third_party/POPE/output/seem/{dataset}/{dataset}_pope_seem_{strategy}.json'
        data=read(path);questions[split]=data;hashes[split]=digest(path)
        assert hashes[split]==aok_audit['files'][str(path.relative_to(ROOT))]['sha256']
        assert len(data)==3000 and len({r['question_id'] for r in data})==3000
        assert sum(r['label']=='yes' for r in data)==1500 and sum(r['label']=='no' for r in data)==1500
        assert len({r['image'] for r in data})==500
        for name in {r['image'] for r in data}:
            path=image_path(dataset,name);value=digest(path)
            assert value==image_manifests[dataset][name]['sha256']
            image_hashes[f'{dataset}/{name}']=value
    config={k:original[k] for k in ['model','revision','dtype','quantization','flash_attention','image_size','max_num',
        'use_thumbnail','batch_size','history','seed','generation']}
    for key in config:assert config[key]==caption_config[key],f'Baseline settings changed: {key}'
    config.update(question_sha256=hashes,image_sha256=image_hashes,
        source_sha256={name:digest(ROOT/'scripts'/name) for name in ['run_appendix_d.py','run_stability.py','model_runtime.py',
            'atomic_io.py','run_pope.py','caption_quality.py','chair_eval.py']},
        packages={name:importlib.metadata.version(name) for name in ['torch','torchvision','transformers']},
        baseline=str(baseline),baseline_config_sha256=digest(baseline/'config.json'),
        caption_config_sha256=digest(captions/'config.json'),
        question_source='Unmodified official output/seem/aokvqa and output/seem/gqa question arrays',
        evaluation='Official first-sentence negative-token parser, all six sets independently verified',
        reuse_policy='Only exact image SHA-256 plus exact prompt under identical inference parameters; labels need not match because source annotation schemes differ. Destination label always retained.',
        output_policy='Include finite nonempty fixed1024 outputs, annotate length/repetition; empty outputs separately saved and mark incomplete',
        scope='InternVL3-2B extension of AppendixD/Table11, not a remeasurement of original baseline models')
    runid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    folder=ROOT/'results'/f'appendixD-{runid}';folder.mkdir(exist_ok=True)
    atomic_json(folder/'config.json',config)
    for name in config['source_sha256']:(folder/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    return questions,config,folder

def image_path(dataset,name):return ROOT/('data/coco/val2014' if dataset=='aokvqa' else 'data/gqa/images')/name

def verify(db,folder,questions):
    saved=rows(db);assert len(saved)==18000
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    raw=[json.loads(line) for line in (folder/'raw-answers.jsonl').read_text(encoding='utf-8').splitlines()]
    keys=lambda values:{(r['prompt_id'],r['question_id']):r for r in values}
    assert len(raw)==len(saved) and keys(raw)==keys(saved)
    for split in SPLITS:
        selected=rows(db,split);assert len(selected)==3000
        for r,q in zip(selected,questions[split]):
            assert all(r[k]==q[k] for k in ['question_id','image','text','label'])
            assert r['prediction']==parse_answer(r['answer']) and r['answer'].strip()
            assert r['image_sha256']==digest(image_path(r['dataset'],r['image']))
            assert 0<len(r['generated_token_ids'])<=1024
            if 'reused_from' in r:
                source=r['reused_from'];sourcefolder=Path(source['folder']).resolve()
                assert sourcefolder.is_relative_to(ROOT/'results')
                with sqlite3.connect(f'file:{sourcefolder/"answers.sqlite"}?mode=ro',uri=True) as previous:
                    prior=previous.execute('SELECT payload FROM answers WHERE split=? AND qid=?',
                                           (source['split'],source['question_id'])).fetchone()
                assert prior
                prior=json.loads(prior[0])
                for key in ['answer','generated_token_ids','image_sha256','text']:assert prior[key]==r[key]
        directory=folder/f'{split}-official-validation';directory.mkdir(exist_ok=True)
        run_pope.official_crosscheck(selected,directory)
        assert keys(read(folder/f'{split}-answers.json'))==keys(selected)
        with (folder/f'{split}-answers.csv').open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
        assert len(csvrows)==3000
        assert {int(r['question_id']):r['answer'] for r in csvrows}=={r['question_id']:r['answer'] for r in selected}
    atomic_json(folder/'validation.json',{'at':now(),'passed':True,'records':18000,'six_sets_complete':True,
        'official_evaluator_and_sklearn':True,'input_labels_images_and_raw_exports_checked':True,
        'all_reused_answer_text_tokens_and_images_unchanged':True})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    questions,config,folder=prepare()
    db=sqlite3.connect(folder/'answers.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS answers(split TEXT,qid INTEGER,payload TEXT,PRIMARY KEY(split,qid))')
    db.execute('CREATE TABLE IF NOT EXISTS failed_questions(split TEXT,qid INTEGER,payload TEXT,PRIMARY KEY(split,qid))');db.commit()
    lock=None
    try:
        repair_journal(db,folder)
        available={}
        baseline=Path(config['baseline'])
        with sqlite3.connect(f'file:{baseline/"answers.sqlite"}?mode=ro',uri=True) as old:
            for item in old.execute('SELECT payload FROM answers ORDER BY split,qid'):
                r=json.loads(item[0]);available.setdefault((r['image_sha256'],r['text']),
                    {'source_record':r,'source':{'folder':str(baseline),'split':r['split'],'question_id':r['question_id']}})
        for r in rows(db):available.setdefault((r['image_sha256'],r['text']),
            {'source_record':r,'source':{'folder':str(folder),'split':r['prompt_id'],'question_id':r['question_id']}})
        reused=0
        for split,items in questions.items():
            dataset,strategy=split.split('_',1)
            for q in items:
                if db.execute('SELECT 1 FROM answers WHERE split=? AND qid=?',(split,q['question_id'])).fetchone():continue
                key=config['image_sha256'][f'{dataset}/{q["image"]}'],q['text']
                if key not in available:continue
                previous=available[key];source=previous['source_record']
                record={**q,'prompt_id':split,'dataset':dataset,'pope_split':strategy,'image_sha256':key[0],
                    'answer':source['answer'],'generated_token_ids':source['generated_token_ids'],
                    'prediction':parse_answer(source['answer']),'tiles':source['tiles'],'peak_gpu_bytes':source['peak_gpu_bytes'],
                    'possible_truncation':source.get('possible_truncation',False),'at':now(),'seconds':0.,
                    'reused_from':previous['source']}
                save(db,folder,annotate(record));reused+=1
        with (folder/'reuse-audit.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'at':now(),'added':reused,'counts':counts(db),'inference_settings_matched':True})+'\n')
        export(db,folder)
        if args.prepare_only:status(db,folder,'appendixD_prepared');print(json.dumps({'prepared':True,'reused':reused,'counts':counts(db)}));return
        assert read(ROOT/'logs/captions-progress.json')['stage']=='captions_complete'
        assert read(ROOT/'logs/stability-progress.json')['stage']=='table4_complete'
        assert read(Path(read(ROOT/'logs/stability-progress.json')['results'])/'validation.json')['passed']
        lock=(ROOT/'logs/gpu.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        if export(db,folder):verify(db,folder,questions);status(db,folder,'appendixD_complete');return
        status(db,folder,'loading_appendixD_model')
        from model_runtime import Runtime
        runtime=Runtime()
        atomic_json(folder/'hardware.json',{'at':now(),'gpu':runtime.torch.cuda.get_device_name(0),
            'free_total':runtime.torch.cuda.mem_get_info(),'system_message':runtime.model.system_message,'template':runtime.model.template})
        tasks=[(split.split('_')[0],q['image'],split,q) for split,items in questions.items() for q in items]
        tasks.sort(key=lambda x:(x[0],x[1],x[2],x[3]['question_id']))
        generated=0
        for dataset,image,split,q in tasks:
            if db.execute('SELECT 1 FROM answers WHERE split=? AND qid=?',(split,q['question_id'])).fetchone():continue
            key=config['image_sha256'][f'{dataset}/{image}'],q['text'];path=image_path(dataset,image)
            common={**q,'prompt_id':split,'dataset':dataset,'pope_split':split.split('_',1)[1],
                    'image_sha256':key[0],'at':now()}
            if key in available:
                previous=available[key];source=previous['source_record']
                record={**source,**common,'prediction':parse_answer(source['answer']),'seconds':0.,'reused_from':previous['source']}
            else:
                start=time.perf_counter();out=runtime.chat(path,q['text'])
                record={**out,**common,'prediction':parse_answer(out['answer']),'seconds':time.perf_counter()-start}
                with (folder/'inference-attempts.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                record=annotate(record)
                if not record['answer'].strip():
                    db.execute('INSERT OR REPLACE INTO failed_questions VALUES(?,?,?)',(split,q['question_id'],json.dumps(record,ensure_ascii=False)));db.commit();continue
                available[key]={'source_record':record,'source':{'folder':str(folder),'split':split,'question_id':q['question_id']}}
                generated+=1
            save(db,folder,record);status(db,folder,'appendixD_running')
            total=sum(counts(db).values())
            print(f'{split}/{q["question_id"]} saved={total}/18000 new_model_calls={generated}',flush=True)
            if total%100==0:export(db,folder)
        complete=export(db,folder)
        if complete:verify(db,folder,questions)
        status(db,folder,'appendixD_complete' if complete else 'appendixD_incomplete')
    except BaseException as error:
        with (folder/'failures.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'at':now(),'error':repr(error),'traceback':traceback.format_exc()})+'\n');f.flush();os.fsync(f.fileno())
        status(db,folder,'requires_diagnosis',repr(error));raise
    finally:
        db.close()
        if lock:lock.close()

if __name__=='__main__':main()
