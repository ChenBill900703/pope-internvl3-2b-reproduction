"""Raw VQA generation on disclosed official-image subsets; no external examiner calls."""
import argparse,csv,hashlib,importlib.metadata,json,msvcrt,os,sqlite3,time,traceback
from pathlib import Path
from run_pope import ROOT,digest,now
from atomic_io import atomic_json
from caption_quality import annotate
from run_stability import repair_journal,save

DATASETS=['aokvqa','gqa']
PARAMETERS=['model','revision','dtype','quantization','flash_attention','image_size','max_num',
            'use_thumbnail','batch_size','history','seed','generation']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def image_path(dataset,name):return ROOT/('data/coco/val2014' if dataset=='aokvqa' else 'data/gqa/images')/name
def records(db,dataset=None):
    query='SELECT payload FROM answers'+(' WHERE split=?' if dataset else '')+' ORDER BY split,qid'
    return [json.loads(r[0]) for r in db.execute(query,(dataset,) if dataset else ())]
def counts(db):return {s:db.execute('SELECT COUNT(*) FROM answers WHERE split=?',(s,)).fetchone()[0] for s in DATASETS}
def status(db,folder,stage,error=None):
    atomic_json(ROOT/'logs/vqa-progress.json',{'updated_at':now(),'stage':stage,'counts':counts(db),
        'results':str(folder),'error':error,'external_examiner_called':False})

def questions():
    aok=read(ROOT/'data/aokvqa/pope-image-vqa-questions.json')
    gqa=read(ROOT/'data/gqa/pope-image-vqa-questions.json')
    result={'aokvqa':[],'gqa':[]}
    for q in aok:
        assert len(q['direct_answers'])==10
        result['aokvqa'].append({'question_id':str(q['question_id']),'image_id':q['image_id'],
            'image':f'COCO_val2014_{q["image_id"]:012d}.jpg','text':q['question'],
            'reference_answers':q['direct_answers'],'question_metadata':q})
    for qid,q in gqa.items():
        result['gqa'].append({'question_id':str(qid),'image_id':str(q['imageId']),
            'image':str(q['imageId'])+'.jpg','text':q['question'],
            'reference_answers':[q['answer']],'question_metadata':q})
    assert len(result['aokvqa'])==515 and len(result['gqa'])==10701
    for dataset,qs in result.items():
        assert len({q['question_id'] for q in qs})==len(qs) and len({q['image'] for q in qs})==500
        assert all(q['text'].strip() and isinstance(q['question_id'],str) for q in qs)
    return result

def prepare():
    qs=questions();sources={}
    for name,statefile in [('table3','progress.json'),('captions','captions-progress.json'),
                           ('table4','stability-progress.json'),('appendixD','appendixD-progress.json')]:
        folder=Path(read(ROOT/'logs'/statefile)['results']);sources[name]=folder
    baseline=read(sources['table3']/'config.json');caption_config=read(sources['captions']/'config.json')
    assert caption_config['runtime_sha256']==digest(ROOT/'scripts/model_runtime.py')
    config={key:baseline[key] for key in PARAMETERS}
    for folder in sources.values():
        previous=read(folder/'config.json')
        for key in PARAMETERS:assert previous[key]==config[key],f'Source inference differs: {folder}/{key}'
    aok={r['image']:r['sha256'] for r in read(ROOT/'data/appendixD-questions-audit.json')['aokvqa_images']}
    gqa=read(ROOT/'data/gqa/images-validation.json');assert gqa['passed'] and gqa['images']==500
    image_hashes={'aokvqa':aok,'gqa':{r['image']:r['sha256'] for r in gqa['manifest']}}
    for dataset in DATASETS:
        assert {q['image'] for q in qs[dataset]}==set(image_hashes[dataset])
        for image,expected in image_hashes[dataset].items():assert digest(image_path(dataset,image))==expected
    config.update(question_sha256={d:digest(ROOT/f'data/{d}/pope-image-vqa-questions.json') for d in DATASETS},
        expected_counts={d:len(qs[d]) for d in DATASETS},image_sha256=image_hashes,
        source_folders={name:str(folder) for name,folder in sources.items()},
        source_config_sha256={name:digest(folder/'config.json') for name,folder in sources.items()},
        source_sha256={name:digest(ROOT/'scripts'/name) for name in ['run_vqa.py','run_stability.py',
            'model_runtime.py','atomic_io.py','run_pope.py','caption_quality.py','chair_eval.py']},
        packages={name:importlib.metadata.version(name) for name in ['torch','torchvision','transformers']},
        input_prompt='<image>\n{verbatim official question}; original native chat template, history=None',
        references_in_model_input=False,choices_in_model_input=False,
        examiner_called=False,grading='Raw answers only; ChatGPT-assisted grading requires separate authorization and provenance',
        scope='Disclosed extension on the original POPE 500 images per dataset. Table6 exact VQA IDs/prompt/grader version not published; do not claim identical Table6 sampling.',
        output_policy='All finite nonempty fixed-budget outputs saved, length/repetition flagged. Empty attempts retained; incomplete sets never marked complete.',
        reuse_policy='Exact image SHA and question text, identical inference parameters; preserve original source answer/tokens. Never copy destination reference answers into a model answer.')
    runid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    folder=ROOT/'results'/f'vqa-raw-{runid}';folder.mkdir(exist_ok=True)
    atomic_json(folder/'config.json',config)
    for name in config['source_sha256']:(folder/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    atomic_json(folder/'original-questions.json',qs)
    return qs,config,folder

def export(db,folder,config):
    summary=[]
    for dataset in DATASETS:
        selected=records(db,dataset);expected=config['expected_counts'][dataset]
        summary.append({'dataset':dataset,'saved':len(selected),'expected':expected,'raw_complete':len(selected)==expected,
            'external_examiner_called':False,'graded':False,
            'length_limited':sum(bool(r['possible_truncation']) for r in selected),
            'repetition_flagged':sum(bool(r['repetitive_generation']) for r in selected)})
        atomic_json(folder/f'{dataset}-answers.json',selected)
        with (folder/f'{dataset}-answers.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            keys=['dataset','question_id','image_id','image','text','answer','generation_finish_reason',
                  'repetitive_generation','at','seconds']
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
            writer.writerows({k:r[k] for k in keys} for r in selected)
    atomic_json(folder/'raw-generation-summary.json',summary)
    with (folder/'raw-generation-summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    return all(r['raw_complete'] for r in summary)

def available_outputs(db,config,folder):
    available={}
    sources=[Path(p) for p in config['source_folders'].values()]+[folder]
    for source in sources:
        with sqlite3.connect(f'file:{source/"answers.sqlite"}?mode=ro',uri=True) as previous:
            for split,qid,payload in previous.execute('SELECT split,qid,payload FROM answers ORDER BY split,qid'):
                r=json.loads(payload);text=r.get('text',r.get('prompt'))
                if not text or not r['answer'].strip():continue
                key=(r['image_sha256'],text)
                fields={k:r[k] for k in ['answer','generated_token_ids','tiles','peak_gpu_bytes']}
                fields['possible_truncation']=r.get('possible_truncation',False)
                available.setdefault(key,{'output':fields,'source':{'folder':str(source),'split':split,'question_id':qid}})
    return available

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    qs,config,folder=prepare()
    db=sqlite3.connect(folder/'answers.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    # IDs must stay text: GQA IDs may begin with zero, A-OKVQA IDs are alphanumeric.
    db.execute('CREATE TABLE IF NOT EXISTS answers(split TEXT,qid TEXT,payload TEXT,PRIMARY KEY(split,qid))')
    db.execute('CREATE TABLE IF NOT EXISTS failed_questions(split TEXT,qid TEXT,payload TEXT,PRIMARY KEY(split,qid))');db.commit()
    lock=None
    try:
        repair_journal(db,folder);available=available_outputs(db,config,folder);reused=0
        for dataset,items in qs.items():
            for q in items:
                if db.execute('SELECT 1 FROM answers WHERE split=? AND qid=?',(dataset,q['question_id'])).fetchone():continue
                key=(config['image_sha256'][dataset][q['image']],q['text'])
                if key not in available:continue
                old=available[key]
                record=annotate({**q,**old['output'],'dataset':dataset,'prompt_id':dataset,'image_sha256':key[0],
                    'at':now(),'seconds':0.,'reused_from':old['source']})
                save(db,folder,record);reused+=1
        with (folder/'reuse-audit.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'at':now(),'added':reused,'counts':counts(db),'inference_parameters_matched':True})+'\n')
        export(db,folder,config)
        if args.prepare_only:status(db,folder,'vqa_prepared');print(json.dumps({'prepared':True,'reused':reused,'counts':counts(db)}));return
        appendix=Path(read(ROOT/'logs/appendixD-progress.json')['results'])
        assert read(ROOT/'logs/appendixD-progress.json')['stage']=='appendixD_complete'
        assert read(appendix/'validation.json')['passed']
        lock=(ROOT/'logs/gpu.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        if not export(db,folder,config):
            status(db,folder,'loading_vqa_model')
            from model_runtime import Runtime
            runtime=Runtime()
            atomic_json(folder/'hardware.json',{'at':now(),'gpu':runtime.torch.cuda.get_device_name(0),
                'free_total':runtime.torch.cuda.mem_get_info(),'system_message':runtime.model.system_message,'template':runtime.model.template})
            tasks=[(d,q) for d,items in qs.items() for q in items]
            tasks.sort(key=lambda item:(item[0],item[1]['image'],item[1]['question_id']))
            for dataset,q in tasks:
                if db.execute('SELECT 1 FROM answers WHERE split=? AND qid=?',(dataset,q['question_id'])).fetchone():continue
                key=(config['image_sha256'][dataset][q['image']],q['text'])
                common={**q,'dataset':dataset,'prompt_id':dataset,'image_sha256':key[0],'at':now()}
                if key in available:
                    old=available[key];record={**common,**old['output'],'seconds':0.,'reused_from':old['source']}
                else:
                    begin=time.perf_counter()
                    # Only verbatim question text goes to the model. Reference
                    # answers, choices and question metadata remain on disk.
                    output=runtime.chat(image_path(dataset,q['image']),q['text'])
                    record={**common,**output,'seconds':time.perf_counter()-begin}
                    with (folder/'inference-attempts.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                record=annotate(record)
                if not record['answer'].strip():
                    db.execute('INSERT OR REPLACE INTO failed_questions VALUES(?,?,?)',(dataset,q['question_id'],json.dumps(record,ensure_ascii=False)));db.commit();continue
                save(db,folder,record)
                fields={k:record[k] for k in ['answer','generated_token_ids','tiles','peak_gpu_bytes','possible_truncation']}
                available[key]={'output':fields,'source':{'folder':str(folder),'split':dataset,'question_id':q['question_id']}}
                status(db,folder,'vqa_running')
                total=sum(counts(db).values());print(f'{dataset}/{q["question_id"]} raw answers saved={total}/11216',flush=True)
                if total%200==0:export(db,folder,config)
        complete=export(db,folder,config)
        status(db,folder,'vqa_raw_complete_needs_examiner' if complete else 'vqa_raw_incomplete')
    except BaseException as error:
        with (folder/'failures.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'at':now(),'error':repr(error),'traceback':traceback.format_exc()})+'\n');f.flush();os.fsync(f.fileno())
        status(db,folder,'requires_diagnosis',repr(error));raise
    finally:
        db.close()
        if lock:lock.close()

if __name__=='__main__':main()
