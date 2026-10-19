"""Durable Table4 evaluation; preparation/reuse can run without a GPU."""
import argparse, csv, hashlib, importlib.metadata, json, msvcrt, os, sqlite3, statistics, sys, time, traceback
from pathlib import Path
import run_pope
from run_pope import ROOT, digest, now, parse_answer, metrics
from atomic_io import atomic_json
from chair_eval import make_parser, caption_detail, aggregate
from caption_quality import annotate, summary as quality_summary

# The original evaluator helper uses this module-level writer.
run_pope.atomic_json=atomic_json
QA=['P1','P2','P3','P4'];CAP=['C1','C2','C3','C4']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def database_rows(db,split=None):
    statement='SELECT payload FROM answers'+(' WHERE split=?' if split else '')+' ORDER BY split,qid'
    return [json.loads(x[0]) for x in db.execute(statement,(split,) if split else ())]
def counts(db):return {p:db.execute('SELECT COUNT(*) FROM answers WHERE split=?',(p,)).fetchone()[0] for p in QA+CAP}

def save(db,folder,record):
    payload=json.dumps(record,ensure_ascii=False)
    db.execute('INSERT INTO answers VALUES(?,?,?)',(record['prompt_id'],record['question_id'],payload));db.commit()
    with (folder/'raw-answers.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(payload+'\n');stream.flush();os.fsync(stream.fileno())

def repair_journal(db,folder):
    """SQLite is authoritative; preserve a mismatching journal before rebuilding."""
    path=folder/'raw-answers.jsonl'
    expected={(r['prompt_id'],r['question_id']):r for r in database_rows(db)}
    try:
        raw=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        actual={(r['prompt_id'],r['question_id']):r for r in raw}
        valid=len(actual)==len(raw) and actual==expected
    except (ValueError,KeyError):valid=False
    if valid:return
    previous=None
    if path.exists():
        previous=folder/f'raw-answers-preserved-{time.time_ns()}.jsonl';previous.write_bytes(path.read_bytes())
    temporary=folder/'raw-answers-reconciled.tmp'
    with temporary.open('w',encoding='utf-8') as stream:
        for r in expected.values():stream.write(json.dumps(r,ensure_ascii=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    with (folder/'journal-reconciliation.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'at':now(),'records':len(expected),'preserved':str(previous) if previous else None})+'\n')

def status(db,folder,stage,error=None):
    atomic_json(ROOT/'logs/stability-progress.json',{'updated_at':now(),'stage':stage,'counts':counts(db),
                'results':str(folder),'error':error})

def export(db,folder):
    summary=[]
    for prompt in QA+CAP:
        rows=database_rows(db,prompt);complete=len(rows)==(3000 if prompt in QA else 2000)
        values=metrics(rows) if prompt in QA else {**aggregate(rows),**quality_summary(rows)}
        summary.append({'prompt_id':prompt,'complete':complete,**values})
        atomic_json(folder/f'{prompt}-answers.json',rows)
        if rows:
            keys=['prompt_id','question_id','image','text','answer']+(['label','prediction'] if prompt in QA else ['generation_finish_reason','repetitive_generation'])
            with (folder/f'{prompt}-answers.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
                writer.writerows({k:r[k] for k in keys} for r in rows)
    atomic_json(folder/'metrics.json',summary)
    for name,prompts in [('pope',QA),('chair',CAP)]:
        selected=[r for r in summary if r['prompt_id'] in prompts]
        with (folder/f'{name}-metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    complete=all(r['complete'] for r in summary)
    result={'complete':complete,'n_pope_per_prompt':3000,'n_caption_per_prompt':2000}
    if complete:
        f1=[r['f1'] for r in summary if r['prompt_id'] in QA]
        chair=[r['CHAIRi'] for r in summary if r['prompt_id'] in CAP]
        result.update(POPE_F1_mean=statistics.mean(f1),POPE_F1_population_std=statistics.pstdev(f1),
                      CHAIRi_mean=statistics.mean(chair),CHAIRi_population_std=statistics.pstdev(chair))
    atomic_json(folder/'table4.json',result)
    if complete:
        with (folder/'table4.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(result));writer.writeheader();writer.writerow(result)
    return complete

def source_setup():
    table3=Path(read(ROOT/'logs/progress.json')['results'])
    captions=Path(read(ROOT/'logs/captions-progress.json')['results'])
    for split in ['random','popular','adversarial']:assert read(table3/f'{split}-validation.json')['passed']
    t=read(table3/'config.json');c=read(captions/'config.json')
    for key in ['model','revision','dtype','quantization','flash_attention','image_size','max_num','use_thumbnail',
                'batch_size','history','seed','generation']:
        assert t[key]==c[key],f'Baseline settings mismatch: {key}'
    assert c['runtime_sha256']==digest(ROOT/'scripts/model_runtime.py')
    assert c['chair_adapter_sha256']==digest(ROOT/'scripts/chair_eval.py')
    protocol=read(ROOT/'data/table4/protocol.json')
    assert protocol['caption_prompts']['C1']==c['prompts']['I1']
    questions={p:read(ROOT/f'data/table4/{p}.json') for p in QA}
    for p,rows in questions.items():
        assert len(rows)==3000 and digest(ROOT/f'data/table4/{p}.json')==protocol['pope'][p]['sha256']
    config={k:t[k] for k in ['model','revision','dtype','quantization','flash_attention','image_size','max_num',
                             'use_thumbnail','batch_size','history','seed','generation']}
    config.update(caption_prompts=protocol['caption_prompts'],table1_image_ids=c['table1_image_ids'],
        question_sha256={p:protocol['pope'][p]['sha256'] for p in QA},
        article_policy={p:protocol['pope'][p]['article_policy'] for p in QA},
        source_table3=str(table3),source_captions=str(captions),
        baseline_config_sha256={'table3':digest(table3/'config.json'),'captions':digest(captions/'config.json')},
        source_sha256={name:digest(ROOT/'scripts'/name) for name in ['run_stability.py','model_runtime.py','chair_eval.py','caption_quality.py','atomic_io.py','run_pope.py']},
        packages={name:importlib.metadata.version(name) for name in ['torch','torchvision','transformers']},
        output_policy='All nonempty finite fixed-budget outputs included; length/repetition flagged, empty outputs separately retained.',
        scope='Original Table4 prompts and author data, new InternVL3-2B model; source LLaVA scores not asserted.')
    runid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    folder=ROOT/'results'/f'table4-{runid}';folder.mkdir(exist_ok=True)
    atomic_json(folder/'config.json',config)
    for name in config['source_sha256']:(folder/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    return table3,captions,questions,config,folder

def reuse_baselines(db,folder,table3,captions,questions,config):
    already={(r['prompt_id'],r['question_id']) for r in database_rows(db)};reused=0
    with sqlite3.connect(f'file:{table3/"answers.sqlite"}?mode=ro',uri=True) as old:
        records=[json.loads(x[0]) for x in old.execute("SELECT payload FROM answers WHERE split='random' ORDER BY qid")]
    assert len(records)==3000
    for r,q in zip(records,questions['P1']):
        assert (r['question_id'],r['image'],r['text'],r['label'])==(q['question_id'],q['image'],q['text'],q['label'])
        assert r['prediction']==parse_answer(r['answer']) and r['image_sha256']==digest(ROOT/'data/coco/val2014'/r['image'])
        if ('P1',r['question_id']) in already:continue
        record={**r,'prompt_id':'P1','kind':'pope','reused_from':{'folder':str(table3),'split':'random','question_id':r['question_id']}}
        save(db,folder,record);reused+=1
    with sqlite3.connect(f'file:{captions/"answers.sqlite"}?mode=ro',uri=True) as old:
        records=[json.loads(x[0]) for x in old.execute("SELECT payload FROM answers WHERE split='I1' ORDER BY qid")]
    selected=set(config['table1_image_ids'])
    for r in records:
        if r['image_id'] not in selected or ('C1',r['image_id']) in already:continue
        assert r['prompt']==config['caption_prompts']['C1']
        assert r['image_sha256']==digest(ROOT/'data/coco/val2014'/r['image'])
        record={**r,'prompt_id':'C1','question_id':r['image_id'],'text':r['prompt'],'kind':'caption',
                'reused_from':{'folder':str(captions),'prompt_id':'I1','image_id':r['image_id']}}
        save(db,folder,record);reused+=1
    with (folder/'reuse-audit.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'at':now(),'added':reused,'counts':counts(db),'inference_settings_matched':True})+'\n')
    print(f'Baseline outputs reused with provenance: {reused}',flush=True)

def verify_pope(db,folder,prompt):
    rows=database_rows(db,prompt)
    destination=folder/f'{prompt}-official-validation';destination.mkdir(exist_ok=True)
    run_pope.official_crosscheck(rows,destination)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--max-new-answers',type=int);args=ap.parse_args()
    table3,captions,questions,config,folder=source_setup()
    db=sqlite3.connect(folder/'answers.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS answers(split TEXT,qid INTEGER,payload TEXT,PRIMARY KEY(split,qid))')
    db.execute('CREATE TABLE IF NOT EXISTS failed_questions(split TEXT,qid INTEGER,payload TEXT,PRIMARY KEY(split,qid))');db.commit()
    lock=None
    try:
        repair_journal(db,folder);reuse_baselines(db,folder,table3,captions,questions,config)
        export(db,folder)
        if args.prepare_only:
            verify_pope(db,folder,'P1');status(db,folder,'table4_prepared');return
        assert read(ROOT/'logs/captions-progress.json')['stage']=='captions_complete','Wait for the active caption job to finish first'
        lock=(ROOT/'logs/gpu.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        if export(db,folder):
            for p in QA:verify_pope(db,folder,p)
            status(db,folder,'table4_complete');return
        status(db,folder,'loading_stability_model')
        from model_runtime import Runtime
        runtime=Runtime();parser,gt=make_parser();done=0
        atomic_json(folder/'hardware.json',{'at':now(),'gpu':runtime.torch.cuda.get_device_name(0),
            'free_total':runtime.torch.cuda.mem_get_info(),'system_message':runtime.model.system_message,'template':runtime.model.template})
        pending=[]
        for prompt in QA[1:]:
            for q in questions[prompt]:pending.append((q['image'],prompt,q['question_id'],q['text'],q))
        for image_id in config['table1_image_ids']:
            image=f'COCO_val2014_{image_id:012d}.jpg'
            for prompt in CAP[1:]:pending.append((image,prompt,image_id,config['caption_prompts'][prompt],None))
        # Adjacent tasks on the same image share only the validated visual features.
        pending.sort(key=lambda x:(x[0],x[1],x[2]))
        for image,prompt,qid,text,q in pending:
            if db.execute('SELECT 1 FROM answers WHERE split=? AND qid=?',(prompt,qid)).fetchone():continue
            path=ROOT/'data/coco/val2014'/image;begin=time.perf_counter();out=runtime.chat(path,text)
            record={**out,'prompt_id':prompt,'question_id':qid,'image':image,'image_sha256':digest(path),
                'text':text,'at':now(),'seconds':time.perf_counter()-begin,'kind':'pope' if q else 'caption'}
            if q:record.update(label=q['label'],prediction=parse_answer(out['answer']))
            else:record.update(image_id=qid,prompt=text)
            with (folder/'inference-attempts.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
            record=annotate(record)
            if not record['answer'].strip():
                db.execute('INSERT OR REPLACE INTO failed_questions VALUES(?,?,?)',(prompt,qid,json.dumps(record,ensure_ascii=False)));db.commit();continue
            if not q:record['chair']=caption_detail(parser,gt,qid,record['answer'])
            save(db,folder,record);done+=1;status(db,folder,'table4_running')
            print(f'{prompt}/{qid} {record["seconds"]:.2f}s {record["answer"]!r}',flush=True)
            if done%100==0:export(db,folder)
            if args.max_new_answers and done>=args.max_new_answers:
                export(db,folder);status(db,folder,'table4_smoke_saved');return
        complete=export(db,folder)
        if complete:
            for p in QA:verify_pope(db,folder,p)
        status(db,folder,'table4_complete' if complete else 'table4_incomplete')
    except BaseException as error:
        with (folder/'failures.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'at':now(),'error':repr(error),'traceback':traceback.format_exc()})+'\n');stream.flush();os.fsync(stream.fileno())
        status(db,folder,'requires_diagnosis',repr(error));raise
    finally:
        db.close()
        if lock:lock.close()

if __name__=='__main__':main()
