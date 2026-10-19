"""Table 1 captions, followed by the POPE image captions for consistency/Appendix F."""
import argparse,csv,json,msvcrt,os,sqlite3,subprocess,sys,time,traceback
from pathlib import Path
from run_pope import ROOT,now,digest,atomic_json
from chair_eval import make_parser,caption_detail,aggregate
from caption_quality import annotate,summary as quality_summary

PROMPTS={'I1':'Generate a short caption of the image.','I2':'Provide a brief description of the given image.'}

def status(stage,counts,folder,error=None):
    p={'updated_at':now(),'stage':stage,'counts':counts,'results':str(folder),'error':error}
    atomic_json(ROOT/'logs/captions-progress.json',p)
    file=ROOT/'STATUS.md';text=file.read_text(encoding='utf-8').split('<!-- CAPTIONS -->')[0]
    file.write_text(text+'<!-- CAPTIONS -->\n\n## 圖片描述即時狀態\n\n```json\n'+json.dumps(p,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')

def export(db,folder,table1_ids,pope_ids):
    summary=[]
    for prompt in PROMPTS:
        rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM answers WHERE split=? ORDER BY qid',(prompt,))]
        atomic_json(folder/f'{prompt}-captions.json',rows)
        with (folder/f'{prompt}-captions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            keys=['image_id','image','prompt','answer','at','seconds','tiles','peak_gpu_bytes',
                  'generation_finish_reason','repetition_4gram_fraction','repetitive_generation']
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
        for selection,ids in [('table1_2000',table1_ids),('pope_500',pope_ids)]:
            subset=[r for r in rows if r['image_id'] in ids]
            summary.append({'prompt_id':prompt,'selection':selection,'expected':len(ids),
                            'complete':len(subset)==len(ids),**aggregate(subset),**quality_summary(subset)})
    atomic_json(folder/'chair-metrics.json',summary)
    with (folder/'chair-metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke-only',action='store_true');args=ap.parse_args()
    original=json.loads((ROOT/'logs/progress.json').read_text(encoding='utf-8'))
    assert original['stage']=='table3_complete'
    for s in ['random','popular','adversarial']:
        assert json.loads((Path(original['results'])/f'{s}-validation.json').read_text(encoding='utf-8'))['passed']
    table1_ids=set(json.loads((ROOT/'data/extended-audit.json').read_text(encoding='utf-8'))['caption_image_ids'])
    pope_ids={int(q['image'].split('_')[-1].split('.')[0]) for q in map(json.loads,(ROOT/'third_party/POPE/output/coco/coco_pope_random.json').read_text().splitlines())}
    config={'model':'OpenGVLab/InternVL3-2B','revision':'899155015275a9b7338c7f4677e19c784e0e5a21',
            'dtype':'bfloat16','quantization':None,'flash_attention':False,'image_size':448,'max_num':12,
            'use_thumbnail':True,'batch_size':1,'history':None,'seed':0,
            'generation':{'do_sample':False,'num_beams':1,'max_new_tokens':1024},'prompts':PROMPTS,
            'table1_image_ids':sorted(table1_ids),'pope_image_ids':sorted(pope_ids),
            'ground_truth_sha256':digest(ROOT/'third_party/POPE/segmentation/coco_ground_truth_segmentation.json'),
            'runner_sha256':digest(__file__),'runtime_sha256':digest(ROOT/'scripts/model_runtime.py'),
            'chair_adapter_sha256':digest(ROOT/'scripts/chair_eval.py'),
            'quality_adapter_sha256':digest(ROOT/'scripts/caption_quality.py'),
            'output_policy':'Fixed 1024-token budget. Include all nonempty finite outputs, flag length/repetition. EOS-only metrics supplementary. Empty outputs saved separately and do not stop other images.',
            'notes':['Original Table 1 prompt texts, author published 2000 caption image IDs',
                     'Use authors published combined caption/segmentation GT and original CHAIR parser',
                     'Reuse official generate visual_features within each image after exact token validation',
                     'Caption length reported with both whitespace and CHAIR-normalized NLTK tokens; original Len tokenization unspecified']}
    import hashlib
    runid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    folder=ROOT/'results'/('captions-'+runid);folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'config.json',config)
    for name in ['run_captions.py','model_runtime.py','chair_eval.py','caption_quality.py']:
        (folder/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    db=sqlite3.connect(folder/'answers.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS answers(split TEXT,qid INTEGER,payload TEXT,PRIMARY KEY(split,qid))');db.commit()
    db.execute('CREATE TABLE IF NOT EXISTS failed_questions(split TEXT,qid INTEGER,payload TEXT,PRIMARY KEY(split,qid))');db.commit()
    counts=lambda:{s:db.execute('SELECT COUNT(*) FROM answers WHERE split=?',(s,)).fetchone()[0] for s in PROMPTS}
    lock=(ROOT/'logs/gpu.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        parser,gt=make_parser()
        settings=ROOT/'logs/caption-settings.json'
        if settings.exists():
            sourcefolder=ROOT/json.loads(settings.read_text(encoding='utf-8'))['resume_from']
            if sourcefolder.resolve()!=folder.resolve():
                old=json.loads((sourcefolder/'config.json').read_text(encoding='utf-8'))
                for key in ['model','revision','dtype','quantization','flash_attention','image_size','max_num','use_thumbnail',
                            'batch_size','history','seed','generation','prompts','table1_image_ids','pope_image_ids',
                            'ground_truth_sha256','runtime_sha256','chair_adapter_sha256']:
                    assert config[key]==old[key],f'Caption migration mismatch: {key}'
                old_db=sqlite3.connect(f'file:{sourcefolder/"answers.sqlite"}?mode=ro',uri=True)
                assert old_db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
                sources=[json.loads(x[0]) for x in old_db.execute('SELECT payload FROM answers')];old_db.close()
                sources.extend(json.loads(l) for l in (sourcefolder/'inference-attempts.jsonl').read_text(encoding='utf-8').splitlines())
                imported=0;recovered=0
                for record in sources:
                    if not record['answer'].strip():continue
                    prompt_id=record['prompt_id'];image_id=record['image_id']
                    if db.execute('SELECT 1 FROM answers WHERE split=? AND qid=?',(prompt_id,image_id)).fetchone():continue
                    assert record['prompt']==PROMPTS[prompt_id] and image_id in table1_ids|pope_ids
                    assert record['image_sha256']==digest(ROOT/'data/coco/val2014'/record['image'])
                    record=annotate(record)
                    if 'chair' not in record:
                        record['chair']=caption_detail(parser,gt,image_id,record['answer']);recovered+=1
                    record['migrated_from']={'folder':str(sourcefolder),'prompt_id':prompt_id,'image_id':image_id}
                    payload=json.dumps(record,ensure_ascii=False)
                    db.execute('INSERT INTO answers VALUES(?,?,?)',(prompt_id,image_id,payload));db.commit()
                    with (folder/'raw-captions.jsonl').open('a',encoding='utf-8') as f:f.write(payload+'\n');f.flush();os.fsync(f.fileno())
                    imported+=1
                audit=folder/'migration-audit.json'
                if not audit.exists():atomic_json(audit,{'at':now(),'source_folder':str(sourcefolder),'imported':imported,
                    'recovered_saved_attempts':recovered,'all_inference_parameters_identical':True,
                    'cap_unchanged':1024,'change':'Output quality policy only; original generated text/tokens unchanged',
                    'repetition_diagnosis':'logs/caption-repetition-diagnosis.json'})
                for name in ['visual-features-validation.json','visual-features-raw-check.json']:
                    if not (folder/name).exists():(folder/name).write_bytes((sourcefolder/name).read_bytes())
                print(f'Caption migration: {imported} actual outputs preserved, {recovered} prior attempts recovered',flush=True)
        status('loading_caption_model',counts(),folder)
        from model_runtime import Runtime
        runtime=Runtime()
        atomic_json(folder/'hardware.json',{'at':now(),'gpu':runtime.torch.cuda.get_device_name(0),
                    'free_total':runtime.torch.cuda.mem_get_info(),'system_message':runtime.model.system_message,
                    'template':runtime.model.template})
        (folder/'official-preprocessing.py').write_text(runtime.preprocess_source,encoding='utf-8')
        cache_audit=folder/'visual-features-validation.json'
        if not cache_audit.exists():
            image_id=min(table1_ids);path=ROOT/f'data/coco/val2014/COCO_val2014_{image_id:012d}.jpg'
            base=runtime.chat(path,PROMPTS['I1'],use_features=False)
            cached=runtime.chat(path,PROMPTS['I1'],use_features=True)
            atomic_json(folder/'visual-features-raw-check.json',{'direct':base,'cached':cached,'image_id':image_id})
            assert base['generated_token_ids']==cached['generated_token_ids'] and base['answer']==cached['answer']
            atomic_json(cache_audit,{'at':now(),'passed':True,'image_id':image_id,'generated_tokens_identical':True})
        for selection,ids in [('table1',sorted(table1_ids)),('pope_caption',sorted(pope_ids-table1_ids))]:
            for image_id in ids:
                image=f'COCO_val2014_{image_id:012d}.jpg';path=ROOT/'data/coco/val2014'/image
                for prompt_id,prompt in PROMPTS.items():
                    if db.execute('SELECT 1 FROM answers WHERE split=? AND qid=?',(prompt_id,image_id)).fetchone():continue
                    begin=time.perf_counter();out=runtime.chat(path,prompt)
                    record={**out,'image_id':image_id,'image':image,'image_sha256':digest(path),
                            'prompt_id':prompt_id,'prompt':prompt,'at':now(),'seconds':time.perf_counter()-begin,
                            'selection_membership':{'table1':image_id in table1_ids,'pope':image_id in pope_ids}}
                    with (folder/'inference-attempts.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                    record=annotate(record)
                    if not record['answer'].strip():
                        db.execute('INSERT OR REPLACE INTO failed_questions VALUES(?,?,?)',(prompt_id,image_id,json.dumps(record,ensure_ascii=False)));db.commit()
                        print(f'Empty caption saved for diagnosis, continue other images: {prompt_id}/{image_id}',flush=True)
                        continue
                    record['chair']=caption_detail(parser,gt,image_id,record['answer'])
                    payload=json.dumps(record,ensure_ascii=False)
                    db.execute('INSERT INTO answers VALUES(?,?,?)',(prompt_id,image_id,payload));db.commit()
                    with (folder/'raw-captions.jsonl').open('a',encoding='utf-8') as f:f.write(payload+'\n');f.flush();os.fsync(f.fileno())
                    c=counts();status(selection,c,folder)
                    print(f'{selection} {prompt_id} {c[prompt_id]} image={image_id} {record["seconds"]:.2f}s {record["answer"]!r}',flush=True)
                    if sum(c.values())%50==0:export(db,folder,table1_ids,pope_ids)
                    if args.smoke_only and sum(c.values())>=4:
                        export(db,folder,table1_ids,pope_ids);status('caption_smoke_passed',c,folder);return
            export(db,folder,table1_ids,pope_ids)
            if selection=='table1':
                complete=all(db.execute('SELECT COUNT(*) FROM answers WHERE split=? AND qid IN ('+','.join('?'*len(table1_ids))+')',[s,*sorted(table1_ids)]).fetchone()[0]==2000 for s in PROMPTS)
                status('table1_captions_complete' if complete else 'table1_incomplete',counts(),folder)
                if complete:
                    with (folder/'analysis-table1.log').open('a',encoding='utf-8') as log:
                        result=subprocess.run([sys.executable,'scripts/analyze_captions.py','--folder',str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                        if result.returncode:print('Caption analysis needs diagnosis; continuing saved caption generation.',flush=True)
        export(db,folder,table1_ids,pope_ids)
        complete=all(n==len(table1_ids|pope_ids) for n in counts().values())
        status('captions_complete' if complete else 'captions_incomplete',counts(),folder)
    except BaseException as e:
        with (folder/'failures.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'at':now(),'error':repr(e),'traceback':traceback.format_exc()})+'\n');f.flush();os.fsync(f.fileno())
        export(db,folder,table1_ids,pope_ids);status('requires_diagnosis',counts(),folder,repr(e));raise
    finally:db.close();lock.close()

if __name__=='__main__':main()
