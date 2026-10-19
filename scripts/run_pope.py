"""Durable, resumable official COCO POPE evaluation for InternVL3-2B."""
import argparse, csv, gc, hashlib, io, json, os, re, sqlite3, sys, time, traceback
from contextlib import redirect_stdout
from pathlib import Path
from datetime import datetime, timezone

for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='backslashreplace')

ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['HF_MODULES_CACHE']=str(ROOT/'.cache/huggingface/modules')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def atomic_json(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,p)

def parse_answer(text):
    # Intentionally reproduce third_party/POPE/evaluate.py, including case sensitivity.
    words=text.split('.')[0].replace(',','').split(' ')
    return 'no' if any(w in words for w in ['No','not','no']) else 'yes'

def metrics(rows):
    tp=sum(r['prediction']=='yes' and r['label']=='yes' for r in rows)
    tn=sum(r['prediction']=='no' and r['label']=='no' for r in rows)
    fp=sum(r['prediction']=='yes' and r['label']=='no' for r in rows)
    fn=sum(r['prediction']=='no' and r['label']=='yes' for r in rows)
    n=len(rows); p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
    return dict(n=n,TP=tp,TN=tn,FP=fp,FN=fn,accuracy=(tp+tn)/n if n else None,
                precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.,
                yes_ratio=(tp+fp)/n if n else None)

def load_rows(split):
    p=ROOT/f'third_party/POPE/output/coco/coco_pope_{split}.json'
    return [json.loads(l) for l in p.read_text().splitlines()]

def official_crosscheck(rows,folder):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    y=[r['label']=='yes' for r in rows]; pred=[r['prediction']=='yes' for r in rows]
    m=metrics(rows)
    for k,value in [('accuracy',accuracy_score(y,pred)),('precision',precision_score(y,pred,zero_division=0)),
                    ('recall',recall_score(y,pred,zero_division=0)),('f1',f1_score(y,pred,zero_division=0))]:
        assert abs(m[k]-value)<1e-12,(k,m[k],value)
    af=folder/'pilot-official-answers.jsonl'; lf=folder/'pilot-official-labels.jsonl'
    af.write_text(''.join(json.dumps({'question':r['text'],'answer':r['answer']})+'\n' for r in rows),encoding='utf-8')
    lf.write_text(''.join(json.dumps({'label':r['label']})+'\n' for r in rows),encoding='utf-8')
    source=(ROOT/'third_party/POPE/evaluate.py').read_text()
    source=source.replace("ans_file = ''",'ans_file = '+repr(str(af))).replace("label_file = ''",'label_file = '+repr(str(lf)))
    scope={}; output=io.StringIO()
    with redirect_stdout(output): exec(compile(source,'official_evaluate.py','exec'),scope)
    for key,official in [('accuracy','acc'),('precision','precision'),('recall','recall'),('f1','f1'),('yes_ratio','yes_ratio')]:
        assert abs(m[key]-scope[official])<1e-12
    (folder/'pilot-official-evaluator.txt').write_text(output.getvalue(),encoding='utf-8')
    atomic_json(folder/'pilot-validation.json',{'at':now(),'passed':True,'n':len(rows),
                'official_evaluator':True,'sklearn':True,'empty_answers':sum(not r['answer'].strip() for r in rows),
                'non_explicit_yes_no':sum(not re.match(r'^\s*(yes|no)\b',r['answer'],re.I) for r in rows)})
    assert all(r['answer'].strip() for r in rows),'Empty response: review before full run'

def progress(stage,counts,folder,error=None):
    obj={'updated_at':now(),'stage':stage,'counts':counts,'results':str(folder),'error':error}
    atomic_json(ROOT/'logs/progress.json',obj)
    p=ROOT/'STATUS.md'; text=p.read_text(encoding='utf-8').split('<!-- LIVE -->')[0]
    text+='<!-- LIVE -->\n\n## 即時執行狀態\n\n```json\n'+json.dumps(obj,ensure_ascii=False,indent=2)+'\n```\n'
    p.write_text(text,encoding='utf-8')

def export(db,folder,stage):
    summary=[]
    for split in ['random','popular','adversarial']:
        rows=[json.loads(x[0]) for x in db.execute('SELECT payload FROM answers WHERE split=? ORDER BY qid',(split,))]
        atomic_json(folder/f'{split}-metrics.json',{'stage':stage,**metrics(rows)})
        p=folder/f'{split}-answers.csv'; tmp=p.with_suffix('.tmp')
        with tmp.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['question_id','image','text','label','answer','prediction','seconds','tiles','peak_gpu_bytes'])
            w.writeheader(); w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
        os.replace(tmp,p)
        atomic_json(folder/f'{split}-answers.json',rows)
        summary.append({'split':split,**metrics(rows)})
    atomic_json(folder/'table3.json',summary)
    with (folder/'table3.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pilot-only',action='store_true'); args=ap.parse_args()
    import torch, torchvision, transformers
    from PIL import Image
    from transformers import AutoModel,AutoTokenizer
    torch.manual_seed(0);torch.set_num_threads(4)
    config={'model':'OpenGVLab/InternVL3-2B','revision':'899155015275a9b7338c7f4677e19c784e0e5a21',
            'dtype':'bfloat16','quantization':None,'flash_attention':False,'batch_size':1,'seed':0,
            'reuse_identical_image_prompt':True,
            'image_size':448,'max_num':12,'use_thumbnail':True,'history':None,
            'prompt':'<image>\n{official_question_text}',
            'generation':{'do_sample':False,'num_beams':1,'max_new_tokens':1024},
            'torch':torch.__version__,'torchvision':torchvision.__version__,'transformers':transformers.__version__,
            'questions':{s:digest(ROOT/f'third_party/POPE/output/coco/coco_pope_{s}.json') for s in ['random','popular','adversarial']},
            'runner_sha256':digest(__file__),
            'adaptations':['InternVL3 not in original paper','BF16 after diagnosed FP16 NaN logits; no OOM and no quantization','FlashAttention disabled on Windows',
                           'Deterministic greedy decoding; original paper does not specify InternVL3 decoding',
                           '1024 new-token cap matching official example; migrated only EOS-complete earlier greedy answers with audit']}
    runid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    folder=ROOT/'results'/('table3-'+runid); folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'config.json',config)
    db=sqlite3.connect(folder/'answers.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS answers(split TEXT,qid INTEGER,payload TEXT,PRIMARY KEY(split,qid))');db.commit()
    counts=lambda:{s:db.execute('SELECT COUNT(*) FROM answers WHERE split=?',(s,)).fetchone()[0] for s in config['questions']}
    lock=(ROOT/'logs/gpu.lock').open('a+b')
    import msvcrt
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        assert torch.cuda.is_available()
        atomic_json(folder/'hardware.json',{'at':now(),'gpu':torch.cuda.get_device_name(0),'free_total':torch.cuda.mem_get_info(),
                    'python':sys.version,'cuda':torch.version.cuda})
        progress('loading_bf16',counts(),folder)
        modelpath=str(ROOT/'models/InternVL3-2B')
        model=AutoModel.from_pretrained(modelpath,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,
                    trust_remote_code=True,use_flash_attn=False,local_files_only=True).eval().cuda()
        tokenizer=AutoTokenizer.from_pretrained(modelpath,trust_remote_code=True,use_fast=False,local_files_only=True)
        generated={}
        original_generate=model.language_model.generate
        def capture_generate(*a,**kw):
            out=original_generate(*a,**kw)
            generated['ids']=out[0].tolist()
            return out
        model.language_model.generate=capture_generate
        def finite_logits(m,a,out):
            if not torch.isfinite(out.logits[:,-1,:]).all():
                raise FloatingPointError('Non-finite language-model logits')
        model.language_model.register_forward_hook(finite_logits)
        atomic_json(folder/'model-loaded.json',{'at':now(),'allocated':torch.cuda.memory_allocated(),
                    'free_total':torch.cuda.mem_get_info(),'system_message':model.system_message,'template':model.template})
        # Copy the exact official preprocessing functions from the pinned model card.
        card=(ROOT/'models/InternVL3-2B/README.md').read_text(encoding='utf-8')
        start=card.index('def build_transform(');end=card.index('\ndef split_model(',start)
        prep=card[start:end]
        (folder/'official-preprocessing.py').write_text(prep,encoding='utf-8')
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        namespace=dict(torch=torch,T=T,Image=Image,InterpolationMode=InterpolationMode,
                       IMAGENET_MEAN=(0.485,0.456,0.406),IMAGENET_STD=(0.229,0.224,0.225))
        exec(compile(prep,'official-preprocessing.py','exec'),namespace)
        # A higher maximum does not alter greedy outputs that already reached EOS.
        # Reuse requires matching inference settings, valid source data, EOS, and live token checks.
        settings_path=ROOT/'logs/run-settings.json'
        if settings_path.exists():
            settings=json.loads(settings_path.read_text(encoding='utf-8'))
            sourcefolder=ROOT/settings['resume_from']
            if sourcefolder.resolve()!=folder.resolve():
                previous=json.loads((sourcefolder/'config.json').read_text(encoding='utf-8'))
                for key in ['model','revision','dtype','quantization','flash_attention','batch_size','seed',
                            'image_size','max_num','use_thumbnail','history','prompt','torch','torchvision','transformers','questions']:
                    assert config[key]==previous[key],f'Migration mismatch: {key}'
                assert {k:v for k,v in config['generation'].items() if k!='max_new_tokens'}=={k:v for k,v in previous['generation'].items() if k!='max_new_tokens'}
                source_db=sqlite3.connect(f'file:{sourcefolder/"answers.sqlite"}?mode=ro',uri=True)
                assert source_db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
                source_rows=[json.loads(x[0]) for x in source_db.execute('SELECT payload FROM answers ORDER BY split,qid')]
                source_db.close()
                questions={s:{q['question_id']:q for q in load_rows(s)} for s in config['questions']}
                for saved in source_rows:
                    q=questions[saved['split']][saved['question_id']]
                    assert all(saved[k]==q[k] for k in q)
                    assert saved['generated_token_ids'][-1]==tokenizer.eos_token_id
                    assert saved['answer'].strip() and not saved['possible_truncation']
                    assert digest(ROOT/'data/coco/val2014'/saved['image'])==saved['image_sha256']
                auditpath=folder/'migration-audit.json'
                if not auditpath.exists():
                    checks=[]
                    for qid in [1,324,835]:
                        saved=next(r for r in source_rows if r['split']=='random' and r['question_id']==qid)
                        pixels=namespace['load_image'](str(ROOT/'data/coco/val2014'/saved['image']),max_num=12).to(torch.bfloat16).cuda()
                        with torch.inference_mode():
                            answer=model.chat(tokenizer,pixels,'<image>\n'+saved['text'],dict(config['generation']),history=None)
                        assert generated['ids']==saved['generated_token_ids'] and answer==saved['answer'],f'Greedy migration mismatch q={qid}'
                        checks.append({'qid':qid,'tokens_identical':True,'answer':answer});del pixels
                    atomic_json(auditpath,{'at':now(),'source':str(sourcefolder),'source_count':len(source_rows),
                                'all_source_answers_reached_eos':True,'matching_inference_settings':True,
                                'old_cap':previous['generation']['max_new_tokens'],'new_cap':config['generation']['max_new_tokens'],
                                'live_crosschecks':checks})
                imported=0
                for saved in source_rows:
                    if db.execute('SELECT 1 FROM answers WHERE split=? AND qid=?',(saved['split'],saved['question_id'])).fetchone():continue
                    saved['migrated_from']={'folder':str(sourcefolder),'split':saved['split'],'question_id':saved['question_id']}
                    saved['generated_max_new_tokens']=previous['generation']['max_new_tokens']
                    payload=json.dumps(saved,ensure_ascii=False)
                    db.execute('INSERT INTO answers VALUES(?,?,?)',(saved['split'],saved['question_id'],payload));db.commit()
                    with (folder/'raw-answers.jsonl').open('a',encoding='utf-8') as f:f.write(payload+'\n');f.flush();os.fsync(f.fileno())
                    imported+=1
                print(f'Migration verified; reused {imported} EOS-complete answers',flush=True)
        answer_cache={}
        for payload, in db.execute('SELECT payload FROM answers'):
            saved=json.loads(payload);answer_cache[(saved['image_sha256'],saved['text'])]=saved
        def run(rows,split,stage):
            for q in rows:
                prior=db.execute('SELECT payload FROM answers WHERE split=? AND qid=?',(split,q['question_id'])).fetchone()
                if prior:
                    saved=json.loads(prior[0]); assert all(saved[k]==q[k] for k in q);continue
                path=ROOT/'data/coco/val2014'/q['image']
                with Image.open(path) as im: im.verify()
                torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
                image_sha=digest(path);cached=answer_cache.get((image_sha,q['text'])) if stage!='pilot100' else None
                pixels=None
                if cached:
                    answer=cached['answer'];token_ids=cached['generated_token_ids']
                    tiles=cached['tiles'];reuse={'split':cached['split'],'question_id':cached['question_id']}
                else:
                    pixels=namespace['load_image'](str(path),max_num=12).to(torch.bfloat16).cuda()
                    with torch.inference_mode():
                        answer=model.chat(tokenizer,pixels,'<image>\n'+q['text'],dict(config['generation']),history=None)
                    token_ids=generated['ids'];tiles=pixels.shape[0];reuse=None
                torch.cuda.synchronize()
                record={**q,'split':split,'answer':answer,'prediction':parse_answer(answer),'at':now(),
                        'seconds':time.perf_counter()-begin,'tiles':tiles,'reused_from':reuse,
                        'generated_token_ids':token_ids,
                        'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'image_sha256':image_sha,
                        'generated_max_new_tokens':cached.get('generated_max_new_tokens',128) if cached else config['generation']['max_new_tokens'],
                        'possible_truncation':len(token_ids)>=config['generation']['max_new_tokens'] and token_ids[-1]!=tokenizer.eos_token_id}
                # Persist every attempt before any output validation can raise.
                with (folder/'inference-attempts.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                assert answer.strip() and not record['possible_truncation'],'Empty/truncated response; halt for diagnosis'
                payload=json.dumps(record,ensure_ascii=False)
                db.execute('INSERT INTO answers VALUES(?,?,?)',(split,q['question_id'],payload));db.commit()
                answer_cache[(image_sha,q['text'])]=record if not cached else cached
                with (folder/'raw-answers.jsonl').open('a',encoding='utf-8') as f:f.write(payload+'\n');f.flush();os.fsync(f.fileno())
                del pixels
                c=counts();progress(stage,c,folder)
                print(f'{stage}: {split} {c[split]}/3000 q={q["question_id"]} {answer!r} {record["seconds"]:.2f}s',flush=True)
                if sum(c.values())%25==0:export(db,folder,stage)
        run(load_rows('random')[:100],'random','pilot100')
        pilot=[json.loads(x[0]) for x in db.execute('SELECT payload FROM answers WHERE split=? ORDER BY qid LIMIT 100',('random',))]
        assert len(pilot)==100
        official_crosscheck(pilot,folder); export(db,folder,'pilot_passed')
        atomic_json(folder/'pilot-metrics.json',metrics(pilot))
        progress('pilot_passed',counts(),folder)
        if not args.pilot_only:
            for split in ['random','popular','adversarial']:
                run(load_rows(split),split,'table3_full');export(db,folder,'table3_full')
            assert sum(counts().values())==9000
            progress('table3_complete',counts(),folder)
        print('FINISHED',folder,flush=True)
    except BaseException as e:
        with (folder/'failures.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'at':now(),'error':repr(e),'traceback':traceback.format_exc()})+'\n');f.flush();os.fsync(f.fileno())
        export(db,folder,'interrupted');progress('requires_diagnosis',counts(),folder,repr(e));raise
    finally:
        db.close();lock.close()

if __name__=='__main__':main()
