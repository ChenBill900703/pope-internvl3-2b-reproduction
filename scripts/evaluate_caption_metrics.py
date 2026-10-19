"""COCO caption metrics for the original 500 POPE images (Appendix F)."""
import argparse, csv, importlib.metadata, json, queue, shutil, sqlite3, subprocess, threading
from pathlib import Path
from run_pope import ROOT, digest, now
from atomic_io import atomic_json

class DeadlineReader:
    def __init__(self, stream, timeout=60):
        self.lines=queue.Queue();self.timeout=timeout
        def read():
            for line in iter(stream.readline,b''):self.lines.put(line)
            self.lines.put(b'')
        threading.Thread(target=read,daemon=True).start()
    def readline(self):
        try:return self.lines.get(timeout=self.timeout)
        except queue.Empty:raise TimeoutError('METEOR did not respond within 60 seconds')

def meteor_engine(log):
    import pycocoevalcap.meteor.meteor as module
    class BoundedMeteor(module.Meteor):
        # Scoring and SCORE/EVAL protocol inherited verbatim. Only Java startup,
        # timeout and cleanup are adapted; do not rewrite the metric itself.
        def __init__(self):
            self.meteor_p=None;self.error_log=None
            jar=Path(module.__file__).parent/'meteor-1.5.jar'
            self.meteor_cmd=[shutil.which('java'),'-Xmx2G','-jar',str(jar),'-','-','-stdio','-l','en','-norm',
                             '-a',str(jar.parent/'data/paraphrase-en.gz')]
            assert self.meteor_cmd[0], 'Existing Java runtime not found'
            self.error_log=log.open('ab')
            self.meteor_p=subprocess.Popen(self.meteor_cmd,cwd=jar.parent,stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=self.error_log)
            self.original_stdout=self.meteor_p.stdout
            self.meteor_p.stdout=DeadlineReader(self.original_stdout)
            self.lock=threading.Lock()
        def close(self):
            if self.meteor_p:
                if self.meteor_p.poll() is None:
                    self.meteor_p.terminate()
                    try:self.meteor_p.wait(timeout=10)
                    except subprocess.TimeoutExpired:self.meteor_p.kill();self.meteor_p.wait(timeout=10)
                try:self.meteor_p.stdin.close()
                except OSError:pass
                self.original_stdout.close();self.meteor_p=None
            if self.error_log:self.error_log.close();self.error_log=None
        def __del__(self):
            self.close()
    return BoundedMeteor()

def score(gts,res,log):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    ids=list(gts);assert set(gts)==set(res) and all(len(v)==1 for v in res.values())
    tokenizer=PTBTokenizer();gts=tokenizer.tokenize(gts);res=tokenizer.tokenize(res)
    assert set(gts)==set(ids) and set(res)==set(ids), 'Tokenizer lost images'
    b,bi=Bleu(2).compute_score(gts,res,verbose=0)
    r,ri=Rouge().compute_score(gts,res)
    engine=meteor_engine(log)
    try:m,mi=engine.compute_score(gts,res)
    finally:engine.close()
    scores={'BLEU-1':float(b[0]),'BLEU-2':float(b[1]),'METEOR':float(m),'ROUGE-L':float(r)}
    per_image=[{'image_id':i,'BLEU-1':float(bi[0][j]),'BLEU-2':float(bi[1][j]),
                'METEOR':float(mi[j]),'ROUGE-L':float(ri[j])} for j,i in enumerate(ids)]
    return scores,per_image,{'references':gts,'hypotheses':res}

def provenance():
    import pycocoevalcap.meteor.meteor as met
    import pycocoevalcap.tokenizer.ptbtokenizer as ptb
    import pycocoevalcap.bleu.bleu as bleu
    import pycocoevalcap.rouge.rouge as rouge
    files=[Path(met.__file__),Path(ptb.__file__),Path(bleu.__file__),Path(rouge.__file__),
           Path(met.__file__).parent/'meteor-1.5.jar',Path(ptb.__file__).parent/'stanford-corenlp-3.4.1.jar',
           Path(met.__file__).parent/'data/paraphrase-en.gz']
    return {'pycocoevalcap':importlib.metadata.version('pycocoevalcap'),
        'java':subprocess.run(['java','-version'],capture_output=True,text=True,timeout=15).stderr,
        'files':{str(p.relative_to(ROOT)):digest(p) for p in files},'script_sha256':digest(__file__),
        'adaptations':'METEOR Java option ordering, explicit identical paraphrase resource path to avoid double-encoded spaces on Windows, bounded response wait, process cleanup. Original metric methods unchanged.',
        'scale':'0-1 in CSV/JSON; multiply by 100 for percentage tables'}

def verify_engine():
    # Explicitly synthetic diagnostics, never mixed with model outputs.
    refs={1:[{'caption':'A small dog sits beside a wooden chair.'}],
          2:[{'caption':'Two people ride bicycles along a quiet street.'}]}
    good={i:[dict(v[0])] for i,v in refs.items()}
    wrong={i:[{'caption':'Purple galaxies beyond ancient universes.'}] for i in refs}
    a,_,_=score(refs,good,ROOT/'logs/caption-metric-engine-java.log')
    b,_,_=score(refs,wrong,ROOT/'logs/caption-metric-engine-java.log')
    assert all(a[k]>b[k] for k in a),(a,b)
    assert a['BLEU-1']>.999 and a['BLEU-2']>.999 and a['ROUGE-L']>.999
    atomic_json(ROOT/'logs/caption-metric-engine-validation.json',{'at':now(),'passed':True,
        'synthetic_diagnostics_only':True,'identity':a,'unrelated':b,'provenance':provenance()})
    print('COCO PTB/BLEU/METEOR/ROUGE scoring engine validated; synthetic diagnostics only.',flush=True)

def evaluate(folder):
    references=ROOT/'data/coco/pope-500-caption-references.json'
    refs=json.loads(references.read_text(encoding='utf-8'));ids={int(i) for i in refs};assert len(ids)==500
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
        rows=[json.loads(x[0]) for x in db.execute('SELECT payload FROM answers ORDER BY split,qid')]
    out=folder/'appendixF';out.mkdir(exist_ok=True);summary=[]
    for prompt in ['I1','I2']:
        selected={r['image_id']:r for r in rows if r['prompt_id']==prompt and r['image_id'] in ids}
        if set(selected)!=ids:
            atomic_json(out/f'{prompt}-pending.json',{'at':now(),'complete':False,'available':len(selected),
                'expected':500,'missing_images':sorted(ids-set(selected))})
            continue
        for i,r in selected.items():assert r['image_sha256']==digest(ROOT/'data/coco/val2014'/r['image'])
        gt={i:[{'caption':text} for text in refs[str(i)]] for i in sorted(ids)}
        predictions={i:[{'caption':selected[i]['answer']}] for i in sorted(ids)}
        scores,per_image,tokenized=score(gt,predictions,out/f'{prompt}-java.log')
        record={'prompt_id':prompt,'n':500,'complete':True,**scores,
            'length_limit_count':sum(bool(r.get('possible_truncation')) for r in selected.values())}
        summary.append(record)
        atomic_json(out/f'{prompt}-raw-evaluation-inputs.json',{'references':gt,'hypotheses':predictions})
        atomic_json(out/f'{prompt}-tokenized.json',tokenized)
        atomic_json(out/f'{prompt}-per-image.json',per_image)
        with (out/f'{prompt}-per-image.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(per_image[0]));writer.writeheader();writer.writerows(per_image)
    atomic_json(out/'metrics.json',summary)
    if summary:
        with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    atomic_json(out/'protocol.json',{'at':now(),'complete':len(summary)==2,'provenance':provenance(),
        'references_sha256':digest(references),'caption_folder':str(folder),'config_sha256':digest(folder/'config.json'),
        'limitation':'Appendix F does not specify its caption prompt. Report I1 and I2 separately; do not choose the better score.',
        'pope_comparison':'Use clearly identified Table3 annotation-based random F1. Original Table12 POPE values mix apparent sources; do not assume identical provenance.'})
    (out/'runner-executed.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'complete':len(summary)==2,'metrics':summary}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--folder');ap.add_argument('--verify-engine',action='store_true');args=ap.parse_args()
    if args.verify_engine:verify_engine()
    else:
        folder=Path(args.folder) if args.folder else Path(json.loads((ROOT/'logs/captions-progress.json').read_text())['results'])
        evaluate(folder)
