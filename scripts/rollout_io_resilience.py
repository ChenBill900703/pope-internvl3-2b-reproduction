"""Controlled operational rollout. Stop only exact project controllers/worker, preserve checkpoints."""
import hashlib,json,sqlite3,sys,time
from pathlib import Path
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[1]

def script_is(p,names):
    try:
        if Path(p.cwd()).resolve()!=ROOT:return False
        for arg in p.cmdline()[1:]:
            if arg.startswith('-'):continue
            path=Path(arg)
            if not path.is_absolute():path=ROOT/path
            if path.resolve() in [ROOT/'scripts'/n for n in names]:return True
    except (psutil.NoSuchProcess,psutil.AccessDenied,OSError,ValueError):pass
    return False

def main():
    ap=sys.argv[1]
    if ap=='controllers':
        names=['watchdog.py','supervise_captions.py']
        workers=[p.pid for p in psutil.process_iter() if script_is(p,['run_captions.py','run_captions_resilient.py'])]
        controllers=[p for p in psutil.process_iter() if script_is(p,names)]
        for p in controllers:
            try:
                if script_is(p,names):p.terminate()
            except psutil.NoSuchProcess:pass
        psutil.wait_procs(controllers,timeout=3)
        assert all(psutil.pid_exists(pid) for pid in workers),'Worker unexpectedly exited during controller swap'
        print(json.dumps({'controllers_stopped':[p.pid for p in controllers],'existing_worker_preserved':workers}))
    elif ap=='worker':
        state=json.loads((ROOT/'logs/captions-progress.json').read_text(encoding='utf-8'));folder=Path(state['results'])
        # Prefer to stop during the next GPU call, outside the short checkpoint write window.
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
                payload=db.execute('SELECT payload FROM answers ORDER BY rowid DESC LIMIT 1').fetchone()[0]
            latest=json.loads(payload);age=(datetime.now(timezone.utc)-datetime.fromisoformat(latest['at'])).total_seconds()
            if latest['prompt_id']=='I1' and .4<age<1.2:break
            time.sleep(.1)
        with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
            saved=[{'split':s,'qid':q,'sha256':hashlib.sha256(payload.encode()).hexdigest()} for s,q,payload in db.execute('SELECT split,qid,payload FROM answers')]
        audit={'at':datetime.now(timezone.utc).isoformat(),'folder':str(folder),'reason':'Apply tested IO resilience only; inference parameters and frozen runner unchanged',
               'saved_before':len(saved),'saved_record_hashes':saved}
        (ROOT/'logs/io-layer-rollout-before.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        workers=[p for p in psutil.process_iter() if script_is(p,['run_captions.py'])]
        for p in workers:
            try:
                if script_is(p,['run_captions.py']):p.terminate()
            except psutil.NoSuchProcess:pass
        psutil.wait_procs(workers,timeout=3)
        print(json.dumps({'controlled_worker_restart':[p.pid for p in workers],'saved_results_preserved':len(saved),'folder':str(folder)}))
    else:raise ValueError(ap)

if __name__=='__main__':main()
