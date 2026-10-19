"""Atomic metadata writes with bounded retries for transient Windows sharing conflicts."""
import json,os,tempfile,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]

def event(kind,**data):
    try:
        with (ROOT/'logs/io-recovery-events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'kind':kind,**data},ensure_ascii=False)+'\n')
    except OSError:pass

def atomic_json(path,value):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name)
    with os.fdopen(fd,'w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    delays=[0,.025,.05,.1,.2,.4,.8,1.,1.5]
    for attempt,delay in enumerate(delays,1):
        if delay:time.sleep(delay)
        try:
            os.replace(temporary,path)
            if attempt>1:event('atomic_replace_recovered',path=str(path),attempts=attempt)
            return {'attempts':attempt}
        except PermissionError:
            if attempt==len(delays):
                event('atomic_replace_deferred',path=str(path),temporary=str(temporary),attempts=attempt)
                raise

