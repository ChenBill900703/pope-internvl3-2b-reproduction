"""Fault injection uses a tiny synthetic worker; no GPU or research data is touched."""
import json,subprocess,sys,time,sqlite3
from pathlib import Path
from run_pope import ROOT,atomic_json,now

testroot=ROOT/'tmp'/('watchdog-test-'+str(time.time_ns()));testroot.mkdir(parents=True)
fake=testroot/'fake_worker.py'
fake.write_text('''import json,sqlite3,time,os
from pathlib import Path
root=Path(__file__).resolve().parent
db=sqlite3.connect(root/'test.sqlite');db.execute('PRAGMA synchronous=FULL')
db.execute('CREATE TABLE IF NOT EXISTS saved(id INTEGER PRIMARY KEY,value TEXT)');db.commit()
for i in range(1,13):
    if db.execute('SELECT 1 FROM saved WHERE id=?',(i,)).fetchone():continue
    db.execute('INSERT INTO saved VALUES(?,?)',(i,'saved-'+str(i)));db.commit()
    tmp=root/'progress.tmp';tmp.write_text(json.dumps({'stage':'running','counts':{'done':i}}),encoding='utf-8');os.replace(tmp,root/'progress.json')
    if i==3 and not (root/'crash-once').exists():
        (root/'crash-once').write_text('yes');os._exit(17)
    if i==6 and not (root/'stall-once').exists():
        (root/'stall-once').write_text('yes');time.sleep(120)
    time.sleep(.15)
tmp=root/'progress.tmp';tmp.write_text(json.dumps({'stage':'complete','counts':{'done':12}}),encoding='utf-8');os.replace(tmp,root/'progress.json')
''',encoding='utf-8')
config={'root':str(testroot),'python':str(ROOT/'.venv/Scripts/python.exe'),'worker_script':'fake_worker.py',
        'progress_file':'progress.json','gpu_lock':'test-gpu.lock','watchdog_lock':'test-watchdog.lock',
        'log_prefix':'watchdog','completed_stages':['complete'],'max_restarts':4,'check_seconds':.25,
        'stall_seconds':2,'restart_delay_seconds':.3,'test_timeout_seconds':40,'reset_after_answers':100}
atomic_json(testroot/'config.json',config)
result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/watchdog.py'),'--config',str(testroot/'config.json')],cwd=ROOT,timeout=50)
assert result.returncode==0,result.returncode
with sqlite3.connect(testroot/'test.sqlite') as db:
    rows=db.execute('SELECT id,value FROM saved ORDER BY id').fetchall()
assert rows==[(i,'saved-'+str(i)) for i in range(1,13)]
events=[json.loads(l) for l in (testroot/'watchdog-events.jsonl').read_text(encoding='utf-8').splitlines()]
assert any(e['kind']=='worker_exited' and e['code']==17 for e in events)
assert any(e['kind']=='stalled_worker_termination' for e in events)
assert sum(e['kind']=='worker_launch' for e in events)>=3
atomic_json(ROOT/'logs/watchdog-fault-test.json',{'at':now(),'passed':True,'fixture':str(testroot),
    'forced_crash_recovered':True,'forced_stall_recovered':True,'saved_ids':len(rows),
    'missing_records':0,'duplicate_records':0,'research_data_touched':False,'gpu_used':False})
print('PASS: forced crash + forced stall recovered; 12/12 records preserved; no duplicates; no GPU used.')
