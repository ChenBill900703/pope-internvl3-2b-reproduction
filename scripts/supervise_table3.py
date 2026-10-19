"""Local bounded restart supervisor. Never runs a second GPU worker."""
import json,subprocess,sys,time,os
from pathlib import Path
import psutil,msvcrt
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='backslashreplace')
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
lock=(ROOT/'logs/supervisor.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
attempts=0
while True:
    state=json.loads((ROOT/'logs/progress.json').read_text(encoding='utf-8'))
    if state['stage']=='table3_complete':
        subprocess.run([sys.executable,'scripts/generate_report.py'],cwd=ROOT,check=True)
        print('Table 3 complete. Ready for remaining experiments by chat automation.',flush=True)
        break
    workers=[]
    for proc in psutil.process_iter(['pid','cmdline']):
        try:
            args=proc.info['cmdline'] or []
            if any(Path(a).name=='run_pope.py' for a in args):workers.append(proc.pid)
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    if workers:
        time.sleep(30);continue
    if attempts>=3 or (state.get('error') and any(s in state['error'] for s in ['OutOfMemory','Non-finite','truncated','FloatingPoint'])):
        print('Stopped for diagnosis; preserved all outputs.',state,flush=True);break
    attempts+=1
    filename=ROOT/f'logs/run-pope-resume-{int(time.time())}.log'
    print('Resuming existing configuration',attempts,filename,flush=True)
    with filename.open('ab') as log:
        result=subprocess.run([sys.executable,'-u','scripts/run_pope.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    print('Worker exit:',result.returncode,flush=True)
    time.sleep(30)
