"""Bounded local restarts for caption inference; respects the shared GPU lock."""
import json,os,subprocess,sys,time,msvcrt
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
for stream in [sys.stdout,sys.stderr]:
    if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='backslashreplace')
lock=(ROOT/'logs/caption-supervisor.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
attempts=0
while True:
    state=json.loads((ROOT/'logs/captions-progress.json').read_text(encoding='utf-8'))
    if state['stage']=='captions_complete':
        print('All queued captions saved. Ready for next experiments.',flush=True);break
    if state['stage']=='captions_incomplete':
        print('Nonempty captions saved; some empty outputs need diagnosis. No automatic loop.',flush=True);break
    workers=[]
    for p in psutil.process_iter(['cmdline']):
        try:
            if any(Path(a).name in ['run_pope.py','run_captions.py'] for a in (p.info['cmdline'] or [])):workers.append(p.pid)
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    if workers:time.sleep(30);continue
    if attempts>=3 or (state.get('error') and any(s in state['error'] for s in ['OutOfMemory','Non-finite','truncated','FloatingPoint','mismatch'])):
        print('Preserved outputs; diagnosis required.',state,flush=True);break
    attempts+=1
    logfile=ROOT/f'logs/caption-worker-{int(time.time())}.log'
    with logfile.open('ab') as f:
        result=subprocess.run([sys.executable,'-u','scripts/run_captions.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    print('Caption worker exited:',result.returncode,flush=True);time.sleep(30)
