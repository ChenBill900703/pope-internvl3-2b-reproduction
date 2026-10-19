"""Reproduce a real Windows file-sharing conflict using isolated synthetic metadata."""
import json,subprocess,sys,time
from pathlib import Path
from run_pope import ROOT,now
from atomic_io import atomic_json
folder=ROOT/'tmp'/('atomic-io-test-'+str(time.time_ns()));folder.mkdir(parents=True)
path=folder/'status.json';atomic_json(path,{'version':0})
holder=folder/'holder.py'
holder.write_text('''import time
from pathlib import Path
p=Path(__file__).parent
with (p/'status.json').open('r',encoding='utf-8') as f:
    (p/'ready').write_text('ready')
    time.sleep(.7)
''',encoding='utf-8')
proc=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),str(holder)],cwd=folder)
deadline=time.monotonic()+5
while not (folder/'ready').exists():
    assert time.monotonic()<deadline;time.sleep(.01)
result=atomic_json(path,{'version':1,'text':'café'})
assert proc.wait(timeout=5)==0
assert json.loads(path.read_text(encoding='utf-8'))=={'version':1,'text':'café'}
assert result['attempts']>1,'Expected a real transient Windows sharing conflict'
atomic_json(ROOT/'logs/atomic-io-fault-test.json',{'at':now(),'passed':True,'attempts':result['attempts'],
    'real_windows_sharing_conflict':True,'fixture':str(folder),'research_data_touched':False})
print('PASS: real Windows sharing conflict recovered; UTF-8 JSON intact;',result['attempts'],'attempts')
