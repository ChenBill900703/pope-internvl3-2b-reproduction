"""Read local checkpoint health without loading a model or starting a worker."""
import json,msvcrt,sqlite3
from datetime import datetime,timezone
from pathlib import Path
from atomic_io import atomic_json
ROOT=Path(__file__).resolve().parents[1]

def held(name):
    if not (ROOT/'logs'/name).exists():return False
    with (ROOT/'logs'/name).open('r+b') as f:
        f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:return True
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        return False

def main():
    p=json.loads((ROOT/'logs/progress.json').read_text(encoding='utf-8'))
    phase='table3';splits=['random','popular','adversarial'];supervisor='supervisor.lock'
    cap=ROOT/'logs/captions-progress.json'
    if p['stage']=='table3_complete' and cap.exists():
        p=json.loads(cap.read_text(encoding='utf-8'));phase='captions';splits=['I1','I2'];supervisor='caption-supervisor.lock'
    stability=ROOT/'logs/stability-progress.json'
    if phase=='captions' and p['stage']=='captions_complete' and stability.exists():
        p=json.loads(stability.read_text(encoding='utf-8'));phase='table4';splits=['P1','P2','P3','P4','C1','C2','C3','C4'];supervisor='stability-supervisor.lock'
    appendix=ROOT/'logs/appendixD-progress.json'
    if phase=='table4' and p['stage']=='table4_complete' and appendix.exists():
        p=json.loads(appendix.read_text(encoding='utf-8'));phase='appendixD'
        splits=[f'{d}_{s}' for d in ['aokvqa','gqa'] for s in ['random','popular','adversarial']];supervisor='appendixD-supervisor.lock'
    vqa=ROOT/'logs/vqa-progress.json'
    if phase=='appendixD' and p['stage']=='appendixD_complete' and vqa.exists():
        p=json.loads(vqa.read_text(encoding='utf-8'));phase='vqa';splits=['aokvqa','gqa'];supervisor='vqa-supervisor.lock'
    folder=Path(p['results'])
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
        counts=dict(db.execute('SELECT split,COUNT(*) FROM answers GROUP BY split'))
        last=db.execute('SELECT payload FROM answers ORDER BY rowid DESC LIMIT 1').fetchone()
    now=datetime.now(timezone.utc)
    last=json.loads(last[0]) if last else None
    age=(now-datetime.fromisoformat(last['at'])).total_seconds() if last else None
    check={'checked_at':now.isoformat(),'automation_id':'pope','phase':phase,'stage':p['stage'],
           'counts':{s:counts.get(s,0) for s in splits},
           'gpu_lock_held':held('gpu.lock'),'supervisor_lock_held':held(supervisor),
           'last_answer_age_seconds':age,'error':p.get('error'),
           'new_gpu_worker_started':False}
    prefix={'captions':'caption','table4':'stability','appendixD':'appendixD','vqa':'vqa'}.get(phase,'table3')
    watchdog=ROOT/'logs'/f'{prefix}-watchdog-state.json'
    check['watchdog_lock_held']=held(f'{prefix}-watchdog.lock') if phase in ['captions','table4','appendixD','vqa'] else False
    if phase in ['captions','table4','appendixD','vqa'] and watchdog.exists():
        wd=json.loads(watchdog.read_text(encoding='utf-8'))
        check['watchdog_state_age_seconds']=(now-datetime.fromisoformat(wd['at'])).total_seconds()
        check['watchdog_attention_required']=wd['attention_required']
    controller=check['supervisor_lock_held'] or (check['watchdog_lock_held'] and check.get('watchdog_state_age_seconds',999)<45)
    check['progress_state_age_seconds']=(now-datetime.fromisoformat(p['updated_at'])).total_seconds()
    activity_age=min(age if age is not None else float('inf'),check['progress_state_age_seconds'])
    check['healthy_running']=check['gpu_lock_held'] and controller and activity_age<300 and not check['error'] and not check.get('watchdog_attention_required',False)
    check['followup_coordinator_lock_held']=held('followup-coordinator.lock')
    check['generation_complete']=phase=='vqa' and p['stage']=='vqa_raw_complete_needs_examiner'
    raw_validation=folder/'raw-validation.json'
    native_validation=folder/'native-evaluation/independent-validation.json'
    check['available_inference_and_local_grading_complete']=bool(check['generation_complete'] and
        raw_validation.exists() and json.loads(raw_validation.read_text(encoding='utf-8'))['passed'] and
        native_validation.exists() and json.loads(native_validation.read_text(encoding='utf-8'))['passed'])
    check['run_state']='completed_available_scope' if check['available_inference_and_local_grading_complete'] else ('running' if check['healthy_running'] else 'needs_check')
    atomic_json(ROOT/'logs/heartbeat-last-check.json',check)
    with (ROOT/'logs/heartbeat-checks.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(check)+'\n')
    print(json.dumps(check))

if __name__=='__main__':main()
