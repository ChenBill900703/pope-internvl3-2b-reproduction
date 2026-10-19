"""Sequential phase handoff only; active-worker recovery belongs to its watchdog."""
import json,msvcrt,os,subprocess,sys,time,traceback
from pathlib import Path
from run_pope import ROOT,now,digest
from atomic_io import atomic_json
from check_run import held

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def event(kind,**values):
    with (ROOT/'logs/followup-coordinator-events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'at':now(),'kind':kind,**values},ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def state(stage,**values):
    atomic_json(ROOT/'logs/followup-coordinator-state.json',{'at':now(),'pid':os.getpid(),'stage':stage,**values})
def step(script,args=(),required=True):
    log=ROOT/'logs'/f'followup-{Path(script).stem}-{time.time_ns()}.log'
    event('cpu_step_started',script=script,log=str(log))
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-u',script,*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                              timeout=1200)
    event('cpu_step_finished',script=script,returncode=result.returncode,log=str(log))
    if required:assert result.returncode==0,f'{script} failed; inspect {log}'
    return result.returncode

def main():
    lock=(ROOT/'logs/followup-coordinator.lock').open('a+b')
    if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        event('coordinator_started',pid=os.getpid())
        while True:
            caps=read(ROOT/'logs/captions-progress.json')
            if caps['stage']=='captions_complete' and not held('gpu.lock') and not held('caption-watchdog.lock'):break
            state('waiting_for_caption_completion',caption_stage=caps['stage'],counts=caps['counts'],
                  launches_gpu_worker=False)
            time.sleep(20)
        state('validating_complete_captions')
        folder=Path(caps['results'])
        if not (folder/'captions-validation.json').exists():
            step('scripts/verify_chair.py')
            step('scripts/verify_captions.py')
        assert read(folder/'captions-validation.json')['passed']
        step('scripts/analyze_consistency.py',required=False)
        # Independent caption metric failures must not prevent other valid tasks.
        if not (folder/'appendixF/protocol.json').exists() or not read(folder/'appendixF/protocol.json')['complete']:
            step('scripts/evaluate_caption_metrics.py',['--folder',str(folder)],required=False)
        step('scripts/generate_report.py',required=False)
        state('preparing_table4')
        step('scripts/run_stability.py',['--prepare-only'])
        step('scripts/verify_stability.py',['--allow-partial'])
        progress=read(ROOT/'logs/stability-progress.json');table4=Path(progress['results']);config=read(table4/'config.json')
        for name,expected in config['source_sha256'].items():
            assert digest(ROOT/'scripts'/name)==expected, f'Source changed; audited migration needed: {name}'
        watchdog={
            'root':str(ROOT),'python':str(ROOT/'.venv/Scripts/python.exe'),
            'worker_script':'scripts/run_stability.py',
            'expected_script_sha256':config['source_sha256']['run_stability.py'],
            'expected_dependency_sha256':{'scripts/'+name:value for name,value in config['source_sha256'].items() if name!='run_stability.py'},
            'progress_file':'logs/stability-progress.json','gpu_lock':'logs/gpu.lock',
            'watchdog_lock':'logs/stability-watchdog.lock','log_prefix':'logs/stability-watchdog',
            'reset_file':'logs/stability-watchdog-reset.json',
            'activity_files':[str((table4/'inference-attempts.jsonl').relative_to(ROOT))],
            'completed_stages':['table4_complete'],'attention_stages':['table4_incomplete'],
            'max_restarts':3,'check_seconds':10,'stall_seconds':300,'restart_delay_seconds':10,'reset_after_answers':50}
        caption_config=read(folder/'config.json')
        ground_truth='third_party/POPE/segmentation/coco_ground_truth_segmentation.json'
        assert digest(ROOT/ground_truth)==caption_config['ground_truth_sha256']
        watchdog['expected_dependency_sha256'][ground_truth]=caption_config['ground_truth_sha256']
        target=ROOT/'logs/stability-watchdog-config.json';atomic_json(target,watchdog)
        state('table4_watchdog_running',results=str(table4))
        if not held('stability-watchdog.lock'):
            log=ROOT/'logs'/f'stability-watchdog-console-{time.time_ns()}.log'
            with log.open('ab') as stream:
                # No timeout: this is the durable phase supervisor, with its own
                # bounded worker retries and source guards. It holds no GPU.
                result=subprocess.run([sys.executable,'-u','scripts/watchdog.py','--config',str(target)],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            assert result.returncode==0,f'Table4 watchdog requires diagnosis: {log}'
        else:
            while held('stability-watchdog.lock'):time.sleep(20)
        assert read(ROOT/'logs/stability-progress.json')['stage']=='table4_complete'
        step('scripts/verify_stability.py')
        step('scripts/generate_report.py',required=False)
        state('table4_verified_preparing_appendixD',results=str(table4))
        gqa=ROOT/'data/gqa/images-validation.json'
        if not gqa.exists() or not read(gqa)['passed']:
            state('appendixD_waiting_for_original_images',next_resource='data/gqa/images-validation.json')
            event('image_preparation_still_needed',resource=str(gqa));return
        step('scripts/run_appendix_d.py',['--prepare-only'])
        p=read(ROOT/'logs/appendixD-progress.json');appendix=Path(p['results']);settings=read(appendix/'config.json')
        appendix_watchdog={
            'root':str(ROOT),'python':str(ROOT/'.venv/Scripts/python.exe'),'worker_script':'scripts/run_appendix_d.py',
            'expected_script_sha256':settings['source_sha256']['run_appendix_d.py'],
            'expected_dependency_sha256':{'scripts/'+name:value for name,value in settings['source_sha256'].items() if name!='run_appendix_d.py'},
            'progress_file':'logs/appendixD-progress.json','gpu_lock':'logs/gpu.lock',
            'watchdog_lock':'logs/appendixD-watchdog.lock','log_prefix':'logs/appendixD-watchdog',
            'reset_file':'logs/appendixD-watchdog-reset.json',
            'activity_files':[str((appendix/'inference-attempts.jsonl').relative_to(ROOT))],
            'completed_stages':['appendixD_complete'],'attention_stages':['appendixD_incomplete'],
            'max_restarts':3,'check_seconds':10,'stall_seconds':300,'restart_delay_seconds':10,'reset_after_answers':50}
        appendix_target=ROOT/'logs/appendixD-watchdog-config.json';atomic_json(appendix_target,appendix_watchdog)
        state('appendixD_watchdog_running',results=str(appendix))
        if not held('appendixD-watchdog.lock'):
            log=ROOT/'logs'/f'appendixD-watchdog-console-{time.time_ns()}.log'
            with log.open('ab') as stream:
                result=subprocess.run([sys.executable,'-u','scripts/watchdog.py','--config',str(appendix_target)],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            assert result.returncode==0,f'AppendixD watchdog requires diagnosis: {log}'
        else:
            while held('appendixD-watchdog.lock'):time.sleep(20)
        assert read(ROOT/'logs/appendixD-progress.json')['stage']=='appendixD_complete'
        assert read(appendix/'validation.json')['passed']
        step('scripts/generate_report.py',required=False)
        state('appendixD_verified_preparing_vqa_raw',results=str(appendix))
        step('scripts/verify_appendix_d.py')
        step('scripts/run_vqa.py',['--prepare-only'])
        step('scripts/verify_vqa.py',['--allow-partial'])
        p=read(ROOT/'logs/vqa-progress.json');vqa=Path(p['results']);settings=read(vqa/'config.json')
        vqa_watchdog={
            'root':str(ROOT),'python':str(ROOT/'.venv/Scripts/python.exe'),'worker_script':'scripts/run_vqa.py',
            'expected_script_sha256':settings['source_sha256']['run_vqa.py'],
            'expected_dependency_sha256':{'scripts/'+name:value for name,value in settings['source_sha256'].items() if name!='run_vqa.py'},
            'progress_file':'logs/vqa-progress.json','gpu_lock':'logs/gpu.lock',
            'watchdog_lock':'logs/vqa-watchdog.lock','log_prefix':'logs/vqa-watchdog',
            'reset_file':'logs/vqa-watchdog-reset.json',
            'activity_files':[str((vqa/'inference-attempts.jsonl').relative_to(ROOT))],
            'completed_stages':['vqa_raw_complete_needs_examiner'],'attention_stages':['vqa_raw_incomplete'],
            'max_restarts':3,'check_seconds':10,'stall_seconds':300,'restart_delay_seconds':10,'reset_after_answers':50}
        for dataset,value in settings['question_sha256'].items():
            vqa_watchdog['expected_dependency_sha256'][f'data/{dataset}/pope-image-vqa-questions.json']=value
        target=ROOT/'logs/vqa-watchdog-config.json';atomic_json(target,vqa_watchdog)
        state('vqa_raw_watchdog_running',results=str(vqa),external_examiner_called=False)
        if not held('vqa-watchdog.lock'):
            log=ROOT/'logs'/f'vqa-watchdog-console-{time.time_ns()}.log'
            with log.open('ab') as stream:
                result=subprocess.run([sys.executable,'-u','scripts/watchdog.py','--config',str(target)],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            assert result.returncode==0,f'VQA watchdog requires diagnosis: {log}'
        else:
            while held('vqa-watchdog.lock'):time.sleep(20)
        assert read(ROOT/'logs/vqa-progress.json')['stage']=='vqa_raw_complete_needs_examiner'
        step('scripts/verify_vqa.py')
        step('scripts/generate_report.py',required=False)
        state('vqa_raw_verified_waiting_examiner_authorization',results=str(vqa),external_examiner_called=False)
        event('raw_vqa_handoff_completed',next_experiments='External examiner needs authorization; original COCO SEEM fixed questions still unresolved')
    except BaseException as error:
        event('requires_diagnosis',error=repr(error),traceback=traceback.format_exc())
        state('requires_diagnosis',error=repr(error));raise
    finally:lock.close()

if __name__=='__main__':main()
