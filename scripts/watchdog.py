"""Monitor durable progress independently of the worker; recover bounded process failures."""
import argparse,hashlib,json,os,subprocess,sys,time,msvcrt
from datetime import datetime,timezone
from pathlib import Path
import psutil
from atomic_io import atomic_json

def utc():return datetime.now(timezone.utc).isoformat()
def read_json(path):
    for _ in range(3):
        try:return json.loads(path.read_text(encoding='utf-8'))
        except (FileNotFoundError,json.JSONDecodeError,PermissionError):time.sleep(.1)
    return None
def write_json(path,data):
    return atomic_json(path,data)

class Watchdog:
    def __init__(self,config):
        self.c=config;self.root=Path(config['root']).resolve();self.script=(self.root/config['worker_script']).resolve()
        self.aliases=[(self.root/name).resolve() for name in config.get('worker_aliases',[])]
        assert self.script.is_relative_to(self.root) and self.script.exists()
        self.progress=self.root/config['progress_file'];self.prefix=self.root/config['log_prefix']
        self.child=None;self.log=None;self.restarts=0;self.last_count=None;self.last_advance=time.monotonic()
        self.started=time.monotonic();self.max_count=0;self.last_launch=0;self.attention=False
        self.baseline=None;self.reset_token=None;self.recovered=0
        self.activity_snapshot=None
        self.lock=(self.root/config['watchdog_lock']).open('a+b')
        if self.lock.seek(0,2)==0:self.lock.write(b'0');self.lock.flush()
        self.lock.seek(0);msvcrt.locking(self.lock.fileno(),msvcrt.LK_NBLCK,1)
    def event(self,kind,**data):
        item={'at':utc(),'kind':kind,**data}
        with Path(str(self.prefix)+'-events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    def matches(self,p):
        try:
            if Path(p.cwd()).resolve()!=self.root:return False
            for arg in p.cmdline()[1:]:
                if arg.startswith('-'):continue
                path=Path(arg)
                if not path.is_absolute():path=self.root/path
                try:
                    if path.resolve() in [self.script,*self.aliases]:return True
                except (OSError,ValueError):pass
        except (psutil.NoSuchProcess,psutil.AccessDenied,OSError):pass
        return False
    def workers(self):return [p for p in psutil.process_iter() if self.matches(p)]
    def gpu_busy(self):
        path=self.root/self.c['gpu_lock']
        with path.open('a+b') as f:
            if f.seek(0,2)==0:f.write(b'0');f.flush()
            f.seek(0)
            try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:return True
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1);return False
    def terminate_stalled(self,workers):
        # Only exact configured script/cwd matches are eligible. Never terminate by process name.
        checked=[p for p in workers if self.matches(p)]
        self.event('stalled_worker_termination',pids=[p.pid for p in checked])
        for p in checked:
            try:
                if self.matches(p):p.terminate()
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        _,alive=psutil.wait_procs(checked,timeout=3)
        for p in alive:
            try:
                if self.matches(p):p.kill()
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    def launch(self):
        expected=self.c.get('expected_script_sha256')
        if expected and hashlib.sha256(self.script.read_bytes()).hexdigest()!=expected:
            self.attention=True;self.event('worker_source_changed_requires_audited_migration');return
        for name,expected in self.c.get('expected_dependency_sha256',{}).items():
            if hashlib.sha256((self.root/name).read_bytes()).hexdigest()!=expected:
                self.attention=True;self.event('dependency_source_changed_requires_audited_migration',path=name);return
        if self.log:self.log.close()
        logfile=Path(str(self.prefix)+f'-worker-{time.time_ns()}.log');self.log=logfile.open('ab')
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        self.child=subprocess.Popen([self.c['python'],'-u',str(self.script),*self.c.get('worker_args',[])],
            cwd=self.root,stdout=self.log,stderr=subprocess.STDOUT,env=env)
        self.last_launch=time.monotonic();self.last_advance=time.monotonic();self.restarts+=1
        self.event('worker_launch',pid=self.child.pid,attempt=self.restarts,log=str(logfile))
    def run(self):
        self.event('watchdog_started',pid=os.getpid(),script=str(self.script))
        try:
            while True:
                state=read_json(self.progress) or {};now=time.monotonic()
                count=sum(state.get('counts',{}).values())
                activity=[]
                for name in self.c.get('activity_files',[]):
                    try:
                        st=(self.root/name).stat();activity.append((name,st.st_mtime_ns,st.st_size))
                    except FileNotFoundError:activity.append((name,None,None))
                if self.activity_snapshot is not None and activity!=self.activity_snapshot:self.last_advance=now
                self.activity_snapshot=activity
                if self.last_count!=count:
                    if self.last_count is not None and count>self.last_count:
                        self.last_advance=now
                        if self.attention:self.event('progress_resumed',count=count)
                        self.attention=False
                    self.last_count=count
                if self.baseline is None:self.baseline=count
                if count-self.baseline>=self.c.get('reset_after_answers',50):
                    self.restarts=0;self.baseline=count;self.attention=False
                resetfile=self.c.get('reset_file')
                reset=read_json(self.root/resetfile) if resetfile else None
                if reset and reset.get('token')!=self.reset_token:
                    self.reset_token=reset['token'];self.restarts=0;self.attention=False;self.last_advance=now
                    self.event('repair_reset_received',token=self.reset_token)
                stage=state.get('stage');workers=self.workers()
                snapshot={'at':utc(),'watchdog_pid':os.getpid(),'stage':stage,'counts':state.get('counts',{}),
                          'worker_pids':[p.pid for p in workers],'seconds_without_progress':round(now-self.last_advance,2),
                          'restarts_since_stable_progress':self.restarts,'attention_required':self.attention}
                try:write_json(Path(str(self.prefix)+'-state.json'),snapshot)
                except PermissionError:self.event('watchdog_status_update_deferred')
                if stage in self.c.get('completed_stages',[]):
                    self.event('job_complete',count=count);return 0
                if stage in self.c.get('attention_stages',[]):
                    if not self.attention:self.event('partial_completion_needs_diagnosis',state=state)
                    self.attention=True
                elif workers:
                    if now-self.last_advance>self.c.get('stall_seconds',300):
                        if self.restarts>=self.c.get('max_restarts',3):
                            if not self.attention:self.event('restart_budget_exhausted',state=state)
                            self.attention=True
                        else:
                            self.terminate_stalled(workers);self.last_advance=time.monotonic()
                elif not self.attention:
                    if self.child and self.child.poll() is not None:
                        self.event('worker_exited',code=self.child.returncode);self.child=None
                    if self.restarts>=self.c.get('max_restarts',3):
                        self.attention=True;self.event('restart_budget_exhausted',state=state)
                    elif not self.gpu_busy() and now-self.last_launch>=self.c.get('restart_delay_seconds',10):
                        self.launch()
                if self.c.get('test_timeout_seconds') and now-self.started>self.c['test_timeout_seconds']:
                    self.event('test_timeout');return 2
                time.sleep(self.c.get('check_seconds',10))
        finally:
            if self.log:self.log.close()
            self.lock.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    raise SystemExit(Watchdog(json.loads(Path(args.config).read_text(encoding='utf-8'))).run())
