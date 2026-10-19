"""Package only the verified public allowlist; replay checks after extraction."""
import csv,hashlib,json,subprocess,sys,tempfile,zipfile
from pathlib import Path
from check_public import ROOT,public_files
def main():
    subprocess.run([sys.executable,str(ROOT/'release_tools/check_public.py')],cwd=ROOT,check=True)
    files=public_files();out=ROOT/'deliverables/GITHUB_PUBLIC_RELEASE_2026-10-19.zip'
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(ROOT/name,name)
    replay=ROOT/'tmp/public-release-extract'
    if replay.exists():replay=Path(tempfile.mkdtemp(prefix='public-release-',dir=ROOT/'tmp'))
    replay.mkdir(exist_ok=True)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(files)
        assert all((replay/n).resolve().is_relative_to(replay.resolve()) for n in z.namelist())
        z.extractall(replay)
    result=subprocess.run([sys.executable,str(replay/'release_tools/check_public.py')],cwd=replay,capture_output=True,text=True,encoding='utf-8',check=True)
    h=hashlib.sha256(out.read_bytes()).hexdigest()
    result_data={'zip':out.name,'bytes':out.stat().st_size,'sha256':h,'files':len(files),'CRC_and_extraction':True,'offline_public_checks':json.loads(result.stdout),'publication_status':'See GITHUB_RELEASE_AUDIT.md; archive validation alone does not establish a GitHub push.'}
    (ROOT/'local_backups/release-2026-10-19/public-package-validation.json').write_text(json.dumps(result_data,ensure_ascii=False,indent=2),encoding='utf-8')
    out.with_suffix('.sha256').write_text(h+'  '+out.name+'\n',encoding='utf-8')
    print(json.dumps(result_data,ensure_ascii=False))
if __name__=='__main__':main()
