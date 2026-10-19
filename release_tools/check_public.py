"""CPU-only public release audit; never reads images, calls models or networks.

Without flags checks all published data and optional manifest; --write-manifest
records the explicit Git allowlist before staging. Private-source revalidation
is a separate, clearly labeled verify_local.py operation.
"""
import argparse,ast,csv,hashlib,importlib.util,json,math,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def public_files():
    # Exactly the same allowlist Git sees; assert no overlooked root directory.
    if not (ROOT/'.git').exists():
        manifest=read(ROOT/'public_data/RELEASE_MANIFEST.json')
        return sorted([*manifest,'public_data/RELEASE_MANIFEST.json','GITHUB_RELEASE_AUDIT.md'])
    data=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT)
    return sorted(set(x.decode('utf-8') for x in data.split(b'\0') if x))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write-manifest',action='store_true');ap.add_argument('--staged',action='store_true');args=ap.parse_args()
    if args.write_manifest and not (ROOT/'public_data/RELEASE_MANIFEST.json').exists():
        (ROOT/'public_data/RELEASE_MANIFEST.json').write_text('{}\n',encoding='utf-8')
    files=public_files();assert files
    forbidden={'results','analysis','data','models','third_party','.cache','.venv','tmp','logs','local_backups','deliverables'}
    for name in files:
        p=ROOT/name
        assert p.parts[len(ROOT.parts)] not in forbidden,name
        assert p.is_file() and not p.is_symlink(),name
        assert p.stat().st_size<2_000_000,(name,'unexpectedly large file')
        assert p.suffix not in {'.pdf','.zip','.sqlite','.jsonl','.safetensors','.bin'},name
        if p.suffix in {'.md','.py','.json','.txt','.csv','.svg'}:
            text=p.read_text(encoding='utf-8-sig')
            for pattern in [r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',r'hf_[A-Za-z0-9]{20,}',r'sk-[A-Za-z0-9]{24,}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'[A-Z]:[\\/]Users[\\/]',r'[\w.+-]+@[\w-]+\.[A-Za-z]{2,}']:
                assert not re.search(pattern,text,re.I),(name,'potential secret/private path',pattern)
        if p.suffix=='.py':compile(p.read_text(encoding='utf-8-sig'),name,'exec')
    # Validate every local Markdown file/image link against the public file set.
    links=0
    for name in files:
        if not name.endswith('.md'):continue
        txt=(ROOT/name).read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',txt):
            target=target.split(' "')[0].strip('<>')
            if re.match(r'^[a-z]+:',target,re.I) or target.startswith('#'):continue
            clean=target.split('#')[0]
            if not clean:continue
            dest=((ROOT/name).parent/clean).resolve()
            assert dest.is_relative_to(ROOT),(name,target)
            assert dest.relative_to(ROOT).as_posix() in files,(name,target,'not publicly tracked')
            links+=1
    metrics=read(ROOT/'public_data/coco_pope.json')+read(ROOT/'public_data/aokvqa_gqa_pope.json')
    for r in metrics:
        assert r['n']==3000 and sum(r[x] for x in ['TP','TN','FP','FN'])==r['n']
        assert r['TP']+r['FN']==r['TN']+r['FP']==1500
        expected=2*r['TP']/(2*r['TP']+r['FP']+r['FN'])
        assert math.isclose(r['f1'],expected,abs_tol=1e-12)
        assert math.isclose(r['accuracy'],(r['TP']+r['TN'])/r['n'],abs_tol=1e-12)
    for p in (ROOT/'public_data').glob('*.csv'):
        with p.open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
        jj=read(p.with_suffix('.json'));assert len(rr)==len(jj),(p,len(rr),len(jj))
        for a,b in zip(rr,jj):
            for k,v in b.items():
                expected=json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else '' if v is None else str(v)
                assert a[k]==expected,(p,k)
    rules=module('public_vqa_rules',ROOT/'research_extensions/vqa/local_rules.py')
    assert rules.self_test()==20
    frozen=read(ROOT/'research_extensions/vqa/FROZEN_GRADING_RULES_v2.json')
    # Parser source must remain the frozen v2.0 original, not a cosmetic rewrite.
    assert rules.VERSION=='2.0'
    for name,h in frozen['code_sha256'].items():assert sha(ROOT/'research_extensions/vqa'/name)==h,(name,'frozen rule changed')
    original=ROOT/'analysis/vqa_diagnostics_2026-10-19/local_rules.py'
    if original.exists():assert sha(original)==sha(ROOT/'research_extensions/vqa/local_rules.py')
    d=module('public_D',ROOT/'research_extensions/semantic_audit/semantic_rules_d.py')
    dfrozen=read(ROOT/'research_extensions/semantic_audit/FROZEN_D_RULES_v2.json')
    assert sha(ROOT/'research_extensions/semantic_audit/semantic_rules_d.py')==dfrozen['configuration']['code_sha256']['semantic_rules_d.py']
    p=ROOT/'research_extensions/semantic_audit/challenge_checks.py';tree=ast.parse(p.read_text(encoding='utf-8'))
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cases' for t in n.targets)]
    cases=ast.literal_eval(assignments[0].value);out=[]
    for name,q,answer,refs,expected in cases:out.append((d.judge(q,answer,refs)['verdict'],expected))
    assert len(cases)==17 and sum(v=='equivalent' and e!='equivalent' for v,e in out)==5
    assert read(ROOT/'public_data/semantic_audit_challenges.json')['do_not_adopt_for_full_dataset']
    # Native output/source preservation. README is the only authorized old edit.
    before=ROOT/'local_backups/release-2026-10-19/preservation-before.json'
    protected=0
    if before.exists():
        changed=[name for name,h in read(before).items() if not (ROOT/name).exists() or sha(ROOT/name)!=h]
        assert changed==['README.md'],changed
        protected=len(read(before))-1
    manifest_path=ROOT/'public_data/RELEASE_MANIFEST.json'
    excluded={'public_data/RELEASE_MANIFEST.json','GITHUB_RELEASE_AUDIT.md'}
    manifest={n:sha(ROOT/n) for n in files if n not in excluded}
    if args.write_manifest:manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    elif manifest_path.exists():assert read(manifest_path)==manifest,'Public release differs from its manifest'
    if args.staged:
        staged=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT)
        names={x.decode('utf-8') for x in staged.split(b'\0') if x}
        assert names==set(files),'Git index does not match the public allowlist'
        for name in names:
            blob=subprocess.check_output(['git','show',':'+name],cwd=ROOT)
            assert blob==(ROOT/name).read_bytes(),(name,'staged bytes differ')
    print(json.dumps({'passed':True,'public_files':len(files),'public_bytes':sum((ROOT/n).stat().st_size for n in files),'markdown_local_links_checked':links,'source_files_unchanged_excluding_README':protected,'pope_metric_rows':len(metrics),'VQA_v2_development_controls':20,'D_postfreeze_false_accepts_reproduced':5,'scope':'public package checks, not fresh model inference or human semantic validation'},ensure_ascii=False))
if __name__=='__main__':main()
