"""Fetch one pinned official snapshot, resume partial weight downloads."""
import hashlib, json, os, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REVISION='899155015275a9b7338c7f4677e19c784e0e5a21'
DEST=ROOT/'models/InternVL3-2B'
DEST.mkdir(parents=True,exist_ok=True)
meta=json.load(urllib.request.urlopen('https://huggingface.co/api/models/OpenGVLab/InternVL3-2B/revision/'+REVISION+'?blobs=true'))
(DEST/'source.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
for item in meta['siblings']:
    name=item['rfilename']
    if name.startswith('examples/') or name.startswith('.'): continue
    p=DEST/name
    expected=item.get('size')
    if p.exists() and (expected is None or p.stat().st_size==expected):
        print('Existing:',name,flush=True); continue
    url=f'https://huggingface.co/OpenGVLab/InternVL3-2B/resolve/{REVISION}/{name}'
    tmp=p.with_name(p.name+'.part')
    for attempt in range(5):
        try:
            offset=tmp.stat().st_size if tmp.exists() else 0
            req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(req,timeout=120) as response:
                append=offset>0 and response.status==206
                if append: assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                with tmp.open('ab' if append else 'wb') as out:
                    while chunk:=response.read(8*1024*1024): out.write(chunk)
                    out.flush(); os.fsync(out.fileno())
            if expected is not None: assert tmp.stat().st_size==expected
            if 'lfs' in item:
                h=hashlib.sha256()
                with tmp.open('rb') as f:
                    while chunk:=f.read(8*1024*1024): h.update(chunk)
                assert h.hexdigest()==item['lfs']['sha256']
            os.replace(tmp,p)
            print('Downloaded:',name,p.stat().st_size,flush=True)
            break
        except Exception as e:
            print('Retry:',name,repr(e),flush=True)
            if attempt==4: raise
            time.sleep(2**attempt)
print('Snapshot ready:',REVISION,flush=True)
