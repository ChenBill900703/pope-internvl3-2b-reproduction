"""Validate the original published POPE files and fetch only missing COCO images."""
import concurrent.futures, hashlib, json, os, re, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/coco/val2014'
OUT.mkdir(parents=True,exist_ok=True)

def download(name):
    path = OUT/name
    if path.exists() and path.stat().st_size > 0:
        return {'image':name,'source':'existing','bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    url = 'http://images.cocodataset.org/val2014/'+name
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'POPE-reproduction/1.0'})
            with urllib.request.urlopen(req,timeout=90) as r:
                data=r.read()
            if not data.startswith(b'\xff\xd8'):
                raise ValueError('Not a JPEG: '+name)
            partial=path.with_suffix('.part')
            partial.write_bytes(data)
            os.replace(partial,path)
            return {'image':name,'source':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        except Exception:
            if attempt == 3: raise
            time.sleep(2**attempt)

if __name__ == '__main__':
    audit={}; images=set()
    gt={x['image']:set(x['objects']) for x in map(json.loads,(ROOT/'third_party/POPE/segmentation/coco_ground_truth_segmentation.json').read_text().splitlines())}
    for split in ['random','popular','adversarial']:
        p=ROOT/f'third_party/POPE/output/coco/coco_pope_{split}.json'
        rows=[json.loads(l) for l in p.read_text().splitlines()]
        assert len(rows)==3000
        assert len({r['question_id'] for r in rows})==3000
        assert sum(r['label']=='yes' for r in rows)==1500
        assert sum(r['label']=='no' for r in rows)==1500
        assert len({r['image'] for r in rows})==500
        disagreements=[]
        for r in rows:
            match=re.fullmatch(r'Is there an? (.+) in the image\?',r['text'])
            assert match, r
            obj=match[1]
            if (obj in gt[r['image']]) != (r['label']=='yes'):
                disagreements.append(r)
        assert not disagreements, disagreements[:3]
        images.update(r['image'] for r in rows)
        audit[split]={'questions':len(rows),'images':500,'yes':1500,'no':1500,
                      'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'annotation_mismatches':disagreements}
    (ROOT/'data/questions-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit),flush=True)
    manifest=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,item in enumerate(pool.map(download,sorted(images)),1):
            manifest.append(item)
            with (ROOT/'data/images-manifest.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(item)+'\n'); f.flush(); os.fsync(f.fileno())
            if i%25==0: print(f'Images verified: {i}/{len(images)}',flush=True)
    print(f'Completed {len(manifest)} images',flush=True)
