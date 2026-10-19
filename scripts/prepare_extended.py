"""Recover the authors' original caption image IDs; fetch missing images only."""
import concurrent.futures,json,os
from pathlib import Path
from prepare_data import download,ROOT

def read_concatenated(path):
    decoder=json.JSONDecoder();text=Path(path).read_text(encoding='utf-8');i=0;rows=[]
    while i<len(text):
        while i<len(text) and text[i].isspace():i+=1
        if i==len(text):break
        row,i=decoder.raw_decode(text,i);rows.append(row)
    return rows

if __name__=='__main__':
    sets={};details={}
    for path in sorted((ROOT/'third_party/POPE/caption_data').glob('*.json')):
        rows=read_concatenated(path);ids={r['image_id'] for r in rows};assert len(ids)==2000
        sets[path.name]=ids
        details[path.name]={'rows':len(rows),'unique_images':len(ids),
                            'prompts':sorted({json.dumps(r.get('prompt',''),ensure_ascii=False) for r in rows})}
    reference=sets['Instruction1_instructblip.json']
    for name,ids in sets.items():assert ids==reference,(name,len(ids&reference))
    pope={r['image'] for r in map(json.loads,(ROOT/'third_party/POPE/output/coco/coco_pope_random.json').read_text().splitlines())}
    images={f'COCO_val2014_{i:012d}.jpg' for i in reference}|pope
    audit={'caption_source':'Author caption_data, all 10 files have identical 2000 image IDs',
           'caption_image_ids':sorted(reference),'pope_overlap':len({f'COCO_val2014_{i:012d}.jpg' for i in reference}&pope),
           'total_images_needed':len(images),'source_files':details}
    (ROOT/'data/extended-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['source_files','caption_image_ids']}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,item in enumerate(pool.map(download,sorted(images)),1):
            with (ROOT/'data/images-manifest-extended.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(item)+'\n');f.flush();os.fsync(f.fileno())
            if i%100==0:print(f'Images verified {i}/{len(images)}',flush=True)
