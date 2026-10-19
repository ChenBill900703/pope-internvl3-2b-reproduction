"""Prepare original-data follow-up experiments without using the GPU."""
import argparse, concurrent.futures, hashlib, json, os, re, shutil, tarfile, urllib.request, zipfile
from pathlib import Path
from run_pope import ROOT, digest, now, load_rows
from atomic_io import atomic_json
from prepare_data import download

COCO_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
COCO_SHA256 = '031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009'
AOK_URL = 'https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz'

def retrieve(url, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected: assert digest(target) == expected, f'Existing checksum mismatch: {target}'
        return
    partial = target.with_suffix(target.suffix + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {'User-Agent': 'POPE-reproduction/1.0'}
    if offset: headers['Range'] = f'bytes={offset}-'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
        append = offset > 0 and response.status == 206
        if append: assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
        with partial.open('ab' if append else 'wb') as stream:
            shutil.copyfileobj(response, stream, length=1024*1024)
            stream.flush(); os.fsync(stream.fileno())
    if expected: assert digest(partial) == expected, f'Download checksum mismatch: {partial}'
    os.replace(partial, target)
    print(f'Asset saved: {target.relative_to(ROOT)} ({target.stat().st_size} bytes)', flush=True)

def find_existing_annotations():
    """One read-only inventory, skip environments and unrelated generated trees."""
    report = ROOT/'logs/followup-asset-search.json'
    if report.exists(): return json.loads(report.read_text(encoding='utf-8'))['matches']
    names = {'captions_val2014.json', 'annotations_trainval2014.zip', 'aokvqa_v1p0.tar.gz', 'aokvqa_v1p0_val.json'}
    excluded = {'.git', '.venv', 'node_modules', '__pycache__', '$recycle.bin', 'system volume information',
                'windows', 'program files', 'program files (x86)', 'appdata', '.cache'}
    hits=[];errors=[]
    for base in [Path('E:/'), Path('D:/'), Path.home()/'Downloads']:
        if not base.exists():continue
        for folder, dirs, files in os.walk(base, onerror=lambda e: errors.append(str(e))):
            dirs[:] = [d for d in dirs if d.lower() not in excluded and not d.lower().startswith('.venv')]
            hits.extend(str(Path(folder)/name) for name in files if name in names)
    atomic_json(report, {'at':now(), 'read_only':True, 'matches':hits, 'scan_errors':errors})
    return hits

def prepare_table4():
    directory = ROOT/'data/table4';directory.mkdir(exist_ok=True)
    templates = {
        'P1': 'Is there a <object> in the image?',
        'P2': 'Does the image contain a <object>?',
        'P3': 'Have you noticed a <object> in the image?',
        'P4': 'Can you see a <object> in the image?',
    }
    captions = {
        'C1':'Generate a short caption of the image.',
        'C2':'Provide a brief description of the image.',
        'C3':'Generate a concise description for the image.',
        'C4':'Create a short textual summary for the image.',
    }
    original = load_rows('random');audit={}
    for prompt, template in templates.items():
        questions=[]
        for row in original:
            match=re.fullmatch(r'Is there an? (.+) in the image\?', row['text']);assert match
            text=row['text'] if prompt=='P1' else template.replace('<object>',match[1])
            questions.append({**row, 'text':text, 'original_text':row['text'], 'object':match[1]})
        path=directory/f'{prompt}.json';atomic_json(path,questions)
        audit[prompt]={'n':len(questions),'sha256':digest(path),'template':template,
                       'article_policy':'P1 exact published question text (a/an); P2-P4 literal a as printed in Table 4'}
    atomic_json(directory/'protocol.json',{'at':now(),'paper_location':'PDF page 8, Table 4',
        'pope':audit,'caption_prompts':captions,
        'caption_image_ids_source':'data/extended-audit.json, original author 2000 IDs',
        'reuse':'P1 from completed Table3 random, C1 from Table1 I1; require image, prompt, model and decoding match',
        'do_not_reuse':'C2 differs from Table1 I2 (given image vs image); must infer C2 separately',
        'standard_deviation':'population; compute from unrounded scores',
        'model_scope':'InternVL3-2B extension; original Table4 model was LLaVA'})
    print('Table4: four original templates prepared; no new images or GPU inference.',flush=True)

def prepare_appendix_d():
    audit={};images=set()
    for path in sorted((ROOT/'third_party/POPE/output/seem').rglob('*.json')):
        rows=json.loads(path.read_text(encoding='utf-8'))
        assert len(rows)==3000 and len({r['question_id'] for r in rows})==3000
        assert len({r['image'] for r in rows})==500
        assert sum(r['label']=='yes' for r in rows)==1500 and sum(r['label']=='no' for r in rows)==1500
        for row in rows: assert re.fullmatch(r'Is there an? (.+) in the image\?',row['text'])
        group={image:[r for r in rows if r['image']==image] for image in {r['image'] for r in rows}}
        assert all(len(rs)==6 and sum(r['label']=='yes' for r in rs)==3 for rs in group.values())
        audit[str(path.relative_to(ROOT))]={'sha256':digest(path),'n':3000,'images':500,'yes':1500,'no':1500}
        if path.parent.name=='aokvqa':images.update(group)
    manifest=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for i,row in enumerate(pool.map(download,sorted(images)),1):
            manifest.append(row)
            if i%100==0:print(f'A-OKVQA original COCO images checked {i}/500',flush=True)
    from PIL import Image
    for row in manifest:
        with Image.open(ROOT/'data/coco/val2014'/row['image']) as image:image.verify()
    atomic_json(ROOT/'data/appendixD-questions-audit.json',{'at':now(),'files':audit,'aokvqa_images':manifest,
        'gqa_images_ready':False,'gqa_note':'Original 500 image IDs verified in questions. Images not substituted with COCO.'})

def prepare_annotations(matches):
    directory=ROOT/'data/coco/annotations';directory.mkdir(parents=True,exist_ok=True)
    target=directory/'captions_val2014.json'
    if not target.exists():
        existing=next((Path(p) for p in matches if Path(p).name==target.name),None)
        if existing:
            shutil.copyfile(existing,target);source=str(existing)
        else:
            archive=next((Path(p) for p in matches if Path(p).name=='annotations_trainval2014.zip'),
                         ROOT/'data/downloads/annotations_trainval2014.zip')
            retrieve(COCO_URL,archive,COCO_SHA256)
            with zipfile.ZipFile(archive) as z:
                with z.open('annotations/captions_val2014.json') as src, target.open('wb') as dst:shutil.copyfileobj(src,dst)
            source=COCO_URL
        atomic_json(directory/'source.json',{'at':now(),'source':source,'sha256':digest(target),
            'archive_expected_sha256':COCO_SHA256,
            'archive_checksum_reference':'https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/coco_captions.txt'})
    annotations=json.loads(target.read_text(encoding='utf-8'));by_image={}
    for row in annotations['annotations']:by_image.setdefault(row['image_id'],[]).append(row['caption'])
    ids={int(row['image'].split('_')[-1].split('.')[0]) for row in load_rows('random')}
    references={str(i):by_image[i] for i in sorted(ids)}
    assert len(references)==500 and all(len(v)>=5 for v in references.values())
    atomic_json(ROOT/'data/coco/pope-500-caption-references.json',references)
    atomic_json(ROOT/'data/coco/caption-references-audit.json',{'at':now(),'source_sha256':digest(target),
        'all_val_images':len(by_image),'all_captions':len(annotations['annotations']),
        'pope_images':500,'references':sum(map(len,references.values())),
        'reference_sha256':digest(ROOT/'data/coco/pope-500-caption-references.json')})
    print('Appendix F: original COCO references verified for all 500 POPE images.',flush=True)

def prepare_aok_annotations(matches):
    directory=ROOT/'data/aokvqa';directory.mkdir(exist_ok=True)
    archive=next((Path(p) for p in matches if Path(p).name=='aokvqa_v1p0.tar.gz'), ROOT/'data/downloads/aokvqa_v1p0.tar.gz')
    target=directory/'aokvqa_v1p0_val.json'
    if not target.exists():
        existing=next((Path(p) for p in matches if Path(p).name==target.name),None)
        if existing:shutil.copyfile(existing,target)
        else:
            retrieve(AOK_URL,archive)
            with tarfile.open(archive) as tar:
                members=[m for m in tar.getmembers() if Path(m.name).name==target.name and m.isfile()];assert len(members)==1
                with tar.extractfile(members[0]) as src, target.open('wb') as dst:shutil.copyfileobj(src,dst)
    rows=json.loads(target.read_text(encoding='utf-8'))
    ids={int(r['image'].split('_')[-1].split('.')[0]) for r in json.loads((ROOT/'third_party/POPE/output/seem/aokvqa/aokvqa_pope_seem_random.json').read_text())}
    subset=[r for r in rows if r['image_id'] in ids]
    atomic_json(directory/'pope-image-vqa-questions.json',subset)
    atomic_json(directory/'audit.json',{'at':now(),'source':AOK_URL,'annotation_sha256':digest(target),
        'validation_questions':len(rows),'questions_on_original_pope_images':len(subset),
        'covered_pope_images':len({r['image_id'] for r in subset}),'pope_images':len(ids),
        'limitation':'Original paper does not publish its selected VQA question IDs or specify whether its VQA subset equals POPE images. This subset is preparation only, not an exact Table6 claim.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--table4-only',action='store_true');args=ap.parse_args()
    prepare_table4()
    if not args.table4_only:
        matches=find_existing_annotations()
        prepare_annotations(matches)
        prepare_aok_annotations(matches)
        prepare_appendix_d()
