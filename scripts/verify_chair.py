"""Cross-check actual saved captions using the original CHAIR compute_chair method."""
import json,sqlite3,textwrap
from functools import partial
from pathlib import Path
from chair_eval import make_parser,aggregate
from run_pope import ROOT,atomic_json,now,digest

p=json.loads((ROOT/'logs/captions-progress.json').read_text(encoding='utf-8'));folder=Path(p['results'])
source=(ROOT/'third_party/CHAIR/utils/chair.py').read_text(encoding='utf-8')
compute=textwrap.dedent(source[source.index('    def compute_chair('):source.index('def load_generated_captions(')])
loader=source[source.index('def load_generated_captions('):source.index('def save_hallucinated_words(')]
with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
    rows=[json.loads(x[0]) for x in db.execute('SELECT payload FROM answers ORDER BY split,qid')]
checks=[]
for prompt in ['I1','I2']:
    selected=[r for r in rows if r['prompt_id']==prompt]
    if not selected:continue
    parser,gt=make_parser();ns=type(parser).caption_to_words.__globals__;ns['json']=json
    # Original loader uses open() without encoding. Keep its body unchanged,
    # explicitly decoding our UTF-8 JSON on Windows rather than the cp950 locale.
    ns['open']=partial(open,encoding='utf-8')
    exec(compile(loader+'\n'+compute,'original_chair_compute','exec'),ns)
    type(parser).compute_chair=ns['compute_chair']
    parser.imid_to_objects={r['image_id']:gt[r['image_id']] for r in selected}
    keys=['Bleu_1','Bleu_2','Bleu_3','Bleu_4','METEOR','CIDEr','SPICE','ROUGE_L']
    caps={'overall':{k:None for k in keys},'imgToEval':{str(r['image_id']):{'image_id':r['image_id'],'caption':r['answer'],**{k:None for k in keys}} for r in selected}}
    path=folder/f'{prompt}-official-chair-input.json';atomic_json(path,caps)
    official=parser.compute_chair(str(path));expected=aggregate(selected)
    for key in ['CHAIRi','CHAIRs']:assert abs(official['overall_metrics'][key]-expected[key])<1e-12
    for actual,r in zip(official['sentences'],selected):
        assert actual['mscoco_generated_words']==r['chair']['mentioned_objects']
        assert [x[1] for x in actual['mscoco_hallucinated_words']]==[h['object'] for h in r['chair']['hallucinated_objects']]
    # None placeholders above are not captioning scores and are never reported as results.
    atomic_json(folder/f'{prompt}-official-chair-output.json',official)
    checks.append({'prompt_id':prompt,'captions_checked':len(selected),'official_CHAIRi':expected['CHAIRi'],
                   'official_CHAIRs':expected['CHAIRs'],'all_mentioned_and_hallucinated_instances_identical':True})
atomic_json(folder/'official-chair-validation.json',{'at':now(),'passed':True,'script_sha256':digest(__file__),
            'original_compute_chair_used':True,'ground_truth':'POPE authors published combined objects','checks':checks})
print(json.dumps(checks))
