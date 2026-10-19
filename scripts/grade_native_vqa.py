"""Free local raw-answer diagnostics; never impersonate ChatGPT-assisted grading."""
import argparse,ast,csv,json,sqlite3
from pathlib import Path
from run_pope import ROOT,digest,now
from atomic_io import atomic_json

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def original_aok_function():
    folder=ROOT/'third_party/A-OKVQA-evaluation';metadata=read(folder/'source.json')
    path=folder/'eval_predictions.py';assert digest(path)==metadata['source_sha256']
    tree=ast.parse(path.read_text(encoding='utf-8'))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='eval_aokvqa']
    assert len(functions)==1
    module=ast.Module(body=functions,type_ignores=[]);namespace={}
    exec(compile(module,str(path),'exec'),namespace)
    return namespace['eval_aokvqa'],metadata

def score(dataset,rows):
    per_question=[]
    for row in rows:
        if dataset=='aokvqa':
            eligible=not row['question_metadata']['difficult_direct_answer']
            matches=sum(row['answer']==answer for answer in row['reference_answers'])
            value=min(1.,matches/3.) if eligible else None
        else:
            eligible=True;matches=int(row['answer']==row['reference_answers'][0]);value=float(matches)
        per_question.append({'dataset':dataset,'question_id':row['question_id'],'image':row['image'],
            'eligible':eligible,'raw_answer':row['answer'],
            'reference_answers_json':json.dumps(row['reference_answers'],ensure_ascii=False),
            'matched_references':matches,'score':value})
    scored=[r for r in per_question if r['eligible']];assert scored
    average=sum(r['score'] for r in scored)/len(scored)
    if dataset=='aokvqa':
        evaluate,_=original_aok_function()
        original=evaluate([r['question_metadata'] for r in rows],{r['question_id']:r['answer'] for r in rows},
                          multiple_choice=False,strict=True)/100.
        assert abs(average-original)<1e-12
    return {'dataset':dataset,'n_total':len(rows),'n_scored':len(scored),'n_excluded':len(rows)-len(scored),
        'score':average,'method':'Official A-OKVQA direct-answer function on raw exact strings' if dataset=='aokvqa' else 'GQA raw exact-match accuracy per official definition',
        'chatgpt_assisted':False,'paper_table6_identical_sampling':False},per_question

def self_check():
    rows=[]
    for qid,answer,references,difficult in [('q1','red',['red']*10,False),
        ('q2','blue',['blue']*2+['red']*8,False),('q3','green',['green']*10,True)]:
        metadata={'question_id':qid,'choices':['red','blue','green'],'correct_choice_idx':0,
                  'direct_answers':references,'difficult_direct_answer':difficult}
        rows.append({'question_id':qid,'image':'synthetic-only','answer':answer,'reference_answers':references,
                     'question_metadata':metadata})
    aok,_=score('aokvqa',rows)
    assert aok['n_scored']==2 and aok['n_excluded']==1 and abs(aok['score']-5/6)<1e-12
    gqa,_=score('gqa',[{'question_id':'001','image':'synthetic-only','answer':'red','reference_answers':['red']},
                       {'question_id':'002','image':'synthetic-only','answer':'Yes.','reference_answers':['yes']}])
    assert gqa['score']==.5
    atomic_json(ROOT/'logs/native-vqa-grading-validation.json',{'at':now(),'passed':True,
        'synthetic_diagnostics_only':True,'not_model_results':True,'aok':aok,'gqa':gqa,
        'checks':['Official function equals per-question scores','Difficult-direct-answer exclusion',
                  'Partial consensus credit','GQA raw exact match, without normalization'],
        'script_sha256':digest(__file__)})
    print('Native grader diagnostic checks passed; no model results or external service calls.')

def evaluate(folder):
    folder=Path(folder);validation=folder/'raw-validation.json'
    if not validation.exists() or not read(validation)['passed'] or not read(validation)['raw_complete']:
        print('Native VQA grading pending: complete verified raw answers required.');return None
    config=read(folder/'config.json');assert not config['examiner_called']
    out=folder/'native-evaluation';out.mkdir(exist_ok=True)
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
        rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM answers ORDER BY split,qid')]
    assert {d:sum(r['dataset']==d for r in rows) for d in config['expected_counts']}==config['expected_counts']
    metrics=[]
    for dataset in ['aokvqa','gqa']:
        selected=[r for r in rows if r['dataset']==dataset]
        summary,per_question=score(dataset,selected);metrics.append(summary)
        atomic_json(out/f'{dataset}-per-question.json',per_question)
        with (out/f'{dataset}-per-question.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(per_question[0]));writer.writeheader();writer.writerows(per_question)
    atomic_json(out/'metrics.json',metrics)
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    _,aok_source=original_aok_function();gqa_source=read(ROOT/'third_party/GQA-evaluation/source.json')
    assert digest(ROOT/'third_party/GQA-evaluation/evaluation-source.html')==gqa_source['source_sha256']
    atomic_json(out/'protocol.json',{'at':now(),'complete':True,'raw_config_sha256':digest(folder/'config.json'),
        'raw_jsonl_sha256':digest(folder/'raw-answers.jsonl'),'raw_validation_sha256':digest(validation),
        'grader_sha256':digest(__file__),'aok_official_source':aok_source,'gqa_official_definition':gqa_source,
        'score_scale':'0-1 in JSON/CSV; multiply by100 for percentage reports',
        'normalization':'None: raw generated strings exactly retained, no trimming, first-word extraction, lowercasing or semantic mapping',
        'scope':'Local diagnostics on disclosed official-data subsets. Not original paper ChatGPT-assisted grading.',
        'limitation':'Open-ended correct prose may fail raw string matching; this is not necessarily a visual reasoning failure.',
        'external_examiner_called':False,'paid_service_called':False})
    (out/'runner-executed.py').write_bytes(Path(__file__).read_bytes())
    (out/'aok-original-eval_predictions.py').write_bytes((ROOT/'third_party/A-OKVQA-evaluation/eval_predictions.py').read_bytes())
    print(json.dumps({'native_only':True,'metrics':metrics}));return metrics

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--self-check',action='store_true');ap.add_argument('--folder');args=ap.parse_args()
    if args.self_check:self_check()
    else:
        folder=Path(args.folder) if args.folder else Path(read(ROOT/'logs/vqa-progress.json')['results'])
        evaluate(folder)
