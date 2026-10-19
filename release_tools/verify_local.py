"""Read-only audit of the private local research sources; publish aggregates only.

Run with the project .venv Python. This never starts a model or rewrites results.
SQLite mode=ro deliberately honors existing committed WAL pages.
"""
import collections,csv,hashlib,importlib.util,json,math,sqlite3,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_pope import metrics,parse_answer
from chair_eval import make_parser,caption_detail,aggregate
OUT=ROOT/'public_data'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def close(a,b):
    assert a.keys()<=b.keys(),(a.keys(),b.keys())
    for k,v in a.items():
        assert math.isclose(v,b[k],abs_tol=1e-12) if isinstance(v,(int,float)) else v==b[k],(k,v,b[k])
def database(folder,n):
    p=ROOT/'results'/folder/'answers.sqlite'
    with sqlite3.connect(p.as_uri()+'?mode=ro',uri=True) as db:
        assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
        rows=[json.loads(x[0]) for x in db.execute('select payload from answers order by split,qid')]
    assert len(rows)==n,(folder,len(rows))
    rp=p.with_name('raw-captions.jsonl' if folder.startswith('captions-') else 'raw-answers.jsonl')
    raw=[json.loads(x) for x in rp.read_text(encoding='utf-8').splitlines()]
    key=lambda r:(r.get('prompt_id',r.get('split')),str(r.get('question_id',r.get('image_id'))))
    assert len({key(r) for r in rows})==n
    assert {key(r):r for r in rows}=={key(r):r for r in raw}
    return rows
def main():
    folders={'table3':'table3-5a3813bb07c0486e','captions':'captions-78966472f82d508d','table4':'table4-22feacf3093f9594','appendixD':'appendixD-d2894d09f402f7a6','vqa':'vqa-raw-c599820ddaf131a9'}
    counts={'table3':9000,'captions':4966,'table4':20000,'appendixD':18000,'vqa':11216}
    allrows={k:database(f,counts[k]) for k,f in folders.items()}
    images={};question_checks=0
    for kind in ['table3','appendixD']:
        folder=ROOT/'results'/folders[kind]
        summaries=read(folder.relative_to(ROOT)/('table3.json' if kind=='table3' else 'table11.json'))
        for m in summaries:
            split=m['split'];rr=[r for r in allrows[kind] if r.get('prompt_id',r.get('split'))==split]
            assert len(rr)==3000
            if kind=='table3':qp=ROOT/f'third_party/POPE/output/coco/coco_pope_{split}.json'
            else:
                dataset,strategy=split.split('_');qp=ROOT/f'third_party/POPE/output/seem/{dataset}/{dataset}_pope_seem_{strategy}.json'
            text=qp.read_text(encoding='utf-8');qs=json.loads(text) if text.lstrip().startswith('[') else [json.loads(l) for l in text.splitlines()]
            byid={q['question_id']:q for q in qs}
            for r in rr:
                assert all(r[k]==byid[r['question_id']][k] for k in ['question_id','image','text','label'])
                assert r['prediction']==parse_answer(r['answer']) and r['answer'].strip()
                base='data/gqa/images' if r.get('dataset')=='gqa' else 'data/coco/val2014'
                ip=ROOT/base/r['image']
                if ip not in images:images[ip]=sha(ip)
                assert images[ip]==r['image_sha256'];question_checks+=1
            close(metrics(rr),m)
            from sklearn.metrics import f1_score,accuracy_score
            assert abs(m['f1']-f1_score([r['label']=='yes' for r in rr],[r['prediction']=='yes' for r in rr]))<1e-12
            assert abs(m['accuracy']-accuracy_score([r['label'] for r in rr],[r['prediction'] for r in rr]))<1e-12
            assert json.loads((folder/f'{split}-answers.json').read_text(encoding='utf-8'))==rr
        save('coco_pope.json' if kind=='table3' else 'aokvqa_gqa_pope.json',summaries)
    parser,gt=make_parser()
    config=read(f'results/{folders["captions"]}/config.json');ids=set(config['table1_image_ids'])
    chair_summaries=read(f'results/{folders["captions"]}/analysis/table1.json')
    for r in allrows['captions']:
        assert caption_detail(parser,gt,r['image_id'],r['answer'])==r['chair']
    for m in chair_summaries:
        rr=[r for r in allrows['captions'] if r['prompt_id']==m['prompt_id'] and r['image_id'] in ids]
        close(aggregate(rr),m)
    save('chair.json',chair_summaries)
    stability=read(f'results/{folders["table4"]}/metrics.json')
    for m in stability:
        rr=[r for r in allrows['table4'] if r['prompt_id']==m['prompt_id']]
        if m['prompt_id'].startswith('P'):
            assert all(r['prediction']==parse_answer(r['answer']) for r in rr);close(metrics(rr),m)
        else:
            for r in rr:assert caption_detail(parser,gt,r['image_id'],r['answer'])==r['chair']
            close(aggregate(rr),m)
    summary=read(f'results/{folders["table4"]}/table4.json')
    close({'POPE_F1_mean':statistics.mean(m['f1'] for m in stability[:4]),'POPE_F1_population_std':statistics.pstdev(m['f1'] for m in stability[:4]),'CHAIRi_mean':statistics.mean(m['CHAIRi'] for m in stability[4:]),'CHAIRi_population_std':statistics.pstdev(m['CHAIRi'] for m in stability[4:])},summary)
    save('prompt_stability.json',stability);save('prompt_stability_summary.json',summary)
    # Independently reconstruct ranks and all three denominator policies.
    freq=collections.Counter();co=collections.defaultdict(collections.Counter)
    for objects in gt.values():
        freq.update(objects)
        for anchor in objects:co[anchor].update(objects-{anchor})
    hr=read(f'results/{folders["captions"]}/analysis/table2-appendixC.json')
    obj=read(f'results/{folders["captions"]}/analysis/all-object-frequencies.json')
    for m in hr:
        rr=[r for r in allrows['captions'] if r['prompt_id']==m['prompt_id'] and r['image_id'] in ids and (not m['anchor'] or m['anchor'] in gt[r['image_id']])]
        f=co[m['anchor']] if m['anchor'] else freq;top=set(sorted(f,key=lambda x:(-f[x],x))[:m['k']])
        hh=[r['chair']['hallucinated_objects'] for r in rr if r['chair']['hallucinated_objects']]
        ratios=[sum(x['object'] in top for x in h)/len(h) for h in hh]
        close({'images':len(rr),'images_with_hallucinations':len(hh),'mean_conditional_nonzero_denominator':statistics.mean(ratios),'mean_all_images_zero_filled':sum(ratios)/len(rr),'pooled_instance_ratio':sum(x['object'] in top for h in hh for x in h)/sum(map(len,hh))},m)
    for m in obj:
        f=co[m['anchor']] if m['anchor'] else freq
        rr=[r for r in allrows['captions'] if r['prompt_id']==m['prompt_id'] and r['image_id'] in ids and (not m['anchor'] or m['anchor'] in gt[r['image_id']])]
        assert sorted(f,key=lambda x:(-f[x],x))[m['rank']-1]==m['object']
        assert m['image_frequency']==f[m['object']]
        assert m['hallucination_count']==sum(h['object']==m['object'] for r in rr for h in r['chair']['hallucinated_objects'])
    save('hra_hrc.json',hr);save('object_frequencies.json',obj)
    for name,src in [('appendixB_cumulative.json','analysis/appendixB-cumulative.json'),('caption_metrics.json','appendixF/metrics.json'),('caption_pope_consistency.json','consistency/summary.json')]:save(name,read(f'results/{folders["captions"]}/{src}'))
    # Recompute all A/B/C aggregates from preserved per-item exports and confirm raw data identity.
    comparisons=[];vqa=allrows['vqa'];leading=0
    rule_path=ROOT/'analysis/vqa_diagnostics_2026-10-19/local_rules.py'
    spec=importlib.util.spec_from_file_location('frozen_vqa_rules',rule_path);rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
    for dataset,fn,n in [('aokvqa','AOKVQA_predictions.csv',515),('gqa','GQA_predictions.csv',10701)]:
        with (ROOT/'analysis/vqa_diagnostics_2026-10-19'/fn).open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
        byid={str(r['question_id']):r for r in vqa if r['dataset']==dataset};assert len(rr)==len(byid)==n
        for r in rr:
            raw=byid[r['question_id']]
            assert r['raw_model_answer']==raw['answer'] and json.loads(r['reference_answer'])==raw['reference_answers']
            assert r['question']==raw['text'] and str(r['image_id'])==str(raw['image_id'])
            assert len(raw['reference_answers'])==(10 if dataset=='aokvqa' else 1)
            assert (r['difficult_direct_answer'].lower()=='true')==(dataset=='aokvqa' and raw['question_metadata']['difficult_direct_answer'])
            parsed=rules.extract(raw['text'],raw['answer'])
            assert parsed['status']==r['parser_status'] and parsed['answer']==r['parsed_answer']
            assert rules.normalize(raw['answer'])==r['normalized_answer']
            assert [rules.normalize(x) for x in raw['reference_answers']]==json.loads(r['normalized_reference_answers'])
            if dataset=='gqa' and r['question_id'].startswith('0'):leading+=1
        eligible=[r for r in rr if r['difficult_direct_answer'].lower()!='true']
        for scheme,field in [('A','A_raw_score'),('B','B_normalized_score'),('C','C_parsed_score')]:
            scores=[]
            for r in eligible:
                candidate=r['raw_model_answer'] if scheme=='A' else r['normalized_answer'] if scheme=='B' else r['parsed_answer']
                refs=json.loads(r['reference_answer'] if scheme=='A' else r['normalized_reference_answers'])
                if scheme=='C' and r['parser_status']!='parsed':v=0.
                else:
                    match=sum(candidate==x for x in refs);v=min(1.,match/3) if dataset=='aokvqa' else float(match>0)
                assert math.isclose(v,float(r[field]),abs_tol=1e-12),(dataset,r['question_id'],scheme)
                scores.append(v)
            comparisons.append({'dataset':dataset,'scheme':scheme,'n_raw':n,'n_eligible':len(eligible),'n_excluded':n-len(eligible),'score_sum':sum(scores),'score_percent':100*sum(scores)/len(eligible),'n_ambiguous':sum(r['parser_status']=='ambiguous' for r in eligible),'n_unparseable':sum(r['parser_status']=='unparseable' for r in eligible)})
    assert leading==5461
    save('vqa_supplementary.json',comparisons)
    save('semantic_audit_challenges.json',read('analysis/vqa_semantic_audit_2026-10-19/POST_FREEZE_CHALLENGE_SUMMARY.json'))
    provenance={}
    for base in folders.values():
        for p in (ROOT/'results'/base).rglob('*'):
            if p.is_file() and not p.name.endswith('-shm'):provenance[p.relative_to(ROOT).as_posix()]=sha(p)
    save('private_source_hashes.json',provenance)
    save('local_validation.json',{'passed':True,'date':'2026-10-19','record_counts':counts,'independent_question_label_image_checks':question_checks,'unique_images_hashed':len(images),'chair_parser_recomputed_records':4966+8000,'hra_hrc_rows_recomputed':len(hr),'object_frequency_rows_recomputed':len(obj),'vqa_raw_export_records_checked':11216,'gqa_leading_zero_ids':leading,'sqlite_read_mode':'ro (WAL honored)','caption_metrics_scope':'Previously saved evaluator aggregates; BLEU/METEOR/ROUGE not rerun by this script','no_gpu_inference':True,'no_source_results_rewritten':True})
    print('Read-only local validation passed; public aggregates saved.',flush=True)
if __name__=='__main__':main()
