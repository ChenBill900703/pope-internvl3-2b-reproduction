"""CPU-only local VQA diagnostics, including portable CSV-based reproduction.
Usage: python run_analysis.py --project ROOT  (first export)
       python run_analysis.py --csv-dir . --output rerun (offline ZIP rerun)
No torch, model, network, paid API or ground-truth-dependent extraction.
"""
import argparse,ast,collections,csv,datetime,hashlib,inspect,json,math,platform,random,re,sqlite3,statistics
from pathlib import Path
import local_rules as rules
HERE=Path(__file__).resolve().parent
SEED=20261019
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def csvwrite(p,rows,fields=None):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),quoting=csv.QUOTE_ALL);w.writeheader();w.writerows(rows)
def official_function():
    path=HERE/'sources/aokvqa_eval_predictions.py'
    assert sha(path)==read(HERE/'sources/aokvqa-source.json')['source_sha256']
    tree=ast.parse(path.read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eval_aokvqa']
    ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns['eval_aokvqa']
def load(project,csv_dir):
    if project:
        path=project/'results/vqa-raw-c599820ddaf131a9/answers.sqlite'
        with sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True) as db:
            rows=[json.loads(x[0]) for x in db.execute('SELECT payload FROM answers ORDER BY split,qid')]
        return [{'dataset':r['dataset'],'question_id':r['question_id'],'image_id':str(r['image_id']),'question':r['text'],
          'refs':r['reference_answers'],'raw':r['answer'],'excluded':r['dataset']=='aokvqa' and r['question_metadata']['difficult_direct_answer'],
          'image_sha256':r['image_sha256'],'types':r['question_metadata'].get('types',{}),'generated_token_count':len(r['generated_token_ids'])} for r in rows]
    rows=[]
    for name in ['AOKVQA_predictions.csv','GQA_predictions.csv']:
        with (csv_dir/name).open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                rows.append({'dataset':r['dataset'],'question_id':r['question_id'],'image_id':r['image_id'],'question':r['question'],
                  'refs':json.loads(r['reference_answer']),'raw':r['raw_model_answer'],'excluded':r['difficult_direct_answer']=='true',
                  'image_sha256':r['image_sha256'],'types':json.loads(r['official_question_types']),'generated_token_count':int(r['generated_token_count'])})
    return rows
def freeze():
    assert rules.self_test()==20
    # Public function signature enforces the reference-blind extraction boundary.
    assert list(inspect.signature(rules.extract).parameters)==['question','raw_prediction']
    values={name:sha(HERE/name) for name in ['local_rules.py','run_analysis.py','sources/aokvqa_eval_predictions.py']}
    path=HERE/'FROZEN_GRADING_RULES_v2.json'
    if path.exists():assert read(path)['code_sha256']==values,'Rules changed after freezing: version before regrading'
    else:
        save(path,{'version':'2.0','frozen_at_before_final_scoring':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'code_sha256':values,'sample_seed':SEED,'sample_target_per_dataset':120,'synthetic_tests_passed':20,
          'extraction_inputs':['question','raw_prediction'],'reference_answers_accessible_to_extractor':False,
          'A':'Unmodified raw strings; AOK consensus min(1,n_exact/3), difficult=True excluded; GQA exact match.',
          'B':'NFKC, lowercase, whitespace, VQA-inspired punctuation/articles/zero-ten digit mapping and six contraction spellings. Entire prediction AND each reference independently normalized; no extraction, synonyms, singularization, fuzzy or substring matching.',
          'C':'Same normalization; deterministic anchored first-sentence syntax/explicit yes-no/number rules; explicit uncertainty, alternatives and recognized contradictions abstain; candidates above 8 words abstain; all eligible records remain denominator. No reference lookup.',
          'official_vs_supplementary':'A uses exact original scoring. B/C preserve dataset score aggregation but transform answers and references; these are supplementary local diagnostics, not official leaderboard scores or ChatGPT-assisted grading.',
          'format_categories':'Overlapping A-I operational text flags, not manual semantic labels. F is an object/person phrase proxy, not proof of object identification.',
          'format_error_definition':'Eligible A=0 and C>0 (rule-recoverable mismatch proxy); not all verbosity and not independently validated semantic accuracy.',
          'uncertainty':'Parsed is a syntax outcome, not proof of correctness. Unrecognized semantic contradictions and wrong target extraction can remain. Manual review required before semantic accuracy claims.',
          'no_paid_api':True,'no_gpu_inference':True})
    return read(path)
def grade_one(d,p,refs,excluded):
    if excluded:return None,0
    matches=sum(p==x for x in refs) if p is not None else 0
    return (min(1.,matches/3.) if d=='aokvqa' else float(matches>0)),matches
def error_topic(row,kind):
    q=row['question'].lower();refs=[rules.normalize(x) for x in row['refs']]
    if kind=='color' or (kind=='choice' and all(set(x.split())<=rules.COLORS|{'dark','light','and'} for x in refs)):return 'color_candidate_mismatch'
    if kind=='number':return 'quantity_candidate_mismatch'
    if kind=='spatial' or re.search(r'\b(left|right|behind|under|above|below|in front of|next to)\b',q):return 'spatial_candidate_mismatch'
    if kind=='reasoning':return 'commonsense_reasoning_candidate_mismatch'
    if kind in ['object_attribute','person'] or row['types'].get('semantic') in ['obj','cat']:return 'object_or_attribute_candidate_mismatch'
    return 'other_candidate_mismatch'
def analyze(inputs,out):
    frozen=freeze();official=official_function();records=[];diagnostics=[]
    for row in inputs:
        d=row['dataset'];raw=row['raw'];refs=row['refs'];excluded=row['excluded']
        parsed=rules.extract(row['question'],raw) # no label/metadata access here
        norm=rules.normalize(raw);nrefs=[rules.normalize(x) for x in refs]
        av,am=grade_one(d,raw,refs,excluded);bv,bm=grade_one(d,norm,nrefs,excluded)
        cv,cm=grade_one(d,parsed['answer'] if parsed['status']=='parsed' else None,nrefs,excluded)
        flags=rules.format_flags(row['question'],raw,parsed)
        if excluded:category='excluded_official_difficult';reason='Official difficult_direct_answer=True; retained, not scored.'
        elif parsed['status']!='parsed':category=parsed['status'];reason=parsed['rule']+'; semantic correctness not established.'
        elif cv>av:category='format_recovered_by_local_rule';reason='Reference-blind syntax extraction earns exact normalized consensus credit; automated preliminary review only.'
        else:category=error_topic(row,parsed['question_kind']);reason='Extracted candidate does not match reference(s); real error vs paraphrase/parser/annotation limitation remains uncertain without review.'
        r={'dataset':d,'question_id':row['question_id'],'image_id':row['image_id'],'question':row['question'],
          'reference_answer':json.dumps(refs,ensure_ascii=False),'raw_model_answer':raw,'normalized_answer':norm,
          'parsed_answer':parsed['answer'],'evaluation_result':json.dumps({'A':av,'B':bv,'C':cv}),
          'evaluation_status':'excluded_official_difficult' if excluded else parsed['status'],
          'difficult_direct_answer':str(excluded).lower(),'reference_count':len(refs),
          'reference_answer_counts':json.dumps(dict(collections.Counter(refs)),ensure_ascii=False),
          'normalized_reference_answers':json.dumps(nrefs,ensure_ascii=False),'A_raw_score':av,'B_normalized_score':bv,'C_parsed_score':cv,
          'A_matched_references':am,'B_matched_references':bm,'C_matched_references':cm,
          'parser_status':parsed['status'],'parser_rule':parsed['rule'],'question_kind':parsed['question_kind'],
          'answer_format_flags':json.dumps(flags,ensure_ascii=False),'error_category':category,'reason':reason,
          'official_question_types':json.dumps(row['types']),'image_sha256':row['image_sha256'],'generated_token_count':row['generated_token_count'],
          'review_status':'automated_preliminary_not_human'}
        records.append(r)
        diagnostics.append({'dataset':d,'question_id':r['question_id'],'question':r['question'],'reference_answer':r['reference_answer'],
          'raw_prediction':raw,'extracted_answer':r['parsed_answer'],'exact_match_result':av,'normalized_result':bv,'parsed_result':cv,
          'error_category':category,'reason':reason,'image_id':r['image_id'],'parser_status':parsed['status'],
          'question_kind':parsed['question_kind'],'review_status':'automated_preliminary_not_human'})
    comparison=[];formats=[];category_counts=[]
    for d,name in [('aokvqa','AOKVQA_predictions.csv'),('gqa','GQA_predictions.csv')]:
        rr=[r for r in records if r['dataset']==d];eligible=[r for r in rr if r['difficult_direct_answer']=='false'];n=len(eligible)
        csvwrite(out/name,rr)
        # Original official AOK function cross-checks *each* complete scheme.
        for method,col in [('A','A_raw_score'),('B','B_normalized_score'),('C','C_parsed_score')]:
            value=math.fsum(r[col] for r in eligible)/n
            if d=='aokvqa':
                original_rows=[];preds={}
                for r in rr:
                    original_rows.append({'question_id':r['question_id'],'difficult_direct_answer':r['difficult_direct_answer']=='true','choices':[],
                      'direct_answers':json.loads(r['reference_answer'] if method=='A' else r['normalized_reference_answers'])})
                    preds[r['question_id']]=r[{'A':'raw_model_answer','B':'normalized_answer','C':'parsed_answer'}[method]] if method!='C' or r['parser_status']=='parsed' else '__ABSTAIN_NO_MATCH__'
                assert abs(official(original_rows,preds,multiple_choice=False,strict=True)/100.-value)<1e-12
            comparison.append({'dataset':d,'scheme':method,'score_0_1':value,'score_percent':100*value,'n_raw':len(rr),'n_eligible':n,'n_excluded':len(rr)-n,
              'score_sum':math.fsum(r[col] for r in eligible),'n_positive_credit':sum(r[col]>0 for r in eligible),
              'n_full_credit':sum(r[col]==1 for r in eligible),'n_ambiguous':sum(r['parser_status']=='ambiguous' for r in eligible),
              'n_unparseable':sum(r['parser_status']=='unparseable' for r in eligible),'n_parsed':sum(r['parser_status']=='parsed' for r in eligible),
              'format_recovered_count':sum(r['A_raw_score']==0 and r['C_parsed_score']>0 for r in eligible),
              'format_recovered_percent':100*sum(r['A_raw_score']==0 and r['C_parsed_score']>0 for r in eligible)/n,
              'denominator_policy':'all_eligible_including_abstentions','official_aggregation_crosschecked':True,
              'grading_status':'raw_exact_baseline' if method=='A' else 'supplementary_local_rules_not_ChatGPT'})
        for flag in ['A_single_word','B_sentence','C_yes_no','D_numeric','E_color','F_object_name_candidate','G_long_explanatory','H_uncertain_refusal','I_abnormal','template_residue','multiple_sentences']:
            total=sum(json.loads(r['answer_format_flags'])[flag] for r in rr);eg=sum(json.loads(r['answer_format_flags'])[flag] for r in eligible)
            affected=[r for r in eligible if json.loads(r['answer_format_flags'])[flag]]
            recovered=sum(r['C_parsed_score']>r['A_raw_score'] for r in affected)
            formats.append({'dataset':d,'category':flag,'count_all':total,'denominator_all':len(rr),'percent_all':100*total/len(rr),
              'count_eligible':eg,'denominator_eligible':n,'percent_eligible':100*eg/n,'format_recovered_in_category':recovered,
              'categories_overlap':True,'semantic_label_is_proxy':True})
        for cat,count in collections.Counter(r['error_category'] for r in rr).items():category_counts.append({'dataset':d,'category':cat,'count':count,'percent_of_all':100*count/len(rr),'automated_preliminary_only':True})
    csvwrite(out/'VQA_GRADING_COMPARISON.csv',comparison);save(out/'VQA_GRADING_COMPARISON.json',comparison)
    csvwrite(out/'VQA_ERROR_ANALYSIS.csv',diagnostics);csvwrite(out/'VQA_ANSWER_FORMATS.csv',formats);csvwrite(out/'VQA_ERROR_CATEGORY_COUNTS.csv',category_counts)
    save(out/'VQA_ANSWER_FORMATS.json',formats)
    # Stratified random selection, 120 per dataset. At least 1 per nonempty
    # stratum; remaining seats allocated proportional to residual population.
    strata=collections.defaultdict(list)
    for r in diagnostics:
        status='excluded' if r['exact_match_result'] is None else (r['parser_status'] if r['parser_status']!='parsed' else ('credit' if r['parsed_result']>0 else 'mismatch'))
        strata[(r['dataset'],status,r['question_kind'])].append(r)
    sample=[];design=[];rng=random.Random(SEED)
    for d in ['aokvqa','gqa']:
        kk=sorted(k for k in strata if k[0]==d);assert len(kk)<=120
        allocation={k:1 for k in kk}
        for _ in range(120-len(kk)):
            available=[k for k in kk if allocation[k]<len(strata[k])]
            # Largest shortfall from proportional target, deterministic tie.
            k=max(available,key=lambda k:(120*len(strata[k])/sum(len(strata[x]) for x in kk)-allocation[k],k))
            allocation[k]+=1
        for k in kk:
            population=sorted(strata[k],key=lambda r:r['question_id']);chosen=rng.sample(population,allocation[k]);label='|'.join(k)
            design.append({'stratum':label,'population_n':len(population),'sample_n':len(chosen),'inclusion_probability':len(chosen)/len(population)})
            for r in chosen:sample.append(dict(r,seed=SEED,stratum=label,stratum_population_n=len(population),stratum_sample_n=len(chosen),inclusion_probability=len(chosen)/len(population),human_judgment='',human_error_category='',human_notes=''))
    assert len(sample)==240 and len({(r['dataset'],r['question_id']) for r in sample})==240
    csvwrite(out/'VQA_MANUAL_REVIEW_SAMPLE.csv',sorted(sample,key=lambda r:(r['dataset'],r['stratum'],r['question_id'])))
    save(out/'MANUAL_SAMPLE_DESIGN.json',{'seed':SEED,'total':240,'per_dataset':120,'strata':design,'no_human_annotations':True,
      'warning':'Oversamples rare strata. Do not interpret unweighted sample accuracy as population semantic accuracy. No true human review completed.'})
    summary={'counts':dict(collections.Counter(r['dataset'] for r in records)),'comparisons':comparison,'all_n':len(records),
      'rules_sha256':sha(HERE/'local_rules.py'),'rules_frozen_sha256':sha(HERE/'FROZEN_GRADING_RULES_v2.json'),
      'human_annotations':0,'manual_sample_n':len(sample),'python':platform.python_version(),
      'parser_failure_by_rule':{d:dict(collections.Counter(r['parser_rule'] for r in records if r['dataset']==d and r['parser_status']!='parsed' and r['difficult_direct_answer']=='false')) for d in ['aokvqa','gqa']}}
    save(out/'ANALYSIS_SUMMARY.json',summary)
    (out/'STATUS.md').write_text('VQA CPU 診斷：全量評分與分層抽樣已完成。尚待獨立人工核對；不宣稱語意準確率。原始資料未更動。\n',encoding='utf-8')
    print(json.dumps({'completed':True,'comparisons':comparison,'manual_sample':len(sample)},ensure_ascii=False))
    return records
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path);ap.add_argument('--csv-dir',type=Path,default=HERE);ap.add_argument('--output',type=Path,default=HERE);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);analyze(load(a.project.resolve() if a.project else None,a.csv_dir),a.output)
