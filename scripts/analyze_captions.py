"""CHAIR, Figure 2, HRA/HRC and Appendix B/C from durable caption object records."""
import argparse,collections,csv,json,sqlite3,statistics
from pathlib import Path
from run_pope import ROOT,atomic_json,now,digest
from chair_eval import make_parser,aggregate
from caption_quality import summary as quality_summary

def csv_save(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(folder):
    parser,gt=make_parser()
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'));ids=set(config['table1_image_ids'])
    freq=collections.Counter();co={}
    for objects in gt.values():
        freq.update(objects)
        for anchor in objects:co.setdefault(anchor,collections.Counter()).update(objects-{anchor})
    ranked=sorted(freq,key=lambda o:(-freq[o],o))
    assert len(ranked)==80
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
        allrows=[json.loads(x[0]) for x in db.execute('SELECT payload FROM answers ORDER BY split,qid')]
    summaries=[];hr=[];cumulative=[];object_stats=[];outputs=folder/'analysis';outputs.mkdir(exist_ok=True)
    for prompt in config['prompts']:
        rows=[r for r in allrows if r['prompt_id']==prompt and r['image_id'] in ids]
        assert len(rows)==2000,(prompt,len(rows))
        summaries.append({'prompt_id':prompt,**aggregate(rows),**quality_summary(rows)})
        hcounts=collections.Counter(h['object'] for r in rows for h in r['chair']['hallucinated_objects'])
        for anchor in [None,'dining table','chair','car']:
            subset=rows if anchor is None else [r for r in rows if anchor in gt[r['image_id']]]
            ranks=ranked if anchor is None else sorted(co[anchor],key=lambda o:(-co[anchor][o],o))
            counts=collections.Counter(h['object'] for r in subset for h in r['chair']['hallucinated_objects'])
            denom=sum(counts.values());eligible=[r for r in subset if r['chair']['hallucinated_objects']]
            for k in [10,20,30]:
                top=set(ranks[:k]);ratios=[sum(h['object'] in top for h in r['chair']['hallucinated_objects'])/len(r['chair']['hallucinated_objects']) for r in eligible]
                hr.append({'prompt_id':prompt,'metric':'HRA' if anchor is None else 'HRC','anchor':anchor or '',
                    'k':k,'images':len(subset),'images_with_hallucinations':len(eligible),
                    'mean_conditional_nonzero_denominator':statistics.mean(ratios) if ratios else None,
                    'mean_all_images_zero_filled':sum(ratios)/len(subset) if subset else None,
                    'pooled_instance_ratio':sum(counts[o] for o in top)/denom if denom else None})
            for k in range(10,81,10):
                cumulative.append({'prompt_id':prompt,'anchor':anchor or '', 'top_k':k,
                    'hallucinated_instances':denom,'cumulative_proportion':sum(counts[o] for o in ranks[:k])/denom if denom else None})
            for i,o in enumerate(ranks,1):
                object_stats.append({'prompt_id':prompt,'anchor':anchor or '','rank':i,'object':o,
                    'image_frequency':freq[o] if anchor is None else co[anchor][o],
                    'hallucination_count':counts[o]})
        # Standard plotting tools, one model with separate original prompt conditions.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
        for ax,anchor,title in [(axes[0],None,'Appearance frequency'),(axes[1],'dining table','Co-occurrence with dining table')]:
            values=[r for r in object_stats if r['prompt_id']==prompt and r['anchor']==(anchor or '')][:10]
            ax.bar([r['object'] for r in values],[r['hallucination_count'] for r in values],color='#336b87')
            ax.set_title(f'InternVL3-2B {prompt}: {title} (ranked high to low)')
            ax.set_ylabel('Hallucinated mentions');ax.tick_params(axis='x',rotation=25)
        fig.savefig(outputs/f'figure2-{prompt}.png',dpi=200);fig.savefig(outputs/f'figure2-{prompt}.svg');plt.close(fig)
    for name,rows in [('table1',summaries),('table2-appendixC',hr),('appendixB-cumulative',cumulative),('all-object-frequencies',object_stats)]:
        atomic_json(outputs/f'{name}.json',rows);csv_save(outputs/f'{name}.csv',rows)
    atomic_json(outputs/'analysis-config.json',{'at':now(),'script_sha256':digest(__file__),
        'rank_corpus':'POPE published combined objects for all 40504 COCO val2014 images',
        'rank_ties':'Alphabetical canonical object name','count_unit':'Mention instances, preserving repeated mentions',
        'hr_zero_denominator':'Original paper does not state handling of images with zero hallucinations. Report conditional mean, zero-filled mean and pooled ratio separately; no claim of exact original Table 2 replication.',
        'caption_source':'New actual InternVL3-2B outputs, not author baseline captions',
        'figure_scope':'InternVL3-2B I1/I2 only; original four baseline models not rerun',
        'generation_quality':'All nonempty finite fixed-budget outputs included in primary metrics; length/repetition counts and EOS-only sensitivity scores reported separately. Original text never cleaned or shortened.',
        'table1_complete':True,'images_per_prompt':2000,'prompts':config['prompts']})
    print(json.dumps({'table1':summaries,'analysis':str(outputs)}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--folder',required=True);args=ap.parse_args();analyze(Path(args.folder))
