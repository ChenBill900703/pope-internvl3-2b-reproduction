"""Paper section 5.3: compare actual POPE answers with actual caption objects."""
import argparse,collections,csv,json,re,sqlite3,time
from pathlib import Path
from run_pope import ROOT,now,digest
from atomic_io import atomic_json
from chair_eval import make_parser

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def read_rows(folder):
    with sqlite3.connect(f'file:{folder/"answers.sqlite"}?mode=ro',uri=True) as db:
        return [json.loads(x[0]) for x in db.execute('SELECT payload FROM answers ORDER BY split,qid')]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial-audit',action='store_true');args=ap.parse_args()
    captions=Path(read(ROOT/'logs/captions-progress.json')['results'])
    table3=Path(read(ROOT/'logs/progress.json')['results'])
    pope=read_rows(table3)
    for split in ['random','popular','adversarial']:
        assert read(table3/f'{split}-validation.json')['passed']
        assert sum(r['split']==split for r in pope)==3000
    ids={int(r['image'].split('_')[-1].split('.')[0]) for r in pope};assert len(ids)==500
    caps={(r['prompt_id'],r['image_id']):r for r in read_rows(captions) if r['image_id'] in ids}
    available={prompt:sum((prompt,i) in caps for i in ids) for prompt in ['I1','I2']}
    complete=all(n==500 for n in available.values())
    folder=captions/'consistency';folder.mkdir(exist_ok=True)
    if not complete and not args.partial_audit:
        atomic_json(folder/'pending.json',{'at':now(),'complete':False,'expected_images_per_prompt':500,
            'available':available,'no_full_sample_metrics':True});return
    if not complete:folder=folder/f'partial-audit-{time.time_ns()}';folder.mkdir()
    parser,_=make_parser();allpairs=[];summaries=[]
    for prompt in ['I1','I2']:
        for split in ['random','popular','adversarial']:
            pairs=[];seen=set()
            for q in pope:
                if q['split']!=split:continue
                image_id=int(q['image'].split('_')[-1].split('.')[0])
                if (prompt,image_id) not in caps:continue
                caption=caps[prompt,image_id]
                assert caption['image_sha256']==q['image_sha256']
                match=re.fullmatch(r'Is there an? (.+) in the image\?',q['text']);assert match
                obj=match[1];assert obj in parser.inverse_synonym_dict, f'Unmapped original object: {obj}'
                canonical=parser.inverse_synonym_dict[obj]
                key=(image_id,canonical);assert key not in seen, f'Ambiguous duplicate probing object: {key}'
                seen.add(key);count=collections.Counter(caption['chair']['mentioned_objects'])[canonical]
                pairs.append({'caption_prompt_id':prompt,'pope_split':split,'question_id':q['question_id'],
                    'image_id':image_id,'image':q['image'],'object':obj,'canonical_object':canonical,
                    'label':q['label'],'prediction':q['prediction'],'caption_mentioned':count>0,
                    'caption_mention_instances':count,'pope_answer':q['answer'],'caption':caption['answer'],
                    'caption_finish_reason':caption['generation_finish_reason']})
            no=[r for r in pairs if r['prediction']=='no'];mentioned=[r for r in pairs if r['caption_mentioned']]
            no_mentioned=sum(r['caption_mentioned'] for r in no)
            yes_mentioned=sum(r['prediction']=='yes' for r in mentioned)
            summaries.append({'caption_prompt_id':prompt,'pope_split':split,'complete':complete,
                'images':available[prompt],'probed_image_object_pairs':len(pairs),
                'no_pairs':len(no),'no_pairs_mentioned_in_caption':no_mentioned,
                'mentioned_given_no':no_mentioned/len(no) if no else None,
                'caption_mentioned_probed_pairs':len(mentioned),'caption_mentioned_pairs_answered_yes':yes_mentioned,
                'yes_given_caption_mentioned':yes_mentioned/len(mentioned) if mentioned else None,
                'no_caption_mention_instances':sum(r['caption_mention_instances'] for r in no),
                'caption_mentioned_probed_instances':sum(r['caption_mention_instances'] for r in mentioned)})
            allpairs.extend(pairs)
    for name,rows in [('summary',summaries),('all-pairs',allpairs)]:
        atomic_json(folder/f'{name}.json',rows)
        if rows:
            with (folder/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(folder/'protocol.json',{'at':now(),'complete':complete,'source_captions':str(captions),
        'source_table3':str(table3),'script_sha256':digest(__file__),
        'source_config_sha256':{'captions':digest(captions/'config.json'),'table3':digest(table3/'config.json')},
        'caption_object_parser':'Original CHAIR and Pattern; recorded mentioned_objects reused',
        'unit':'Unique (image, canonical object) pairs within each split; repeated caption instances separately reported',
        'scope':'Only objects actually probed in that split; unprobed caption objects are not assumed yes or no',
        'settings_limitation':'Original section5.3 does not state its POPE split or caption prompt explicitly. Report all three splits crossed with I1/I2, not selected favorable conditions.',
        'parsing_limitation':'Original POPE first-sentence negative-token parser retained, including known parsing artifacts',
        'partial_audit_only':not complete})
    print(json.dumps({'complete':complete,'available':available,'pairs':len(allpairs),'folder':str(folder)}))

if __name__=='__main__':main()
