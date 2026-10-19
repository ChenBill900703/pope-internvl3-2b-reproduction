"""Use the original CHAIR word parser and Pattern singularize with POPE's published GT."""
import json,re,hashlib
from pathlib import Path
from run_pope import ROOT,digest,atomic_json,now

def make_parser():
    import nltk
    nltk.data.path.insert(0,str(ROOT/'data/nltk'))
    inflect=(ROOT/'third_party/pattern/inflect.py').read_text(encoding='utf-8')
    prep=inflect[inflect.index('plural_prepositions ='):inflect.index('# Inflection rules')]
    singular=inflect[inflect.index('#### SINGULARIZE'):inflect.index('#### VERB CONJUGATION')]
    ns={'re':re,'NOUN':'NN','ADJECTIVE':'JJ'}
    exec(compile(prep+'\n'+singular,'original_pattern_singularize','exec'),ns)
    source=(ROOT/'third_party/CHAIR/utils/chair.py').read_text(encoding='utf-8')
    source=source[source.index('class CHAIR(object):'):source.index('    def get_annotations_from_segments(')]
    source=source.replace("open('data/synonyms.txt')",'open('+repr(str(ROOT/'third_party/CHAIR/data/synonyms.txt'))+", encoding='utf-8')")
    ns.update(nltk=nltk,singularize=ns['singularize'])
    exec(compile(source,'original_CHAIR_parser','exec'),ns)
    parser=ns['CHAIR']([],None)
    gt={}
    for row in map(json.loads,(ROOT/'third_party/POPE/segmentation/coco_ground_truth_segmentation.json').read_text().splitlines()):
        gt[row['image_id']]={parser.inverse_synonym_dict[x] for x in row['objects']}
    return parser,gt

def caption_detail(parser,gt,image_id,text):
    words,nodes,indices,raw=parser.caption_to_words(text)
    hallucinated=[{'word':w,'object':n,'index':i} for w,n,i in zip(words,nodes,indices) if n not in gt[image_id]]
    return {'mentioned_objects':nodes,'hallucinated_objects':hallucinated,'ground_truth_objects':sorted(gt[image_id]),
            'CHAIRs':int(bool(hallucinated)),'CHAIRi':len(hallucinated)/len(nodes) if nodes else 0.,
            'word_length_whitespace':len(text.split()),'word_length_nltk':len(raw)}

def aggregate(rows):
    n=len(rows);mentioned=sum(len(r['chair']['mentioned_objects']) for r in rows)
    hallucinated=sum(len(r['chair']['hallucinated_objects']) for r in rows)
    return {'n':n,'CHAIRi':hallucinated/mentioned if mentioned else None,
            'CHAIRs':sum(r['chair']['CHAIRs'] for r in rows)/n if n else None,
            'mentioned_instances':mentioned,'hallucinated_instances':hallucinated,
            'average_length_whitespace':sum(r['chair']['word_length_whitespace'] for r in rows)/n if n else None,
            'average_length_nltk':sum(r['chair']['word_length_nltk'] for r in rows)/n if n else None}

if __name__=='__main__':
    parser,gt=make_parser()
    examples=[('A baby bird beside a toilet seat and two wine glasses.',{'bird','toilet','wine glass'}),
              ('A passenger jet and a bow tie.',{'airplane','tie'}),
              ('Three people with dogs and a dining table.',{'person','dog','dining table'})]
    for text,expected in examples:
        got=set(parser.caption_to_words(text)[1]);assert got==expected,(text,got,expected)
    atomic_json(ROOT/'logs/chair-parser-validation.json',{'at':now(),'passed':True,
        'source_chair_commit':'6e4d33c4bedf6dfcd24c61bf527f40714717be47',
        'chair_source_sha256':digest(ROOT/'third_party/CHAIR/utils/chair.py'),
        'pattern_source_sha256':digest(ROOT/'third_party/pattern/inflect.py'),
        'synonyms_sha256':digest(ROOT/'third_party/CHAIR/data/synonyms.txt'),
        'gt_images':len(gt),'checks':len(examples),'modifications':['Resolve synonyms file path only',
        'Extract Python3-compatible original CHAIR constructor and word parser unchanged',
        'Use original Pattern singularize and its rules without unrelated Pattern dependencies',
        'Use POPE authors published combined segmentation/caption objects as GT']})
    print('Original CHAIR parser checks passed;',len(gt),'ground-truth images')
