"""Record generation quality without changing or selectively shortening model output."""
from chair_eval import aggregate

def annotate(record):
    ids=record['generated_token_ids'];grams=[tuple(ids[i:i+4]) for i in range(max(0,len(ids)-3))]
    ratio=1-len(set(grams))/len(grams) if grams else 0.
    record['generation_finish_reason']='empty' if not record['answer'].strip() else ('length' if record['possible_truncation'] else 'eos')
    record['repetition_4gram_fraction']=ratio
    record['repetitive_generation']=len(ids)>=64 and ratio>.5
    return record

def summary(rows):
    eos=[r for r in rows if r.get('generation_finish_reason','eos')=='eos']
    m=aggregate(eos)
    return {'length_limit_count':sum(r.get('generation_finish_reason')=='length' for r in rows),
            'repetition_flagged_count':sum(r.get('repetitive_generation',False) for r in rows),
            'sensitivity_eos_only_n':len(eos),'sensitivity_eos_only_CHAIRi':m['CHAIRi'],
            'sensitivity_eos_only_CHAIRs':m['CHAIRs']}
