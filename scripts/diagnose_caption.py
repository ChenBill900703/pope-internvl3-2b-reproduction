"""Check whether a capped/repetitive caption is caused by visual feature reuse."""
import json
from pathlib import Path
from run_pope import ROOT,atomic_json,now
from model_runtime import Runtime
p=json.loads((ROOT/'logs/captions-progress.json').read_text(encoding='utf-8'));folder=Path(p['results'])
attempt=json.loads((folder/'inference-attempts.jsonl').read_text(encoding='utf-8').splitlines()[-1])
runtime=Runtime();checks={'calls':0,'all_finite':True}
def inspect(m,a,out):
    checks['calls']+=1
    checks['all_finite']=checks['all_finite'] and bool(runtime.torch.isfinite(out.logits[:,-1,:]).all())
runtime.model.language_model.register_forward_hook(inspect)
direct=runtime.chat(ROOT/'data/coco/val2014'/attempt['image'],attempt['prompt'],max_new_tokens=1024,use_features=False)
result={'at':now(),'source_folder':str(folder),'image':attempt['image'],'prompt':attempt['prompt'],
        'cached_attempt':attempt,'direct_output':direct,'logits':checks,
        'tokens_identical':direct['generated_token_ids']==attempt['generated_token_ids'],
        'answers_identical':direct['answer']==attempt['answer']}
atomic_json(ROOT/'logs/caption-repetition-diagnosis.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['cached_attempt','direct_output']},ensure_ascii=True))
