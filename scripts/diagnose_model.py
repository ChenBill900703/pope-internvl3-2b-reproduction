"""Numerical diagnosis; stores invalid outputs without treating them as benchmark results."""
import os,sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['HF_MODULES_CACHE']=str(ROOT/'.cache/huggingface/modules')
os.environ['HF_HUB_OFFLINE']='1'
import torch, torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel,AutoTokenizer
dtype=getattr(torch,sys.argv[1] if len(sys.argv)>1 else 'bfloat16')
torch.set_num_threads(4);torch.manual_seed(0)
model=AutoModel.from_pretrained(str(ROOT/'models/InternVL3-2B'),trust_remote_code=True,
    torch_dtype=dtype,low_cpu_mem_usage=True,use_flash_attn=False,local_files_only=True).eval().cuda()
tok=AutoTokenizer.from_pretrained(str(ROOT/'models/InternVL3-2B'),trust_remote_code=True,use_fast=False,local_files_only=True)
card=(ROOT/'models/InternVL3-2B/README.md').read_text(encoding='utf-8')
start=card.index('def build_transform(');end=card.index('\ndef split_model(',start)
ns=dict(torch=torch,T=T,Image=Image,InterpolationMode=InterpolationMode,IMAGENET_MEAN=(.485,.456,.406),IMAGENET_STD=(.229,.224,.225))
exec(card[start:end],ns)
q=json.loads((ROOT/'third_party/POPE/output/coco/coco_pope_random.json').read_text().splitlines()[0])
record={'dtype':str(dtype),'question':q,'bf16_supported':torch.cuda.is_bf16_supported(),'finite':{}}
def check(name,value):
    record['finite'][name]={'all_finite':bool(torch.isfinite(value).all()),'abs_max':float(value.abs().max())}
model.vision_model.register_forward_hook(lambda m,a,o:check('vision',o.last_hidden_state))
model.mlp1.register_forward_hook(lambda m,a,o:check('projector',o))
model.language_model.register_forward_hook(lambda m,a,o:check('last_logits',o.logits[:,-1,:]))
original=model.language_model.generate
def generated(*a,**kw):
    out=original(*a,**kw);record['generated_ids']=out[0].tolist();return out
model.language_model.generate=generated
pixels=ns['load_image'](str(ROOT/'data/coco/val2014'/q['image']),max_num=12).to(dtype).cuda()
start=time.perf_counter()
with torch.inference_mode():
    record['answer']=model.chat(tok,pixels,'<image>\n'+q['text'],{'do_sample':False,'num_beams':1,'max_new_tokens':32},history=None)
record.update(seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated())
(ROOT/f'logs/diagnostic-{str(dtype).split(".")[-1]}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=True),flush=True)
