"""Pinned InternVL runtime with official preprocessing and optional feature reuse."""
import os,time
from pathlib import Path
from run_pope import ROOT,digest

class Runtime:
    def __init__(self):
        import torch,torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from PIL import Image
        from transformers import AutoModel,AutoTokenizer
        self.torch=torch;torch.set_num_threads(4);torch.manual_seed(0)
        self.model=AutoModel.from_pretrained(str(ROOT/'models/InternVL3-2B'),torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,use_flash_attn=False,trust_remote_code=True,local_files_only=True).eval().cuda()
        self.tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/InternVL3-2B'),trust_remote_code=True,
                                                    use_fast=False,local_files_only=True)
        card=(ROOT/'models/InternVL3-2B/README.md').read_text(encoding='utf-8')
        a=card.index('def build_transform(');b=card.index('\ndef split_model(',a)
        self.preprocess_source=card[a:b]
        ns=dict(torch=torch,T=T,Image=Image,InterpolationMode=InterpolationMode,
                IMAGENET_MEAN=(.485,.456,.406),IMAGENET_STD=(.229,.224,.225))
        exec(compile(self.preprocess_source,'official_preprocess','exec'),ns)
        self.load_image=ns['load_image'];self.cached_image=None;self.pixels=None;self.features=None
        self.tokens=[];original=self.model.language_model.generate
        def capture(*a,**kw):
            out=original(*a,**kw);self.tokens=out[0].tolist();return out
        self.model.language_model.generate=capture
        def finite(m,a,out):
            if not torch.isfinite(out.logits[:,-1,:]).all():raise FloatingPointError('Non-finite logits')
        self.model.language_model.register_forward_hook(finite)

    def image(self,path):
        if self.cached_image!=str(path):
            self.pixels=None;self.features=None
            self.pixels=self.load_image(str(path),max_num=12).to(self.torch.bfloat16).cuda()
            with self.torch.inference_mode():self.features=self.model.extract_feature(self.pixels)
            self.cached_image=str(path)

    def chat(self,path,question,max_new_tokens=1024,use_features=True):
        self.image(path);self.torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        generation={'do_sample':False,'num_beams':1,'max_new_tokens':max_new_tokens}
        if use_features:generation['visual_features']=self.features
        with self.torch.inference_mode():
            answer=self.model.chat(self.tokenizer,self.pixels,'<image>\n'+question,generation,history=None)
        self.torch.cuda.synchronize()
        return {'answer':answer,'generated_token_ids':list(self.tokens),'seconds_generation':time.perf_counter()-begin,
                'tiles':len(self.pixels),'peak_gpu_bytes':self.torch.cuda.max_memory_allocated(),
                'possible_truncation':len(self.tokens)>=max_new_tokens and self.tokens[-1]!=self.tokenizer.eos_token_id}
