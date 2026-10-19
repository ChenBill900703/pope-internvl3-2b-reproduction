"""Recover official balanced validation questions only, then retain POPE image IDs."""
import hashlib,json,os,re,urllib.request,zipfile,zlib
from pathlib import Path
import prepare_gqa as remote
from run_pope import ROOT,digest,now
from atomic_io import atomic_json

QUESTION_URL='https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip'

class QuestionRanges(remote.OfficialRanges):
    def __init__(self):
        import threading
        remote.URL=QUESTION_URL
        self.position=0;self.network_bytes=0;self.request_count=0;self.lock=threading.Lock();self.segments=[]
        req=urllib.request.Request(QUESTION_URL,headers={'Range':'bytes=-65536','User-Agent':'POPE-reproduction/1.0'})
        with urllib.request.urlopen(req,timeout=90) as response:
            assert response.status==206,'Refuse downloading the whole question archive'
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers['Content-Range']);assert match
            start,end,self.size=map(int,match.groups());self.etag=response.headers['ETag']
            self.modified=response.headers.get('Last-Modified');data=response.read(end-start+2)
            assert len(data)==end-start+1 and len(data)<=65536
        self.network_bytes=len(data);self.request_count=1;self.segments.append((start,data))
        key=hashlib.sha256((QUESTION_URL+self.etag).encode()).hexdigest()[:16]
        self.cache=ROOT/'data/downloads/gqa-question-zip-ranges'/key;self.cache.mkdir(parents=True,exist_ok=True)
        atomic_json(self.cache/'source.json',{'url':QUESTION_URL,'size':self.size,'etag':self.etag,'last_modified':self.modified})
        (self.cache/f'{start}-{end}.bin').write_bytes(data)

def main():
    reader=QuestionRanges();directory=ROOT/'data/gqa';directory.mkdir(exist_ok=True)
    with zipfile.ZipFile(reader) as archive:
        entries=[{'name':i.filename,'compressed_bytes':i.compress_size,'uncompressed_bytes':i.file_size,'crc32':i.CRC}
                 for i in archive.infolist()]
        atomic_json(directory/'official-question-zip-index.json',{'at':now(),'source':QUESTION_URL,
            'archive_bytes':reader.size,'etag':reader.etag,'entries':entries})
        found=[i for i in archive.infolist() if Path(i.filename).name=='val_balanced_questions.json'];assert len(found)==1
        info=found[0];assert info.compress_size<80*1024*1024,'Unexpected large validation member; stop for inspection'
        target=directory/'val_balanced_questions.json'
        if not target.exists():
            temporary=target.with_suffix('.part')
            with archive.open(info) as source,temporary.open('wb') as stream:
                while chunk:=source.read(1024*1024):stream.write(chunk)
                stream.flush();os.fsync(stream.fileno())
            assert temporary.stat().st_size==info.file_size
            os.replace(temporary,target)
        data=target.read_bytes();assert len(data)==info.file_size and zlib.crc32(data)&0xffffffff==info.CRC
    questions=json.loads(data)
    ids={Path(r['image']).stem for r in json.loads((ROOT/'third_party/POPE/output/seem/gqa/gqa_pope_seem_random.json').read_text())}
    selected={qid:row for qid,row in questions.items() if str(row['imageId']) in ids}
    assert selected and all(row['isBalanced'] for row in selected.values())
    atomic_json(directory/'pope-image-vqa-questions.json',selected)
    atomic_json(directory/'vqa-questions-audit.json',{'at':now(),'source':QUESTION_URL,'etag':reader.etag,
        'whole_archive_downloaded':False,'archive_bytes':reader.size,'member':info.filename,'member_crc32':info.CRC,
        'member_sha256':digest(target),'validation_questions':len(questions),'pope_image_questions':len(selected),
        'pope_images':500,'covered_pope_images':len({str(q['imageId']) for q in selected.values()}),
        'network_bytes_this_run':reader.network_bytes,
        'limitation':'Original Table6 VQA sampling IDs and prompt are not published. Official balanced val questions on author POPE images are a transparent extension, not asserted to be the identical Table6 subset.'})
    print(json.dumps({'official_val_questions':len(questions),'on_pope_images':len(selected),
                      'covered_pope_images':len({str(q['imageId']) for q in selected.values()}),
                      'network_bytes_this_run':reader.network_bytes}),flush=True)

if __name__=='__main__':main()
