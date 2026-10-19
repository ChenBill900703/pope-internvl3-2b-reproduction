"""Retrieve only POPE's original 500 GQA images from the official ZIP ranges."""
import concurrent.futures, hashlib, io, json, os, re, struct, threading, time, urllib.request, zipfile, zlib
from pathlib import Path
from PIL import Image
from run_pope import ROOT,digest,now
from atomic_io import atomic_json

URL='https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip'

class OfficialRanges(io.RawIOBase):
    def __init__(self):
        self.position=0;self.network_bytes=0;self.request_count=0;self.lock=threading.Lock();self.segments=[]
        index=ROOT/'data/gqa/original-500-zip-index.json'
        if index.exists():
            previous=json.loads(index.read_text(encoding='utf-8'));assert previous['source']==URL
            self.size=previous['archive_bytes'];self.etag=previous['etag']
            key=hashlib.sha256((URL+self.etag).encode()).hexdigest()[:16]
            self.cache=ROOT/'data/downloads/gqa-zip-ranges'/key
            source=json.loads((self.cache/'source.json').read_text(encoding='utf-8'))
            self.modified=source['last_modified']
            return
        req=urllib.request.Request(URL,headers={'Range':'bytes=-65536','User-Agent':'POPE-reproduction/1.0'})
        with urllib.request.urlopen(req,timeout=60) as response:
            assert response.status==206,'Official server did not honor Range; refuse whole-archive download'
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers['Content-Range']);assert match
            first,last,self.size=map(int,match.groups());self.etag=response.headers['ETag']
            self.modified=response.headers.get('Last-Modified');body=response.read(last-first+2)
            assert len(body)==last-first+1 and len(body)<=65536
        self.network_bytes=len(body);self.request_count=1;self.segments.append((first,body))
        key=hashlib.sha256((URL+self.etag).encode()).hexdigest()[:16]
        self.cache=ROOT/'data/downloads/gqa-zip-ranges'/key;self.cache.mkdir(parents=True,exist_ok=True)
        atomic_json(self.cache/'source.json',{'url':URL,'size':self.size,'etag':self.etag,'last_modified':self.modified})
        (self.cache/f'{first}-{last}.bin').write_bytes(body)

    def fetch(self,start,size):
        assert 0<=start<=self.size and 0<=size<=32*1024*1024,'Refuse an excessive range'
        size=min(size,self.size-start)
        if size==0:return b''
        for offset,data in self.segments:
            if offset<=start and start+size<=offset+len(data):return data[start-offset:start-offset+size]
        end=start+size-1;target=self.cache/f'{start}-{end}.bin'
        if target.exists():
            data=target.read_bytes();assert len(data)==size;return data
        for attempt in range(4):
            try:
                req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','If-Range':self.etag,
                                                       'User-Agent':'POPE-reproduction/1.0'})
                with urllib.request.urlopen(req,timeout=90) as response:
                    assert response.status==206,'Refuse non-range download'
                    assert response.headers['Content-Range']==f'bytes {start}-{end}/{self.size}'
                    assert response.headers.get('ETag')==self.etag,'Archive changed during extraction'
                    data=response.read(size+1);assert len(data)==size
                with self.lock:
                    self.network_bytes+=len(data);self.request_count+=1
                    assert self.network_bytes<250*1024*1024,'Download budget exceeded; no whole dataset downloads'
                temporary=target.with_suffix('.tmp');temporary.write_bytes(data);os.replace(temporary,target)
                return data
            except Exception as error:
                with self.lock:
                    with (ROOT/'logs/gqa-range-retries.jsonl').open('a',encoding='utf-8') as log:
                        log.write(json.dumps({'at':now(),'offset':start,'bytes':size,'attempt':attempt+1,'error':repr(error)})+'\n')
                if attempt==3:raise
                time.sleep(5*2**attempt)

    def seek(self,offset,whence=0):
        self.position=(0 if whence==0 else self.position if whence==1 else self.size)+offset
        if self.position<0:raise ValueError('Negative seek')
        return self.position
    def tell(self):return self.position
    def seekable(self):return True
    def readable(self):return True
    def read(self,size=-1):
        if size<0:size=self.size-self.position
        data=self.fetch(self.position,size);self.position+=len(data);return data

def existing_images(names):
    report=ROOT/'logs/gqa-asset-search.json'
    if report.exists():return json.loads(report.read_text(encoding='utf-8'))['matches']
    excluded={'.git','.venv','node_modules','__pycache__','$recycle.bin','system volume information',
              'windows','program files','program files (x86)','appdata','.cache'}
    matches={};errors=[]
    for base in [Path('E:/'),Path('D:/'),Path.home()/'Downloads']:
        if not base.exists():continue
        for folder,dirs,files in os.walk(base,onerror=lambda e:errors.append(str(e))):
            dirs[:]=[d for d in dirs if d.lower() not in excluded and not d.lower().startswith('.venv')]
            for name in set(files)&names:matches.setdefault(name,[]).append(str(Path(folder)/name))
    atomic_json(report,{'at':now(),'read_only':True,'matches':matches,'scan_errors':errors})
    return matches

def decode_member(reader,info):
    length=30+len(info.filename.encode('utf-8'))+max(256,len(info.extra))+info.compress_size
    blob=reader.fetch(info.header_offset,length)
    header=struct.unpack('<4s5H3I2H',blob[:30])
    signature,version,flags,method,mt,md,crc,cs,us,nl,el=header
    assert signature==b'PK\x03\x04' and not flags&1 and method==info.compress_type
    name=blob[30:30+nl].decode('utf-8' if flags&0x800 else 'cp437');assert name==info.filename
    start=30+nl+el
    if start+info.compress_size>len(blob):blob=reader.fetch(info.header_offset,start+info.compress_size)
    encoded=blob[start:start+info.compress_size]
    if method==zipfile.ZIP_STORED:data=encoded
    elif method==zipfile.ZIP_DEFLATED:data=zlib.decompress(encoded,-15)
    else:raise ValueError(f'Unsupported ZIP compression {method}')
    assert len(data)==info.file_size and zlib.crc32(data)&0xffffffff==info.CRC
    return data

def valid_data(data,info):
    if len(data)!=info.file_size or zlib.crc32(data)&0xffffffff!=info.CRC:return False
    with Image.open(io.BytesIO(data)) as image:image.verify()
    return True

def main():
    question_path=ROOT/'third_party/POPE/output/seem/gqa/gqa_pope_seem_random.json'
    rows=json.loads(question_path.read_text(encoding='utf-8'));names={r['image'] for r in rows};assert len(names)==500
    assert all(re.fullmatch(r'(?:n)?\d+\.jpg',name) for name in names)
    matches=existing_images(names);reader=OfficialRanges()
    with zipfile.ZipFile(reader) as archive:
        entries={}
        for info in archive.infolist():
            name=Path(info.filename).name
            if name in names:
                assert name not in entries,'Ambiguous image basename in official archive'
                entries[name]=info
        assert set(entries)==names,sorted(names-set(entries))
        # Retain index evidence before downloading any image payloads.
        directory=ROOT/'data/gqa';directory.mkdir(exist_ok=True);out=directory/'images';out.mkdir(exist_ok=True)
        atomic_json(directory/'original-500-zip-index.json',{'source':URL,'etag':reader.etag,'archive_bytes':reader.size,
            'question_sha256':digest(question_path),'entries':{name:{'filename':info.filename,'crc32':info.CRC,
                'uncompressed_bytes':info.file_size,'compressed_bytes':info.compress_size,'offset':info.header_offset,
                'method':info.compress_type} for name,info in entries.items()}})
        print(f'Official archive indexed ({reader.size} bytes); extracting only 500 selected images.',flush=True)
        def download(name):
            info=entries[name];target=out/name;source=None;data=None
            candidates=([target] if target.exists() else [])+[Path(p) for p in matches.get(name,[]) if Path(p)!=target]
            for candidate in candidates:
                candidate_data=candidate.read_bytes()
                if valid_data(candidate_data,info):data=candidate_data;source='existing:'+str(candidate);break
            if data is None:data=decode_member(reader,info);source=URL+'#'+info.filename
            assert valid_data(data,info)
            if not target.exists():
                temporary=target.with_suffix('.part')
                with temporary.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
                os.replace(temporary,target)
            else:assert target.read_bytes()==data,'Existing project image differs; preserve it for diagnosis'
            return {'image':name,'source':source,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
                    'official_zip_crc32':info.CRC,'official_zip_member':info.filename,'jpeg_verified':True}
        manifest={};pending=sorted(names)
        for round_id in range(3):
            failed=[]
            # A single HTTP stream avoids overloading the academic data host.
            for name in pending:
                try:record=download(name)
                except Exception as error:
                    failed.append(name)
                    with (directory/'image-download-failures.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps({'at':now(),'image':name,'round':round_id+1,'error':repr(error)})+'\n')
                    continue
                manifest[name]=record
                with (directory/'images-manifest.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
                if len(manifest)%25==0:
                    print(f'Original GQA images verified {len(manifest)}/500',flush=True)
                    atomic_json(directory/'download-progress.json',{'at':now(),'verified':len(manifest),'expected':500,
                        'network_bytes_this_run':reader.network_bytes,'complete':False})
            pending=failed
            if not pending:break
            print(f'GQA host retry round needed for {len(pending)} images; existing successes retained.',flush=True)
            time.sleep(20)
    atomic_json(ROOT/'data/gqa/images-validation.json',{'at':now(),'passed':not pending,'expected':500,'images':len(manifest),
        'source':URL,'etag':reader.etag,'archive_bytes':reader.size,'whole_archive_downloaded':False,
        'network_bytes_this_run':reader.network_bytes,'http_range_requests_this_run':reader.request_count,
        'all_original_archive_crc_and_sizes_match':True,'all_jpegs_verified':True,'manifest':list(manifest.values()),
        'missing_images':pending,
        'script_sha256':digest(__file__)})
    print(json.dumps({'passed':not pending,'images':len(manifest),'downloaded_bytes':reader.network_bytes,'avoided_full_archive_bytes':reader.size}),flush=True)
    if pending:raise SystemExit(2)

if __name__=='__main__':main()
