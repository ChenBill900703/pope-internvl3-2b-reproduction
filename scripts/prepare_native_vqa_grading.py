"""Cache readily available public native graders; never call a paid examiner."""
import json,urllib.request
from pathlib import Path
from run_pope import ROOT,digest,now
from atomic_io import atomic_json

def request(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'POPE-reproduction/1.0'}),timeout=40).read()

def main():
    folder=ROOT/'third_party/A-OKVQA-evaluation';folder.mkdir(exist_ok=True)
    metadata=folder/'source.json'
    if not metadata.exists():
        commit=json.loads(request('https://api.github.com/repos/allenai/aokvqa/commits/main'))['sha']
        url=f'https://raw.githubusercontent.com/allenai/aokvqa/{commit}/evaluation/eval_predictions.py'
        target=folder/'eval_predictions.py';target.write_bytes(request(url))
        # Retain a license when the repository exposes one. No extended search.
        listing=json.loads(request(f'https://api.github.com/repos/allenai/aokvqa/contents?ref={commit}'))
        licenses=[]
        for entry in listing:
            if entry['type']=='file' and entry['name'].lower().startswith(('license','copying')):
                dest=folder/entry['name'];dest.write_bytes(request(entry['download_url']));licenses.append(entry['name'])
        atomic_json(metadata,{'at':now(),'repository':'https://github.com/allenai/aokvqa','commit':commit,
            'source_url':url,'source_sha256':digest(target),'license_files':licenses,
            'used_function':'eval_aokvqa, extracted from original AST without changing its function body',
            'score_policy':'Original exact-string direct-answer matching; original difficult_direct_answer exclusion; no ChatGPT'})
    gqa=ROOT/'third_party/GQA-evaluation';gqa.mkdir(exist_ok=True)
    if not (gqa/'source.json').exists():
        url='https://cs.stanford.edu/people/dorarad/gqa/evaluate.html'
        target=gqa/'evaluation-source.html';target.write_bytes(request(url))
        atomic_json(gqa/'source.json',{'at':now(),'source_url':url,'source_sha256':digest(target),
            'method':'Documented standard accuracy: exact predicted-answer equality to reference, mean over questions',
            'implementation_scope':'Local raw exact-match diagnostic, not the complete GQA evaluation suite',
            'normalization':'None: no lowercasing, punctuation stripping or semantic answer mapping',
            'large_evaluation_bundle_downloaded':False})
    print('Official A-OKVQA function and GQA exact-match definition cached; no GPU or external examiner used.')

if __name__=='__main__':main()
