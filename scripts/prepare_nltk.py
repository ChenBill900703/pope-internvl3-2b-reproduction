from pathlib import Path
import nltk
ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'data/nltk';target.mkdir(parents=True,exist_ok=True)
nltk.data.path.insert(0,str(target))
for package,resource in [('punkt_tab','tokenizers/punkt_tab/english/')]:
    try:nltk.data.find(resource);print('Already available:',package)
    except LookupError:
        assert nltk.download(package,download_dir=str(target),quiet=False,raise_on_error=True)
print(nltk.word_tokenize('The people see a bird.'))
