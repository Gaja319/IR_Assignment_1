import re
from typing import Dict, List
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def ensure_nltk():
    try:
        stopwords.words('english')
    except Exception:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')


def load_documents(folder_path: str) -> Dict[str, str]:
    from pathlib import Path
    docs = {}
    p = Path(folder_path)
    for f in sorted(p.glob('*.txt')):
        docs[f.stem] = f.read_text(encoding='utf-8')
    return docs


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\w+", text)
    return tokens


def preprocess_docs(docs: Dict[str, str]):
    ensure_nltk()
    stops = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    docs_tokens = {}
    docs_stem = {}
    docs_lem = {}
    for doc_id, text in docs.items():
        # hyphen handling: replace hyphens with spaces
        text2 = text.replace('-', ' ')
        toks = [t.lower() for t in tokenize(text2)]
        toks_nostop = [t for t in toks if t not in stops]
        stemmed = [stemmer.stem(t) for t in toks_nostop]
        lemm = [lemmatizer.lemmatize(t) for t in toks_nostop]
        docs_tokens[doc_id] = toks_nostop
        docs_stem[doc_id] = stemmed
        docs_lem[doc_id] = lemm

    return docs_tokens, docs_stem, docs_lem
