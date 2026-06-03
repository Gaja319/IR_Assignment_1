import math
from typing import Dict, List


def build_inverted_index(docs_tokens: Dict[str, List[str]]):
    inv = {}
    for docid, toks in docs_tokens.items():
        for t in toks:
            inv.setdefault(t, set()).add(docid)
    return inv


def compute_tf(docs_tokens: Dict[str, List[str]]):
    tf = {}
    for docid, toks in docs_tokens.items():
        freqs = {}
        for t in toks:
            freqs[t] = freqs.get(t, 0) + 1
        tf[docid] = freqs
    return tf


def compute_idf(inv_index: Dict[str, set], N: int):
    idf = {}
    for term, docs in inv_index.items():
        idf[term] = math.log((N) / (1 + len(docs)))
    return idf


def build_tfidf_vectors(tf: Dict[str, Dict[str, int]], idf: Dict[str, float]):
    vecs = {}
    for docid, freqs in tf.items():
        vec = {}
        for t, f in freqs.items():
            vec[t] = (1 + math.log(f)) * idf.get(t, 0.0)
        vecs[docid] = vec
    return vecs


def cosine_sim(vec1: Dict[str, float], vec2: Dict[str, float]):
    num = 0.0
    for k, v in vec1.items():
        num += v * vec2.get(k, 0.0)
    norm1 = math.sqrt(sum(v*v for v in vec1.values()))
    norm2 = math.sqrt(sum(v*v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return num / (norm1 * norm2)


def rank_query_tfidf(query_terms: List[str], idf: Dict[str, float], doc_vecs: Dict[str, Dict[str, float]], tf_docs: Dict[str, Dict[str,int]]):
    # build query vector
    qfreq = {}
    for t in query_terms:
        qfreq[t] = qfreq.get(t, 0) + 1
    qvec = {}
    for t, f in qfreq.items():
        qvec[t] = (1 + math.log(f)) * idf.get(t, 0.0)
    scores = []
    for docid, vec in doc_vecs.items():
        scores.append((docid, cosine_sim(qvec, vec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
