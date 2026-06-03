from collections import defaultdict
from typing import Dict, List, Tuple


def build_positional_index(docs_tokens: Dict[str, List[str]]):
    # term -> {docid: [positions]}
    idx = defaultdict(lambda: defaultdict(list))
    for doc_id, tokens in docs_tokens.items():
        for pos, t in enumerate(tokens):
            idx[t][doc_id].append(pos)
    return idx


def build_biword_index(docs_tokens: Dict[str, List[str]]):
    bi = defaultdict(set)
    for doc_id, tokens in docs_tokens.items():
        for i in range(len(tokens)-1):
            pair = tokens[i] + ' ' + tokens[i+1]
            bi[pair].add(doc_id)
    return bi


def phrase_query_biword(query: str, biword_index):
    return biword_index.get(query, set())


def phrase_query_positional(query: str, positional_index):
    terms = query.split()
    if not terms:
        return set()
    # start with docs containing first term
    first = positional_index.get(terms[0], {})
    results = set(first.keys())
    for i in range(1, len(terms)):
        term = terms[i]
        posting = positional_index.get(term, {})
        docs_with = set(posting.keys())
        results &= docs_with
    final = set()
    for d in results:
        positions = [positional_index[terms[0]][d]]
        # check consecutive positions
        for p in positional_index[terms[0]][d]:
            ok = True
            for i in range(1, len(terms)):
                needed = p + i
                if needed not in positional_index[terms[i]][d]:
                    ok = False
                    break
            if ok:
                final.add(d)
                break
    return final
