from src.preprocess import load_documents, preprocess_docs
from src.indexes import build_positional_index, build_biword_index, phrase_query_biword, phrase_query_positional
from src.tolerant import build_kgram_index, spell_correction, edit_distance
from src.trees import BST, BTree, time_search_structure
from src.tfidf import build_inverted_index, compute_tf, compute_idf, build_tfidf_vectors, rank_query_tfidf
import json
from pathlib import Path


def main():
    base = Path('Dataset')
    out = Path('outputs')
    out.mkdir(exist_ok=True)

    docs = load_documents(base)
    print(f'Loaded {len(docs)} documents')

    toks, stem, lem = preprocess_docs(docs)
    Path(out/'tokens.json').write_text(json.dumps(toks), encoding='utf-8')

    pos_idx = build_positional_index(toks)
    bi_idx = build_biword_index(toks)
    inv_idx = build_inverted_index(toks)
    tf = compute_tf(toks)
    idf = compute_idf(inv_idx, len(docs))
    doc_vecs = build_tfidf_vectors(tf, idf)

    # save simple stats
    stats = {
        'num_docs': len(docs),
        'vocab_size': len(pos_idx)
    }
    Path(out/'stats.json').write_text(json.dumps(stats), encoding='utf-8')

    # sample queries
    queries = ['information retrieval', 'data mining', 'machine learning']
    results = {}
    for q in queries:
        results[q] = {
            'biword': list(phrase_query_biword(q, bi_idx)),
            'positional': list(phrase_query_positional(q, pos_idx))
        }

    Path(out/'phrase_results.json').write_text(json.dumps(results), encoding='utf-8')

    # build vocab and trees
    vocab = sorted(pos_idx.keys())
    bst = BST()
    btree = BTree(t=3)
    for term in vocab:
        bst.insert(term)
        btree.insert(term)

    # time searches
    sample_terms = vocab[:100]
    t_bst = time_search_structure(bst, sample_terms)
    t_btree = time_search_structure(btree, sample_terms)
    Path(out/'tree_times.json').write_text(json.dumps({'bst':t_bst,'btree':t_btree}), encoding='utf-8')

    # tolerant retrieval: k-gram and spell correction
    kidx = build_kgram_index(vocab, k=2)
    misspell = 'retrival'
    corrections = spell_correction(misspell, vocab, kidx, max_edit=2)
    Path(out/'tolerant.json').write_text(json.dumps({'misspell':misspell,'corrections':corrections}), encoding='utf-8')

    # TF-IDF sample ranking and evaluation using simple exact-match relevance
    sample_queries = ['information retrieval', 'data mining', 'machine learning']
    eval_results = {}
    for q in sample_queries:
        q_terms = [t for t in q.split()]
        ranked = rank_query_tfidf(q_terms, idf, doc_vecs, tf)
        top5 = [r[0] for r in ranked[:5]]
        # simple relevance: doc is relevant if contains all query terms
        relevant = set()
        for docid, toks_doc in toks.items():
            if all(term in toks_doc for term in q_terms):
                relevant.add(docid)
        retrieved = set(top5)
        tp = len(retrieved & relevant)
        fp = len(retrieved - relevant)
        fn = len(relevant - retrieved)
        precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
        recall = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        eval_results[q] = {'top5':top5,'precision':precision,'recall':recall,'f1':f1,'relevant_count':len(relevant)}
    Path(out/'tfidf_eval.json').write_text(json.dumps(eval_results), encoding='utf-8')

    print('Pipeline complete. Outputs in outputs/')


if __name__ == '__main__':
    main()
