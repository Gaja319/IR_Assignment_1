import streamlit as st
from src.preprocess import load_documents, preprocess_docs
from src.indexes import build_positional_index, build_biword_index, phrase_query_biword, phrase_query_positional
from src.tfidf import build_inverted_index, compute_tf, compute_idf, build_tfidf_vectors, rank_query_tfidf
from pathlib import Path
import zipfile, shutil
from src.trees import BST, BTree, time_search_structure
from src.tolerant import build_kgram_index, spell_correction, edit_distance
import time

st.title('IR Assignment - End-to-End Information Retrieval System')

# Initialize session state for persistence across reruns
if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = 'Dataset'
if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'toks' not in st.session_state:
    st.session_state.toks = None
if 'stem' not in st.session_state:
    st.session_state.stem = None
if 'lem' not in st.session_state:
    st.session_state.lem = None
if 'chosen' not in st.session_state:
    st.session_state.chosen = None
if 'pos_idx' not in st.session_state:
    st.session_state.pos_idx = None
if 'bi_idx' not in st.session_state:
    st.session_state.bi_idx = None
if 'doc_vecs' not in st.session_state:
    st.session_state.doc_vecs = None
if 'idf' not in st.session_state:
    st.session_state.idf = None

st.write('📂 Upload a ZIP file or use the default Dataset/ folder. Click "Run Pipeline" to preprocess.')

# Handle ZIP upload
uploaded = st.file_uploader('Upload a ZIP of text files (optional)', type=['zip'])
if uploaded is not None:
    try:
        uploaded_path = Path('uploaded_dataset.zip')
        with uploaded_path.open('wb') as f:
            f.write(uploaded.getbuffer())
        extract_dir = Path('uploaded_dataset')
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir()
        with zipfile.ZipFile(uploaded_path, 'r') as z:
            z.extractall(extract_dir)
        st.session_state.dataset_path = str(extract_dir)
        st.success(f'✓ ZIP extracted to {extract_dir}/')
    except Exception as e:
        st.error(f'Error extracting ZIP: {e}')

st.info(f'📁 Using dataset: {st.session_state.dataset_path}')

# Sidebar options
st.sidebar.header('Options')
preproc = st.sidebar.selectbox('Preprocessing', ['tokens','stem','lemma'])
retrieval = st.sidebar.selectbox('Retrieval method', ['TF-IDF','Biword','Positional'])

# Pipeline execution
if st.sidebar.button('Run Pipeline'):
    try:
        with st.spinner('Loading and preprocessing documents...'):
            st.session_state.docs = load_documents(st.session_state.dataset_path)
            toks, stem, lem = preprocess_docs(st.session_state.docs)
            
            # store all variants
            st.session_state.toks = toks
            st.session_state.stem = stem
            st.session_state.lem = lem
            
            if preproc == 'tokens':
                chosen = toks
            elif preproc == 'stem':
                chosen = stem
            else:
                chosen = lem
            
            st.session_state.chosen = chosen
            st.session_state.pos_idx = build_positional_index(chosen)
            st.session_state.bi_idx = build_biword_index(chosen)
            inv_idx = build_inverted_index(chosen)
            tf = compute_tf(chosen)
            st.session_state.idf = compute_idf(inv_idx, len(st.session_state.docs))
            st.session_state.doc_vecs = build_tfidf_vectors(tf, st.session_state.idf)
        
        st.success(f'✓ Pipeline complete. Loaded {len(st.session_state.docs)} documents, vocab: {len(st.session_state.pos_idx)} terms')
        st.write('**Sample document (first 400 chars):**')
        st.write(next(iter(st.session_state.docs.values()))[:400])
    except Exception as e:
        st.error(f'❌ Pipeline error: {e}')
        import traceback
        st.write(traceback.format_exc())

# Detailed comparison panels
if st.session_state.docs is not None:
    st.divider()
    st.subheader('Comparisons & Detailed Outputs')

    with st.expander('Preprocessing Comparison (tokens / stem / lemma)', expanded=True):
        docs_list = list(st.session_state.docs.keys())
        sel = st.selectbox('Select document id', docs_list)
        st.write('Original (first 400 chars):')
        st.write(st.session_state.docs[sel][:400])
        st.write('Tokens (after stopword removal):')
        st.write(st.session_state.toks[sel][:60])
        st.write('Stemmed tokens:')
        st.write(st.session_state.stem[sel][:60])
        st.write('Lemmatized tokens:')
        st.write(st.session_state.lem[sel][:60])
        # show simple top terms
        from collections import Counter
        ct_tok = Counter(st.session_state.toks[sel])
        ct_stem = Counter(st.session_state.stem[sel])
        ct_lem = Counter(st.session_state.lem[sel])
        st.write('Top tokens (token/stem/lemma):')
        st.write(ct_tok.most_common(8))
        st.write(ct_stem.most_common(8))
        st.write(ct_lem.most_common(8))

    with st.expander('Phrase Query: Biword vs Positional', expanded=True):
        phrase = st.text_input('Enter phrase query for phrase-index comparison', key='phrase_query')
        if phrase:
            bi_res = phrase_query_biword(phrase, st.session_state.bi_idx)
            pos_res = phrase_query_positional(phrase, st.session_state.pos_idx)
            st.write('Biword matches (count):', len(bi_res))
            st.write(list(bi_res)[:100])
            st.write('Positional matches (count):', len(pos_res))
            st.write(list(pos_res)[:100])
            st.write('Biword-only (possible false positives):')
            st.write(list(set(bi_res) - set(pos_res))[:100])
            st.write('Positional-only (missed by biword):')
            st.write(list(set(pos_res) - set(bi_res))[:100])

    with st.expander('TF-IDF Ranking & Simple Evaluation', expanded=True):
        q2 = st.text_input('Enter query for TF-IDF evaluation', key='tfidf_query')
        if q2:
            q_terms = [t.lower() for t in q2.split()]
            tf_local = compute_tf(st.session_state.toks)
            ranked = rank_query_tfidf(q_terms, st.session_state.idf, st.session_state.doc_vecs, tf_local)
            st.write('Top 10 TF-IDF results:')
            for i, (d, s) in enumerate(ranked[:10], 1):
                st.write(f'{i}. {d} — score={s:.4f}')
            # exact-match relevance heuristic
            relevant = {docid for docid, toks in st.session_state.toks.items() if all(term in toks for term in q_terms)}
            retrieved = [d for d, _ in ranked[:10]]
            tp = len(set(retrieved) & relevant)
            fp = len(set(retrieved) - relevant)
            fn = len(relevant - set(retrieved))
            precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
            recall = tp / (tp+fn) if (tp+fn)>0 else 0.0
            f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
            st.write({'precision@10': precision, 'recall': recall, 'f1': f1, 'relevant_docs': len(relevant)})

    with st.expander('Dictionary Structures: BST vs B-Tree timings', expanded=True):
        vocab = sorted(st.session_state.pos_idx.keys())
        bst = BST()
        btree = BTree(t=3)
        for term in vocab:
            bst.insert(term)
            btree.insert(term)
        sample_terms = vocab[:min(100, len(vocab))]
        t_bst = time_search_structure(bst, sample_terms)
        t_btree = time_search_structure(btree, sample_terms)
        st.write({'bst_time_s': t_bst, 'btree_time_s': t_btree})

    with st.expander('Tolerant Retrieval (k-gram + edit distance)', expanded=True):
        vocab = sorted(st.session_state.pos_idx.keys())
        kidx = build_kgram_index(vocab, k=2)
        miss = st.text_input('Enter misspelled word', key='misspell')
        if miss:
            cands = spell_correction(miss, set(vocab), kidx, max_edit=2)
            st.write('Candidates (k-gram + edit distance):', cands[:50])
            st.write('Edit distances:')
            for c in cands[:50]:
                st.write(c, edit_distance(miss, c))

    st.write('---')
    st.info('Use the panels above to inspect preprocessing differences, compare phrase indexes, run TF-IDF ranking, measure dictionary timings, and test tolerant retrieval.')
else:
    st.warning('⚠ Click "Run Pipeline" button (left sidebar) to preprocess the dataset before using comparison panels.')
