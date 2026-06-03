import streamlit as st
from src.preprocess import load_documents, preprocess_docs
from src.indexes import build_positional_index, build_biword_index, phrase_query_biword, phrase_query_positional
from src.tfidf import build_inverted_index, compute_tf, compute_idf, build_tfidf_vectors, rank_query_tfidf

st.title('IR Assignment - End-to-End Demo')

uploaded = st.file_uploader('Upload a ZIP of text files (optional)', type=['zip'])
dataset_path = 'Dataset'
if uploaded is not None:
    import zipfile, shutil
    from pathlib import Path
    # save uploaded zip to workspace and extract to uploaded_dataset/
    uploaded_path = Path('uploaded_dataset.zip')
    with uploaded_path.open('wb') as f:
        f.write(uploaded.getbuffer())
    extract_dir = Path('uploaded_dataset')
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()
    with zipfile.ZipFile(uploaded_path, 'r') as z:
        z.extractall(extract_dir)
    st.success(f'Uploaded ZIP extracted to {extract_dir}/ — using it as dataset')
    dataset_path = str(extract_dir)

st.sidebar.header('Options')
preproc = st.sidebar.selectbox('Preprocessing', ['tokens','stem','lemma'])
retrieval = st.sidebar.selectbox('Retrieval method', ['TF-IDF','Biword','Positional'])

if st.sidebar.button('Run Pipeline'):
    docs = load_documents(dataset_path)
    toks, stem, lem = preprocess_docs(docs)
    if preproc == 'tokens':
        chosen = toks
    elif preproc == 'stem':
        chosen = stem
    else:
        chosen = lem
    st.write(f'Loaded {len(docs)} documents')
    st.write('Sample document (doc1):')
    st.write(next(iter(docs.values()))[:400])
    pos_idx = build_positional_index(chosen)
    bi_idx = build_biword_index(chosen)
    inv_idx = build_inverted_index(chosen)
    tf = compute_tf(chosen)
    idf = compute_idf(inv_idx, len(docs))
    doc_vecs = build_tfidf_vectors(tf, idf)

    q = st.text_input('Enter query')
    if q:
        q_terms = [t for t in q.split()]
        if retrieval == 'TF-IDF':
            ranked = rank_query_tfidf(q_terms, idf, doc_vecs, tf)
            st.write('Top results (TF-IDF):')
            for docid, score in ranked[:10]:
                st.write(docid, score)
        elif retrieval == 'Biword':
            res = phrase_query_biword(q, bi_idx)
            st.write('Biword results:', list(res))
        else:
            res = phrase_query_positional(q, pos_idx)
            st.write('Positional results:', list(res))
