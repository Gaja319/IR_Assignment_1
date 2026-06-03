import streamlit as st
from src.preprocess import load_documents, preprocess_docs
from src.indexes import build_positional_index, build_biword_index, phrase_query_biword, phrase_query_positional
from src.tfidf import build_inverted_index, compute_tf, compute_idf, build_tfidf_vectors, rank_query_tfidf
from pathlib import Path
import zipfile, shutil

st.title('IR Assignment - End-to-End Information Retrieval System')

# Initialize session state for persistence across reruns
if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = 'Dataset'
if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'toks' not in st.session_state:
    st.session_state.toks = None
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
            
            if preproc == 'tokens':
                chosen = toks
            elif preproc == 'stem':
                chosen = stem
            else:
                chosen = lem
            
            st.session_state.toks = chosen
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

# Query section (shown after pipeline runs)
if st.session_state.docs is not None:
    st.divider()
    st.subheader('🔍 Query Results')
    q = st.text_input('Enter query')
    if q:
        try:
            q_terms = [t.lower() for t in q.split()]
            if retrieval == 'TF-IDF':
                tf = compute_tf(st.session_state.toks)
                ranked = rank_query_tfidf(q_terms, st.session_state.idf, st.session_state.doc_vecs, tf)
                st.write(f'**Top 10 results (TF-IDF):**')
                for i, (docid, score) in enumerate(ranked[:10], 1):
                    st.write(f'  {i}. {docid}: score={score:.4f}')
            elif retrieval == 'Biword':
                res = phrase_query_biword(q, st.session_state.bi_idx)
                st.write(f'**Biword results ({len(res)} docs):**')
                st.write(list(res)[:20])
            else:
                res = phrase_query_positional(q, st.session_state.pos_idx)
                st.write(f'**Positional results ({len(res)} docs):**')
                st.write(list(res)[:20])
        except Exception as e:
            st.error(f'❌ Query error: {e}')
            import traceback
            st.write(traceback.format_exc())
else:
    st.warning('⚠ Click "Run Pipeline" button (left sidebar) to preprocess the dataset before entering queries.')
