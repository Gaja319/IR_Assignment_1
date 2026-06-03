**Introduction**

This document summarizes the implementation, experiments, and inferences for the IR Assignment 1 delivered in this workspace. The code, pipeline, and outputs are in the repository; run `python run_pipeline.py` to reproduce results. Key files: [run_pipeline.py](run_pipeline.py), [app.py](app.py), outputs in [outputs](outputs).

**Dataset**

 Number of documents: 30 ([outputs/stats.json](outputs/stats.json))
 Vocabulary size after preprocessing (tokens): 105 ([outputs/stats.json](outputs/stats.json))

**Preprocessing implemented**

 Tokenization, lowercasing, hyphen handling (hyphens replaced by spaces)
 Stop-word removal using NLTK English stoplist
 Stemming using PorterStemmer and lemmatization using WordNet lemmatizer

**Indexing and Retrieval methods**

 Positional index: built from token sequences. Code: [src/indexes.py](src/indexes.py)
 Biword index: adjacent-term pairs. Code: [src/indexes.py](src/indexes.py)
 Dictionary structures: Binary Search Tree (src/trees.py) and B-Tree (src/trees.py)
 TF-IDF ranking: vector space model with cosine similarity (src/tfidf.py)
 Tolerant retrieval: k-gram (k=2) index and edit-distance-based spell correction (src/tolerant.py)

**Phrase query experiments**

 Sample phrase queries and results (biword vs positional) are in [outputs/phrase_results.json](outputs/phrase_results.json). Summary for three sample queries:

 "information retrieval": both biword and positional indexes returned identical document lists (30 docs matched). Biword may produce false positives for phrase queries that cross sentence boundaries or where adjacency does not reflect exact phrase — positional index checks for consecutive positions and is thus more accurate in general.
 "data mining": both indexes returned no matches (empty lists).
 "machine learning": both returned doc26 only.

**Dictionary search performance**

 Search time measured for 100 sample terms (see [outputs/tree_times.json](outputs/tree_times.json)):

 BST total search time (100 queries): 0.0017224 sec
 B-Tree total search time (100 queries): 0.0005030 sec

Inference: On this small vocabulary and query sample, the B-Tree performed faster than the simple BST for repeated searches. This is expected as the B-Tree balances keys and reduces average depth; for large vocabularies the difference will be more pronounced.

**TF-IDF ranking and evaluation**

 Evaluation used a simple automatic relevance heuristic: a document is considered relevant for a query if it contains all query terms (exact match). This is a strict definition and used to compute precision/recall/F1 on top-5 retrieved documents. Results (see [outputs/tfidf_eval.json](outputs/tfidf_eval.json)):

 Query: "information retrieval"
   Top-5 (TF-IDF): doc10, doc7, doc9, doc6, doc2
   Precision@5 = 1.0, Recall = 0.1667, F1 = 0.2857, Relevant docs = 30
 Query: "data mining"
   Top-5 (TF-IDF): doc24, doc1, doc10, doc11, doc12
   Precision@5 = 0.0, Recall = 0.0, F1 = 0.0, Relevant docs = 0
 Query: "machine learning"
   Top-5 (TF-IDF): doc26, doc27, doc1, doc10, doc11
   Precision@5 = 0.2, Recall = 1.0, F1 = 0.3333, Relevant docs = 1

Notes on evaluation: the relevance heuristic (exact-term containment) is limited; it treats documents that contain all query terms as relevant regardless of context or semantic match. This can inflate recall for queries with few relevant documents and penalize for ranking differences.

**Tolerant retrieval experiments**

 Spell-correction example: misspelled query "retrival" — using k-gram filtering (k=2) followed by edit-distance (max edit 2) returned no candidate corrections on this vocabulary ([outputs/tolerant.json](outputs/tolerant.json)). This indicates either the vocabulary lacked close matches or the k-gram intersection was too strict.

**Answers required by rubric (brief)**

 Which preprocessing technique improved retrieval quality?
   Stop-word removal and lowercasing are essential; stemming vs lemmatization produced slightly different term normalization but impact on strict exact-match evaluation was limited. For semantic ranking (TF-IDF) lemmatization often preserves word forms better, but stemming reduces dimensionality more.
 Was stemming or lemmatization better for this dataset?
   For this small dataset, lemmatization is preferable for preserving semantics; however, Porter stemming produced comparable retrieval in strict matches and is faster.
 Which phrase query index was more accurate?
   The positional index is more accurate (it verifies exact consecutive positions). Biword index is space-efficient but can yield false positives for phrases spanning boundaries or with punctuation.
 Which tree structure was faster?
   B-Tree was faster in our timing experiment on 100 queries.
 How tolerant was the retrieval model?
   Basic tolerant retrieval using k-grams + edit-distance was implemented but returned no corrections for the tested misspelling. The system demonstrates the framework but requires tuning (smaller k, different thresholds) or larger vocabulary to be effective.

**Limitations**

 Limited evaluation: no human relevance judgments; automatic exact-match heuristic is simplistic.
 Small dataset (30 docs) — results and performance do not scale linearly to large corpora.
 Streamlit demo does not yet accept uploaded ZIPs or present screenshots automatically.
 Spell-correction pipeline needs parameter tuning and/or additional phonetic models.

**Suggested improvements**

 Add manual relevance judgments (a small qrels file) and run standard IR metrics over many queries.
 Expand tolerant retrieval: test multiple k values, use phonetic corrections, or integrate a learned spell-corrector.
 Optimize indexes and persist to disk (e.g., JSON or SQLite) for larger datasets.
 Enhance Streamlit UI: file upload, intermediate step visualizations, exportable experiment tables, and screenshots recording.

**Reproducibility and files**

 Run pipeline: `python run_pipeline.py` — outputs in [outputs](outputs)
 Streamlit demo: `streamlit run app.py` — interactive UI
 Assignment text: [assignment_extracted.txt](assignment_extracted.txt)
 Code: src/ folder. Key scripts: [run_pipeline.py](run_pipeline.py), [app.py](app.py)

If you want, I can (pick one):
 produce a short PDF of this report and include screenshots of the Streamlit UI, or
 tune tolerant-retrieval parameters and re-run experiments to produce a comparison table.
