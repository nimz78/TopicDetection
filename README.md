# Topic Detection and Multi-Label Assignment of Scientific Papers using Topic Modeling and SSFuzzyART Clustering

> Master's thesis project — a framework for clustering and multi-label topic assignment of bilingual (Persian–English) scientific papers.

## Overview

Scientific papers are frequently interdisciplinary: a single paper may belong to several topics at once (for example, a biomedical-engineering paper relates simultaneously to medicine, computer science, and artificial intelligence). Assigning such a paper to a single category discards valuable information.

This project presents a framework that automatically clusters bilingual (Persian–English) scientific papers by topic and assigns **multiple** labels per paper, so that interdisciplinary work can be represented across several related clusters. The framework combines topic modeling and deep contextual embeddings with a **semi-supervised Fuzzy ART (SSFuzzyART)** clustering algorithm, and does not require large-scale manual labeling.

## Method

The pipeline consists of five stages:

1. **Preprocessing.** Persian text is normalized, tokenized, and cleaned using the Hazm library (character normalization, ZWNJ handling, stopword removal); English text is cleaned and stopword-filtered. Author keywords and paper titles are additionally incorporated (weighted) to strengthen the topical signal.
2. **Feature extraction (two competing representations).**
   - *LDA:* documents are represented as topic-mixture vectors, with the number of topics *K* selected automatically via **topic coherence**.
   - *Deep embeddings:* **ParsBERT** encodes Persian text and an English sentence-transformer encodes English text; the two vectors are concatenated into a single bilingual representation.
3. **Topic stabilization (TS-ENTM).** Redundant LDA topics are merged. This stage applies **only** to the LDA representation; it is not meaningful for dense embeddings, since individual embedding dimensions carry no standalone semantic meaning.
4. **Clustering (SSFuzzyART).** A semi-supervised Fuzzy ART variant groups the papers. The algorithm is dual-mode — a fuzzy similarity with complement coding for LDA vectors, and cosine similarity for dense embeddings — and requires no preset number of clusters (the vigilance parameter *ρ* is selected automatically). It operates unsupervised by default and becomes semi-supervised when a small set of seed labels is available.
5. **Multi-label assignment and evaluation.** Each paper receives up to three cluster labels whose membership degrees exceed a threshold, so single-topic papers get one label and interdisciplinary papers get several. Quality is assessed with internal indices, topic coherence/diversity, and manual semantic inspection of cluster contents.

## Experimental Scenarios

Three scenarios are compared to isolate the effect of the representation and the topic-stabilization stage:

| Scenario | Representation | Topic stabilization | Pipeline |
|----------|----------------|---------------------|----------|
| S1 | LDA | Yes (TS-ENTM) | LDA → TS-ENTM → SSFuzzyART |
| S2 | LDA | No | LDA → SSFuzzyART |
| S3 | BERT / ParsBERT | Not applicable | Embeddings → SSFuzzyART |

The BERT-with-TS-ENTM scenario is intentionally excluded, since topic stabilization is not applicable to dense embeddings.

## Key Findings

- **Deep embeddings substantially outperform classical topic modeling** for semantic clustering of this corpus. On manual inspection, embedding-based clusters were topically coherent (each cluster mapping to a clear subject), whereas LDA clusters frequently mixed unrelated fields.
- **Topic stabilization (TS-ENTM) is representation-specific.** On this corpus the LDA topics were already well separated, so no merging occurred; and the technique is not appropriate for embeddings at all. Consequently the LDA scenarios with and without this stage produced identical results — itself a methodological finding.
- **The silhouette score is unreliable for high-dimensional text.** It remained near zero or negative even for coherent clusters, so evaluation relied primarily on semantic coherence and inspection rather than on that index alone.

## Repository Contents

- `m1_clean_naming.ipynb` — the complete, runnable pipeline (preprocessing, feature extraction, TS-ENTM, SSFuzzyART, multi-label assignment, evaluation, and output generation).
- Generated outputs include per-scenario cluster previews, cluster topic names, a cross-scenario comparison table, and evaluation metrics.

## Usage

The notebook is designed to run in Google Colab.

1. Install dependencies (Hazm, scikit-learn, sentence-transformers, pandas, numpy).
2. Mount the data source and set the dataset path in the configuration block.
3. Run the notebook; all outputs are written to the configured output directory.

The dataset is expected to contain, per paper: paper id, Persian title, English title, Persian text, English text, Persian keywords, and English keywords.

> **Note.** Preprocessing results are cached; when preprocessing settings change, clear the cache (or set the force-rebuild flag) so the data is reprocessed.

## Technologies

Python · scikit-learn · Hazm · Sentence-Transformers (ParsBERT) · Gensim-free coherence · pandas · NumPy

## Citation

If you use this work, please cite the associated thesis. *(Add full citation details here.)*

## License

*(Add your chosen license here — e.g., MIT.)*
