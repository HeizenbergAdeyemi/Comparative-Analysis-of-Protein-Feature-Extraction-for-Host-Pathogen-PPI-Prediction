# Part 2 — Feature extraction methods

Each protein is converted to a fixed-length vector. Dimensions below are those of the standard definitions; the **Implementation** column must be confirmed against `feature_modules.py` and the extraction scripts.

| Method | Type | Dimension | Captures |
|---|---|---|---|
| AAC | Traditional | 20 | Residue composition |
| DPC | Traditional | 400 | Adjacent-residue pair composition |
| K-mer | Traditional | 20^k (k = 3 → 8000) | Local k-residue patterns |
| CTriad | Traditional | 343 | Composition of triads of residue *classes* |
| PSSM | Traditional (evolutionary) | 20 or 400 after aggregation | Position-specific conservation |
| ESM2 | PLM | 320–5120 (model-dependent) | Learned contextual representation |
| ProtBert | PLM | 1024 | Learned contextual representation |
| ProtT5 | PLM | 1024 | Learned contextual representation |

> **TODO (author):** fill in the values used in this study: the `k` for K-mer, the PSSM aggregation (column mean → 20-d, or 400-d), the exact ESM2 checkpoint, and the pooling used for PLM embeddings.

## Traditional features

### Amino Acid Composition (AAC)
Relative frequency of each of the 20 standard amino acids: `AAC(a) = count(a) / L`, where `L` is sequence length. Fixed 20-d regardless of length. Discards all order information.

### Dipeptide Composition (DPC)
Relative frequency of each of the 400 ordered residue pairs: `DPC(a,b) = count(ab) / (L − 1)`. Retains nearest-neighbour order. Sequences shorter than 2 residues give an undefined denominator and must be handled explicitly (Part 6).

### K-mer
Generalisation of DPC to k consecutive residues, `20^k` features, normalised by `L − k + 1`. Dimensionality grows quickly with `k` and the vectors become sparse.

### Conjoint Triad (CTriad)
The 20 amino acids are grouped into 7 classes by dipole moment and side-chain volume (Shen et al., 2007): {A,G,V}, {I,L,F,P}, {Y,M,T,S}, {H,N,Q,W}, {R,K}, {D,E}, {C}. Each sequence becomes a string over 7 symbols, and the frequency of each of the 7³ = 343 consecutive triads is counted and normalised. It trades residue-level detail for a compact, physicochemically motivated vector.

### Position-Specific Scoring Matrix (PSSM)
Run PSI-BLAST for each sequence against a protein database (SwissProt here) for several iterations to obtain an `L × 20` log-odds matrix of position-specific substitution scores. It is aggregated to a fixed-length vector (typically a 20-d column mean or a 400-d residue-type aggregation). It encodes evolutionary conservation, but is the most expensive and fragile traditional feature: it depends on BLAST+, a local database and per-sequence output files. In this study every PSSM model produced a constant prediction; see Part 6.

## Protein language model embeddings

PLMs are pretrained on very large protein corpora; the final hidden states for each residue are pooled over the sequence (commonly a mean over residues) into one vector per protein. No task-specific training is involved in feature extraction.

### ESM2
Transformer family from Meta AI. Embedding width depends on the checkpoint (8M → 320, 35M → 480, 150M → 640, 650M → 1280, 3B → 2560, 15B → 5120). Long sequences may need truncation or chunking to the model's context window.

### ProtBert
BERT-style model (`Rostlab/prot_bert`), 1024-d. Requires residues separated by spaces and rare residues (`U, Z, O, B`) mapped to `X` before tokenisation.

### ProtT5
T5 encoder (`Rostlab/prot_t5_xl_uniref50`), 1024-d, with the same space-separation/`X`-mapping convention as ProtBert. Only the encoder is needed for embeddings.

## Representing a host–pathogen pair

> **TODO (author):** state how two per-protein vectors become one pair vector (e.g. concatenation `[host ‖ pathogen]`, element-wise product, or both), since this determines the classifier input dimension and should be identical across features.

## Where each feature is implemented

| Features | Calculation | Driver |
|---|---|---|
| AAC, DPC, K-mer, CTriad, PSSM | `feature_modules.py` | `extract_features_traditional.py` |
| ESM2, ProtBert, ProtT5 | model loading + pooling | `extract_features_plm.py` |
