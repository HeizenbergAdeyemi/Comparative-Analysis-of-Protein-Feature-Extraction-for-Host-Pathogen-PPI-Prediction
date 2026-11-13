# Part 2 — Feature extraction methods

Each protein is converted to a fixed-length vector; a **pair** is represented by concatenating the host vector and the pathogen vector (`[host ‖ pathogen]`), so the classifier input is twice the per-protein dimension.

| Method (CLI name) | Type | Per protein | Per pair | Implemented in |
|---|---|---|---|---|
| AAC | Traditional | 20 | 40 | `amino_acid_composition` |
| DPC | Traditional | 400 | 800 | `dipeptide_composition` |
| KMER (k = 3) | Traditional | 8000 | 16000 | `kmer_composition` |
| CTRIAD | Traditional | 343 | 686 | `conjoint_triad` |
| PSSM | Traditional (evolutionary) | 40 | 80 | `pssm_summary` + PSI-BLAST |
| ESM2 (`esm2_t6_8M_UR50D`) | PLM | 320 | 640 | `embed_sequences` |
| ProtBert (`Rostlab/prot_bert`) | PLM | 1024 | 2048 | `embed_sequences` |
| ProtT5 (`Rostlab/prot_t5_xl_uniref50`) | PLM | 1024 | 2048 | `embed_sequences` |

## Common cleaning

`clean_sequence` upper-cases a sequence and keeps only the 20 canonical amino acids (`ACDEFGHIKLMNPQRSTVWY`), so gaps, stop symbols and ambiguous residues (`X, B, Z, U, O`) are removed *before* any length-based normalisation.

## Traditional features

### Amino Acid Composition (AAC)
Relative frequency of each canonical residue: `count(a) / L`, where `L` is the cleaned length. Discards all order information.

### Dipeptide Composition (DPC)
Frequency of each of the 400 ordered residue pairs over overlapping windows, divided by `L − 1`. Keeps nearest-neighbour order.

### K-mer
Generalises DPC to `k` consecutive residues (default `k = 3`, set with `--k`), giving `20^k` features divided by `L − k + 1`. At `k = 3` the vectors have 8000 dimensions per protein and are sparse.

### Conjoint Triad (CTriad)
Residues are mapped to 7 classes by dipole moment and side-chain volume (Shen et al.): {A,G,V}, {I,L,F,P}, {Y,M,T,S}, {H,N,Q,W}, {R,K}, {D,E}, {C}. Consecutive triads of classes (7³ = 343) are counted and divided by the number of valid triads, so the vector is a relative frequency. A `gap` argument (default 0) allows skipping residues between triad members.

### Position-Specific Scoring Matrix (PSSM)
For each sequence, `psiblast` is run against a local SwissProt BLAST database (3 iterations, e-value 0.001) with `-out_ascii_pssm`. The residue-by-20 log-odds matrix is parsed from the ASCII file and summarised as the **per-column mean and standard deviation** across residues, giving a 40-dimensional vector per protein. PSSM is the slowest and most fragile feature: it needs BLAST+ and a local database, and two PSI-BLAST runs per pair.

## Protein language model embeddings

For each sequence the last hidden states are **mean-pooled over residues, excluding special tokens and padding**, giving one vector per protein. No task-specific training is involved. Sequences are truncated at `--max-length` tokens (default 1024), embeddings are computed in batches (`--plm-batch-size`, default 8) and cached as `.npy` files under `outputs/cache/`.

| Model | Checkpoint | Dimension | Input format |
|---|---|---|---|
| ESM2 | `facebook/esm2_t6_8M_UR50D` (smallest, 8M parameters) | 320 | raw sequence |
| ProtBert | `Rostlab/prot_bert` | 1024 | residues separated by spaces |
| ProtT5 | `Rostlab/prot_t5_xl_uniref50` | 1024 | residues separated by spaces |

Notes for interpreting the PLM results:

- **ESM2 here is the smallest checkpoint.** Conclusions about ESM2 apply to the 8M model, not to the 650M or 3B variants, which are usually considerably stronger.
- Rare residues (`U, Z, O, B`) are **not** mapped to `X` before ProtBert/ProtT5 tokenisation; only non-letters are stripped.
- ProtT5 is loaded with `AutoModel`, which for T5 returns the full encoder–decoder model. See Part 6 for why this very likely fails and how to fix it.

## Where each feature is implemented

| Features | Calculation | Driver |
|---|---|---|
| AAC, DPC, K-mer, CTriad, PSSM summary | `feature_modules.py` | `extract_features_traditional.py` |
| ESM2, ProtBert, ProtT5 | model loading and pooling | `extract_features_plm.py` |
