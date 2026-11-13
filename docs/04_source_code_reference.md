# Part 4 — Complete Python source codes

The pipeline consists of eight modules in `src/`. This page is the reference for each: purpose, imports, functions and dependencies.

> **Status.** The responsibilities below are as defined for the project. The *Imports*, *Functions* and *Inputs/outputs* fields are templates to be completed from the actual files, because function names and signatures should be copied from the code, not inferred.

## Module map

| File | Responsibility | Depends on |
|---|---|---|
| `main_pipeline.py` | Coordinates the complete workflow | all other modules |
| `preprocess_fasta.py` | Loads, cleans and prepares protein sequences | Biopython (expected) |
| `feature_modules.py` | Implements the underlying feature calculations | NumPy, itertools (expected) |
| `extract_features_traditional.py` | Extracts AAC, DPC, K-mer, CTriad and PSSM | `feature_modules.py`, BLAST+ (PSSM) |
| `extract_features_plm.py` | Extracts ESM2, ProtBert and ProtT5 embeddings | PyTorch, transformers / esm |
| `models.py` | Defines DNN, RNN and CNN architectures | TensorFlow/Keras |
| `train_model.py` | Handles model training | scikit-learn, xgboost, `models.py` |
| `evaluate_model.py` | Evaluates models and generates visualisations | scikit-learn, matplotlib, seaborn |

Dependencies marked *expected* follow from the module's job; confirm against each file's import block.

## Per-file reference

Use this block for each file.

### `main_pipeline.py`
- **Purpose:** Coordinates the complete workflow.
- **Imports:** _TODO_
- **Public functions / entry point:** _TODO_
- **Inputs → outputs:** _TODO_ (organism list, feature list, model list → results tree)
- **Notes:** _TODO_ (CLI arguments, what is skipped if outputs already exist)

### `preprocess_fasta.py`
- **Purpose:** Loads, cleans and prepares protein sequences.
- **Imports:** _TODO_
- **Public functions:** _TODO_
- **Inputs → outputs:** FASTA files → cleaned sequences keyed by ID (_confirm_)
- **Notes:** _TODO_ (duplicate handling, non-standard residues, length filtering). Sequences dropped here must also be dropped from the labels, or features and labels will mismatch (Part 6).

### `feature_modules.py`
- **Purpose:** Implements the underlying feature calculations.
- **Imports:** _TODO_
- **Public functions:** _TODO_ (one per feature: AAC, DPC, K-mer, CTriad, PSSM parsing)
- **Notes:** _TODO_ (value of `k`, CTriad class mapping, guards for short sequences)

### `extract_features_traditional.py`
- **Purpose:** Extracts AAC, DPC, K-mer, CTriad and PSSM.
- **Imports:** _TODO_
- **Public functions:** _TODO_
- **Outputs:** _TODO_ (file format/location of feature matrices)
- **Notes:** _TODO_ (PSI-BLAST command, database path, iteration count, PSSM aggregation)

### `extract_features_plm.py`
- **Purpose:** Extracts ESM2, ProtBert and ProtT5 embeddings.
- **Imports:** _TODO_
- **Public functions:** _TODO_
- **Notes:** _TODO_ (checkpoints, max length, pooling, device/precision)

### `models.py`
- **Purpose:** Defines DNN, RNN and CNN architectures.
- **Imports:** _TODO_
- **Public functions:** _TODO_ (one builder per architecture)
- **Notes:** _TODO_ (input shape, layer sizes, activation/loss)

### `train_model.py`
- **Purpose:** Handles model training.
- **Imports:** _TODO_
- **Public functions:** _TODO_
- **Notes:** _TODO_ (split, scaling, class weights, seeds, early stopping)

### `evaluate_model.py`
- **Purpose:** Evaluates models and generates visualisations.
- **Imports:** _TODO_
- **Public functions:** _TODO_
- **Outputs:** confusion-matrix and ROC figures titled `"{MODEL} - {organism} ({feature})"` and `"ROC - {MODEL} - {organism} ({feature})"` (as seen in the saved plots) plus any metrics file.
- **Notes:** see Part 5 for what is computed and how it is saved.

## Adding the full source listings

For a self-contained document, append each file's code below its reference block:

````markdown
#### Source
```python
# contents of src/<file>.py
```
````
