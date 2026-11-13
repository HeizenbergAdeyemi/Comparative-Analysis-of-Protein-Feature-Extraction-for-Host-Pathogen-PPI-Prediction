# Source code

Place the eight pipeline modules here:

| File | Responsibility |
|---|---|
| `main_pipeline.py` | Coordinates the complete workflow |
| `preprocess_fasta.py` | Loads, cleans and prepares protein sequences |
| `feature_modules.py` | Implements the underlying feature calculations |
| `extract_features_traditional.py` | Extracts AAC, DPC, K-mer, CTriad and PSSM |
| `extract_features_plm.py` | Extracts ESM2, ProtBert and ProtT5 embeddings |
| `models.py` | Defines DNN, RNN and CNN architectures |
| `train_model.py` | Handles model training |
| `evaluate_model.py` | Evaluates models and generates visualisations |

See [`../docs/04_source_code_reference.md`](../docs/04_source_code_reference.md) for the per-file reference.
