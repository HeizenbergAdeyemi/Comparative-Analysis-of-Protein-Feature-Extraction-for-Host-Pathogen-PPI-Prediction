# Source code

The eight pipeline modules. Run from the repository root, e.g. `python src/main_pipeline.py --help`.

| File | Responsibility |
|---|---|
| `main_pipeline.py` | Coordinates the complete workflow |
| `preprocess_fasta.py` | Loads, cleans and prepares protein sequences |
| `feature_modules.py` | Implements the underlying feature calculations |
| `extract_features_traditional.py` | Extracts AAC, DPC, K-mer, CTriad and PSSM |
| `extract_features_plm.py` | Extracts ESM2, ProtBert and ProtT5 embeddings |
| `models.py` | Defines the classical models and the DNN, RNN and CNN architectures |
| `train_model.py` | Handles model training |
| `evaluate_model.py` | Evaluates models and generates visualisations |

Per-file reference and full listings: [`../docs/04_source_code_reference.md`](../docs/04_source_code_reference.md).
