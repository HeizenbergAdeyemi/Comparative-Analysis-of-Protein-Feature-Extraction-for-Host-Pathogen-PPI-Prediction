# Part 7 — Reproducibility

## Directory structure

```
.
├── README.md
├── requirements.txt
├── docs/
├── src/
│   ├── main_pipeline.py
│   ├── preprocess_fasta.py
│   ├── feature_modules.py
│   ├── extract_features_traditional.py
│   ├── extract_features_plm.py
│   ├── models.py
│   ├── train_model.py
│   └── evaluate_model.py
├── data/<organism>/            # four FASTA files each; git-ignored by default
├── outputs/                    # pipeline output; git-ignored
└── results/                    # published figures and metrics
```

## Environment configuration

```bash
conda create -n hpi-ppi python=3.10 -y     # confirm the Python version used for the study
conda activate hpi-ppi
pip install -r requirements.txt
```

| Need | For | Check |
|---|---|---|
| BLAST+ and the SwissProt database | PSSM | `psiblast -version`, `blastdbcmd -db <prefix> -info` |
| GPU + CUDA (optional, `PPI_USE_GPU=1`) | PLM embedding, neural models | `torch.cuda.is_available()` |
| Internet on first run | PLM weights from Hugging Face | enough disk for the checkpoints (ProtT5-XL is several GB) |

> **TODO (author):** record the exact Python, PyTorch, TensorFlow, scikit-learn and XGBoost versions (`pip freeze > requirements.lock.txt`). Random seeds are fixed in the code (42).

## Execution instructions

```bash
# smoke test (minutes): one organism, cheap features, two models
python src/main_pipeline.py --data-root data --output-root outputs \
    --organisms plant --features AAC DPC --models RF XGB

# everything
python src/main_pipeline.py --data-root data --output-root outputs \
    --pssm-database /path/to/blastdb/swissprot

# with GPU
PPI_USE_GPU=1 python src/main_pipeline.py --features ESM2 ProtBert --organisms plant
```

Feature names: `AAC DPC KMER CTRIAD PSSM ESM2 ProtBert ProtT5`. Model names: `RF SVM XGB DNN RNN CNN`.

## Expected outputs

Per organism and feature: the files listed in Part 5. The completed study in `results/` contains **5 organisms × 6 features × 3 models = 90 combinations and 180 figures** (RF, SVM, XGBoost on AAC, DPC, K-mer, PSSM, ESM2, ProtBert), with reference values in `results/metrics/results_summary.csv`. After any run, check `outputs/pipeline_summary.json` for `"error"` entries and the log for `skipped` messages, which show features or models that did not run.

## Known issues and recommendations

1. **Test for leakage with a grouped split (highest priority).** `train_model.py` splits *pairs* at random, so a protein appearing in several pairs is in both train and test. Composition vectors are close to unique protein identifiers, which would explain AUC ≈ 1.00 for AAC/DPC/K-mer on HIV-1 and SARS-CoV-2 (few pathogen proteins, many partners), and the lower scores of the PLMs. A first check using the IDs already stored on each pair:

   ```python
   from sklearn.model_selection import GroupShuffleSplit
   groups = [s["pathogen_id"] for s in samples]
   gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
   train_idx, test_idx = next(gss.split(X, y, groups=groups))
   ```

   This separates pathogen proteins only; for a strict test, also cluster by sequence identity (CD-HIT or MMseqs2) and split by cluster so that neither host nor pathogen homologues cross the split. Also check that negatives were drawn from the same protein pool as positives. If AUC drops sharply, the earlier scores were a dataset effect.
2. **Fix ProtT5.** Replace the `AutoModel`/`AutoTokenizer` loading for ProtT5 with the encoder only:

   ```python
   import re
   from transformers import T5Tokenizer, T5EncoderModel
   tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
   model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
   seq = " ".join(re.sub(r"[UZOB]", "X", sequence))
   ```

   and apply the same rare-residue mapping for ProtBert.
3. **Decide on GPU use deliberately.** With the current import order, PLM embedding runs on CPU unless `PPI_USE_GPU=1`.
4. **Avoid repeated work.** Embeddings and PSSMs are computed per pair. Computing them once per unique sequence and indexing into the result would cut PLM and PSI-BLAST time sharply, because proteins recur across pairs.
5. **Re-run with the current code and a fixed PSSM pipeline.** This gives figures and metrics that match the code (Part 5), and shows whether the PSSM collapse disappears.
6. **Complete the matrix.** Run CTRIAD, ProtT5 and DNN/RNN/CNN and check `outputs/pipeline_summary.json` for errors.
7. **Use a stronger ESM2 checkpoint** (for example `esm2_t33_650M_UR50D`) before concluding that PLMs do not help; the current runs use the 8M model.
8. **Report more than accuracy and AUC.** With strong imbalance, add precision–recall AUC and Matthews correlation coefficient, and use repeated or cross-validated splits for confidence intervals. Consider class weighting for XGBoost to match RF and SVM.
