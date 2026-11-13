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
├── data/            # not tracked
└── results/
```

## Environment configuration

```bash
conda create -n hpi-ppi python=3.10 -y     # confirm the Python version actually used
conda activate hpi-ppi
pip install -r requirements.txt
```

External requirements:

| Need | For | Check |
|---|---|---|
| BLAST+ and the SwissProt database | PSSM | `psiblast -version`, `blastdbcmd -db swissprot -info` |
| NVIDIA GPU + CUDA (recommended) | ESM2/ProtBert/ProtT5, DNN/RNN/CNN | `torch.cuda.is_available()`, `tf.config.list_physical_devices('GPU')` |
| Model weights (downloaded on first use) | PLMs | disk space and internet access on first run |

> **TODO (author):** record the exact Python, CUDA, PyTorch, TensorFlow and package versions (`pip freeze > requirements.lock.txt`) and the random seeds.

## Execution instructions

```bash
python src/main_pipeline.py
```

> **TODO (author):** document the command-line arguments (organisms, features, models, paths) and a minimal example that runs end-to-end on one organism and one feature in a few minutes.

## Expected outputs

For every organism × feature × model the pipeline writes the confusion matrix and ROC curve to `results/figures/<organism>/<feature>/`. The completed study has **5 organisms × 6 features × 3 models = 90 combinations and 180 figures** for RF, SVM and XGBoost (the shipped `results/` folder). The reference values are in `results/metrics/results_summary.csv`.

## Recommendations for future development

1. **Fix and re-run PSSM.** Verify the PSI-BLAST outputs and the PSSM feature matrix (Part 6) before reporting PSSM results.
2. **Test the HIV-1 and SARS-CoV-2 results for leakage.** Re-split with a sequence-identity-aware method (cluster with CD-HIT or MMseqs2 at a fixed identity threshold and split by cluster, so no homologous protein appears in both train and test), and check that negatives were sampled from the same distribution as positives. If AUC falls sharply, the earlier near-perfect scores were a dataset effect.
3. **Complete the matrix.** Add CTriad and ProtT5, and the DNN/RNN/CNN models, so all eight features and six models are compared.
4. **Report more than accuracy and AUC.** With strong imbalance, add precision–recall AUC and Matthews correlation coefficient, and report confidence intervals from repeated or cross-validated splits.
5. **Control for the representation, not just the learner.** Use the same pair-construction rule and the same scaling across features, and tune each model comparably.
6. **Automate provenance.** Have `evaluate_model.py` write the metrics CSV directly, so figures and numbers can never drift apart.
