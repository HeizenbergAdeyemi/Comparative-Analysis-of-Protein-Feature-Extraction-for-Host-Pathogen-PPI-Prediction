# Part 6 — Error handling and troubleshooting

Each entry gives the symptom, the usual cause and the fix.

## Feature–label sample mismatches
- **Symptom:** `ValueError: Found input variables with inconsistent numbers of samples: [N, M]`, or silently shifted labels.
- **Cause:** sequences were dropped (failed PSSM, too long, invalid characters) from features but not from labels, or features and labels were sorted differently.
- **Fix:** key everything by protein/pair ID, drop failures from *both* sides in one step, and assert `X.shape[0] == len(y)` and ID equality before splitting. Log every dropped ID.

## Division-by-zero errors
- **Symptom:** `ZeroDivisionError`, `RuntimeWarning: invalid value encountered in divide`, `nan` in features or metrics.
- **Cause:** (a) composition features on sequences shorter than the window (`L − k + 1 ≤ 0`); (b) normalising an all-zero vector; (c) precision when a model predicts no positives.
- **Fix:** skip or flag sequences shorter than `k`; guard denominators (`max(denom, 1)`); use `zero_division=0` in `precision_score`, `recall_score` and `f1_score`; finish with `assert np.isfinite(X).all()`.

## PSSM and PSI-BLAST configuration
- **Symptom:** missing/empty PSSM files, `command not found: psiblast`, "BLAST Database error", or PSSM features that are constant.
- **Cause:** BLAST+ not installed or not on `PATH`; database path wrong; PSSM not requested in the output; failed runs silently converted to zeros.
- **Fix:** install BLAST+ and check `psiblast -version`; run with an ASCII PSSM output (`-out_ascii_pssm`), a fixed iteration count (`-num_iterations`) and the database (`-db`); treat a missing or empty PSSM as an error rather than substituting zeros.

### Diagnosing the PSSM collapse seen in the results
All 15 PSSM models (RF, SVM, XGBoost × 5 organisms) predict a single class and have ROC-AUC 0.50 (`results/summary/pssm_collapse_confusion_matrices.png`). Three unrelated learners failing identically means the input carries no usable signal. Check, in order:

1. `np.unique(X, axis=0).shape[0]`: if it is 1 or very small, rows are identical.
2. `X.std(axis=0).max()`: if it is ~0, the matrix is constant (often all zeros from failed PSI-BLAST runs).
3. Fraction of proteins with a valid PSSM file vs. total.
4. Whether the aggregation step runs on the parsed matrix or on an empty placeholder.

This is a likely explanation, not a confirmed one; fix it before reporting PSSM performance.

## SwissProt database configuration
- **Symptom:** PSI-BLAST reports it cannot find or open the database.
- **Cause:** database not downloaded, partially downloaded, or `BLASTDB` unset.
- **Fix:** download with BLAST+'s `update_blastdb.pl --decompress swissprot`; set `BLASTDB` to the directory or pass the full path to `-db`; verify with `blastdbcmd -db swissprot -info`.

## PLM tokenizer and embedding issues
- **Symptom:** tokenizer errors, `index out of range`, CUDA out-of-memory, embeddings of the wrong width, or `nan` embeddings.
- **Cause:** residues not space-separated for ProtBert/ProtT5; rare residues (`U, Z, O, B`) not mapped to `X`; sequences longer than the model context; batch too large; wrong pooling over padding tokens.
- **Fix:** preprocess exactly as the model card specifies; truncate or chunk long sequences; embed with a small batch (or one sequence at a time) under `torch.no_grad()`; mask padding when mean-pooling; assert the embedding width matches the model.

## TensorFlow CUDA compatibility
- **Symptom:** `Could not load dynamic library 'libcudart...'`, "Cannot dlopen some GPU libraries", or TensorFlow silently falling back to CPU.
- **Cause:** TensorFlow built for a different CUDA/cuDNN than the installed driver stack; TensorFlow and PyTorch needing different CUDA versions in one environment.
- **Fix:** install a TensorFlow build whose CUDA requirement matches the driver (the `tensorflow[and-cuda]` extra bundles it), or keep TensorFlow and PyTorch in separate environments; to force CPU, set `CUDA_VISIBLE_DEVICES=-1`. Confirm with `tf.config.list_physical_devices('GPU')`.

## Missing dependencies and imports
- **Symptom:** `ModuleNotFoundError`, `ImportError`, `cannot import name ...`.
- **Cause:** package not installed, version mismatch, or circular imports between modules.
- **Fix:** install from `requirements.txt` into a clean environment; pin versions with `pip freeze`; keep shared helpers in `feature_modules.py` so the extraction scripts import from it and not from each other.

> **TODO (author):** replace the generic symptoms above with the exact error messages and fixes you encountered, where they differ.
