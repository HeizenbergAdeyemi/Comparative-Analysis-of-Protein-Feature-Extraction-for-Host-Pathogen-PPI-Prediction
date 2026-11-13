# Part 6 — Error handling and troubleshooting

Each entry gives the symptom (with the exact message from the code where there is one), the cause and the fix.

## Feature–label sample mismatches
- **Messages:** `Feature/label mismatch for <organism>/<method>: N rows vs M labels.` (`main_pipeline.py`), `X/y mismatch: X=…, y=….` (`train_model.py`), `<host_key>/<pathogen_key> counts differ … Pairing by file order would be unsafe` (`preprocess_fasta.py`).
- **Cause:** features and labels went out of step, or the host and pathogen FASTA files have different numbers of records.
- **How the code prevents it:** `extract_traditional_features` returns `(X, retained_samples)` and a pair is retained only if both partner vectors were generated; labels are rebuilt from the retained samples. The PLM path does not drop samples, so a bad sequence aborts that method instead.
- **Fix for count errors:** make each host/pathogen file pair have the same number of records in the same order (an empty record is dropped by the reader, so one empty sequence on one side shifts the counts).

## Division-by-zero errors
- **Where handled:** `_normalise` in `feature_modules.py` returns a zero vector when the denominator is ≤ 0; `conjoint_triad` only divides when there is at least one valid triad; `_pool_hidden` clamps the denominator to ≥ 1; the metrics use `zero_division=0`; ROC-AUC is `NaN` when only one class is present.
- **Side effect to watch:** a sequence shorter than the window (for example fewer than 3 residues for k = 3) silently becomes an all-zero vector rather than an error. Count all-zero rows after extraction if short sequences are possible.

## PSSM and PSI-BLAST configuration
- **Messages and causes:**
  - `PSSM requires a local SwissProt BLAST database prefix.`: `--pssm-database` was not given.
  - `'psiblast' executable not found. Install NCBI BLAST+ …`: BLAST+ missing or not on `PATH` (or pass `--psiblast /full/path/psiblast`).
  - `PSI-BLAST failed. Check the database prefix and BLAST+ installation.`: wrong database prefix, unreadable database, or a timeout (600 s per protein).
  - `No residue score rows parsed from PSSM`: the ASCII PSSM was empty or in an unexpected format.
  - `PSSM produced zero feature rows. Do not train a model on empty features.`: every pair failed.
- **Behaviour:** failed pairs are excluded and the first 10 reasons are printed; the run stops for that feature only if nothing survives.
- **Speed:** PSI-BLAST runs twice per pair, and the cache is keyed by `sample_id`, so a protein that appears in many pairs is searched again each time. Caching by sequence hash would remove most of the work.

### Diagnosing the PSSM collapse in the shipped results
All 15 PSSM models (RF, SVM, XGBoost × 5 organisms) predict one class and have AUC 0.50 (`results/summary/pssm_collapse_confusion_matrices.png`). Three unrelated learners failing identically means the input carried no usable signal. The current `extract_features_traditional.py` cannot silently produce an uninformative matrix: failed proteins are excluded, non-finite vectors are rejected, and an empty result raises. So the collapse most likely came from the earlier pipeline version that produced the shipped figures (for example, failed PSSM runs replaced by zeros). To confirm on any run, check:

1. `np.unique(X, axis=0).shape[0]`: 1 or very small means identical rows.
2. `X.std(axis=0).max()`: about 0 means a constant matrix.
3. The fraction of pairs retained versus total (printed as "excluded N samples").

## SwissProt database configuration
- **Cause:** the database was not downloaded, is incomplete, or `--pssm-database` points at the archive instead of the database. The CLI help states it must be the *prefix*, not the downloaded `tar.gz` filename.
- **Fix:** download with `update_blastdb.pl --decompress swissprot`, then pass the path without any extension, for example `--pssm-database /data/blastdb/swissprot`. Verify with `blastdbcmd -db /data/blastdb/swissprot -info`.

## PLM tokenizer and embedding issues
- **Messages:** `Empty protein sequence.`, `Unknown PLM <key>; choose [...]`.
- **What the code does:** keeps letters only, inserts spaces between residues for ProtBert/ProtT5, truncates at `--max-length` tokens, mean-pools excluding special tokens and padding, and caches embeddings (the cache is accepted when the row count matches).
- **Known gaps:**
  - Rare residues `U, Z, O, B` are not mapped to `X`.
  - **ProtT5 will very likely fail as written.** It is loaded with `AutoModel`, which for T5 returns the full encoder–decoder `T5Model`; calling it with only `input_ids` and `attention_mask` raises an error asking for `decoder_input_ids`. This matches the absence of ProtT5 results. Fix: load `T5EncoderModel` and `T5Tokenizer` (requires `sentencepiece`) and map rare residues to `X` (snippet in Part 7). Test on two sequences before a full run.
  - With the cache check on row count only, delete `outputs/cache/` if you change checkpoints or preprocessing for the same sequences.
- **Memory:** lower `--plm-batch-size` if you hit out-of-memory errors.

## TensorFlow CUDA compatibility and the GPU switch
- **Behaviour:** `models.py` sets `CUDA_VISIBLE_DEVICES=-1` at import to avoid CUDA/PTX incompatibilities, unless `PPI_USE_GPU=1` is set.
- **Consequence:** `main_pipeline.py` imports `train_model` (and so `models.py`) at start-up, before any embedding runs, so, as far as the import order shows, PyTorch also sees no GPU and PLM embedding runs on CPU. This is much slower for ESM2/ProtBert/ProtT5. Set `PPI_USE_GPU=1` to use the GPU, and confirm with `torch.cuda.is_available()`.
- **Neural failures are non-fatal:** an exception while building or training a DNN/RNN/CNN prints `[<organism>/<feature>/<model>] skipped: …` and the run continues. If no DNN/RNN/CNN results appear, search the log for `skipped`.
- **Fix for CUDA errors:** match the TensorFlow build to the driver/CUDA version, or keep TensorFlow and PyTorch in separate environments; CPU-only (the default here) always works.

## Missing dependencies and imports
- `torch` and `transformers` are imported at the top of `extract_features_plm.py`, which `main_pipeline.py` imports, so the pipeline will not start without them even for traditional-only runs.
- `xgboost`: if absent, `xgboost is not installed; XGB will be omitted.` is printed and XGB is skipped.
- `tensorflow`: imported lazily; if absent the neural models are skipped (see above).
- Biopython: optional; a built-in FASTA parser is used if it is missing.
- `ModuleNotFoundError` for project modules: run from the repository root as `python src/main_pipeline.py`, so `src/` is on the import path.
