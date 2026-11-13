# Part 4 — Complete Python source codes

The pipeline consists of eight modules in [`src/`](../src/). Each section below gives the file's purpose, imports, functions, inputs and outputs, and notes, followed by the complete source.

## Module map and dependencies

| File | Responsibility | Imports from this project |
|---|---|---|
| `main_pipeline.py` | Coordinates the complete workflow | `preprocess_fasta`, `extract_features_traditional`, `extract_features_plm`, `train_model` |
| `preprocess_fasta.py` | Loads, cleans and prepares protein sequences | none |
| `feature_modules.py` | Implements the underlying feature calculations | none |
| `extract_features_traditional.py` | Extracts AAC, DPC, K-mer, CTriad and PSSM | `feature_modules` |
| `extract_features_plm.py` | Extracts ESM2, ProtBert and ProtT5 embeddings | none |
| `models.py` | Defines the classical models and DNN, RNN and CNN architectures | none |
| `train_model.py` | Handles model training | `models`, `evaluate_model` |
| `evaluate_model.py` | Evaluates models and generates visualisations | none |

Call chain: `main_pipeline` → (`preprocess_fasta`, `extract_features_*`) → `train_model` → (`models`, `evaluate_model`).

External packages: NumPy, scikit-learn, joblib, matplotlib (all runs); PyTorch and transformers (imported at start-up by `main_pipeline`); XGBoost (optional); TensorFlow (optional, neural models only); Biopython (optional); NCBI BLAST+ with a SwissProt database (PSSM only).

## `main_pipeline.py`

- **Purpose:** Coordinates the complete workflow.
- **Imports:** `argparse`, `extract_features_plm`, `extract_features_traditional`, `json`, `numpy`, `pathlib`, `preprocess_fasta`, `train_model`
- **Functions:** `run_pipeline(data_root='datasets', output_root='outputs', organisms=None, feature_methods=None, models=None, pssm_database=None, psiblast_executable='psiblast', k=3, plm_batch_size=8, max_length=1024, neural_epochs=30, device=None)`, `parse_args()`
- **Inputs → outputs:** Reads FASTA via `preprocess_fasta`; for every organism × feature method extracts features, validates alignment and calls `train_and_evaluate`. Writes `outputs/<organism>_pipeline_summary.json` and `outputs/pipeline_summary.json`.
- **Notes:** Command-line entry point (`--data-root`, `--output-root`, `--organisms`, `--features`, `--models`, `--pssm-database`, `--psiblast`, `--k`, `--plm-batch-size`, `--max-length`, `--neural-epochs`, `--device`). Errors in one feature method are caught and stored as `{"error": ...}` so the rest of the run continues. It imports `torch` at start-up (through `extract_features_plm`), so PyTorch must be installed even for traditional-only runs.

<details>
<summary>Full source: <code>main_pipeline.py</code> (102 lines)</summary>

```python
"""Entry point for the host–pathogen PPI feature extraction and ML pipeline."""
from pathlib import Path
import argparse
import json
import numpy as np

from preprocess_fasta import DEFAULT_ORGANISMS, load_organism_pairs
from extract_features_traditional import (
    TRADITIONAL_METHODS, extract_traditional_features
)
from extract_features_plm import PLM_MODELS, extract_plm_features
from train_model import train_and_evaluate, ALL_MODELS

def run_pipeline(data_root="datasets", output_root="outputs",
                 organisms=None, feature_methods=None, models=None,
                 pssm_database=None, psiblast_executable="psiblast",
                 k=3, plm_batch_size=8, max_length=1024,
                 neural_epochs=30, device=None):
    organisms = organisms or DEFAULT_ORGANISMS
    feature_methods = feature_methods or TRADITIONAL_METHODS + list(PLM_MODELS)
    models = models or ALL_MODELS
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    overall = {}

    for organism in organisms:
        print(f"\n{'='*72}\nProcessing organism: {organism}\n{'='*72}")
        samples = load_organism_pairs(data_root, organism)
        organism_results = {}

        for method in feature_methods:
            print(f"\n--- Feature method: {method} ---")
            try:
                if method.upper() in TRADITIONAL_METHODS:
                    X, retained = extract_traditional_features(
                        samples, method, k=k, pssm_database=pssm_database,
                        psiblast_executable=psiblast_executable,
                        cache_dir=output_root / "cache" / organism / method
                    )
                elif method in PLM_MODELS:
                    X, retained = extract_plm_features(
                        samples, method, batch_size=plm_batch_size,
                        max_length=max_length, device=device,
                        cache_dir=output_root / "cache"
                    )
                else:
                    print(f"Unknown feature method '{method}'; skipping.")
                    continue

                y = np.asarray([s["label"] for s in retained])
                if X.shape[0] != len(y):
                    raise RuntimeError(
                        f"Feature/label mismatch for {organism}/{method}: "
                        f"{X.shape[0]} rows vs {len(y)} labels."
                    )
                if len(np.unique(y)) != 2:
                    raise RuntimeError("Both classes must remain after feature extraction.")
                method_results = train_and_evaluate(
                    X, retained, organism, method, output_root=output_root,
                    models_to_run=models, epochs=neural_epochs
                )
                organism_results[method] = method_results
            except Exception as exc:
                print(f"[ERROR] {organism}/{method}: {type(exc).__name__}: {exc}")
                organism_results[method] = {"error": str(exc)}

        overall[organism] = organism_results
        with (output_root / f"{organism}_pipeline_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(organism_results, handle, indent=2, allow_nan=True)

    with (output_root / "pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2, allow_nan=True)
    return overall

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--organisms", nargs="+", default=DEFAULT_ORGANISMS)
    parser.add_argument("--features", nargs="+",
                        default=TRADITIONAL_METHODS + list(PLM_MODELS))
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--pssm-database", default=None,
                        help="BLAST database prefix, not the downloaded tar.gz filename.")
    parser.add_argument("--psiblast", default="psiblast")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--plm-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--neural-epochs", type=int, default=30)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_root=args.data_root, output_root=args.output_root,
        organisms=args.organisms, feature_methods=args.features,
        models=args.models, pssm_database=args.pssm_database,
        psiblast_executable=args.psiblast, k=args.k,
        plm_batch_size=args.plm_batch_size, max_length=args.max_length,
        neural_epochs=args.neural_epochs, device=args.device
    )
```

</details>

## `preprocess_fasta.py`

- **Purpose:** Loads, cleans and prepares protein sequences.
- **Imports:** `Bio`, `pathlib`, `typing`, `warnings`
- **Functions:** `read_fasta(path)`, `_find_fasta(directory, stem)`, `load_organism_pairs(data_root='datasets', organism='plant')`, `load_all_organisms(data_root='datasets', organisms=None)`
- **Module constants:** `DEFAULT_ORGANISMS`, `FASTA_NAMES`
- **Inputs → outputs:** `datasets/<organism>/{interacting,non_interacting}_{host,pathogen}.fasta` → list of pair dictionaries (`sample_id`, `host_id`, `pathogen_id`, `host_sequence`, `pathogen_sequence`, `label`).
- **Notes:** Pairs host and pathogen records by file order and raises if counts differ. Biopython is optional (fallback parser included). Upper-cases sequences, strips `*`, drops empty records.

<details>
<summary>Full source: <code>preprocess_fasta.py</code> (116 lines)</summary>

```python
"""Load and validate paired host/pathogen FASTA files for PPI prediction."""
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

try:
    from Bio import SeqIO
except ImportError:
    SeqIO = None

DEFAULT_ORGANISMS = [
    "bacillus", "plant", "human_hiv1", "human_sars_cov2", "plasmodium"
]
FASTA_NAMES = {
    "interacting_host": "interacting_host",
    "interacting_pathogen": "interacting_pathogen",
    "non_interacting_host": "non_interacting_host",
    "non_interacting_pathogen": "non_interacting_pathogen",
}

def read_fasta(path: Path) -> List[Tuple[str, str]]:
    """Return (record_id, uppercase_sequence) records from a FASTA file."""
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    records = []
    if SeqIO is not None:
        for record in SeqIO.parse(str(path), "fasta"):
            seq = str(record.seq).upper().replace("*", "")
            if seq:
                records.append((str(record.id), seq))
    else:
        header, chunks = None, []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None and chunks:
                        records.append((header, "".join(chunks).upper().replace("*", "")))
                    header, chunks = line[1:].split()[0], []
                else:
                    chunks.append(line)
            if header is not None and chunks:
                records.append((header, "".join(chunks).upper().replace("*", "")))
    return [(rid, seq) for rid, seq in records if seq]

def _find_fasta(directory: Path, stem: str) -> Path:
    """Accept exact stem, .fasta/.fa/.faa extensions, and common FASTA suffixes."""
    candidates = [
        directory / stem, directory / f"{stem}.fasta",
        directory / f"{stem}.fa", directory / f"{stem}.faa",
        directory / f"{stem}.fasta.gz", directory / f"{stem}.fa.gz",
    ]
    for candidate in candidates:
        if candidate.exists() and not candidate.name.endswith(".gz"):
            return candidate
    matches = sorted(p for p in directory.glob(f"{stem}*")
                     if p.is_file() and not p.name.endswith(".gz"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not locate '{stem}' in {directory}. Expected {stem}.fasta (or .fa/.faa)."
    )

def load_organism_pairs(data_root="datasets", organism="plant"):
    """
    Load positive and negative host/pathogen pairs.

    Expected directory:
      datasets/<organism>/
        interacting_host.fasta
        interacting_pathogen.fasta
        non_interacting_host.fasta
        non_interacting_pathogen.fasta

    Records are paired by their order in the corresponding files. If your files
    use explicit pair identifiers, ensure the two files have matching order.
    """
    directory = Path(data_root) / organism
    if not directory.is_dir():
        raise FileNotFoundError(f"Organism directory not found: {directory}")

    def load_pair(host_key, pathogen_key, label):
        host_records = read_fasta(_find_fasta(directory, host_key))
        pathogen_records = read_fasta(_find_fasta(directory, pathogen_key))
        if len(host_records) != len(pathogen_records):
            raise ValueError(
                f"{organism}: {host_key}/{pathogen_key} counts differ "
                f"({len(host_records)} vs {len(pathogen_records)}). "
                "Pairing by file order would be unsafe; correct the input files."
            )
        pairs = []
        for i, ((hid, hseq), (pid, pseq)) in enumerate(zip(host_records, pathogen_records)):
            pairs.append({
                "sample_id": f"{organism}_{'positive' if label else 'negative'}_{i+1}",
                "host_id": hid, "pathogen_id": pid,
                "host_sequence": hseq, "pathogen_sequence": pseq,
                "label": int(label),
            })
        return pairs

    positive = load_pair("interacting_host", "interacting_pathogen", 1)
    negative = load_pair("non_interacting_host", "non_interacting_pathogen", 0)
    samples = positive + negative
    if not positive or not negative:
        raise ValueError(f"{organism}: both positive and negative pairs are required.")
    warnings.warn(
        f"{organism}: loaded {len(positive)} positive and {len(negative)} negative pairs.",
        RuntimeWarning
    )
    return samples

def load_all_organisms(data_root="datasets", organisms=None):
    organisms = organisms or DEFAULT_ORGANISMS
    return {org: load_organism_pairs(data_root, org) for org in organisms}
```

</details>

## `feature_modules.py`

- **Purpose:** Implements the underlying feature calculations.
- **Imports:** `collections`, `itertools`, `numpy`
- **Functions:** `clean_sequence(sequence)`, `_normalise(counts, denominator)`, `amino_acid_composition(sequence)`, `dipeptide_composition(sequence)`, `kmer_composition(sequence, k=3)`, `conjoint_triad(sequence, gap=0)`, `pssm_summary(pssm)`
- **Module constants:** `AMINO_ACIDS`, `AA_SET`, `AA_INDEX`, `_CT_GROUPS`
- **Inputs → outputs:** Protein sequence string → NumPy vector (AAC 20, DPC 400, k-mer 20^k, CTriad 343, PSSM summary 40).
- **Notes:** `clean_sequence` keeps only canonical residues. `_normalise` returns an all-zero vector when the denominator is zero (for example a sequence shorter than `k`), rather than raising.

<details>
<summary>Full source: <code>feature_modules.py</code> (81 lines)</summary>

```python
"""Low-level protein sequence descriptors used by the traditional pipeline."""
from collections import Counter
from itertools import product
import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AMINO_ACIDS)
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def clean_sequence(sequence):
    """Keep canonical amino acids; remove gaps, stop symbols and ambiguous residues."""
    return "".join(aa for aa in str(sequence).upper() if aa in AA_SET)

def _normalise(counts, denominator):
    if denominator <= 0:
        return np.zeros(len(counts), dtype=np.float32)
    return np.asarray(counts, dtype=np.float32) / float(denominator)

def amino_acid_composition(sequence):
    """20-dimensional AAC vector."""
    seq = clean_sequence(sequence)
    counts = [seq.count(aa) for aa in AMINO_ACIDS]
    return _normalise(counts, len(seq))

def dipeptide_composition(sequence):
    """400-dimensional overlapping dipeptide frequency vector."""
    seq = clean_sequence(sequence)
    keys = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
    counts = Counter(seq[i:i+2] for i in range(max(0, len(seq)-1)))
    return _normalise([counts[k] for k in keys], max(0, len(seq)-1))

def kmer_composition(sequence, k=3):
    """Canonical k-mer frequency vector; default k=3 (8,000 dimensions)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    seq = clean_sequence(sequence)
    keys = ["".join(p) for p in product(AMINO_ACIDS, repeat=k)]
    counts = Counter(seq[i:i+k] for i in range(max(0, len(seq)-k+1)))
    return _normalise([counts[key] for key in keys], max(0, len(seq)-k+1))

# Standard Conjoint Triad grouping (Shen et al.); 7 physicochemical groups.
_CT_GROUPS = {
    **{aa: 1 for aa in "AGV"},
    **{aa: 2 for aa in "ILFP"},
    **{aa: 3 for aa in "YMTS"},
    **{aa: 4 for aa in "HNQW"},
    **{aa: 5 for aa in "RK"},
    **{aa: 6 for aa in "DE"},
    **{aa: 7 for aa in "C"},
}

def conjoint_triad(sequence, gap=0):
    """343-dimensional normalized Conjoint Triad descriptor."""
    if gap < 0:
        raise ValueError("gap must be non-negative")
    seq = clean_sequence(sequence)
    mapped = [_CT_GROUPS.get(aa) for aa in seq]
    counts = np.zeros(7 ** 3, dtype=np.float32)
    step = gap + 1
    valid_triplets = 0
    for i in range(max(0, len(mapped) - 2 * step)):
        triad = mapped[i], mapped[i + step], mapped[i + 2 * step]
        if None in triad:
            continue
        index = (triad[0]-1) * 49 + (triad[1]-1) * 7 + (triad[2]-1)
        counts[index] += 1
        valid_triplets += 1
    if valid_triplets:
        counts /= valid_triplets
    return counts

def pssm_summary(pssm):
    """
    Convert a residue-by-20 PSI-BLAST PSSM matrix to 40 features:
    per-amino-acid mean and standard deviation across residues.
    """
    matrix = np.asarray(pssm, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != 20:
        raise ValueError(f"Invalid PSSM matrix shape: {matrix.shape}")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([matrix.mean(axis=0), matrix.std(axis=0)]).astype(np.float32)
```

</details>

## `extract_features_traditional.py`

- **Purpose:** Extracts AAC, DPC, K-mer, CTriad and PSSM.
- **Imports:** `feature_modules`, `numpy`, `os`, `pathlib`, `shutil`, `subprocess`, `tempfile`
- **Functions:** `_pair_descriptor(function, host, pathogen)`, `_run_psiblast(sequence, database, psiblast_executable='psiblast', iterations=3, evalue=0.001, timeout=600)`, `shutil_which(executable)`, `_parse_ascii_pssm(path)`, `extract_traditional_features(samples, method, k=3, pssm_database=None, psiblast_executable='psiblast', cache_dir=None)`
- **Module constants:** `TRADITIONAL_METHODS`
- **Inputs → outputs:** List of pairs + method name → `(X, retained_samples)`, where each row is `[host ‖ pathogen]`.
- **Notes:** A pair is kept only if both partner vectors are generated and finite; failures are counted and the first 10 are printed. Raises if no row survives. PSSM runs `psiblast` per protein (3 iterations, e-value 0.001) and caches the pair result per `sample_id`.

<details>
<summary>Full source: <code>extract_features_traditional.py</code> (131 lines)</summary>

```python
"""Pair-level traditional feature extraction, including real PSI-BLAST PSSMs."""
from pathlib import Path
import os
import subprocess
import tempfile
import numpy as np
from feature_modules import (
    amino_acid_composition, dipeptide_composition, kmer_composition,
    conjoint_triad, pssm_summary, clean_sequence
)

TRADITIONAL_METHODS = ["AAC", "DPC", "KMER", "CTRIAD", "PSSM"]

def _pair_descriptor(function, host, pathogen, **kwargs):
    return np.concatenate([function(host, **kwargs), function(pathogen, **kwargs)])

def _run_psiblast(sequence, database, psiblast_executable="psiblast",
                  iterations=3, evalue=0.001, timeout=600):
    """Run PSI-BLAST and parse its ASCII PSSM output."""
    seq = clean_sequence(sequence)
    if not seq:
        raise ValueError("Cannot generate PSSM for an empty/invalid sequence.")
    if not database:
        raise ValueError("PSSM requires a local SwissProt BLAST database prefix.")
    output = shutil_which(psiblast_executable)
    if output is None:
        raise FileNotFoundError(
            f"'{psiblast_executable}' executable not found. Install NCBI BLAST+ "
            "and ensure psiblast is available on PATH."
        )
    with tempfile.TemporaryDirectory(prefix="ppi_pssm_") as tmp:
        query = Path(tmp) / "query.fasta"
        pssm_file = Path(tmp) / "query.pssm"
        query.write_text(f">query\n{seq}\n", encoding="utf-8")
        command = [
            output, "-query", str(query), "-db", str(database),
            "-num_iterations", str(iterations), "-evalue", str(evalue),
            "-out_ascii_pssm", str(pssm_file), "-out", os.devnull,
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0 or not pssm_file.exists() or pssm_file.stat().st_size == 0:
            raise RuntimeError(
                "PSI-BLAST failed. Check the database prefix and BLAST+ installation. "
                f"stderr: {result.stderr[-1500:]}"
            )
        matrix = _parse_ascii_pssm(pssm_file)
        return pssm_summary(matrix)

def shutil_which(executable):
    from shutil import which
    return which(executable)

def _parse_ascii_pssm(path):
    """Parse the 20 score columns from an NCBI ASCII PSSM file."""
    rows = []
    started = False
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 22 and parts[0].isdigit():
                try:
                    values = [float(x) for x in parts[2:22]]
                except ValueError:
                    continue
                rows.append(values)
                started = True
            elif started and rows:
                break
    if not rows:
        raise ValueError(f"No residue score rows parsed from PSSM: {path}")
    return np.asarray(rows, dtype=np.float32)

def extract_traditional_features(samples, method, k=3, pssm_database=None,
                                 psiblast_executable="psiblast", cache_dir=None):
    """
    Return (X, retained_samples). A sample is retained only if BOTH partner
    feature vectors are successfully generated. Labels are therefore aligned.
    """
    method = method.upper()
    if method not in TRADITIONAL_METHODS:
        raise ValueError(f"Unknown method {method}; choose from {TRADITIONAL_METHODS}")
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    feature_rows, retained = [], []
    failures = []
    for sample in samples:
        try:
            host, pathogen = sample["host_sequence"], sample["pathogen_sequence"]
            if method == "AAC":
                vector = _pair_descriptor(amino_acid_composition, host, pathogen)
            elif method == "DPC":
                vector = _pair_descriptor(dipeptide_composition, host, pathogen)
            elif method == "KMER":
                vector = _pair_descriptor(kmer_composition, host, pathogen, k=k)
            elif method == "CTRIAD":
                vector = _pair_descriptor(conjoint_triad, host, pathogen)
            else:
                cache_key = sample["sample_id"].replace("/", "_")
                cache_file = cache_dir / f"{cache_key}_pssm.npy" if cache_dir else None
                if cache_file and cache_file.exists():
                    host_vec, pathogen_vec = np.load(cache_file, allow_pickle=False)
                else:
                    host_vec = _run_psiblast(host, pssm_database, psiblast_executable)
                    pathogen_vec = _run_psiblast(pathogen, pssm_database, psiblast_executable)
                    if cache_file:
                        np.save(cache_file, np.stack([host_vec, pathogen_vec]))
                vector = np.concatenate([host_vec, pathogen_vec])
            if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                raise ValueError("Feature vector is not finite and one-dimensional.")
            feature_rows.append(vector.astype(np.float32))
            retained.append(sample)
        except Exception as exc:
            failures.append((sample.get("sample_id", "unknown"), str(exc)))

    if failures:
        print(f"[{method}] excluded {len(failures)} samples due to feature errors.")
        for sample_id, reason in failures[:10]:
            print(f"  - {sample_id}: {reason}")
        if len(failures) > 10:
            print(f"  ... and {len(failures)-10} additional failures.")
    if not feature_rows:
        raise RuntimeError(
            f"{method} produced zero feature rows. Do not train a model on empty features."
        )
    X = np.vstack(feature_rows)
    y_count = len(retained)
    if X.shape[0] != y_count:
        raise RuntimeError("Internal alignment error: feature rows and retained samples differ.")
    return X, retained
```

</details>

## `extract_features_plm.py`

- **Purpose:** Extracts ESM2, ProtBert and ProtT5 embeddings.
- **Imports:** `hashlib`, `numpy`, `pathlib`, `torch`, `transformers`
- **Functions:** `_prepare_sequence(sequence, model_name)`, `_pool_hidden(hidden, attention_mask, special_tokens_mask=None)`, `embed_sequences(sequences, model_key='ESM2', batch_size=8, max_length=1024, device=None, cache_path=None)`, `extract_plm_features(samples, model_key='ESM2', batch_size=8, max_length=1024, device=None, cache_dir='outputs/cache')`
- **Module constants:** `PLM_MODELS`
- **Inputs → outputs:** List of pairs + model key → `(X, samples)` with mean-pooled host and pathogen embeddings concatenated.
- **Notes:** Model checkpoints are defined in `PLM_MODELS`. Embeddings are cached as `.npy` under `outputs/cache/`, keyed by an MD5 signature of all sequences and the model name. Sequences are embedded per pair, not de-duplicated.

<details>
<summary>Full source: <code>extract_features_plm.py</code> (90 lines)</summary>

```python
"""Protein language model embeddings for paired host/pathogen sequences."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

PLM_MODELS = {
    "ESM2": "facebook/esm2_t6_8M_UR50D",
    "ProtBert": "Rostlab/prot_bert",
    "ProtT5": "Rostlab/prot_t5_xl_uniref50",
}

def _prepare_sequence(sequence, model_name):
    seq = "".join(ch for ch in str(sequence).upper() if ch.isalpha())
    if not seq:
        raise ValueError("Empty protein sequence.")
    # ProteinBERT tokenizer convention: whitespace-separated residues.
    if "prot_bert" in model_name.lower() or "prot_t5" in model_name.lower():
        seq = " ".join(seq)
    return seq

def _pool_hidden(hidden, attention_mask, special_tokens_mask=None):
    mask = attention_mask.bool()
    if special_tokens_mask is not None:
        mask = mask & (~special_tokens_mask.bool())
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (hidden * weights).sum(dim=1) / denom

def embed_sequences(sequences, model_key="ESM2", batch_size=8, max_length=1024,
                    device=None, cache_path=None):
    """Return one mean-pooled embedding per sequence."""
    if model_key not in PLM_MODELS:
        raise ValueError(f"Unknown PLM {model_key}; choose {list(PLM_MODELS)}")
    model_name = PLM_MODELS[model_key]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if cached.shape[0] == len(sequences):
            return cached

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    prepared = [_prepare_sequence(s, model_name) for s in sequences]
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start:start+batch_size]
            encoded = tokenizer(
                batch, add_special_tokens=True, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
                return_special_tokens_mask=True
            )
            special = encoded.pop("special_tokens_mask", None)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            if special is not None:
                special = special.to(device)
            output = model(**encoded)
            pooled = _pool_hidden(output.last_hidden_state,
                                  encoded["attention_mask"], special)
            embeddings.append(pooled.cpu().numpy().astype(np.float32))
            print(f"[{model_key}] embedded {min(start+len(batch), len(prepared))}/{len(prepared)}")
    result = np.vstack(embeddings)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, result)
    return result

def extract_plm_features(samples, model_key="ESM2", batch_size=8, max_length=1024,
                         device=None, cache_dir="outputs/cache"):
    """Concatenate host and pathogen embeddings while preserving sample order."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hosts = [s["host_sequence"] for s in samples]
    pathogens = [s["pathogen_sequence"] for s in samples]
    signature = hashlib.md5(
        ("|".join(hosts) + "||" + "|".join(pathogens) + model_key).encode()
    ).hexdigest()[:12]
    host_cache = cache_dir / f"{model_key}_{signature}_host.npy"
    pathogen_cache = cache_dir / f"{model_key}_{signature}_pathogen.npy"
    host_emb = embed_sequences(hosts, model_key, batch_size, max_length, device, host_cache)
    pathogen_emb = embed_sequences(pathogens, model_key, batch_size, max_length, device, pathogen_cache)
    if host_emb.shape[0] != pathogen_emb.shape[0]:
        raise RuntimeError("Host and pathogen embedding counts differ.")
    return np.concatenate([host_emb, pathogen_emb], axis=1), samples
```

</details>

## `models.py`

- **Purpose:** Defines the classical models and the DNN, RNN and CNN architectures.
- **Imports:** `os`, `sklearn.ensemble`, `sklearn.svm`, `tensorflow`, `tensorflow.keras`, `tensorflow.keras.layers`, `xgboost`
- **Functions:** `get_classical_models(random_state=42)`, `build_neural_model(model_name, input_dim, random_state=42)`
- **Inputs → outputs:** `get_classical_models(random_state)` → dict of RF, SVM, XGB; `build_neural_model(name, input_dim)` → compiled Keras model.
- **Notes:** Sets `CUDA_VISIBLE_DEVICES=-1` at import unless `PPI_USE_GPU=1`. XGBoost and TensorFlow are imported lazily.

<details>
<summary>Full source: <code>models.py</code> (72 lines)</summary>

```python
"""Classical ML and neural network model definitions."""
import os
# Avoid known CUDA/PTX incompatibilities by default; set PPI_USE_GPU=1 to opt in.
if os.environ.get("PPI_USE_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_classical_models(random_state=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    models = {
        "RF": RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            random_state=random_state, n_jobs=-1
        ),
        "SVM": SVC(
            C=1.0, kernel="rbf", probability=True,
            class_weight="balanced", random_state=random_state
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGB"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="logloss",
            random_state=random_state, n_jobs=-1
        )
    except ImportError:
        print("xgboost is not installed; XGB will be omitted.")
    return models

def build_neural_model(model_name, input_dim, random_state=42):
    """Build a compact binary classifier. TensorFlow is imported lazily."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D,
        GlobalMaxPooling1D, SimpleRNN
    )
    tf.keras.utils.set_random_seed(random_state)
    model_name = model_name.upper()
    model = Sequential(name=model_name)
    if model_name == "DNN":
        model.add(Input(shape=(input_dim,)))
        model.add(Dense(256, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation="sigmoid"))
    elif model_name in ("RNN", "CNN"):
        model.add(Input(shape=(input_dim, 1)))
        if model_name == "RNN":
            model.add(SimpleRNN(64, return_sequences=True))
            model.add(Dropout(0.3))
            model.add(SimpleRNN(32))
        else:
            model.add(Conv1D(64, 5, padding="same", activation="relu"))
            model.add(MaxPooling1D(2))
            model.add(Conv1D(32, 3, padding="same", activation="relu"))
            model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="sigmoid"))
    else:
        raise ValueError("Neural model must be DNN, RNN, or CNN.")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model
```

</details>

## `train_model.py`

- **Purpose:** Handles model training.
- **Imports:** `evaluate_model`, `joblib`, `json`, `models`, `numpy`, `pathlib`, `sklearn.base`, `sklearn.impute`, `sklearn.metrics`, `sklearn.model_selection`, `sklearn.pipeline`, `sklearn.preprocessing`, `tensorflow`
- **Functions:** `_scores(model, X)`, `train_and_evaluate(X, samples, organism, feature_name, output_root='outputs', models_to_run=None, test_size=0.2, random_state=42, epochs=30, batch_size=32)`
- **Module constants:** `ALL_MODELS`
- **Inputs → outputs:** `(X, samples, organism, feature_name)` → trained models saved to `outputs/<organism>/<feature>/`, evaluation files from `save_evaluation`, and a `summary_metrics.json`.
- **Notes:** Stratified 80/20 split of pairs (seed 42); classical models in an imputer–scaler–classifier `Pipeline`; neural models with early stopping. Saves `<MODEL>_model.joblib` or `<MODEL>_model.keras` (+ scaler).

<details>
<summary>Full source: <code>train_model.py</code> (107 lines)</summary>

```python
"""Train classical ML and neural classifiers with leakage-safe preprocessing."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from models import get_classical_models, build_neural_model
from evaluate_model import save_evaluation

ALL_MODELS = ["RF", "SVM", "XGB", "DNN", "RNN", "CNN"]

def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    return model.predict(X).astype(float)

def train_and_evaluate(X, samples, organism, feature_name, output_root="outputs",
                       models_to_run=None, test_size=0.2, random_state=42,
                       epochs=30, batch_size=32):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray([s["label"] for s in samples], dtype=np.int64)
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError(f"X/y mismatch: X={X.shape}, y={y.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Feature matrix contains NaN/Inf values.")
    if len(np.unique(y)) != 2:
        raise ValueError("Binary PPI training requires both classes.")
    if min(np.bincount(y)) < 2:
        raise ValueError("Each class needs at least two samples for stratified splitting.")

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    run_dir = Path(output_root) / organism / feature_name
    run_dir.mkdir(parents=True, exist_ok=True)

    requested = models_to_run or ALL_MODELS
    classical = get_classical_models(random_state)
    results = {}

    for name in requested:
        name = name.upper()
        if name in classical:
            # Scaling is fitted on training data only. Tree models tolerate scaling too.
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", clone(classical[name]))
            ])
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            score = _scores(pipeline, X_test)
            joblib.dump(pipeline, run_dir / f"{name}_model.joblib")
        elif name in ("DNN", "RNN", "CNN"):
            try:
                import tensorflow as tf
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_train).astype(np.float32)
                Xte = scaler.transform(X_test).astype(np.float32)
                neural = build_neural_model(name, X.shape[1], random_state)
                if name == "DNN":
                    train_input, test_input = Xtr, Xte
                else:
                    train_input, test_input = Xtr[..., None], Xte[..., None]
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=5, restore_best_weights=True
                    )
                ]
                neural.fit(
                    train_input, y_train, validation_split=0.15,
                    epochs=epochs, batch_size=batch_size, verbose=1,
                    callbacks=callbacks
                )
                score = neural.predict(test_input, verbose=0).ravel()
                pred = (score >= 0.5).astype(int)
                neural.save(run_dir / f"{name}_model.keras")
                joblib.dump(scaler, run_dir / f"{name}_scaler.joblib")
            except Exception as exc:
                print(f"[{organism}/{feature_name}/{name}] skipped: {exc}")
                continue
        else:
            print(f"Model {name} unavailable or not recognised; skipping.")
            continue

        metrics = save_evaluation(
            y_test, pred, score, run_dir, f"{feature_name}_{name}"
        )
        metrics["n_train"] = int(len(train_idx))
        metrics["n_test"] = int(len(test_idx))
        results[name] = metrics
        print(f"{organism} | {feature_name} | {name}: {metrics}")

    with (run_dir / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=True)
    return results
```

</details>

## `evaluate_model.py`

- **Purpose:** Evaluates models and generates visualisations.
- **Imports:** `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `pathlib`, `sklearn.metrics`
- **Functions:** `evaluate_predictions(y_true, y_pred, y_score)`, `save_evaluation(y_true, y_pred, y_score, output_dir, prefix)`
- **Inputs → outputs:** `(y_true, y_pred, y_score, output_dir, prefix)` → accuracy, precision, recall, F1, ROC-AUC; writes `<prefix>_confusion_matrix.png`, `<prefix>_roc_curve.png` and `<prefix>_metrics.json`.
- **Notes:** Uses `zero_division=0`; ROC-AUC is NaN and the ROC plot is skipped if only one class is present in the test set. Figures are 300 dpi.

<details>
<summary>Full source: <code>evaluate_model.py</code> (55 lines)</summary>

```python
"""Metrics and publication-friendly evaluation plots."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

def evaluate_predictions(y_true, y_pred, y_score):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, y_score))
        if len(np.unique(y_true)) == 2 else float("nan")
    )
    return metrics

def save_evaluation(y_true, y_pred, y_score, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_predictions(y_true, y_pred, y_score)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Non-interacting", "Interacting"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion matrix: {prefix}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=300)
    plt.close(fig)

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title=f"ROC curve: {prefix}")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_roc_curve.png", dpi=300)
        plt.close(fig)

    with (output_dir / f"{prefix}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics
```

</details>
