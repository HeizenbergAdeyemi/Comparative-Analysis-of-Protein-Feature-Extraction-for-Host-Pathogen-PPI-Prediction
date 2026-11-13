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
