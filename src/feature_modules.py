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
