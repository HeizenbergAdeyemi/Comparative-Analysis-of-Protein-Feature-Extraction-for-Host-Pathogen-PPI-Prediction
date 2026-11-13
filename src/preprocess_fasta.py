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
