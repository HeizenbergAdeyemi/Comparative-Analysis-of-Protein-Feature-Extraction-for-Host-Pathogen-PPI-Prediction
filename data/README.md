# Data

Pass this folder to the pipeline with `--data-root data`. Each organism group has its own sub-folder containing four FASTA files:

```
data/<organism>/
    interacting_host.fasta
    interacting_pathogen.fasta
    non_interacting_host.fasta
    non_interacting_pathogen.fasta
```

Organism identifiers: `bacillus`, `plant`, `human_hiv1`, `human_sars_cov2`, `plasmodium`.

Records are paired **by their order** in the matching host/pathogen files, so record *i* of `interacting_host.fasta` interacts with record *i* of `interacting_pathogen.fasta` (the loader refuses files whose record counts differ). Extensions `.fasta`, `.fa` and `.faa` are accepted.

FASTA inputs are git-ignored by default (see `.gitignore`).

> **TODO (author):** add the source database or publication, version/download date and licence for each of the five datasets, and how the non-interacting pairs were generated.
