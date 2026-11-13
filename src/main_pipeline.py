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
