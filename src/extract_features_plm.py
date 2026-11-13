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
