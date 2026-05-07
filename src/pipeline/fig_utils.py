"""Shared utilities for figure scripts (fig1–fig3)."""

import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

EMB_DIR  = BASE_DIR / "data/embeddings"
QUAL_DIR = BASE_DIR / "data/quality_scores"
FIG_DIR  = BASE_DIR / "figures"

# Embedding model used in the paper: text-embedding-3-small.
# Dicts kept as single-entry for compatibility with plot_*.py CLI flags.
EMB_ROOTS = {"small": BASE_DIR / "data/embeddings"}
FIG_DIRS  = {"small": BASE_DIR / "figures"}

MODELS = [
    "meta-llama_Llama-3.1-8B-Instruct",
    "mistralai_Mistral-7B-Instruct-v0.3",
    "Qwen_Qwen2.5-14B-Instruct",
    "google_gemma-4-E4B-it",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B",
]

MODEL_LABELS = {
    "meta-llama_Llama-3.1-8B-Instruct":        "Llama-3.1-8B",
    "mistralai_Mistral-7B-Instruct-v0.3":       "Mistral-7B",
    "Qwen_Qwen2.5-14B-Instruct":                "Qwen2.5-14B",
    "google_gemma-4-E4B-it":                    "Gemma-4-4B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-R1-14B",
}

# config_name -> (persona: bool, temperature: int)
CONFIG_META = {
    "baseline":      (False, 1),
    "no_persona_t2": (False, 2),
    "no_persona_t4": (False, 4),
    "no_persona_t8": (False, 8),
    "no_persona_t16":(False, 16),
    "no_persona_t32":(False, 32),
    "persona_t1":    (True,  1),
    "persona_t2":    (True,  2),
    "persona_t4":    (True,  4),
    "persona_t8":    (True,  8),
    "persona_t16":   (True,  16),
    "persona_t32":   (True,  32),
}

TEMPERATURES = [1, 2, 4, 8, 16, 32]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_ircs(embeddings):
    """Mean pairwise cosine similarity excluding self-pairs (intra-group IRCS)."""
    emb = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = emb / np.maximum(norms, 1e-10)
    sim = normed @ normed.T
    N = len(emb)
    return float((sim.sum() - np.trace(sim)) / (N * (N - 1)))


def compute_intermodel_ircs(emb_a, emb_b):
    """Mean cosine similarity between all pairs from two different sets."""
    a = np.array(emb_a, dtype=np.float32)
    b = np.array(emb_b, dtype=np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-10)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-10)
    return float((a @ b.T).mean())


def load_embeddings_by_prompt(model_dir, config, emb_root=EMB_DIR):
    """Return dict {prompt: embeddings_list} or None if file missing."""
    path = emb_root / model_dir / f"{config}.jsonl"
    if not path.exists():
        return None
    return {row["prompt"]: row["embeddings"] for row in load_jsonl(path)}


def load_persona_embeddings_by_prompt(model_dir, config, emb_root=EMB_DIR):
    """Return dict {prompt: persona_embeddings_list} or None."""
    path = emb_root / model_dir / f"{config}.jsonl"
    if not path.exists():
        return None
    rows = load_jsonl(path)
    if not rows or "persona_embeddings" not in rows[0]:
        return None
    return {row["prompt"]: row["persona_embeddings"] for row in rows}


def load_quality_by_prompt(model_dir, config):
    """Return dict {prompt: mean_score} or None if file missing."""
    path = QUAL_DIR / model_dir / f"{config}.jsonl"
    if not path.exists():
        return None
    return {row["prompt"]: row["mean_score"] for row in load_jsonl(path)}


def mean_ircs(emb_by_prompt):
    """Mean IRCS across all prompts."""
    if emb_by_prompt is None:
        return None
    vals = [compute_ircs(embs) for embs in emb_by_prompt.values()]
    return float(np.mean(vals)) if vals else None


def mean_quality(qual_by_prompt):
    """Mean quality score across all prompts."""
    if qual_by_prompt is None:
        return None
    vals = list(qual_by_prompt.values())
    return float(np.mean(vals)) if vals else None
