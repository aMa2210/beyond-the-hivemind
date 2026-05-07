"""
Figure 1 (paper): PCA projection of model responses for one prompt.
Left panel:  baseline (no persona, T=1)
Right panel: persona_t32 (persona + FTS)
Colors: one per model.

Picks the prompt with fewest missing embeddings across all models × configs.
Use --rank 2 to get the second-cleanest prompt (used for fig_pca_2.pdf).

Usage:
    python -m src.pipeline.plot_fig_pca
    python -m src.pipeline.plot_fig_pca --rank 2
"""
import json
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

from src.pipeline.fig_utils import MODELS, MODEL_LABELS, EMB_DIR, FIG_DIR

CONFIGS = ["baseline", "persona_t32"]
CONFIG_TITLES = {
    "baseline":    "Baseline (No Persona, T = 1)",
    "persona_t32": "Proposed (Persona + FTS, T = 32)",
}
EXPECTED = 50

MODEL_COLORS = [
    "#e74c3c",  # Llama    – red
    "#3498db",  # Mistral  – blue
    "#2ecc71",  # Qwen     – green
    "#f39c12",  # Gemma    – orange
    "#9b59b6",  # DeepSeek – purple
]


def _pick_cleanest_prompt(emb_dir=EMB_DIR, rank=1):
    """Pass 1: load only embedding counts to select the prompt with fewest failures.
    rank=1 returns the cleanest, rank=2 the second cleanest, etc."""
    counts = {}
    for model in MODELS:
        for config in CONFIGS:
            path = emb_dir / model / f"{config}.jsonl"
            result = {}
            if path.exists():
                with open(path) as f:
                    for line in f:
                        d = json.loads(line)
                        result[d["prompt"]] = len(d.get("embeddings", []))
            counts[(model, config)] = result
    all_prompts = sorted(set.intersection(*[set(v.keys()) for v in counts.values()]))
    scored = [
        (sum(EXPECTED - counts[(m, c)].get(prompt, 0) for m in MODELS for c in CONFIGS), prompt)
        for prompt in all_prompts
    ]
    scored.sort(key=lambda x: x[0])
    missing, prompt = scored[rank - 1]
    print(f"Selected prompt rank={rank} (total_missing={missing}):\n  {prompt[:100]}")
    return prompt


def _load_prompt_embeddings(prompt, emb_dir=EMB_DIR):
    """Pass 2: load full embedding vectors for ONE prompt only."""
    data = {}
    for model in MODELS:
        for config in CONFIGS:
            path = emb_dir / model / f"{config}.jsonl"
            data[(model, config)] = []
            if path.exists():
                with open(path) as f:
                    for line in f:
                        d = json.loads(line)
                        if d["prompt"] == prompt:
                            data[(model, config)] = d.get("embeddings", [])
                            break
    return data


def main(rank=1):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    prompt = _pick_cleanest_prompt(rank=rank)
    data = _load_prompt_embeddings(prompt)

    all_vecs, all_model_idx, all_config = [], [], []
    for ci, config in enumerate(CONFIGS):
        for mi, model in enumerate(MODELS):
            embs = data[(model, config)]
            if embs:
                all_vecs.append(np.array(embs, dtype=np.float32))
                all_model_idx.extend([mi] * len(embs))
                all_config.extend([ci] * len(embs))

    X = np.vstack(all_vecs)
    model_idx = np.array(all_model_idx)
    config_idx = np.array(all_config)

    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(X)
    xlabel, ylabel = "First Principal Component", "Second Principal Component"

    wrapped_prompt = "\n".join(textwrap.wrap(f'"{prompt}"', width=90))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.18, top=0.78, wspace=0.3)

    for ax, ci, config in zip(axes, range(2), CONFIGS):
        mask = config_idx == ci
        for mi, model in enumerate(MODELS):
            sel = mask & (model_idx == mi)
            ax.scatter(
                coords[sel, 0], coords[sel, 1],
                c=MODEL_COLORS[mi],
                label=MODEL_LABELS[model],
                alpha=0.7, s=25, linewidths=0,
            )
        ax.set_title(CONFIG_TITLES[config], fontsize=12, pad=6)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=8, labelleft=True)

    handles = [
        mpatches.Patch(color=MODEL_COLORS[mi], label=MODEL_LABELS[model])
        for mi, model in enumerate(MODELS)
    ]
    fig.legend(handles=handles, title="Model", loc="lower center",
               ncol=len(MODELS), bbox_to_anchor=(0.5, 0.0), fontsize=9)

    fig.suptitle(wrapped_prompt, fontsize=11, y=0.93)

    suffix = "" if rank == 1 else f"_{rank}"
    out = FIG_DIR / f"fig_pca{suffix}.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=1, help="1=cleanest, 2=second cleanest, ...")
    args = parser.parse_args()
    main(rank=args.rank)
