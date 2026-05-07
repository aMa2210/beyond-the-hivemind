"""
Figure 3: Inter-model IRCS matrix (N×N).
  Upper triangle: baseline (no persona, T=1) inter-model IRCS.
  Lower triangle: persona_t32 (persona, T=32) inter-model IRCS.
  Diagonal:       intra-model IRCS under baseline condition.

  Inter-model IRCS for models A and B = mean cosine similarity over all
  N_A × N_B response pairs, averaged across all shared prompts.

Usage:
    python -m src.pipeline.plot_fig3
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.pipeline.fig_utils import (
    MODELS, MODEL_LABELS, FIG_DIR, EMB_DIR, EMB_ROOTS, FIG_DIRS,
    load_embeddings_by_prompt,
    compute_ircs, compute_intermodel_ircs,
)

UPPER_CONFIG = "baseline"
LOWER_CONFIG = "persona_t32"


def compute_matrix(config, emb_root=EMB_DIR):
    """Return (N, N) matrix of pairwise inter-model IRCS; diagonal = intra-model IRCS."""
    n = len(MODELS)
    all_embs = {m: load_embeddings_by_prompt(m, config, emb_root=emb_root) for m in MODELS}

    # Intersect available prompts across all models
    prompt_sets = [set(d.keys()) for d in all_embs.values() if d is not None]
    common_prompts = sorted(set.intersection(*prompt_sets))
    print(f"  [{config}] {len(common_prompts)} common prompts")

    mat = np.full((n, n), np.nan)
    for i, mi in enumerate(MODELS):
        for j, mj in enumerate(MODELS):
            vals = []
            for p in common_prompts:
                ei, ej = all_embs[mi][p], all_embs[mj][p]
                vals.append(compute_ircs(ei) if i == j else compute_intermodel_ircs(ei, ej))
            mat[i, j] = float(np.mean(vals))
    return mat


def main(emb_root=EMB_DIR, fig_dir=FIG_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Computing upper-triangle matrix (baseline)...")
    upper = compute_matrix(UPPER_CONFIG, emb_root=emb_root)
    print("Computing lower-triangle matrix (persona_t32)...")
    lower = compute_matrix(LOWER_CONFIG, emb_root=emb_root)

    n = len(MODELS)
    combined = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i < j:
                combined[i, j] = upper[i, j]   # upper triangle from baseline
            elif i > j:
                combined[i, j] = lower[i, j]   # lower triangle from persona_t32
            else:
                combined[i, j] = upper[i, i]   # diagonal: intra-model (baseline)

    vmin, vmax = float(np.nanmin(combined)), float(np.nanmax(combined))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    masked = np.ma.array(combined, mask=np.isnan(combined))
    im = ax.imshow(masked, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")

    labels = [MODEL_LABELS[m] for m in MODELS]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate all cells
    for i in range(n):
        for j in range(n):
            val = combined[i, j]
            if not np.isnan(val):
                brightness = (val - vmin) / (vmax - vmin + 1e-10)
                tc = "white" if brightness > 0.65 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=tc)

    # Blue border around diagonal cells to distinguish intra-model reference
    for i in range(n):
        ax.add_patch(mpatches.FancyBboxPatch(
            (i - 0.48, i - 0.48), 0.96, 0.96,
            boxstyle="square,pad=0", fill=False,
            edgecolor="steelblue", linewidth=2.5, zorder=5,
        ))


    fig.colorbar(im, ax=ax, label="IRCS", shrink=0.82, pad=0.01)

    ax.set_title(
        "Inter-model IRCS\n"
        r"Upper $\triangle$: baseline (no persona, T=1)  ·  "
        r"Lower $\triangle$: persona, T$_{\max}$=32  ·  "
        "Diagonal: intra-model (baseline)",
        fontsize=9, pad=8,
    )
    plt.tight_layout()
    out = fig_dir / "fig3_intermodel_ircs.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default="small", choices=list(EMB_ROOTS.keys()))
    args = parser.parse_args()
    main(emb_root=EMB_ROOTS[args.embedding_model], fig_dir=FIG_DIRS[args.embedding_model])
