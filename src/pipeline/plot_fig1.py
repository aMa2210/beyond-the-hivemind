"""
Figure 1: Per-model 2×2 IRCS heatmap.
  Rows: no persona / persona
  Cols: T=1 / T=Tmax (32)
  5 models side by side, shared colorbar.

Usage:
    python -m src.pipeline.plot_fig1
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.pipeline.fig_utils import (
    MODELS, MODEL_LABELS, FIG_DIR, EMB_DIR, EMB_ROOTS, FIG_DIRS,
    load_embeddings_by_prompt, mean_ircs,
)

ROW_LABELS = ["No persona", "Persona"]
COL_LABELS = ["T = 1", "T = 32"]

# (row, col) -> config name
CELL_CONFIGS = {
    (0, 0): "baseline",      # no persona, T=1
    (0, 1): "no_persona_t32",# no persona, Tmax=32
    (1, 0): "persona_t1",    # persona, T=1
    (1, 1): "persona_t32",   # persona, Tmax=32
}


def main(emb_root=EMB_DIR, fig_dir=FIG_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-model IRCS values
    values = {}
    for model in MODELS:
        values[model] = {}
        for (r, c), config in CELL_CONFIGS.items():
            if config is None:
                values[model][(r, c)] = None
            else:
                embs = load_embeddings_by_prompt(model, config, emb_root=emb_root)
                values[model][(r, c)] = mean_ircs(embs)

    all_vals = [v for m in values.values() for v in m.values() if v is not None]
    vmin, vmax = min(all_vals), max(all_vals)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("white")

    n = len(MODELS)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2), constrained_layout=True)

    im = None
    for ax, model in zip(axes, MODELS):
        grid = np.full((2, 2), np.nan)
        for (r, c), v in values[model].items():
            if v is not None:
                grid[r, c] = v

        masked = np.ma.array(grid, mask=np.isnan(grid))
        im = ax.imshow(masked, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")

        for r in range(2):
            for c in range(2):
                val = grid[r, c]
                if np.isnan(val):
                    ax.text(c, r, "N/A", ha="center", va="center",
                            fontsize=10, color="gray", style="italic")
                    # Gray background for N/A cell
                    ax.add_patch(mpatches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        color="lightgray", zorder=0,
                    ))
                else:
                    brightness = (val - vmin) / (vmax - vmin + 1e-10)
                    text_color = "white" if brightness > 0.6 else "black"
                    ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                            fontsize=11, fontweight="bold", color=text_color)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(COL_LABELS, fontsize=11)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(ROW_LABELS, fontsize=11)
        ax.set_title(MODEL_LABELS[model], fontsize=12, pad=5)

    fig.colorbar(im, ax=axes.tolist(), label="IRCS", shrink=0.82, pad=0.02)
    fig.suptitle(
        "Intra-model IRCS: No Persona vs Persona  ×  T = 1 vs T = 32",
        fontsize=12, y=1.01,
    )
    out = fig_dir / "fig1_2x2_ircs.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default="small", choices=list(EMB_ROOTS.keys()))
    args = parser.parse_args()
    main(emb_root=EMB_ROOTS[args.embedding_model], fig_dir=FIG_DIRS[args.embedding_model])
