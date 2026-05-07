"""
Figure 2.3: Persona description IRCS heatmap.
  Same layout as Figure 2.1 but metric = IRCS of persona description embeddings.
  Only persona configs carry persona_embeddings; no-persona rows are left white.

Usage:
    python -m src.pipeline.plot_fig2_3
"""
import numpy as np
import matplotlib.pyplot as plt

from src.pipeline.fig_utils import (
    MODELS, MODEL_LABELS, TEMPERATURES, CONFIG_META, FIG_DIR, EMB_DIR, EMB_ROOTS, FIG_DIRS,
    load_persona_embeddings_by_prompt, mean_ircs,
)

TEMP_COLS = TEMPERATURES
CONDITIONS = [(True, "Persona")]
_PERSONA_TEMP_TO_CONFIG = {v: k for k, v in CONFIG_META.items()}


def build_grid(emb_root=EMB_DIR):
    rows = [(m, persona) for m in MODELS for (persona, _) in CONDITIONS]
    temp_idx = {t: i for i, t in enumerate(TEMP_COLS)}
    grid = np.full((len(rows), len(TEMP_COLS)), np.nan)

    for ri, (model, persona) in enumerate(rows):
        for temp in TEMP_COLS:
            config = _PERSONA_TEMP_TO_CONFIG.get((persona, temp))
            if config is None:
                continue
            val = mean_ircs(load_persona_embeddings_by_prompt(model, config, emb_root=emb_root))
            if val is not None:
                grid[ri, temp_idx[temp]] = val

    row_labels = [
        f"{MODEL_LABELS[m]} · {'Persona' if persona else 'No persona'}"
        for m, persona in rows
    ]
    return grid, row_labels


def main(emb_root=EMB_DIR, fig_dir=FIG_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    grid, row_labels = build_grid(emb_root=emb_root)

    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("white")

    n_rows, n_cols = grid.shape
    fig_h = max(4.0, 0.5 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    masked = np.ma.array(grid, mask=np.isnan(grid))
    im = ax.imshow(masked, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")

    for r in range(n_rows):
        for c in range(n_cols):
            val = grid[r, c]
            if not np.isnan(val):
                brightness = (val - vmin) / (vmax - vmin + 1e-10)
                tc = "white" if brightness > 0.65 else "black"
                ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                        fontsize=7.5, color=tc)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(t) for t in TEMP_COLS], fontsize=9)
    ax.set_xlabel("Temperature", fontsize=10)

    for i in range(1, len(MODELS)):
        ax.axhline(i * len(CONDITIONS) - 0.5, color="black", linewidth=1.8)

    fig.colorbar(im, ax=ax, label="Persona Description IRCS", shrink=0.85)
    ax.set_title("Persona Description IRCS by Temperature", fontsize=12, pad=8)
    plt.tight_layout()
    out = fig_dir / "fig2_3_persona_sim_heatmap.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default="small", choices=list(EMB_ROOTS.keys()))
    args = parser.parse_args()
    main(emb_root=EMB_ROOTS[args.embedding_model], fig_dir=FIG_DIRS[args.embedding_model])
