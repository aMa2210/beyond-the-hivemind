"""
Figure 2.2: Mean LLM quality score heatmap.
  Same layout as Figure 2.1 but metric = mean GPT quality score (1–10).

Usage:
    python -m src.pipeline.plot_fig2_2
"""
import numpy as np
import matplotlib.pyplot as plt

from src.pipeline.fig_utils import (
    MODELS, MODEL_LABELS, TEMPERATURES, CONFIG_META, FIG_DIR,
    load_quality_by_prompt, mean_quality,
)

TEMP_COLS = TEMPERATURES
CONDITIONS = [(False, "No persona"), (True, "Persona")]
_PERSONA_TEMP_TO_CONFIG = {v: k for k, v in CONFIG_META.items()}


def build_grid():
    rows = [(m, persona) for m in MODELS for (persona, _) in CONDITIONS]
    temp_idx = {t: i for i, t in enumerate(TEMP_COLS)}
    grid = np.full((len(rows), len(TEMP_COLS)), np.nan)

    for ri, (model, persona) in enumerate(rows):
        for temp in TEMP_COLS:
            config = _PERSONA_TEMP_TO_CONFIG.get((persona, temp))
            if config is None:
                continue
            val = mean_quality(load_quality_by_prompt(model, config))
            if val is not None:
                grid[ri, temp_idx[temp]] = val

    row_labels = [
        f"{MODEL_LABELS[m]} · {'Persona' if persona else 'No persona'}"
        for m, persona in rows
    ]
    return grid, row_labels


def main():
    FIG_DIR.mkdir(exist_ok=True)
    grid, row_labels = build_grid()

    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    cmap = plt.cm.YlGn.copy()
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
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=tc)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(t) for t in TEMP_COLS], fontsize=9)
    ax.set_xlabel("Temperature", fontsize=10)

    for i in range(1, len(MODELS)):
        ax.axhline(i * len(CONDITIONS) - 0.5, color="black", linewidth=1.8)

    fig.colorbar(im, ax=ax, label="Mean LLM Quality Score (1–10)", shrink=0.85)
    ax.set_title("Mean LLM Quality Score by Temperature and Condition", fontsize=12, pad=8)
    plt.tight_layout()
    out = FIG_DIR / "fig2_2_quality_heatmap.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
