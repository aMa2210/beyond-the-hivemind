"""
Run all paper figure scripts (heatmaps). For Figure 1 (PCA), run plot_fig_pca separately.

Usage:
    python -m src.pipeline.plot_all_figs
"""
import importlib

from src.pipeline.fig_utils import EMB_ROOTS, FIG_DIRS

PLOT_MODULES = [
    "src.pipeline.plot_fig1",
    "src.pipeline.plot_fig2_1",
    "src.pipeline.plot_fig2_2",
    "src.pipeline.plot_fig2_3",
    "src.pipeline.plot_fig3",
    "src.pipeline.plot_fig4",
]


def main():
    emb_root = EMB_ROOTS["small"]
    fig_dir = FIG_DIRS["small"]
    print(f"\n=== Generating figures → {fig_dir} ===")
    for mod_name in PLOT_MODULES:
        mod = importlib.import_module(mod_name)
        print(f"  {mod_name.split('.')[-1]} ...", end=" ", flush=True)
        try:
            try:
                mod.main(emb_root=emb_root, fig_dir=fig_dir)
            except TypeError:
                mod.main()
        except Exception as e:
            print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
