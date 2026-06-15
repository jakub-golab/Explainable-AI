"""End-to-end analysis pipeline for ligand-receptor fingerprint classification."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.cv_stability import (
    cv_residue_stability,
    plot_cv_stability,
    plot_fold_heatmap,
    stability_summary,
)
from src.data_loading import dataset_summary, load_receptor_dataset
from src.homology import (
    add_bw_annotation,
    compare_homologous_residues,
    homology_overlap_summary,
)
from src.interpretability import (
    compare_fi_vs_shap,
    compare_method_overlap,
    compare_shap_vs_permutation,
    top_features_by_method,
)
from src.models import select_best_model, train_and_evaluate
from src.statistics import aggregate_stats_to_residues, compare_ml_vs_statistics, feature_statistical_tests


def run_full_analysis(
    data_dir: str | Path = ".",
    output_dir: str | Path = "results",
    top_n: int = 20,
    use_undersampling_for_d4: bool = True,
) -> dict:
    """Run complete analysis for D2 and D4 receptors; save tables and figures."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {"summaries": {}, "receptors": {}}

    receptor_outputs: dict[str, dict] = {}
    d2_ml_residues = d4_ml_residues = None
    d2_stat_residues = d4_stat_residues = None

    for receptor in ("d2", "d4"):
        print(f"\n{'='*60}\nProcessing receptor {receptor.upper()}\n{'='*60}")
        summary = dataset_summary(receptor, data_dir)
        all_results["summaries"][receptor] = summary
        print(json.dumps(summary, indent=2))

        X, y = load_receptor_dataset(receptor, data_dir)
        use_undersampling = use_undersampling_for_d4 and receptor == "d4"

        model_results = train_and_evaluate(X, y, receptor, use_undersampling=use_undersampling)
        best = select_best_model(model_results)

        print(f"\nBest model: {best.name} (balanced_accuracy={best.cv_metrics['balanced_accuracy']:.3f})")
        print(f"  Imbalance strategy: {best.imbalance_strategy}")
        for r in model_results:
            print(f"  {r.name}: {r.cv_metrics}")

        # CV fold stability: are top amino acids consistent across folds?
        stability = cv_residue_stability(
            X,
            y,
            model_name=best.name,
            receptor=receptor,
            use_undersampling=use_undersampling,
            top_n=15,
        )
        stab_summary = stability_summary(stability)
        print(f"\n  CV stability (top-15 residues across folds):")
        print(f"    Core residues (all folds): {stab_summary['core_residues_in_all_folds']}")
        print(f"    Mean pairwise Jaccard: {stab_summary['mean_pairwise_jaccard']:.3f}")
        print(f"    Hypothesis supported: {stab_summary['hypothesis_supported']}")

        interp = top_features_by_method(
            best.model, X, y, top_n=top_n, include_shap=True, include_permutation=True
        )
        stats = feature_statistical_tests(X, y)
        stat_residues = aggregate_stats_to_residues(stats, top_n=top_n)

        shap_residues = interp["shap_residues"]
        fi_vs_shap = compare_fi_vs_shap(interp, top_n=15)
        shap_vs_permutation = compare_shap_vs_permutation(interp, top_n=15)
        ml_vs_stat = compare_ml_vs_statistics(shap_residues, stat_residues, top_n=15)
        method_overlap = compare_method_overlap(interp, top_n=15)

        receptor_out = {
            "best_model": best.name,
            "cv_metrics": {r.name: r.cv_metrics for r in model_results},
            "imbalance_strategy": best.imbalance_strategy,
            "cv_stability": stab_summary,
            "fi_vs_shap": fi_vs_shap,
            "shap_vs_permutation": shap_vs_permutation,
            "shap_vs_statistics": ml_vs_stat,
        }
        all_results["receptors"][receptor] = receptor_out
        receptor_outputs[receptor] = {
            "interp": interp,
            "stats": stats,
            "stat_residues": stat_residues,
            "shap_residues": shap_residues,
            "fi_vs_shap": fi_vs_shap,
            "shap_vs_permutation": shap_vs_permutation,
            "method_overlap": method_overlap,
            "stability": stability,
            "X": X,
            "y": y,
        }

        _save_receptor_outputs(
            output_dir / receptor,
            receptor,
            interp,
            stats,
            stat_residues,
            method_overlap,
            stability,
            best.name,
        )

        if receptor == "d2":
            d2_ml_residues = shap_residues
            d2_stat_residues = stat_residues
        else:
            d4_ml_residues = shap_residues
            d4_stat_residues = stat_residues

    # Cross-receptor homology comparison
    if d2_ml_residues is not None and d4_ml_residues is not None:
        homology_ml = homology_overlap_summary(d2_ml_residues, d4_ml_residues, top_n=top_n)
        homology_stat = homology_overlap_summary(d2_stat_residues, d4_stat_residues, top_n=top_n)
        all_results["homology"] = {
            "ml": {k: v for k, v in homology_ml.items() if k != "comparison_table"},
            "statistics": {k: v for k, v in homology_stat.items() if k != "comparison_table"},
        }

        comp_ml = compare_homologous_residues(d2_ml_residues, d4_ml_residues, top_n=top_n)
        comp_stat = compare_homologous_residues(d2_stat_residues, d4_stat_residues, top_n=top_n)
        comp_ml.to_csv(output_dir / "homology_ml_comparison.csv", index=False)
        comp_stat.to_csv(output_dir / "homology_stat_comparison.csv", index=False)

        _plot_homology_comparison(comp_ml, output_dir / "homology_ml_plot.png")

    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Presentation-quality interpretability figures
    try:
        from src.generate_figures import generate_all_figures

        generate_all_figures(
            data_dir=data_dir,
            results_dir=output_dir,
            figures_dir=Path("presentation/figures"),
            use_undersampling_for_d4=use_undersampling_for_d4,
            top_n=15,
        )
    except Exception as exc:
        print(f"Figure generation skipped: {exc}")

    print(f"\nResults saved to {output_dir}/")
    return all_results


def _save_receptor_outputs(
    out_dir: Path,
    receptor: str,
    interp: dict,
    stats: pd.DataFrame,
    stat_residues: pd.DataFrame,
    method_overlap: pd.DataFrame,
    stability,
    model_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / "statistical_tests.csv", index=False)
    stat_residues.to_csv(out_dir / "top_residues_statistics.csv", index=False)
    method_overlap.to_csv(out_dir / "method_overlap.csv", index=False)

    stability.frequency.to_csv(out_dir / "cv_stability_frequency.csv", index=False)
    stability.pairwise_jaccard.to_csv(out_dir / "cv_stability_pairwise_jaccard.csv", index=False)
    fold_rows = [
        {"fold": fold, "rank": rank + 1, "residue": residue}
        for fold, residues in stability.fold_residues.items()
        for rank, residue in enumerate(residues)
    ]
    pd.DataFrame(fold_rows).to_csv(out_dir / "cv_stability_per_fold.csv", index=False)
    plot_cv_stability(stability, out_dir / "cv_stability_plot.png", receptor, model_name)
    plot_fold_heatmap(stability, out_dir / "cv_stability_heatmap.png", receptor)

    for name, df in interp.items():
        if isinstance(df, pd.DataFrame):
            annotated = add_bw_annotation(df, receptor) if "residue" in df.columns else df
            annotated.to_csv(out_dir / f"{name}.csv", index=False)

    _plot_top_residues_comparison(interp, stat_residues, out_dir, receptor)


def _plot_top_residues_comparison(interp, stat_residues, out_dir, receptor):
    ml_key = "shap_residues"
    ml_df = interp[ml_key].head(15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=ml_df, y="residue", x="importance", ax=axes[0], palette="Blues_r")
    axes[0].set_title(f"{receptor.upper()} – SHAP (top residues)")
    axes[0].set_xlabel("Średnia |SHAP|")

    stat_plot = stat_residues.head(15).copy()
    stat_plot["neg_log_p"] = -stat_plot["p_fisher_adj"].clip(lower=1e-300).apply(lambda x: __import__("math").log10(x))
    sns.barplot(data=stat_plot, y="residue", x="neg_log_p", ax=axes[1], palette="Oranges_r")
    axes[1].set_title(f"{receptor.upper()} – Statistical analysis (top residues)")
    axes[1].set_xlabel("-log10(adjusted p-value)")

    plt.tight_layout()
    fig.savefig(out_dir / "top_residues_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_homology_comparison(comp_df, path):
    plot_df = comp_df.dropna(subset=["d2_score", "d4_score"])
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_df["d2_score"], plot_df["d4_score"], s=80, alpha=0.7)
    for _, row in plot_df.iterrows():
        ax.annotate(row["bw_id"], (row["d2_score"], row["d4_score"]), fontsize=8)
    ax.set_xlabel("D2 SHAP")
    ax.set_ylabel("D4 SHAP")
    ax.set_title("Homologous positions (BW numbering) – D2 vs D4")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_full_analysis()
