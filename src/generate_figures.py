"""Generate interpretability figures focused on Feature Importance vs SHAP."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_loading import load_receptor_dataset
from src.homology import add_bw_annotation
from src.interpretability import (
    compare_fi_vs_shap,
    compare_method_overlap,
    compare_shap_vs_permutation,
    model_feature_importance,
    top_features_by_method,
)
from src.models import select_best_model, train_and_evaluate
from src.statistics import (
    aggregate_stats_to_residues,
    compare_shap_vs_statistics_features,
    feature_statistical_tests,
    top_statistical_features,
)
from src.plotting import (
    plot_cv_stability_presentation,
    plot_feature_importance_top15,
    plot_fi_vs_shap_heatmap,
    plot_fi_vs_shap_presence,
    plot_fi_vs_shap_scatter,
    plot_homology_paired,
    plot_method_jaccard,
    plot_ml_vs_statistics,
    plot_permutation_top15,
    plot_residue_importance,
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_shap_top15_features,
    plot_shap_vs_permutation_heatmap,
    plot_shap_vs_permutation_presence,
    plot_shap_vs_permutation_scatter,
    plot_shap_vs_statistics_features,
)


def generate_all_figures(
    data_dir: str | Path = ".",
    results_dir: str | Path = "results",
    figures_dir: str | Path = "presentation/figures",
    use_undersampling_for_d4: bool = True,
    top_n: int = 15,
) -> dict:
    """Train models and save FI / SHAP interpretability plots."""
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "analysis_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    generated: dict[str, list[str]] = {"d2": [], "d4": [], "cross": []}
    interp_cache: dict[str, dict] = {}
    fi_shap_cache: dict[str, dict] = {}
    shap_perm_cache: dict[str, dict] = {}

    for receptor in ("d2", "d4"):
        out = figures_dir / receptor
        out.mkdir(parents=True, exist_ok=True)

        X, y = load_receptor_dataset(receptor, data_dir)
        use_undersampling = use_undersampling_for_d4 and receptor == "d4"
        best = select_best_model(
            train_and_evaluate(X, y, receptor, use_undersampling=use_undersampling)
        )

        interp = top_features_by_method(
            best.model, X, y, top_n=top_n, include_shap=True, include_permutation=True
        )
        interp_cache[receptor] = interp
        fi_shap = compare_fi_vs_shap(interp, top_n=top_n)
        fi_shap_cache[receptor] = fi_shap
        shap_perm = compare_shap_vs_permutation(interp, top_n=top_n)
        shap_perm_cache[receptor] = shap_perm

        native = model_feature_importance(best.model, list(X.columns))
        shap_features = interp["shap"]
        perm_features = interp["permutation"]
        fi_res = add_bw_annotation(interp["native_residues"], receptor)
        shap_res = add_bw_annotation(interp["shap_residues"], receptor)
        perm_res = add_bw_annotation(interp["permutation_residues"], receptor)

        stats = feature_statistical_tests(X, y)
        stat_features = top_statistical_features(stats, top_n=top_n)
        stat_res = aggregate_stats_to_residues(stats, top_n=top_n)
        shap_vs_stat_features = compare_shap_vs_statistics_features(shap_features, stats, top_n=top_n)

        # --- Top-15 feature-level plots (requested) ---
        p = out / f"{receptor}_top15_feature_importance"
        plot_feature_importance_top15(native, receptor.upper(), p, top_n=top_n)
        generated[receptor].append(str(p.with_suffix(".png")))

        p = out / f"{receptor}_top15_shap"
        plot_shap_top15_features(shap_features, receptor.upper(), p, top_n=top_n)
        generated[receptor].append(str(p.with_suffix(".png")))

        p = out / f"{receptor}_top15_shap_vs_statistics"
        plot_shap_vs_statistics_features(shap_features, stats, receptor.upper(), p, top_n=top_n)
        generated[receptor].append(str(p.with_suffix(".png")))

        p = out / f"{receptor}_top15_permutation"
        plot_permutation_top15(perm_features, receptor.upper(), p, top_n=top_n)
        generated[receptor].append(str(p.with_suffix(".png")))

        # Residue-level FI
        for plot_fn, name, df, title, color in [
            (
                plot_residue_importance,
                f"{receptor}_feature_importance_residues",
                fi_res,
                f"{receptor.upper()} – Feature importance (Random Forest)",
                "#2c7bb6",
            ),
        ]:
            p = out / name
            plot_fn(df, title, p, color=color)
            generated[receptor].append(str(p.with_suffix(".png")))

        # SHAP beeswarm + residue aggregation
        p = out / f"{receptor}_shap_bar"
        plot_shap_bar(best.model, X, receptor.upper(), p, top_n=top_n)
        generated[receptor].append(str(p.with_suffix(".png")))

        p = out / f"{receptor}_shap_beeswarm"
        plot_shap_beeswarm(best.model, X, receptor.upper(), p, top_n=top_n)
        generated[receptor].append(str(p.with_suffix(".png")))

        p = out / f"{receptor}_shap_residues"
        plot_residue_importance(
            shap_res,
            f"{receptor.upper()} – SHAP (agregacja na poziomie aminokwasu)",
            p,
            color="#1a9641",
            xlabel="Średnia |SHAP|",
        )
        generated[receptor].append(str(p.with_suffix(".png")))

        # Hypothesis 1: FI vs SHAP
        for plot_fn, suffix in [
            (plot_fi_vs_shap_heatmap, "fi_vs_shap_heatmap"),
            (plot_fi_vs_shap_scatter, "fi_vs_shap_scatter"),
            (plot_fi_vs_shap_presence, "fi_vs_shap_presence"),
        ]:
            p = out / f"{receptor}_{suffix}"
            plot_fn(fi_res, shap_res, receptor.upper(), p, top_n=top_n)
            generated[receptor].append(str(p.with_suffix(".png")))

        p = out / f"{receptor}_fi_vs_shap_jaccard"
        plot_method_jaccard(fi_shap, receptor.upper(), p)
        generated[receptor].append(str(p.with_suffix(".png")))

        # SHAP vs permutation
        for plot_fn, suffix in [
            (plot_shap_vs_permutation_heatmap, "shap_vs_permutation_heatmap"),
            (plot_shap_vs_permutation_scatter, "shap_vs_permutation_scatter"),
            (plot_shap_vs_permutation_presence, "shap_vs_permutation_presence"),
        ]:
            p = out / f"{receptor}_{suffix}"
            plot_fn(shap_res, perm_res, receptor.upper(), p, top_n=top_n)
            generated[receptor].append(str(p.with_suffix(".png")))

        p = out / f"{receptor}_shap_vs_permutation_jaccard"
        plot_method_jaccard(
            shap_perm,
            receptor.upper(),
            p,
            label="SHAP vs Permutation",
            bar_color="#7b3294",
        )
        generated[receptor].append(str(p.with_suffix(".png")))

        p = out / f"{receptor}_permutation_residues"
        plot_residue_importance(
            perm_res,
            f"{receptor.upper()} – Permutation importance (agregacja na poziomie aminokwasu)",
            p,
            color="#d7191c",
            xlabel="Permutation importance",
        )
        generated[receptor].append(str(p.with_suffix(".png")))

        # Hypothesis 2: SHAP vs statistics (residue + feature metrics in manifest)
        p = out / f"{receptor}_shap_vs_statistics"
        plot_ml_vs_statistics(shap_res, stat_res, receptor.upper(), p, top_n=top_n)
        generated[receptor].append(str(p.with_suffix(".png")))

        fi_shap_cache[receptor]["shap_vs_statistics_features"] = shap_vs_stat_features

        # Hypothesis 4: CV stability
        stab_path = results_dir / receptor / "cv_stability_frequency.csv"
        if stab_path.exists():
            freq = pd.read_csv(stab_path)
            cv_stab = summary.get("receptors", {}).get(receptor, {}).get("cv_stability", {})
            p = out / f"{receptor}_cv_stability"
            plot_cv_stability_presentation(
                freq, cv_stab.get("mean_pairwise_jaccard", 0.0), receptor.upper(), p, top_n=top_n
            )
            generated[receptor].append(str(p.with_suffix(".png")))

    # Hypothesis 3: homology based on SHAP rankings
    d2_shap = interp_cache["d2"]["shap_residues"]
    d4_shap = interp_cache["d4"]["shap_residues"]
    hom_path = results_dir / "homology_ml_comparison.csv"
    if hom_path.exists():
        hom = pd.read_csv(hom_path)
        p = figures_dir / "homology_d2_d4_paired"
        plot_homology_paired(hom, "D2 / D4 (SHAP)", p)
        generated["cross"].append(str(p.with_suffix(".png")))

        plot_df = hom.dropna(subset=["d2_score", "d4_score"])
        if not plot_df.empty:
            import matplotlib.pyplot as plt

            p = figures_dir / "homology_d2_d4_scatter"
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(plot_df["d2_score"], plot_df["d4_score"], s=100, c="#7b3294", alpha=0.8)
            for _, row in plot_df.iterrows():
                ax.annotate(
                    f"{row['bw_id']}",
                    (row["d2_score"], row["d4_score"]),
                    fontsize=9,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
            ax.set_xlabel("D2 – SHAP")
            ax.set_ylabel("D4 – SHAP")
            ax.set_title("Pozycje homologiczne (BW): D2 vs D4")
            ax.grid(alpha=0.3)
            fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
            fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
            generated["cross"].append(str(p.with_suffix(".png")))

    manifest = figures_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {**generated, "fi_vs_shap": fi_shap_cache, "shap_vs_permutation": shap_perm_cache},
            indent=2,
            default=str,
        )
    )
    print(f"Figures saved to {figures_dir}/")
    return generated


if __name__ == "__main__":
    generate_all_figures()
