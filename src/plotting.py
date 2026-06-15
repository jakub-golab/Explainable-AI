"""Publication-quality interpretability plots for presentation and reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from src.data_loading import feature_to_residue_key
from src.interpretability import (
    aggregate_to_residues,
    model_feature_importance,
    shap_feature_importance,
)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

PALETTE = {
    "native": "#2c7bb6",
    "permutation": "#d7191c",
    "shap": "#1a9641",
    "statistics": "#fdae61",
    "ml": "#2c7bb6",
    "overlap": "#7b3294",
    "d2": "#2166ac",
    "d4": "#b2182b",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_residue_importance(
    residues_df: pd.DataFrame,
    title: str,
    out_path: Path,
    color: str = PALETTE["native"],
    xlabel: str = "Importance",
) -> None:
    """Horizontal bar chart of top residues."""
    df = residues_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["residue"], df["importance"], color=color, edgecolor="white", height=0.7)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_path)


def plot_top15_features(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    title: str,
    out_path: Path,
    color: str,
    xlabel: str = "Score",
) -> None:
    """Generic horizontal bar chart for top-15 features."""
    plot_df = df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df[label_col], plot_df[value_col], color=color, edgecolor="white", height=0.7)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_path)


def plot_shap_top15_features(
    shap_df: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Top-15 interaction features by mean |SHAP|."""
    plot_top15_features(
        shap_df,
        value_col="importance",
        label_col="feature",
        title=f"{receptor.upper()} – Top-{top_n} cech (SHAP)",
        out_path=out_path,
        color=PALETTE["shap"],
        xlabel="Średnia |SHAP|",
    )


def plot_feature_importance_top15(
    importance_df: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Top-15 interaction features by Random Forest feature importance."""
    plot_top15_features(
        importance_df,
        value_col="importance",
        label_col="feature",
        title=f"{receptor.upper()} – Top-{top_n} cech (Feature importance)",
        out_path=out_path,
        color=PALETTE["native"],
        xlabel="Feature importance",
    )


def plot_shap_vs_statistics_features(
    shap_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Presence plot for top-N features: SHAP vs statistical test."""
    shap_set = set(shap_df.head(top_n)["feature"])
    stat_set = set(stats_df.sort_values("p_fisher_adj").head(top_n)["feature"])
    only_shap = shap_set - stat_set
    only_stat = stat_set - shap_set
    both = shap_set & stat_set
    union = sorted(both | only_shap | only_stat)
    jaccard = len(both) / len(shap_set | stat_set) if (shap_set | stat_set) else 0.0

    colors = []
    for f in union:
        if f in both:
            colors.append(PALETTE["overlap"])
        elif f in only_shap:
            colors.append(PALETTE["shap"])
        else:
            colors.append(PALETTE["statistics"])

    fig, ax = plt.subplots(figsize=(9, max(4, len(union) * 0.32)))
    ax.barh(np.arange(len(union)), [1] * len(union), color=colors, edgecolor="white")
    ax.set_yticks(np.arange(len(union)))
    ax.set_yticklabels(union, fontsize=8)
    ax.set_xticks([])
    ax.set_title(
        f"{receptor.upper()} – Top-{top_n} cech: SHAP vs statystyka\n"
        f"wspólne: {len(both)}/{top_n}, Jaccard = {jaccard:.2f}"
    )
    legend = [
        mpatches.Patch(color=PALETTE["overlap"], label="Obie metody"),
        mpatches.Patch(color=PALETTE["shap"], label="Tylko SHAP"),
        mpatches.Patch(color=PALETTE["statistics"], label="Tylko statystyka"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    _save(fig, out_path)


def plot_feature_importance_top_interactions(
    importance_df: pd.DataFrame,
    title: str,
    out_path: Path,
    top_n: int = 15,
    color: str = PALETTE["native"],
) -> None:
    """Bar chart at interaction-feature level (not aggregated to residues)."""
    df = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["feature"], df["importance"], color=color, edgecolor="white", height=0.7)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_path)


def _extract_shap_for_active_class(shap_values: np.ndarray, n_features: int) -> np.ndarray:
    """Handle SHAP outputs: list, 2D, or 3D (n_samples, n_features, n_classes)."""
    if isinstance(shap_values, list):
        return shap_values[1]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # Binary classifier: use active class (index 1)
        return arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def compute_shap_matrix(pipeline, X: pd.DataFrame, max_samples: int = 400):
    """Return (shap_values_for_active_class, X_sample, feature_names)."""
    from src.interpretability import _get_classifier

    clf = _get_classifier(pipeline)
    scaler = pipeline.named_steps.get("scaler")
    X_sample = X.sample(min(max_samples, len(X)), random_state=42)
    X_in = scaler.transform(X_sample) if scaler is not None else X_sample.values

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_in)
    values = _extract_shap_for_active_class(shap_values, X_sample.shape[1])
    return values, X_sample, list(X.columns)


def plot_shap_bar(
    pipeline,
    X: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
    max_samples: int = 400,
) -> pd.DataFrame:
    """Mean |SHAP| bar plot at residue level."""
    shap_df = shap_feature_importance(pipeline, X, max_samples=max_samples)
    residues = aggregate_to_residues(shap_df, top_n=top_n)
    plot_residue_importance(
        residues,
        f"{receptor.upper()} – SHAP (średnia |wartość| na poziomie reszty)",
        out_path,
        color=PALETTE["shap"],
        xlabel="Średnia |SHAP|",
    )
    return residues


def plot_shap_beeswarm(
    pipeline,
    X: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
    max_samples: int = 300,
) -> None:
    """SHAP beeswarm for top interaction features."""
    values, X_sample, feature_names = compute_shap_matrix(pipeline, X, max_samples=max_samples)
    mean_abs = np.abs(values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]

    X_plot = X_sample.iloc[:, top_idx].copy()
    values_plot = values[:, top_idx]
    names_plot = [feature_names[i] for i in top_idx]

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        values_plot,
        X_plot,
        feature_names=names_plot,
        show=False,
        plot_size=None,
        max_display=top_n,
    )
    plt.title(f"{receptor.upper()} – wykres SHAP (klasa: aktywny)")
    _save(fig, out_path)


def plot_permutation_top15(
    perm_df: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Top-15 interaction features by permutation importance."""
    plot_top15_features(
        perm_df,
        value_col="importance",
        label_col="feature",
        title=f"{receptor.upper()} – Top-{top_n} cech (Permutation importance)",
        out_path=out_path,
        color=PALETTE["permutation"],
        xlabel="Permutation importance",
    )


def _plot_two_method_heatmap(
    left_res: pd.DataFrame,
    right_res: pd.DataFrame,
    left_label: str,
    right_label: str,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
    title: str | None = None,
) -> None:
    """Heatmap comparing rankings from two interpretability methods."""
    methods = {left_label: left_res, right_label: right_res}
    all_residues: set[str] = set()
    for df in methods.values():
        all_residues.update(df.head(top_n)["residue"].tolist())

    residue_list = sorted(all_residues)
    matrix = np.zeros((len(residue_list), 2))
    method_names = list(methods.keys())

    for j, df in enumerate(methods.values()):
        ranked = {r: i for i, r in enumerate(df.head(top_n)["residue"].tolist())}
        for i, res in enumerate(residue_list):
            if res in ranked:
                matrix[i, j] = top_n - ranked[res]

    fig, ax = plt.subplots(figsize=(5, max(5, len(residue_list) * 0.28)))
    sns.heatmap(
        matrix,
        yticklabels=residue_list,
        xticklabels=method_names,
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Rank score (wyżej = ważniejsze)"},
        ax=ax,
    )
    ax.set_title(title or f"{receptor.upper()} – {left_label} vs {right_label} (top-{top_n})")
    _save(fig, out_path)


def plot_fi_vs_shap_heatmap(
    fi_res: pd.DataFrame,
    shap_res: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Heatmap: Feature importance vs SHAP rankings (hypothesis 1)."""
    _plot_two_method_heatmap(
        fi_res,
        shap_res,
        "Feature importance",
        "SHAP",
        receptor,
        out_path,
        top_n=top_n,
    )


def plot_shap_vs_permutation_heatmap(
    shap_res: pd.DataFrame,
    perm_res: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Heatmap: SHAP vs permutation importance rankings."""
    _plot_two_method_heatmap(
        shap_res,
        perm_res,
        "SHAP",
        "Permutation",
        receptor,
        out_path,
        top_n=top_n,
    )


def _plot_two_method_scatter(
    left_res: pd.DataFrame,
    right_res: pd.DataFrame,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Scatter comparing importance scores from two methods."""
    left_top = left_res.head(top_n).set_index("residue")
    right_top = right_res.head(top_n).set_index("residue")
    residues = sorted(set(left_top.index) | set(right_top.index))

    left_vals = [left_top.loc[r, "importance"] if r in left_top.index else 0 for r in residues]
    right_vals = [right_top.loc[r, "importance"] if r in right_top.index else 0 for r in residues]

    fig, ax = plt.subplots(figsize=(7, 6))
    in_both = set(left_top.index) & set(right_top.index)
    colors = [PALETTE["overlap"] if r in in_both else left_color for r in residues]
    ax.scatter(left_vals, right_vals, c=colors, s=80, alpha=0.85, edgecolors="white")
    for r, x, y in zip(residues, left_vals, right_vals):
        if x > 0 or y > 0:
            ax.annotate(r, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(left_label)
    ax.set_ylabel(right_label)
    ax.set_title(f"{receptor.upper()} – zgodność rankingów {left_label} i {right_label}")
    ax.grid(alpha=0.3)
    legend = [
        mpatches.Patch(color=PALETTE["overlap"], label="W top-N obu metod"),
        mpatches.Patch(color=left_color, label="Tylko jedna metoda"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    _save(fig, out_path)


def plot_fi_vs_shap_scatter(
    fi_res: pd.DataFrame,
    shap_res: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Scatter of FI vs SHAP importance per residue (union of top-N)."""
    _plot_two_method_scatter(
        fi_res,
        shap_res,
        "Feature importance",
        "Średnia |SHAP|",
        PALETTE["native"],
        PALETTE["shap"],
        receptor,
        out_path,
        top_n=top_n,
    )


def plot_shap_vs_permutation_scatter(
    shap_res: pd.DataFrame,
    perm_res: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Scatter of SHAP vs permutation importance per residue."""
    _plot_two_method_scatter(
        shap_res,
        perm_res,
        "Średnia |SHAP|",
        "Permutation importance",
        PALETTE["shap"],
        PALETTE["permutation"],
        receptor,
        out_path,
        top_n=top_n,
    )


def _plot_two_method_presence(
    left_res: pd.DataFrame,
    right_res: pd.DataFrame,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Presence plot for top-N residues from two methods."""
    left_set = set(left_res.head(top_n)["residue"])
    right_set = set(right_res.head(top_n)["residue"])
    both = left_set & right_set
    only_left = left_set - right_set
    only_right = right_set - left_set
    union = sorted(both | only_left | only_right)
    jaccard = len(both) / len(left_set | right_set) if (left_set | right_set) else 0.0

    colors = []
    for r in union:
        if r in both:
            colors.append(PALETTE["overlap"])
        elif r in only_left:
            colors.append(left_color)
        else:
            colors.append(right_color)

    fig, ax = plt.subplots(figsize=(8, max(4, len(union) * 0.3)))
    ax.barh(np.arange(len(union)), [1] * len(union), color=colors, edgecolor="white")
    ax.set_yticks(np.arange(len(union)))
    ax.set_yticklabels(union)
    ax.set_xticks([])
    ax.set_title(
        f"{receptor.upper()} – {left_label} vs {right_label} (top-{top_n})\n"
        f"wspólne: {len(both)}/{top_n}, Jaccard = {jaccard:.2f}"
    )
    legend = [
        mpatches.Patch(color=PALETTE["overlap"], label="Obie metody"),
        mpatches.Patch(color=left_color, label=f"Tylko {left_label}"),
        mpatches.Patch(color=right_color, label=f"Tylko {right_label}"),
    ]
    ax.legend(handles=legend, loc="lower right")
    _save(fig, out_path)


def plot_fi_vs_shap_presence(
    fi_res: pd.DataFrame,
    shap_res: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Bar presence plot: which residues are in FI top-N, SHAP top-N, or both."""
    _plot_two_method_presence(
        fi_res,
        shap_res,
        "Feature importance",
        "SHAP",
        PALETTE["native"],
        PALETTE["shap"],
        receptor,
        out_path,
        top_n=top_n,
    )


def plot_shap_vs_permutation_presence(
    shap_res: pd.DataFrame,
    perm_res: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Presence plot for SHAP vs permutation top-N residues."""
    _plot_two_method_presence(
        shap_res,
        perm_res,
        "SHAP",
        "Permutation",
        PALETTE["shap"],
        PALETTE["permutation"],
        receptor,
        out_path,
        top_n=top_n,
    )


# Backward-compatible alias
plot_method_comparison = plot_fi_vs_shap_heatmap


def plot_method_jaccard(
    comparison: dict | pd.DataFrame,
    receptor: str,
    out_path: Path,
    label: str = "FI vs SHAP",
    bar_color: str | None = None,
) -> None:
    """Single bar showing Jaccard overlap between two interpretability methods."""
    if isinstance(comparison, pd.DataFrame):
        jaccard = float(comparison["jaccard"].iloc[0]) if not comparison.empty else 0.0
        overlap = int(comparison["overlap"].iloc[0]) if not comparison.empty else 0
    else:
        jaccard = comparison.get("jaccard", 0.0)
        overlap = comparison.get("n_overlap", 0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([label], [jaccard], color=bar_color or PALETTE["overlap"], width=0.5, edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Współczynnik Jaccarda")
    ax.set_title(f"{receptor.upper()} – pokrywanie {label}\n({overlap} wspólnych reszt w top-15)")
    ax.text(0, jaccard + 0.02, f"{jaccard:.2f}", ha="center", fontsize=12)
    _save(fig, out_path)


def plot_ml_vs_statistics(
    ml_residues: pd.DataFrame,
    stat_residues: pd.DataFrame,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """
    Side-by-side presence plot for ML vs statistical top residues.
    Hypothesis 2.
    """
    ml_set = set(ml_residues.head(top_n)["residue"])
    stat_set = set(stat_residues.head(top_n)["residue"])
    only_ml = ml_set - stat_set
    only_stat = stat_set - ml_set
    both = ml_set & stat_set
    union = sorted(both | only_ml | only_stat)

    colors = []
    for r in union:
        if r in both:
            colors.append(PALETTE["overlap"])
        elif r in only_ml:
            colors.append(PALETTE["shap"])
        else:
            colors.append(PALETTE["statistics"])

    fig, ax = plt.subplots(figsize=(8, max(4, len(union) * 0.3)))
    y = np.arange(len(union))
    ax.barh(y, [1] * len(union), color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(union)
    ax.set_xticks([])
    ax.set_title(
        f"{receptor.upper()} – SHAP vs statystyka (top-{top_n})\n"
        f"wspólne: {len(both)}/{top_n}, Jaccard = {len(both)/len(ml_set|stat_set):.2f}"
    )
    legend = [
        mpatches.Patch(color=PALETTE["overlap"], label="Obie metody"),
        mpatches.Patch(color=PALETTE["shap"], label="Tylko SHAP"),
        mpatches.Patch(color=PALETTE["statistics"], label="Tylko statystyka"),
    ]
    ax.legend(handles=legend, loc="lower right")
    _save(fig, out_path)


def plot_homology_paired(
    homology_df: pd.DataFrame,
    receptor_pair: str,
    out_path: Path,
) -> None:
    """
    Paired bar chart for homologous BW positions with scores in D2 and D4.
    Hypothesis 3.
    """
    df = homology_df[homology_df["both_significant"]].copy()
    if df.empty:
        df = homology_df.dropna(subset=["d2_score"]).dropna(subset=["d4_score"])
    if df.empty:
        return

    labels = [f"{row['bw_id']}\n{row.get('d2_residue','')}/{row.get('d4_residue','')}" for _, row in df.iterrows()]
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, df["d2_score"], width, label="D2 (ML)", color=PALETTE["d2"])
    ax.bar(x + width / 2, df["d4_score"], width, label="D4 (ML)", color=PALETTE["d4"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Ważność SHAP")
    ax.set_title(f"{receptor_pair} – aminokwasy na pozycjach homologicznych (numeracja BW)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_cv_stability_presentation(
    frequency_df: pd.DataFrame,
    mean_jaccard: float,
    receptor: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """CV fold stability plot for hypothesis 4."""
    df = frequency_df.head(top_n + 5).iloc[::-1]
    colors = [PALETTE["shap"] if s else PALETTE["native"] for s in df["stable"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["residue"], df["folds_in_top_n"], color=colors, edgecolor="white")
    ax.set_xlabel("Liczba foldów CV (z 5) w top-15")
    ax.set_title(
        f"{receptor.upper()} – stabilność ważności cech między foldami CV\n"
        f"średni Jaccard = {mean_jaccard:.2f}"
    )
    ax.set_xlim(0, 5.5)
    ax.axvline(5, color="gray", linestyle="--", alpha=0.5)
    _save(fig, out_path)
