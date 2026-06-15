"""Regenerate only SHAP figures (faster than full generate_figures)."""

from src.data_loading import load_receptor_dataset
from src.models import select_best_model, train_and_evaluate
from src.plotting import plot_shap_bar, plot_shap_beeswarm

for receptor in ("d2", "d4"):
    X, y = load_receptor_dataset(receptor, ".")
    use_us = receptor == "d4"
    best = select_best_model(train_and_evaluate(X, y, receptor, use_undersampling=use_us))
    out = f"presentation/figures/{receptor}"
    plot_shap_bar(best.model, X, receptor.upper(), __import__("pathlib").Path(out) / f"{receptor}_shap_bar")
    plot_shap_beeswarm(best.model, X, receptor.upper(), __import__("pathlib").Path(out) / f"{receptor}_shap_beeswarm")
    print(f"SHAP done: {receptor}")
