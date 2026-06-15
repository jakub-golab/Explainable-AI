# Explainable AI - metody wyjaśnialności w analizie kompleksów ligand-receptor

### Authors: Yury Sarosek, Palina Posakh, Jakub Gołąb, Dorian Rząsa

Machine learning classification and interpretability analysis of ProLIF structural interaction fingerprints for dopamine receptors **D2** and **D4**.

## Project goals

1. **Classification** – predict biological activity (active/inactive) from binary interaction fingerprints
2. **Interpretability** – SHAP (primary) + feature importance comparison
3. **Statistical comparison** – Fisher's exact tests on interaction frequencies (active vs inactive)
4. **Method comparison** – overlap between ML interpretability and simple statistics
5. **Homology analysis** – compare key residues at aligned Ballesteros-Weinstein positions between D2 and D4

## Data

| File | Description |
|------|-------------|
| `fingerprint_active_d2.pkl` | Active ligand–D2 complexes (3061 samples) |
| `fingerprint_inactive_d2.pkl` | Inactive ligand–D2 complexes (3090 samples) |
| `fingerprint_active_d4.pkl` | Active ligand–D4 complexes (1053 samples) |
| `fingerprint_inactive_d4.pkl` | Inactive ligand–D4 complexes (489 samples) |

**Class balance:** D2 is balanced (~1:1). **D4 is imbalanced** (~2.15:1 active:inactive) — handled via random **undersampling** of the majority (active) class in each training fold (features stay binary). Alternative: `class_weight` (set `use_undersampling_for_d4=False`).

**CV stability:** tests whether top amino acids are consistent across cross-validation folds (`results/*/cv_stability_*.csv`).

Each feature is a binary interaction, e.g. `PHE389.A.Hydrophobic` (amino acid + position + chain + interaction type).

## Setup

```bash
pip install -r requirements.txt
```

## Run analysis

```bash
python -m src.run_analysis
```

Results are written to `results/`:
- `results/d2/` and `results/d4/` – per-receptor CSVs and plots
- `results/homology_ml_comparison.csv` – homologous position comparison
- `results/analysis_summary.json` – metrics and overlap statistics

## Generate presentation figures

```bash
python -m src.generate_figures
```



## Research hypotheses addressed

| Hypothesis | Implementation |
|------------|----------------|
| Compare amino acids from different interpretability methods | `method_overlap.csv`, Jaccard index |
| Compare ML vs statistical analysis | `ml_vs_statistics` in summary JSON |
| Compare homologous positions in D2 vs D4 | BW numbering in `homology.py`, comparison CSVs |

## Project structure

```
src/
  data_loading.py      # Load ProLIF pickles → DataFrames
  models.py            # RF, XGBoost, L1-logistic + CV + undersampling
  cv_stability.py      # Per-fold residue stability across CV
  interpretability.py  # SHAP + feature importance, FI vs SHAP comparison
  statistics.py        # Fisher's exact, FDR correction
  homology.py          # D2↔D4 BW position mapping
  run_analysis.py      # End-to-end pipeline
notebooks/
  analysis.ipynb
results/               # Generated outputs
```
