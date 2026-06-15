"""Load ProLIF fingerprint pickles and convert to binary feature matrices."""

from __future__ import annotations

import re
import sys
import types
import warnings
from pathlib import Path

import dill
import inspect
import pandas as pd

warnings.filterwarnings("ignore")


def _setup_prolif_compat() -> None:
    """Patch prolif modules so legacy v1.x pickles can be deserialized."""
    import prolif.fingerprint as fp_mod
    from prolif import interactions as prolif_interactions

    interactions_module = types.ModuleType("prolif.interactions.interactions")
    for name, obj in inspect.getmembers(prolif_interactions, inspect.isclass):
        if name[0].isupper():
            setattr(interactions_module, name, obj)
    sys.modules["prolif.interactions.interactions"] = interactions_module

    if not hasattr(fp_mod, "first_occurence"):
        fp_mod.first_occurence = lambda interaction: next(interaction)
    if not hasattr(fp_mod, "all_occurences"):
        fp_mod.all_occurences = lambda interaction: (x for x in interaction)


def residue_to_str(res) -> str:
    """Convert a ProLIF ResidueId to 'PHE389.A' format."""
    chain = f".{res.chain}" if res.chain else ""
    return f"{res.name}{res.number}{chain}"


def parse_feature_name(col: str) -> tuple[str, int, str, str] | None:
    """Parse 'PHE389.A.Hydrophobic' -> (aa, position, chain, interaction)."""
    parts = col.rsplit(".", 1)
    if len(parts) != 2:
        return None
    interaction = parts[1]
    res_part = parts[0]
    chain_parts = res_part.rsplit(".", 1)
    chain = chain_parts[1] if len(chain_parts) > 1 else ""
    res_name_num = chain_parts[0]
    match = re.match(r"([A-Z]+)(\d+)", res_name_num)
    if match:
        return match.group(1), int(match.group(2)), chain, interaction
    return None


def feature_to_residue_key(col: str) -> str:
    """Return residue identifier without interaction type, e.g. 'PHE389.A'."""
    return col.rsplit(".", 1)[0]


def fp_to_dataframe(fp) -> pd.DataFrame:
    """Convert a loaded ProLIF Fingerprint object to a 0/1 DataFrame."""
    rows: list[dict[str, int]] = []
    indices: list[int] = []

    for frame_idx, ifp in fp.ifp.items():
        row: dict[str, int] = {}
        for (_lig_res, prot_res), interaction_dict in ifp.items():
            res_str = residue_to_str(prot_res)
            for interaction_name in interaction_dict:
                row[f"{res_str}.{interaction_name}"] = 1
        rows.append(row)
        indices.append(frame_idx)

    return pd.DataFrame(rows, index=indices).fillna(0).astype(int)


def load_fingerprint_pickle(path: str | Path) -> pd.DataFrame:
    """Load a ProLIF pickle file and return a binary interaction DataFrame."""
    _setup_prolif_compat()
    with open(path, "rb") as f:
        fp = dill.load(f)
    return fp_to_dataframe(fp)


def load_receptor_dataset(
    receptor: str,
    data_dir: str | Path = ".",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load active/inactive fingerprints for a receptor (d2 or d4).

    Returns (X, y) where y=1 for active and y=0 for inactive.
    """
    data_dir = Path(data_dir)
    receptor = receptor.lower()
    active = load_fingerprint_pickle(data_dir / f"fingerprint_active_{receptor}.pkl")
    inactive = load_fingerprint_pickle(data_dir / f"fingerprint_inactive_{receptor}.pkl")

    active["label"] = 1
    inactive["label"] = 0

    combined = pd.concat([active, inactive], axis=0)
    feature_cols = [col for col in combined.columns if col != "label"]
    combined[feature_cols] = combined[feature_cols].fillna(False).astype(int)
    combined = combined.drop_duplicates(subset=feature_cols, keep=False).reset_index(drop=True)

    y = combined["label"].astype(int)
    X = combined[feature_cols].astype(int)

    return X, y


def dataset_summary(receptor: str, data_dir: str | Path = ".") -> dict:
    """Return basic statistics for a receptor dataset."""
    X, y = load_receptor_dataset(receptor, data_dir)
    n_active = int((y == 1).sum())
    n_inactive = int((y == 0).sum())
    return {
        "receptor": receptor,
        "n_samples": len(y),
        "n_active": n_active,
        "n_inactive": n_inactive,
        "balance_ratio": n_active / n_inactive if n_inactive else float("inf"),
        "n_features": X.shape[1],
        "mean_interactions_per_sample": float(X.sum(axis=1).mean()),
    }
