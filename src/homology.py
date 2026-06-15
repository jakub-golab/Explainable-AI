"""Homology mapping and cross-receptor comparison for D2 vs D4 dopamine receptors."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# Ballesteros-Weinstein (BW) numbering for human DRD2 vs DRD4.
# Positions derived from sequence alignment of binding-site region (TM3-TM7).
# Format: receptor_position -> BW_id
DRD2_BW_MAP: dict[int, str] = {
    114: "3.32",  # D114 (Asp, conserved)
    184: "4.60",
    190: "4.66",
    389: "5.43",  # Phe
    393: "5.47",
    408: "5.62",
    416: "6.48",
    100: "3.28",
    103: "3.29",
    178: "4.56",
    181: "4.59",
    192: "4.68",
    195: "4.71",
    404: "5.58",
    410: "5.64",
}

DRD4_BW_MAP: dict[int, str] = {
    91: "5.43",   # Phe (homologous to D2 PHE389)
    101: "3.28",  # Trp
    111: "4.60",  # Leu
    115: "3.32",  # Asp
    195: "4.66",
    196: "4.67",
    410: "5.64",  # Phe
    412: "5.66",
    419: "6.48",
    103: "3.29",
    178: "4.56",
    181: "4.59",
    192: "4.68",
    404: "5.58",
}


@dataclass
class HomologyMapping:
    d2_position: int
    d4_position: int
    bw_id: str
    d2_aa: str | None = None
    d4_aa: str | None = None


def build_bw_homology_table() -> pd.DataFrame:
    """Build D2<->D4 mapping table via shared Ballesteros-Weinstein IDs."""
    d2_inv = {bw: pos for pos, bw in DRD2_BW_MAP.items()}
    d4_inv = {bw: pos for pos, bw in DRD4_BW_MAP.items()}
    shared_bw = set(d2_inv) & set(d4_inv)

    rows = []
    for bw in sorted(shared_bw):
        rows.append(
            {
                "bw_id": bw,
                "d2_position": d2_inv[bw],
                "d4_position": d4_inv[bw],
            }
        )
    return pd.DataFrame(rows)


def residue_str(aa: str, position: int, chain: str = "A") -> str:
    return f"{aa}{position}.{chain}"


def map_residue_to_bw(residue_key: str, receptor: str) -> str | None:
    """Map 'PHE389.A' to BW ID for the given receptor."""
    match = re.match(r"([A-Z]+)(\d+)", residue_key.split(".")[0])
    if not match:
        return None
    pos = int(match.group(2))
    bw_map = DRD2_BW_MAP if receptor.lower() == "d2" else DRD4_BW_MAP
    return bw_map.get(pos)


def add_bw_annotation(residue_df: pd.DataFrame, receptor: str) -> pd.DataFrame:
    """Add BW numbering column to a residue-level importance/stats DataFrame."""
    out = residue_df.copy()
    out["bw_id"] = out["residue"].apply(lambda r: map_residue_to_bw(r, receptor))
    return out


def _score_column(df: pd.DataFrame) -> str:
    """Pick the primary ranking column from a residue-level DataFrame."""
    for col in ("importance", "p_fisher_adj", "freq_diff"):
        if col in df.columns:
            return col
    raise ValueError(f"No known score column in: {list(df.columns)}")


def compare_homologous_residues(
    d2_residues: pd.DataFrame,
    d4_residues: pd.DataFrame,
    d2_score_col: str | None = None,
    d4_score_col: str | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Compare key residues at homologous BW positions between D2 and D4.

    Returns a table with both receptors' scores at aligned positions.
    """
    d2_score_col = d2_score_col or _score_column(d2_residues)
    d4_score_col = d4_score_col or _score_column(d4_residues)
    homology = build_bw_homology_table()
    d2_ann = add_bw_annotation(d2_residues, "d2")
    d4_ann = add_bw_annotation(d4_residues, "d4")

    d2_top = d2_ann.dropna(subset=["bw_id"]).head(top_n)
    d4_top = d4_ann.dropna(subset=["bw_id"]).head(top_n)

    merged = homology.merge(
        d2_top[["bw_id", "residue", "aa", "position", d2_score_col]].rename(
            columns={
                "residue": "d2_residue",
                "aa": "d2_aa",
                "position": "d2_position_data",
                d2_score_col: "d2_score",
            }
        ),
        on="bw_id",
        how="left",
    ).merge(
        d4_top[["bw_id", "residue", "aa", "position", d4_score_col]].rename(
            columns={
                "residue": "d4_residue",
                "aa": "d4_aa",
                "position": "d4_position_data",
                d4_score_col: "d4_score",
            }
        ),
        on="bw_id",
        how="left",
    )

    merged["both_significant"] = merged["d2_score"].notna() & merged["d4_score"].notna()
    return merged.sort_values("bw_id").reset_index(drop=True)


def homology_overlap_summary(
    d2_residues: pd.DataFrame,
    d4_residues: pd.DataFrame,
    top_n: int = 15,
) -> dict:
    """Summarize how many top residues fall on homologous BW positions."""
    comparison = compare_homologous_residues(d2_residues, d4_residues, top_n=top_n)
    d2_top_bw = set(
        add_bw_annotation(d2_residues, "d2").dropna(subset=["bw_id"]).head(top_n)["bw_id"]
    )
    d4_top_bw = set(
        add_bw_annotation(d4_residues, "d4").dropna(subset=["bw_id"]).head(top_n)["bw_id"]
    )
    overlap = d2_top_bw & d4_top_bw
    return {
        "d2_top_bw_positions": sorted(d2_top_bw),
        "d4_top_bw_positions": sorted(d4_top_bw),
        "shared_bw_positions": sorted(overlap),
        "n_shared": len(overlap),
        "jaccard": len(overlap) / len(d2_top_bw | d4_top_bw) if (d2_top_bw | d4_top_bw) else 0.0,
        "comparison_table": comparison,
    }
