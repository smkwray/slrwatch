from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import derived_data_path
from ..pipeline import read_table, write_frame


def build_market_overlay_panel(
    *,
    primary_dealer_path: Path | None = None,
    trace_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    pieces: list[pd.DataFrame] = []
    if primary_dealer_path is not None and primary_dealer_path.exists():
        pieces.append(read_table(primary_dealer_path))
    if trace_path is not None and trace_path.exists():
        pieces.append(read_table(trace_path))
    if not pieces:
        raise FileNotFoundError("No market overlay inputs were provided")

    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, how="outer", on="quarter_end")

    merged["quarter_end"] = pd.to_datetime(merged["quarter_end"]).dt.strftime("%Y-%m-%d")
    merged = merged.sort_values("quarter_end").reset_index(drop=True)
    destination = output_path or derived_data_path("market_overlay_panel.parquet")
    return write_frame(merged, destination)
