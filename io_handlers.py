"""MIT License

Copyright (c) 2024 The TCM Data Cleaner Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from processing import ProcessedData, compute_rhoa

DELIMITERS = [",", "\t", ";", "|"]


def _detect_delimiter(text: str) -> str:
    counts = {delim: text.count(delim) for delim in DELIMITERS}
    return max(counts, key=counts.get) if counts else ","


def read_table(path: str) -> pd.DataFrame:
    """Read a tabular file supporting CSV, DAT, and Excel formats."""

    file_path = pathlib.Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(file_path)
        return df

    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = _detect_delimiter(sample)
        df = pd.read_csv(handle, delimiter=delimiter, engine="python")
    if df.columns.str.contains("Unnamed").all():
        df.columns = [f"C{i+1}" for i in range(df.shape[1])]
    return df


def export_processed_csv(
    processed: ProcessedData,
    output_path: str,
    mode: str,
    visible_channels: Sequence[str],
) -> None:
    """Write the processed data to a wide CSV file."""

    data = {
        "Distance_m": processed.distance,
    }
    if processed.elevation is not None:
        data["Elevation_m"] = processed.elevation
    for channel in visible_channels:
        values = processed.channels.get(channel)
        if values is None:
            continue
        data[f"{channel}_mSperm"] = values
        rhoa = compute_rhoa(values)
        data[f"{channel}_rhoa_ohm_m"] = rhoa
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


def export_surfer_dat(
    processed: ProcessedData,
    output_dir: str,
    base_name: str,
    mode: str,
    use_xy: bool,
    include_header: bool,
    visible_channels: Sequence[str],
) -> List[str]:
    """Export Surfer DAT files for the selected channels."""

    exported: List[str] = []
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for channel in visible_channels:
        values = processed.channels.get(channel)
        if values is None:
            continue
        if mode == "log":
            values = compute_rhoa(values)
            suffix = "rhoa"
        else:
            suffix = "mSperm"
        filename = f"{base_name}__{channel}_{suffix}.dat"
        target = output_dir / filename
        if use_xy and processed.xy_projected is not None:
            arr = np.column_stack([processed.xy_projected, values])
            header = "X_m\tY_m\tValue"
        else:
            arr = np.column_stack([processed.distance, values])
            header = "Distance_m\tValue"
        with open(target, "w", encoding="utf-8") as handle:
            if include_header:
                handle.write(header + "\n")
            np.savetxt(handle, arr, fmt="%.6f", delimiter="\t")
        exported.append(str(target))
    return exported


def export_session(
    output_path: str,
    session_payload: Dict[str, object],
) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(session_payload, handle, indent=2)


def load_session(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
