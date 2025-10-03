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

import dataclasses
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover - optional dependency fallback
    CRS = None
    Transformer = None


@dataclasses.dataclass
class LoadedData:
    """Container for the raw dataframe and metadata."""

    dataframe: pd.DataFrame
    time_column: str
    latitude_column: str
    longitude_column: str
    elevation_column: Optional[str]
    channel_columns: List[str]


@dataclasses.dataclass
class ProcessedData:
    """Processed arrays returned by :func:`process_dataset`."""

    distance: np.ndarray
    raw_distance: np.ndarray
    channels: Dict[str, np.ndarray]
    raw_channels: Dict[str, np.ndarray]
    elevation: Optional[np.ndarray]
    xy_projected: Optional[np.ndarray]


def auto_utm_epsg(latitudes: Sequence[float], longitudes: Sequence[float]) -> Optional[int]:
    """Return the EPSG code for the auto-selected UTM zone.

    Parameters
    ----------
    latitudes:
        Iterable of latitudes in degrees.
    longitudes:
        Iterable of longitudes in degrees.

    Returns
    -------
    Optional[int]
        EPSG code for UTM zone or ``None`` if inputs are empty or PyProj is
        unavailable.
    """

    if Transformer is None:
        return None

    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)
    if latitudes.size == 0 or longitudes.size == 0:
        return None

    mean_lat = float(np.nanmean(latitudes))
    mean_lon = float(np.nanmean(longitudes))
    zone = int(math.floor((mean_lon + 180.0) / 6.0) + 1)
    if mean_lat >= 0:
        return int(f"326{zone:02d}")
    return int(f"327{zone:02d}")


def _project_to_xy(latitudes: Sequence[float], longitudes: Sequence[float]) -> Optional[np.ndarray]:
    """Project latitude/longitude to planar XY coordinates in meters.

    Returns ``None`` if :mod:`pyproj` is unavailable.
    """

    epsg = auto_utm_epsg(latitudes, longitudes)
    if epsg is None or Transformer is None:
        return None

    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
    x, y = transformer.transform(longitudes, latitudes)
    return np.column_stack([x, y])


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance between two points in meters."""

    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def compute_chainage(latitudes: Sequence[float], longitudes: Sequence[float]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute cumulative distance from latitude/longitude series.

    Parameters
    ----------
    latitudes:
        Latitudes in degrees.
    longitudes:
        Longitudes in degrees.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        Distance array in meters and optional projected XY coordinates. If
        :mod:`pyproj` is not available the XY component is ``None``.
    """

    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    if latitudes.size == 0:
        return np.array([], dtype=float), None

    xy = _project_to_xy(latitudes, longitudes)
    if xy is not None:
        dx = np.diff(xy[:, 0], prepend=xy[0, 0])
        dy = np.diff(xy[:, 1], prepend=xy[0, 1])
        segment = np.hypot(dx, dy)
    else:
        segment = np.zeros_like(latitudes)
        segment[1:] = [
            _haversine_distance(latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i])
            for i in range(1, latitudes.size)
        ]
    distance = np.cumsum(segment)
    distance[0] = 0.0
    return distance, xy


def _resample_series(distance: np.ndarray, values: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a 1D series to an evenly spaced distance grid.

    Parameters
    ----------
    distance:
        Input distance coordinates in meters.
    values:
        Raw values aligned with ``distance``.
    spacing:
        Output spacing in meters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(grid, resampled_values)`` pair.
    """

    if distance.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    dmin = float(distance.min())
    dmax = float(distance.max())
    grid = np.arange(dmin, dmax + spacing * 0.5, spacing, dtype=float)
    valid = np.isfinite(distance) & np.isfinite(values)
    if valid.sum() < 2:
        resampled = np.full_like(grid, np.nan, dtype=float)
        return grid, resampled

    resampled = np.interp(grid, distance[valid], values[valid], left=np.nan, right=np.nan)
    return grid, resampled


def _running_mean(distance: np.ndarray, values: np.ndarray, width: float, spacing: float) -> np.ndarray:
    """Compute running mean with a sliding window defined in meters."""

    if values.size == 0:
        return values.copy()

    if width <= 0:
        return values.copy()

    # convert window size to number of samples
    samples = max(int(round(width / spacing)), 1)
    kernel = np.ones(samples, dtype=float)
    valid = np.isfinite(values)
    padded = np.where(valid, values, 0.0)
    counts = np.convolve(valid.astype(float), kernel, mode="same")
    summed = np.convolve(padded, kernel, mode="same")
    with np.errstate(invalid="ignore"):
        result = np.where(counts > 0, summed / np.maximum(counts, 1.0), np.nan)
    return result


class MaskManager:
    """Manage mask arrays and undo/redo history."""

    def __init__(self) -> None:
        self.masks: Dict[str, np.ndarray] = {}
        self._undo_stack: List[List[Tuple[str, np.ndarray, np.ndarray]]] = []
        self._redo_stack: List[List[Tuple[str, np.ndarray, np.ndarray]]] = []

    def initialise(self, channels: Iterable[str], size: int) -> None:
        self.masks = {name: np.zeros(size, dtype=bool) for name in channels}
        self.clear_history()

    def clear_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def apply_batch(self, updates: Dict[str, Tuple[np.ndarray, bool]]) -> bool:
        history: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for channel, (indices, mask_value) in updates.items():
            mask = self.masks.get(channel)
            if mask is None:
                continue
            arr = np.asarray(indices, dtype=int)
            valid = arr[(0 <= arr) & (arr < mask.size)]
            if valid.size == 0:
                continue
            previous = mask[valid].copy()
            if np.all(previous == mask_value):
                continue
            mask[valid] = mask_value
            history.append((channel, valid, previous))
        if history:
            self._undo_stack.append(history)
            self._redo_stack.clear()
            return True
        return False

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        changes = self._undo_stack.pop()
        redo_entry: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for channel, indices, previous in changes:
            mask = self.masks.get(channel)
            if mask is None:
                continue
            redo_entry.append((channel, indices, mask[indices].copy()))
            mask[indices] = previous
        if redo_entry:
            self._redo_stack.append(redo_entry)
            return True
        return False

    def redo(self) -> None:
        if not self._redo_stack:
            return False
        changes = self._redo_stack.pop()
        undo_entry: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for channel, indices, previous in changes:
            mask = self.masks.get(channel)
            if mask is None:
                continue
            undo_entry.append((channel, indices, mask[indices].copy()))
            mask[indices] = previous
        if undo_entry:
            self._undo_stack.append(undo_entry)
            return True
        return False

    def masked_percentage(self, channel: str) -> float:
        mask = self.masks.get(channel)
        if mask is None or mask.size == 0:
            return 0.0
        return float(mask.mean() * 100.0)

    def masked_intervals(self, channel: str, distance: np.ndarray) -> List[Tuple[float, float, int]]:
        mask = self.masks.get(channel)
        if mask is None or mask.size == 0:
            return []
        intervals: List[Tuple[float, float, int]] = []
        current_start = None
        for i, flagged in enumerate(mask):
            if flagged and current_start is None:
                current_start = i
            elif not flagged and current_start is not None:
                start_idx = current_start
                end_idx = i - 1
                intervals.append((float(distance[start_idx]), float(distance[end_idx]), end_idx - start_idx + 1))
                current_start = None
        if current_start is not None:
            start_idx = current_start
            end_idx = mask.size - 1
            intervals.append((float(distance[start_idx]), float(distance[end_idx]), end_idx - start_idx + 1))
        return intervals


def process_dataset(
    loaded: LoadedData,
    drop_negatives: bool,
    spacing: float,
    running_mean_width: float,
    mask_manager: Optional[MaskManager] = None,
) -> ProcessedData:
    """Process a dataset according to the configured parameters.

    The processing pipeline follows the order specified in the project
    requirements: drop negatives, resample, running mean, and mask application.
    """

    df = loaded.dataframe.sort_values(loaded.time_column).reset_index(drop=True)
    latitudes = df[loaded.latitude_column].to_numpy(dtype=float)
    longitudes = df[loaded.longitude_column].to_numpy(dtype=float)
    elevation = df[loaded.elevation_column].to_numpy(dtype=float) if loaded.elevation_column else None

    chainage, xy = compute_chainage(latitudes, longitudes)
    raw_channels: Dict[str, np.ndarray] = {}
    processed_channels: Dict[str, np.ndarray] = {}
    resampled_distance: Optional[np.ndarray] = None

    for column in loaded.channel_columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        raw_channels[column] = values.copy()
        if drop_negatives:
            values = np.where(values < 0, np.nan, values)
        if resampled_distance is None:
            resampled_distance, resampled = _resample_series(chainage, values, spacing)
        else:
            _, resampled = _resample_series(chainage, values, spacing)
        processed_channels[column] = resampled

    if resampled_distance is None:
        resampled_distance = np.array([], dtype=float)

    if mask_manager is not None:
        if not mask_manager.masks or next(iter(mask_manager.masks.values())).size != resampled_distance.size:
            mask_manager.initialise(loaded.channel_columns, resampled_distance.size)

    if resampled_distance.size > 0:
        for column in loaded.channel_columns:
            mask = mask_manager.masks.get(column) if mask_manager else None
            if running_mean_width > 0:
                data = processed_channels[column]
                filtered = data.copy()
                if mask is not None:
                    filtered = filtered.copy()
                    filtered[mask] = np.nan
                smoothed = _running_mean(resampled_distance, filtered, running_mean_width, spacing)
                processed_channels[column] = smoothed
            if mask is not None:
                processed_channels[column][mask] = np.nan
    return ProcessedData(
        distance=resampled_distance,
        raw_distance=chainage,
        channels=processed_channels,
        raw_channels=raw_channels,
        elevation=elevation,
        xy_projected=xy,
    )


def compute_rhoa(conductivity_ms: np.ndarray) -> np.ndarray:
    """Convert conductivity from mS/m to apparent resistivity in ohm-m."""

    conductivity_s = conductivity_ms * 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        rhoa = np.where(conductivity_s > 0, 1.0 / conductivity_s, np.nan)
    return rhoa


def enforce_spacing_consistency(spacing: float, running_mean_width: float) -> Tuple[float, float, bool]:
    """Ensure running mean width is not smaller than spacing.

    Returns the validated spacing, width, and a flag indicating if an
    adjustment was made.
    """

    if running_mean_width < spacing:
        return spacing, spacing, True
    return spacing, running_mean_width, False


def mask_from_distance(distance: np.ndarray, selection: Tuple[float, float]) -> np.ndarray:
    """Return boolean mask for indices within the distance selection."""

    if distance.size == 0:
        return np.array([], dtype=bool)
    start, end = selection
    return (distance >= min(start, end)) & (distance <= max(start, end))


def apply_selection_mask(
    mask_manager: MaskManager,
    distance: np.ndarray,
    selection: Tuple[float, float],
    channels: Iterable[str],
    mask_value: bool,
) -> bool:
    """Apply a selection mask to the provided channels.

    Returns ``True`` when any channel mask changed.
    """

    indices = np.flatnonzero(mask_from_distance(distance, selection))
    updates = {channel: (indices, mask_value) for channel in channels}
    return mask_manager.apply_batch(updates)


def mask_statistics(mask_manager: MaskManager, distance: np.ndarray) -> Dict[str, Dict[str, object]]:
    """Return summary statistics for all channel masks."""

    summary: Dict[str, Dict[str, object]] = {}
    for channel, mask in mask_manager.masks.items():
        summary[channel] = {
            "percent": mask_manager.masked_percentage(channel),
            "intervals": mask_manager.masked_intervals(channel, distance),
        }
    return summary
