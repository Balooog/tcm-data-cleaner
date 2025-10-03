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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import PySimpleGUI as sg

from io_handlers import export_processed_csv, export_session, export_surfer_dat, load_session, read_table
from plotting import PlotManager
from processing import (
    LoadedData,
    MaskManager,
    ProcessedData,
    apply_selection_mask,
    enforce_spacing_consistency,
    mask_statistics,
    process_dataset,
)

sg.theme("DarkBlue3")


@dataclass
class ColumnMapping:
    time: Optional[str] = None
    latitude: Optional[str] = None
    longitude: Optional[str] = None
    elevation: Optional[str] = None
    channels: List[str] = field(default_factory=list)


@dataclass
class AppState:
    dataframe: Optional[pd.DataFrame] = None
    loaded: Optional[LoadedData] = None
    processed: Optional[ProcessedData] = None
    mapping: ColumnMapping = field(default_factory=ColumnMapping)
    mask_manager: MaskManager = field(default_factory=MaskManager)
    visible_channels: List[str] = field(default_factory=list)
    drop_negatives: bool = True
    spacing: float = 1.0
    running_mean: float = 1.0
    log_mode: bool = False
    apply_visible_only: bool = True
    buffer_width: float = 50.0
    buffer_center: float = 0.0
    selection_range: Optional[tuple] = None
    filename: Optional[str] = None

    def mapped(self) -> bool:
        return all(
            [
                self.mapping.time,
                self.mapping.latitude,
                self.mapping.longitude,
                self.mapping.channels,
            ]
        )


class TCMDataCleaner:
    def __init__(self) -> None:
        self.state = AppState()
        self.window = self._build_window()
        canvas_elem = self.window["-CANVAS-"]
        self.plot_manager = PlotManager(canvas_elem, self._on_selection)
        self._log("Welcome to TCM Data Cleaner — load a GF .dat file to begin.")

    # ------------------------------------------------------------------ UI
    def _build_window(self) -> sg.Window:
        toolbar = [
            sg.Button("Load", key="-LOAD-", bind_return_key=True),
            sg.Button("Cut (x)", key="-CUT-"),
            sg.Button("Restore (r)", key="-RESTORE-"),
            sg.Button("Undo (z)", key="-UNDO-"),
            sg.Button("Redo (y)", key="-REDO-"),
            sg.Button("Toggle log/linear (l)", key="-TOGGLE-"),
            sg.Button("Export", key="-EXPORT-"),
            sg.Button("Save Session", key="-SAVESESSION-"),
            sg.Button("Open Session", key="-OPENSESSION-"),
        ]

        column_mapping = [
            [sg.Text("Time"), sg.Combo([], key="-TIME-", size=(20, 1), enable_events=True)],
            [sg.Text("Latitude"), sg.Combo([], key="-LAT-", size=(20, 1), enable_events=True)],
            [sg.Text("Longitude"), sg.Combo([], key="-LON-", size=(20, 1), enable_events=True)],
            [sg.Text("Elevation"), sg.Combo([], key="-ELEV-", size=(20, 1), enable_events=True)],
            [
                sg.Text("Channels"),
                sg.Listbox(
                    values=[],
                    select_mode=sg.SELECT_MODE_EXTENDED,
                    key="-CHANNELS-",
                    size=(20, 6),
                    enable_events=True,
                ),
            ],
            [sg.Button("Apply Mapping", key="-APPLY-")],
        ]

        controls = [
            [sg.Checkbox("Drop negatives", default=True, key="-DROPNEG-", enable_events=True)],
            [
                sg.Text("Spacing (m)"),
                sg.Input("1.0", size=(8, 1), key="-SPACING-", enable_events=True),
                sg.Text("Run-mean (m)"),
                sg.Input("1.0", size=(8, 1), key="-RUNMEAN-", enable_events=True),
            ],
            [sg.Checkbox("Apply to visible channels only", default=True, key="-VISIBLEONLY-", enable_events=True)],
            [sg.Text("Buffer width (m)"), sg.Input("50", size=(8, 1), key="-BUFFERWIDTH-", enable_events=True)],
            [
                sg.Text("Scrub"),
                sg.Slider(range=(0, 100), orientation="h", size=(30, 15), key="-SCRUB-", enable_events=True),
                sg.Text("", key="-SCRUBREADOUT-"),
            ],
            [sg.Text("Visible channels"), sg.Listbox(values=[], size=(20, 6), select_mode=sg.SELECT_MODE_EXTENDED, key="-VISIBLE-", enable_events=True)],
            [sg.Text("Mask summary"), sg.Multiline("", size=(30, 8), key="-MASKSUMMARY-", disabled=True)],
            [sg.Text("Log"), sg.Multiline("", size=(40, 12), key="-LOG-", disabled=True, autoscroll=True)],
        ]

        layout = [
            toolbar,
            [
                sg.Column([[sg.Canvas(size=(800, 600), key="-CANVAS-")]], expand_x=True, expand_y=True),
                sg.VerticalSeparator(),
                sg.Column(column_mapping + controls, expand_y=True, scrollable=True, vertical_scroll_only=True),
            ],
        ]
        window = sg.Window(
            "TCM Data Cleaner",
            layout,
            finalize=True,
            resizable=True,
            return_keyboard_events=True,
            enable_close_attempted_event=True,
        )
        return window

    def _log(self, message: str) -> None:
        log_elem = self.window["-LOG-"]
        current = log_elem.get() if log_elem else ""
        log_elem.update(current + message + "\n")

    # ---------------------------------------------------------------- Processing helpers
    def _reprocess(self) -> None:
        if not self.state.loaded:
            return
        try:
            spacing = float(self.window["-SPACING-"].get())
            runmean = float(self.window["-RUNMEAN-"].get())
        except (TypeError, ValueError):
            self._log("Invalid spacing or running mean value.")
            return
        if spacing <= 0:
            spacing = 0.5
            self.window["-SPACING-"].update("0.5")
            self._log("Spacing must be positive; defaulting to 0.5 m.")
        spacing, runmean, adjusted = enforce_spacing_consistency(spacing, runmean)
        if adjusted:
            self.window["-RUNMEAN-"].update(f"{runmean:.3f}")
            self._log("Running mean increased to match spacing.")
        self.state.spacing = spacing
        self.state.running_mean = runmean
        self.state.drop_negatives = bool(self.window["-DROPNEG-"].get())
        processed = process_dataset(
            self.state.loaded,
            drop_negatives=self.state.drop_negatives,
            spacing=spacing,
            running_mean_width=runmean,
            mask_manager=self.state.mask_manager,
        )
        self.state.processed = processed
        if not self.state.visible_channels:
            self.state.visible_channels = list(self.state.loaded.channel_columns)
            if self.state.visible_channels:
                self.window["-VISIBLE-"].update(
                    values=self.state.visible_channels, set_to_index=list(range(len(self.state.visible_channels)))
                )
        if processed.distance.size > 0 and self.state.buffer_center == 0.0:
            self.state.buffer_center = float(processed.distance.mean())
        self._update_plot()
        self._update_mask_summary()

    def _update_plot(self) -> None:
        if not self.state.processed or not self.state.loaded:
            return
        processed = self.state.processed
        visible = self.state.visible_channels or self.state.loaded.channel_columns
        selection = self.state.selection_range
        self.plot_manager.set_log_scale(self.state.log_mode)
        self.plot_manager.update_data(
            processed.raw_distance,
            processed.raw_channels,
            processed.distance,
            processed.channels,
            visible,
            masked=self.state.mask_manager.masks,
            elevation=processed.elevation,
            distance_elev=processed.distance if processed.elevation is not None else None,
        )
        center = self.state.buffer_center or (processed.distance.mean() if processed.distance.size else 0.0)
        self.plot_manager.update_buffer(center, self.state.buffer_width)
        self.plot_manager.highlight_selection(selection)
        if processed.distance.size > 0:
            slider = self.window["-SCRUB-"]
            slider.update(range=(processed.distance.min(), processed.distance.max()))
            slider.update(value=center)
            self.window["-SCRUBREADOUT-"].update(f"{center:.2f} m")
            if selection is None:
                half = self.state.buffer_width / 2.0
                self.state.selection_range = (center - half, center + half)
                self.plot_manager.highlight_selection(self.state.selection_range)

    def _update_mask_summary(self) -> None:
        if not self.state.processed:
            return
        stats = mask_statistics(self.state.mask_manager, self.state.processed.distance)
        lines = []
        for channel, payload in stats.items():
            percent = payload["percent"]
            intervals = payload["intervals"]
            lines.append(f"{channel}: {percent:.2f}% masked")
            for start, end, count in intervals:
                lines.append(f"  {start:.1f} – {end:.1f} m ({count} samples)")
        self.window["-MASKSUMMARY-"].update("\n".join(lines))

    # ---------------------------------------------------------------- Selection callbacks
    def _on_selection(self, selection: Optional[tuple]) -> None:
        self.state.selection_range = selection
        self.plot_manager.highlight_selection(selection)
        if selection and self.state.processed:
            center = sum(selection) / 2.0
            self.state.buffer_center = center
            self.plot_manager.update_buffer(center, self.state.buffer_width)
            if self.state.processed.distance.size > 0:
                self.window["-SCRUB-"].update(value=center)
                self.window["-SCRUBREADOUT-"].update(f"{center:.2f} m")

    def _selection_indices(self) -> np.ndarray:
        if not self.state.processed or not self.state.selection_range:
            return np.array([], dtype=int)
        start, end = self.state.selection_range
        distance = self.state.processed.distance
        mask = (distance >= min(start, end)) & (distance <= max(start, end))
        return np.flatnonzero(mask)

    # ---------------------------------------------------------------- Event handling
    def run(self) -> None:
        while True:
            event, values = self.window.read(timeout=100)
            if event in (sg.WIN_CLOSED, sg.WINDOW_CLOSE_ATTEMPTED_EVENT):
                break
            if event == "-LOAD-":
                self._handle_load()
            elif event == "-APPLY-":
                self._handle_apply_mapping()
            elif event in ("-DROPNEG-", "-SPACING-", "-RUNMEAN-", "-VISIBLE-", "-BUFFERWIDTH-"):
                self._handle_parameter_change(event)
            elif event == "-VISIBLEONLY-":
                self.state.apply_visible_only = bool(values["-VISIBLEONLY-"])
            elif event == "-CUT-":
                self._handle_mask_operation(True)
            elif event == "-RESTORE-":
                self._handle_mask_operation(False)
            elif event == "-UNDO-":
                if self.state.mask_manager.undo():
                    self._reprocess()
            elif event == "-REDO-":
                if self.state.mask_manager.redo():
                    self._reprocess()
            elif event == "-TOGGLE-":
                self.state.log_mode = not self.state.log_mode
                self._log(f"Plot mode set to {'log10(ρa)' if self.state.log_mode else 'conductivity'}.")
                self._update_plot()
            elif event == "-EXPORT-":
                self._handle_export()
            elif event == "-SAVESESSION-":
                self._handle_save_session()
            elif event == "-OPENSESSION-":
                self._handle_open_session()
            elif event == "-SCRUB-":
                self._handle_scrub(values)
            elif isinstance(event, str) and len(event) == 1:
                self._handle_shortcut(event)
        self.window.close()

    # ---------------------------------------------------------------- Event helpers
    def _handle_load(self) -> None:
        file = sg.popup_get_file("Open GF .dat or table", file_types=(("Data", "*.dat;*.csv;*.txt;*.xls;*.xlsx"),))
        if not file:
            return
        try:
            df = read_table(file)
        except Exception as exc:  # pragma: no cover - GUI feedback
            self._log(f"Failed to open file: {exc}")
            return
        self.state.dataframe = df
        self.state.filename = file
        columns = list(df.columns)
        for key in ("-TIME-", "-LAT-", "-LON-", "-ELEV-"):
            self.window[key].update(values=columns)
        self.window["-CHANNELS-"].update(values=columns)
        self._log(f"Loaded {Path(file).name} with {len(df)} rows and columns {columns}.")

    def _handle_apply_mapping(self) -> None:
        if self.state.dataframe is None:
            sg.popup_error("Load data first.")
            return
        all_channels = getattr(self.window["-CHANNELS-"], "Values", []) or []
        mapping = ColumnMapping(
            time=self.window["-TIME-"].get(),
            latitude=self.window["-LAT-"].get(),
            longitude=self.window["-LON-"].get(),
            elevation=self.window["-ELEV-"].get() or None,
            channels=list(all_channels),
        )
        channels_selected = self.window["-CHANNELS-"].get()
        if not mapping.time or not mapping.latitude or not mapping.longitude:
            sg.popup_error("Time, latitude, and longitude columns are required.")
            return
        if not channels_selected:
            sg.popup_error("Select at least one channel column.")
            return
        mapping.channels = list(channels_selected)
        loaded = LoadedData(
            dataframe=self.state.dataframe,
            time_column=mapping.time,
            latitude_column=mapping.latitude,
            longitude_column=mapping.longitude,
            elevation_column=mapping.elevation,
            channel_columns=mapping.channels,
        )
        self.state.mapping = mapping
        self.state.loaded = loaded
        self.state.mask_manager = MaskManager()
        self.state.visible_channels = list(mapping.channels)
        self.state.buffer_center = 0.0
        self.state.selection_range = None
        self.window["-VISIBLE-"].update(values=self.state.visible_channels, set_to_index=list(range(len(self.state.visible_channels))))
        self._log(
            "Applied column mapping: "
            + json.dumps(
                {
                    "time": mapping.time,
                    "lat": mapping.latitude,
                    "lon": mapping.longitude,
                    "elev": mapping.elevation,
                    "channels": mapping.channels,
                }
            )
        )
        self._reprocess()

    def _handle_parameter_change(self, event: str) -> None:
        if event == "-VISIBLE-":
            selection = self.window["-VISIBLE-"].get()
            if selection:
                self.state.visible_channels = list(selection)
                self._update_plot()
            return
        if event == "-BUFFERWIDTH-":
            try:
                width = float(self.window["-BUFFERWIDTH-"].get())
                self.state.buffer_width = max(width, 1.0)
                if self.state.processed:
                    self.plot_manager.update_buffer(self.plot_manager.buffer_center, self.state.buffer_width)
            except (TypeError, ValueError):
                self._log("Invalid buffer width.")
            return
        self._reprocess()

    def _handle_mask_operation(self, mask_value: bool) -> None:
        if not self.state.processed or not self.state.loaded:
            return
        indices = self._selection_indices()
        if indices.size == 0:
            self._log("No samples selected. Use the box selector or scrub window to choose a region.")
            return
        channels = self.state.visible_channels if self.state.apply_visible_only else self.state.loaded.channel_columns
        selection = (self.state.processed.distance[min(indices)], self.state.processed.distance[max(indices)])
        changed = apply_selection_mask(
            self.state.mask_manager,
            self.state.processed.distance,
            selection,
            channels,
            mask_value,
        )
        if not changed:
            self._log("Selection already in desired state; no mask changes made.")
            return
        self._reprocess()
        self._log(("Masked" if mask_value else "Restored") + f" {indices.size} samples across {len(channels)} channel(s).")

    def _handle_export(self) -> None:
        if not self.state.processed or not self.state.loaded:
            sg.popup_error("Nothing to export yet.")
            return
        folder = sg.popup_get_folder("Select export directory")
        if not folder:
            return
        mode = "log" if self.state.log_mode else "cond"
        base_name = Path(self.state.filename or "export").stem
        export_processed_csv(self.state.processed, os.path.join(folder, f"{base_name}__processed.csv"), mode, self.state.visible_channels)
        use_xy = sg.popup_yes_no("Export XY columns for Surfer DAT?", default_yes=True) == "Yes"
        include_header = sg.popup_yes_no("Include header row?", default_yes=True) == "Yes"
        files = export_surfer_dat(
            self.state.processed,
            output_dir=folder,
            base_name=base_name,
            mode=mode,
            use_xy=use_xy,
            include_header=include_header,
            visible_channels=self.state.visible_channels,
        )
        self._log("Exported files:\n" + "\n".join(files))

    def _handle_save_session(self) -> None:
        if not self.state.loaded or not self.state.processed:
            sg.popup_error("Load and process data before saving a session.")
            return
        file = sg.popup_get_file("Save session", save_as=True, default_extension=".json")
        if not file:
            return
        payload = {
            "mapping": self.state.mapping.__dict__,
            "parameters": {
                "drop_negatives": self.state.drop_negatives,
                "spacing": self.state.spacing,
                "running_mean": self.state.running_mean,
                "log_mode": self.state.log_mode,
                "apply_visible_only": self.state.apply_visible_only,
                "buffer_width": self.state.buffer_width,
            },
            "masks": {ch: mask.tolist() for ch, mask in self.state.mask_manager.masks.items()},
            "visible_channels": self.state.visible_channels,
            "selection": self.state.selection_range,
            "filename": self.state.filename,
        }
        export_session(file, payload)
        self._log(f"Session saved to {file}.")

    def _handle_open_session(self) -> None:
        file = sg.popup_get_file("Open session", file_types=(("Session", "*.json"),))
        if not file:
            return
        try:
            payload = load_session(file)
        except Exception as exc:
            sg.popup_error(f"Failed to load session: {exc}")
            return
        data_file = payload.get("filename")
        if not data_file or not Path(data_file).exists():
            sg.popup_error("Original data file not found. Load it manually first.")
            return
        self.state = AppState()
        self.window.close()
        self.__init__()
        self.state.dataframe = read_table(data_file)
        self.state.filename = data_file
        columns = list(self.state.dataframe.columns)
        for key in ("-TIME-", "-LAT-", "-LON-", "-ELEV-"):
            self.window[key].update(values=columns)
        self.window["-CHANNELS-"].update(values=columns)
        mapping_payload = payload.get("mapping", {})
        mapping = ColumnMapping(**mapping_payload)
        self.state.mapping = mapping
        loaded = LoadedData(
            dataframe=self.state.dataframe,
            time_column=mapping.time,
            latitude_column=mapping.latitude,
            longitude_column=mapping.longitude,
            elevation_column=mapping.elevation,
            channel_columns=mapping.channels,
        )
        self.window["-TIME-"].update(value=mapping.time)
        self.window["-LAT-"].update(value=mapping.latitude)
        self.window["-LON-"].update(value=mapping.longitude)
        self.window["-ELEV-"].update(value=mapping.elevation)
        if mapping.channels:
            indices = [columns.index(ch) for ch in mapping.channels if ch in columns]
            self.window["-CHANNELS-"].update(set_to_index=indices)
        self.state.loaded = loaded
        self.state.mask_manager = MaskManager()
        masks_payload = {
            channel: np.array(mask, dtype=bool)
            for channel, mask in payload.get("masks", {}).items()
        }
        params = payload.get("parameters", {})
        self.state.drop_negatives = params.get("drop_negatives", True)
        self.state.spacing = params.get("spacing", 1.0)
        self.state.running_mean = params.get("running_mean", 1.0)
        self.state.log_mode = params.get("log_mode", False)
        self.state.apply_visible_only = params.get("apply_visible_only", True)
        self.state.buffer_width = params.get("buffer_width", 50.0)
        self.state.visible_channels = payload.get("visible_channels", mapping.channels)
        self.state.selection_range = payload.get("selection")
        self._reprocess()
        if masks_payload:
            size = self.state.processed.distance.size if self.state.processed else 0
            for channel, mask in masks_payload.items():
                if size and mask.size != size:
                    resized = np.zeros(size, dtype=bool)
                    length = min(size, mask.size)
                    resized[:length] = mask[:length]
                    mask = resized
                self.state.mask_manager.masks[channel] = mask
            self._reprocess()
        self._log(f"Session loaded from {file}.")

    def _handle_scrub(self, values: Dict) -> None:
        if not self.state.processed:
            return
        center = float(values["-SCRUB-"])
        self.state.buffer_center = center
        self.plot_manager.update_buffer(center, self.state.buffer_width)
        self.window["-SCRUBREADOUT-"].update(f"{center:.2f} m")
        half = self.state.buffer_width / 2.0
        self.state.selection_range = (center - half, center + half)
        self.plot_manager.highlight_selection(self.state.selection_range)

    def _handle_shortcut(self, key: str) -> None:
        mapping = {
            "x": lambda: self._handle_mask_operation(True),
            "r": lambda: self._handle_mask_operation(False),
            "z": self.state.mask_manager.undo,
            "y": self.state.mask_manager.redo,
            "l": lambda: self.window.write_event_value("-TOGGLE-", None),
        }
        action = mapping.get(key.lower())
        if not action:
            return
        result = action()
        if key.lower() in {"z", "y"} and result:
            self._reprocess()


def main() -> None:
    app = TCMDataCleaner()
    app.run()


if __name__ == "__main__":
    main()
