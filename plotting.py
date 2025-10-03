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

import itertools
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import numpy as np

COLOR_CYCLE = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


class PlotManager:
    """Handle matplotlib drawing and interactive tools."""

    def __init__(
        self,
        canvas_parent,
        selection_callback: Callable[[Tuple[float, float]], None],
    ) -> None:
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax_raw = self.figure.add_subplot(211)
        self.ax_processed = self.figure.add_subplot(212, sharex=self.ax_raw)
        self.ax_raw.set_title("Raw")
        self.ax_processed.set_title("Processed")
        self.ax_raw.grid(True, alpha=0.3)
        self.ax_processed.grid(True, alpha=0.3)
        self.ax_processed.set_xlabel("Distance (m)")
        self.ax_raw.set_ylabel("Conductivity (mS/m)")
        self.ax_processed.set_ylabel("Conductivity (mS/m)")
        self.figure.subplots_adjust(hspace=0.15)

        self.canvas_parent = canvas_parent
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_parent.TKCanvas)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        self.selection_callback = selection_callback
        self.selector = RectangleSelector(
            self.ax_raw,
            self._on_select,
            drawtype="box",
            interactive=False,
            button=[1],
            minspanx=0.01,
            minspany=0.01,
            useblit=False,
        )
        self.buffer_width = 50.0
        self.buffer_center = 0.0
        self.buffer_raw_patch = Rectangle((0, 0), 1, 1, facecolor="tab:orange", alpha=0.15)
        self.buffer_processed_patch = Rectangle((0, 0), 1, 1, facecolor="tab:orange", alpha=0.15)
        self.buffer_raw_patch.set_transform(self.ax_raw.get_xaxis_transform())
        self.buffer_processed_patch.set_transform(self.ax_processed.get_xaxis_transform())
        self.ax_raw.add_patch(self.buffer_raw_patch)
        self.ax_processed.add_patch(self.buffer_processed_patch)
        self._selection_distance: Optional[Tuple[float, float]] = None
        self._selection_artist_raw = None
        self._selection_artist_processed = None
        self._channel_lines_raw: Dict[str, List] = {}
        self._channel_lines_processed: Dict[str, List] = {}
        self._log_scale = False
        self._elevation_axis = None
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event) -> None:
        if event.key in {"left", "right"}:
            step = self.buffer_width / 5.0
            if event.key == "left":
                center = self.buffer_center - step
            else:
                center = self.buffer_center + step
            self.update_buffer(center, self.buffer_width)
            self._selection_distance = (center - self.buffer_width / 2.0, center + self.buffer_width / 2.0)
            if self.selection_callback:
                self.selection_callback(self._selection_distance)

    def _on_scroll(self, event) -> None:
        if event.key == "shift":
            # pan
            shift = -event.step * self.buffer_width * 0.1
            center = self.buffer_center + shift
            self.update_buffer(center, self.buffer_width)
        else:
            # zoom
            factor = 0.9 if event.step > 0 else 1.1
            new_width = max(1.0, self.buffer_width * factor)
            center = self.buffer_center
            self.update_buffer(center, new_width)
        self._selection_distance = (self.buffer_center - self.buffer_width / 2.0, self.buffer_center + self.buffer_width / 2.0)
        if self.selection_callback:
            self.selection_callback(self._selection_distance)

    def _on_select(self, eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        selection = (float(eclick.xdata), float(erelease.xdata))
        self._selection_distance = selection
        if self.selection_callback:
            self.selection_callback(selection)

    # Data management -----------------------------------------------------
    def set_log_scale(self, enable: bool) -> None:
        self._log_scale = enable
        if enable:
            self.ax_raw.set_ylabel("log10(ρa) [Ω·m]")
            self.ax_processed.set_ylabel("log10(ρa) [Ω·m]")
        else:
            self.ax_raw.set_ylabel("Conductivity (mS/m)")
            self.ax_processed.set_ylabel("Conductivity (mS/m)")
        self.canvas.draw_idle()

    def update_data(
        self,
        distance_raw: np.ndarray,
        raw_channels: Dict[str, np.ndarray],
        distance_processed: np.ndarray,
        processed_channels: Dict[str, np.ndarray],
        visible_channels: Sequence[str],
        selection_indices: Optional[np.ndarray] = None,
        masked: Optional[Dict[str, np.ndarray]] = None,
        elevation: Optional[np.ndarray] = None,
        distance_elev: Optional[np.ndarray] = None,
    ) -> None:
        """Update plot data."""

        for ax in (self.ax_raw, self.ax_processed):
            ax.cla()
            ax.grid(True, alpha=0.3)
        self.ax_raw.set_title("Raw")
        self.ax_processed.set_title("Processed")
        self.ax_processed.set_xlabel("Distance (m)")
        if self._log_scale:
            self.ax_raw.set_ylabel("log10(ρa) [Ω·m]")
            self.ax_processed.set_ylabel("log10(ρa) [Ω·m]")
        else:
            self.ax_raw.set_ylabel("Conductivity (mS/m)")
            self.ax_processed.set_ylabel("Conductivity (mS/m)")

        self.ax_raw.add_patch(self.buffer_raw_patch)
        self.ax_processed.add_patch(self.buffer_processed_patch)

        color_iter = itertools.cycle(COLOR_CYCLE)
        self._channel_lines_raw.clear()
        self._channel_lines_processed.clear()
        for channel in visible_channels:
            color = next(color_iter)
            raw_values = raw_channels.get(channel)
            processed_values = processed_channels.get(channel)
            if raw_values is None or processed_values is None:
                continue
            if self._log_scale:
                raw_plot = np.where(raw_values > 0, np.log10(raw_values), np.nan)
                processed_plot = np.where(processed_values > 0, np.log10(processed_values), np.nan)
            else:
                raw_plot = raw_values
                processed_plot = processed_values

            lines_raw = self.ax_raw.plot(distance_raw, raw_plot, label=channel, color=color, alpha=0.8)
            lines_processed = self.ax_processed.plot(
                distance_processed,
                processed_plot,
                label=channel,
                color=color,
                alpha=0.9,
            )
            if masked and channel in masked:
                mask = masked[channel]
                if mask.size == processed_values.size:
                    masked_distance = distance_processed[mask]
                    masked_values = processed_plot[mask]
                    self.ax_processed.scatter(masked_distance, masked_values, color=color, marker="x", alpha=0.9)
            self._channel_lines_raw[channel] = lines_raw
            self._channel_lines_processed[channel] = lines_processed

        if self._elevation_axis:
            self._elevation_axis.remove()
            self._elevation_axis = None
        if elevation is not None and distance_elev is not None and elevation.size == distance_elev.size:
            self._elevation_axis = self.ax_processed.twinx()
            self._elevation_axis.plot(distance_elev, elevation, color="tab:gray", linewidth=1.0, alpha=0.7)
            self._elevation_axis.set_ylabel("Elevation (m)")
            self._elevation_axis.grid(False)

        if self.ax_raw.lines:
            self.ax_raw.legend(loc="upper right")
        if self.ax_processed.lines:
            self.ax_processed.legend(loc="upper right")
        self._update_buffer_patch()
        self.highlight_selection(self._selection_distance)
        self.canvas.draw_idle()

    def _update_buffer_patch(self) -> None:
        for patch in (self.buffer_raw_patch, self.buffer_processed_patch):
            patch.set_x(self.buffer_center - self.buffer_width / 2.0)
            patch.set_width(self.buffer_width)
            patch.set_y(0)
            patch.set_height(1)
        self.ax_raw.relim()
        self.ax_raw.autoscale(axis="y")
        self.ax_processed.relim()
        self.ax_processed.autoscale(axis="y")

    def update_buffer(self, center: float, width: float) -> None:
        self.buffer_center = center
        self.buffer_width = max(width, 1.0)
        self._update_buffer_patch()
        self.canvas.draw_idle()

    # Selection feedback --------------------------------------------------
    def highlight_selection(
        self,
        selection: Optional[Tuple[float, float]],
    ) -> None:
        self._selection_distance = selection
        if self._selection_artist_raw:
            self._selection_artist_raw.remove()
            self._selection_artist_raw = None
        if self._selection_artist_processed:
            self._selection_artist_processed.remove()
            self._selection_artist_processed = None
        if selection is None:
            self.canvas.draw_idle()
            return
        start, end = selection
        x0, x1 = min(start, end), max(start, end)
        self._selection_artist_raw = self.ax_raw.axvspan(x0, x1, color="tab:blue", alpha=0.1)
        self._selection_artist_processed = self.ax_processed.axvspan(x0, x1, color="tab:blue", alpha=0.1)
        self.canvas.draw_idle()

    @property
    def current_selection(self) -> Optional[Tuple[float, float]]:
        return self._selection_distance
