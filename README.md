# TCM Data Cleaner

TCM Data Cleaner is a desktop GUI (PySimpleGUI + matplotlib) for reviewing GF CMD Explorer `.dat` profiles. The tool mirrors Aarhus Workbench's Profile workflow with synchronized raw/processed plots, a movable buffer window, and interactive box-selection for cutting bad data. Processing updates immediately when parameters change so you can iterate quickly before exporting Surfer `.dat` grids or a wide CSV deliverable.

## Key capabilities

- Load GF `.dat`, CSV, or Excel workbooks and map columns for time, latitude, longitude, elevation, and 1–6 conductivity channels.
- Auto-compute along-track distance (chainage) from GPS coordinates using pyproj (auto UTM) with a haversine fallback.
- Fully interactive processing pipeline: drop-negative handling, spacing control, linear resampling, and running-mean smoothing.
- Dual plot view with synchronized zoom/pan, keyboard shortcuts, and a slider-driven buffer window shown on both axes.
- Box-select regions on the raw plot and apply masks (cut/restore) per-channel with undo/redo history.
- Optional log10 apparent resistivity view. Apparent resistivity (`ρa = 1/σ`) is computed alongside conductivity.
- Export processed data to wide CSV and Surfer `.dat` (distance or projected XY) with mask-aware values, plus JSON session snapshots.

## Installation

1. Clone the repository and create a Python 3.9+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:

   ```bash
   python app.py
   ```

PySimpleGUI uses Tkinter by default. On Linux you may need the Tk development packages installed.

## Workflow

1. **Load data** – Click **Load** and choose a GF `.dat`, CSV, or Excel file. The log pane prints a preview of the detected columns.
2. **Map columns** – Use the combos to assign time, latitude, longitude, optional elevation, and the conductivity channels (mS/m). Press **Apply Mapping** to begin processing.
3. **Tune processing** – Adjust spacing, running-mean width, and the drop-negative toggle. The processed profile recomputes automatically (running mean width is clamped to the current spacing for stability).
4. **Scrub data** – Drag the buffer slider or use the rectangle selector on the raw plot. Apply **Cut (x)** or **Restore (r)** to the active selection. Undo (`z`) and redo (`y`) maintain mask history. Toggle between conductivity and log10(ρa) with `l`.
5. **Review masks** – The mask summary shows masked percentages and interval ranges per channel. Use the legend or visible channel list to focus on specific traces.
6. **Export** – Choose **Export** to write a wide CSV and Surfer `.dat` files (distance-only or projected XY). Sessions can be saved to JSON for repeatable QA runs.

## Controls & shortcuts

| Action | Control | Shortcut |
|--------|---------|----------|
| Load file | `Load` | — |
| Apply mapping | `Apply Mapping` | — |
| Cut selection | `Cut (x)` | `x` |
| Restore selection | `Restore (r)` | `r` |
| Undo | `Undo (z)` | `z` |
| Redo | `Redo (y)` | `y` |
| Toggle conductivity / log10(ρa) | `Toggle log/linear (l)` | `l` |
| Export data | `Export` | — |
| Save session | `Save Session` | — |
| Open session | `Open Session` | — |
| Move buffer | Scrub slider | Arrow keys (left/right), Shift + mouse wheel |
| Zoom buffer width | Mouse wheel | — |

Box selection is always active on the raw plot. Draw a rectangle with the left mouse button to highlight the region across both plots.

## Processing details

- Distances are computed from latitude/longitude using auto-selected UTM zones via `pyproj`. If `pyproj` is unavailable, a pure Python haversine fallback is used (less accurate but robust).
- **Pipeline order**: drop negatives → resample to uniform spacing → running-mean smoothing → apply masks (masked samples become `NaN`). Changing any parameter triggers a fast recompute.
- Running mean windows operate in meters and must be ≥ the resample spacing. Violations automatically bump the width and note it in the log pane.
- Apparent resistivity values are generated for each channel and are used for log plotting when the log mode is active.

## Export formats

- **Processed CSV** – Wide format with `Distance_m`, optional `Elevation_m`, per-channel conductivity (`*_mSperm`), and apparent resistivity (`*_rhoa_ohm_m`). Masked samples export as `NaN`.
- **Surfer `.dat`** – One file per channel. Choose between `Distance_m<TAB>Value` or projected `X_m<TAB>Y_m<TAB>Value` layouts. Apparent resistivity is exported when the UI is in log mode; otherwise conductivity is exported.
- **Session JSON** – Captures column mappings, processing parameters, masks, visible channels, and the last viewport centre/width.
- **PNG snapshot (optional)** – Use your OS screenshot tools; plotting canvases are standard Tk windows.

## PyInstaller build

A helper script is included for Windows packaging:

```bat
build_exe.bat
```

This runs PyInstaller in `--onefile` mode and writes the executable to `dist/`. Ensure the Visual C++ runtime is installed on target machines. SmartScreen may flag unsigned binaries; choose “More info” → “Run anyway”.

## Known limitations

- Matplotlib is used for plotting; extremely large datasets (> ~200k samples) may feel sluggish. The plotting adapter was written to allow a future swap to PyQtGraph.
- The haversine fallback lacks the positional accuracy of the UTM projection. Enable `pyproj` for best results.
- Elevation overlays are rendered as a light secondary axis; scaling is global, so extreme outliers may flatten subtle terrain variations.
- Multi-file batch processing and advanced masking tools (polyline deletion, statistics per selection) are not yet implemented but the code is structured to accept future extensions.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
