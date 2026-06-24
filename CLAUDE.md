# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LigEdit is a desktop application for viewing and editing lightning waveform data stored in proprietary `.lig` binary files. Built with PyQt5 + pyqtgraph. It provides a 5-step batch processing pipeline (distance + day/night classification) and three analytics modules: multi-station trace matching, waveform clustering, and lightning data analysis.

### Environment

Conda environment `lightning` — Python 3.11 with PyQt5, pyqtgraph, numpy, scipy, pandas, scikit-learn, openpyxl.

```bash
conda activate lightning
```

## Commands

```bash
# Run the application
python lig_editor.py

# Run with lig files as CLI arguments
python lig_editor.py data1.lig data2.lig

# Build standalone EXE (Windows only)
build.bat
```

No test suite, no linting configuration.

## Architecture

### Shared module: `lig_parser.py`

Extracted from the original `lig_editor.py` to eliminate duplication across modules. All other files import from here:

- **Binary I/O:** `ReadLigFile`, `ReadLigFileWithOffsets`, `repacklig`, `PieceWriter` class
- **Signal processing:** `ButterFilter` (4th-order Butterworth lowpass, 300kHz @ 5MHz), `CutPieceTo16000`
- **Time utilities:** `compute_final_time`, `format_time_display`, `time_classifier_display`, `format_txt_time`
- **Station matching:** `load_station_coords`, `match_station_name` (Chebyshev distance, 0.02° tolerance)
- **Geo math:** `deg2rad`, `haversine_distance`
- **Resource:** `_resource_path()` — resolves paths for dev and PyInstaller-frozen environments

### Binary .lig file format

Three supported versions (1001, 2001, 3001). Each file has:

- **File header** (28 bytes): version, NumOfPiece, cache counts, GPS times for first/last piece
- **Pieces** (variable length): metadata struct (sampling rate, channels, GPS location) followed by raw waveform samples as uint16 arrays (`struct.unpack('{cnt}H', ...)`)

`ReadLigFileWithOffsets()` tracks byte offsets for non-destructive editing; `ReadLigFile()` is a simpler version for pipeline use.

### Module map

| Module               | Role |
| -------------------- | ---- |
| `lig_editor.py` | Entry point (`main()`), `SaveLigFile()`, `MergeLigFiles()`. Imports everything else from `lig_parser.py`. |
| `lig_parser.py` | Shared .lig binary parsing, signal processing, time utilities, station matching (~280 lines). |
| `main_window.py` | PyQt5 `MainWindow`: multi-file tree view (file → timestamp pieces), check/delete state per piece, waveform preview, export/save. Data model: `file_data` / `deleted_sets` / `checked_sets` dicts keyed by filepath. |
| `waveform_widget.py` | Dual-panel pyqtgraph widget: detail view (top) + overview with draggable region (bottom). `SCOPE_STYLE` dict defines dark oscilloscope theme. ~33fps throttled linkage. |
| `pipeline.py` | 5-step batch processing: extract timestamps → match WWLLN `.loc` data → distance range selection → extract/repack → day/night classification. `PieceWriter` auto-splits output into 512-piece `.lig` files. |
| `pipeline_dialog.py` | `DistanceClassifyDialog` and `DayNightClassifyDialog` — Qt dialogs running pipeline steps on `QThread` worker. |

### `analytics/` package

Three integrated modules (originally standalone projects: LigTrace, LigCluster, LigAnalyse). All use the `QDialog` + `QThread` worker pattern and import shared utilities from `lig_parser.py`.

| Module | Role |
| ------- | ---- |
| `analytics/trace_core.py` | Multi-station event matching: loads station timelines + WWLLN data, binary-searches closest unused piece within time window, writes matched `.lig` + `.txt` output. |
| `analytics/trace_dialog.py` | QDialog for trace matching: scrollable station config (add/remove rows), WWLLN/output directory selectors, min_stations/time_window params, progress bar, log area. |
| `analytics/cluster_core.py` | Waveform clustering: 16-dim feature extraction (peak, rise/fall time, zero crossings, kurtosis, spectral centroid), sklearn algorithms (KMeans, DBSCAN, Agglomerative, GMM), dimensionality reduction (t-SNE, PCA), cluster evaluation. |
| `analytics/cluster_dialog.py` | QDialog for clustering: algorithm/feature/dim-reduction selectors, pyqtgraph scatter + waveform preview, evaluation scores, export to `.lig` or CSV. |
| `analytics/analyse_core.py` | Lightning data analysis: distance distribution, current distribution, independent event classification. Supports WWLLN `.loc`, NBE `.loc`, and xlsx data sources. |
| `analytics/analyse_dialog.py` | QDialog for analysis: path selectors, pie chart widget, distance/current/independent pyqtgraph plots, CSV export. |

### Data files (bundled at app root)

- `站点经纬度.txt` — station names and coordinates (name on one line, lat/lon on next)
- `LigHead.lig` — binary header template for new lig file creation
- `Limitbyt` — binary template used by `repacklig()` for piece packaging

### Key behaviors

- Deleting pieces marks them in `deleted_sets`; data is only removed on save via `SaveLigFile()`, which rebuilds the file excluding deleted byte ranges
- Checking pieces (double-click or right-click) marks them for batch export
- Waveform preview shows raw data (pink) and Butterworth-filtered data (white), with color changes for deleted (red) and checked (cyan) states
- Time display uses UTC→Beijing (+8h) conversion for day/night classification (5:30–19:00 Beijing = daytime)