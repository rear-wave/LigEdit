# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LigEdit is a desktop application for viewing and editing lightning waveform data stored in proprietary `.lig` binary files. Built with PyQt5 + pyqtgraph. A 5-step data processing pipeline classifies lightning events by distance and day/night.

## Commands

```bash
# Run the application
python lig_editor.py

# Build standalone EXE (Windows)
build.bat
```

No test suite, no linting configuration.

## Architecture

### Binary .lig file format

Three supported versions (1001, 2001, 3001). Each file has:
- **File header** (28 bytes): version, NumOfPiece, cache counts, GPS times for first/last piece
- **Pieces** (variable length): each piece has metadata struct (sampling rate, channels, GPS location, etc.) followed by raw waveform samples as uint16 arrays (`struct.unpack('{cnt}H', ...)`)

Key parsing functions: `ReadLigFileWithOffsets()` tracks byte offsets for non-destructive editing; `ReadLigFile()` is a simpler version for pipeline use.

### Module map

| Module | Role |
|--------|------|
| `lig_editor.py` | Entry point (`main()`), .lig binary parser, Butterworth lowpass filter (300kHz @ 5MHz), station coordinate matching, time formatting, `SaveLigFile()` / `repacklig()` |
| `main_window.py` | PyQt5 `MainWindow`: multi-file tree view (file → timestamp pieces), check/delete state per piece, waveform preview, export/save operations. Data model: `file_data` dict keyed by filepath, `deleted_sets` / `checked_sets` dicts keyed by filepath with `set()` of indices |
| `waveform_widget.py` | Dual-panel pyqtgraph widget: top = detail view (Y-axis zoom via scroll wheel, X-pan via drag), bottom = overview with draggable region box. Uses downsampling + throttled (~33fps) linkage between views. `SCOPE_STYLE` dict defines the dark oscilloscope theme |
| `pipeline.py` | 5-step batch processing: (1) extract timestamps from lig files, (2) match against WWLLN `.loc` data using dual-pointer algorithm, (3) distance range selection, (4) extract and repack matching waveforms, (5) day/night classification. `_PieceWriter` auto-splits output into 512-piece `.lig` files |
| `pipeline_dialog.py` | `DistanceClassifyDialog` and `DayNightClassifyDialog` — Qt dialogs that run pipeline steps on a `QThread` worker |

### Data files (bundled at app root)

- `站点经纬度.txt` — station names and coordinates (name on one line, lat/lon on next)
- `LigHead.lig` — binary header template for new lig file creation (`lig_file_head_path`)
- `Limitbyt` — binary template used by `repacklig()` for piece packaging (`lig_head_path`)

### Key behaviors

- Deleting pieces marks them in `deleted_sets`; actual data is only removed on save via `SaveLigFile()`, which rebuilds the file excluding deleted byte ranges
- Checking pieces (double-click or right-click) marks them for batch export
- Waveform preview shows both raw data (pink) and Butterworth-filtered data (white), with color changes for deleted (red) and checked (cyan) states
- Station matching is nearest-neighbor with Chebyshev distance, tolerance 0.02 degrees
- Time display uses UTC→Beijing (+8h) conversion for day/night classification, with 5:30–19:00 Beijing time counted as daytime
- The `_resource_path()` helper resolves file paths for both dev and PyInstaller-frozen environments
