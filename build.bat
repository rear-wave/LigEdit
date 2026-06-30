@echo off
echo ============================================
echo   LigEdit - Build EXE (Windows)
echo ============================================
echo.

pip show PyInstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

echo Building LigEdit.exe ...
echo.

pyinstaller --onefile --windowed --name LigEdit --clean ^
    --runtime-hook runtime_hook_onnx.py ^
    --hidden-import=PyQt5 --hidden-import=PyQt5.QtCore --hidden-import=PyQt5.QtGui --hidden-import=PyQt5.QtWidgets ^
    --hidden-import=PyQt5.sip ^
    --hidden-import=pyqtgraph ^
    --hidden-import=numpy --hidden-import=scipy --hidden-import=scipy.signal ^
    --hidden-import=scipy.stats --hidden-import=scipy.ndimage ^
    --hidden-import=pandas --hidden-import=pandas._libs ^
    --hidden-import=sklearn --hidden-import=sklearn.cluster --hidden-import=sklearn.mixture ^
    --hidden-import=sklearn.manifold --hidden-import=sklearn.preprocessing ^
    --hidden-import=sklearn.decomposition --hidden-import=sklearn.neighbors ^
    --hidden-import=sklearn.metrics ^
    --hidden-import=lig_parser --hidden-import=pipeline --hidden-import=pipeline_dialog ^
    --hidden-import=waveform_widget --hidden-import=main_window ^
    --hidden-import=analytics --hidden-import=analytics.trace_core ^
    --hidden-import=analytics.trace_dialog --hidden-import=analytics.cluster_core ^
    --hidden-import=analytics.cluster_dialog --hidden-import=analytics.analyse_core ^
    --hidden-import=analytics.analyse_dialog ^
    --hidden-import=classify_module --hidden-import=classify_dialog ^
    --collect-binaries onnxruntime ^
    --add-data "LigHead.lig;." --add-data "Limitbyt;." --add-data "站点经纬度.txt;." ^
    --add-data "checkpoints/resnet.onnx;checkpoints" ^
    --exclude-module=matplotlib --exclude-module=tkinter ^
    --exclude-module=torch --exclude-module=IPython --exclude-module=PIL ^
    main_window.py

echo.
if exist "dist\LigEdit.exe" (
    echo ============================================
    echo   SUCCESS: dist\LigEdit.exe
    echo ============================================
) else (
    echo ============================================
    echo   FAILED - check errors above
    echo ============================================
)

pause