@echo off
echo ============================================
echo   LigEdit - Build EXE (PyQt5 + pyqtgraph)
echo ============================================
echo.

pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

pip show PyQt5 >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyQt5...
    pip install PyQt5
)

pip show pyqtgraph >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing pyqtgraph...
    pip install pyqtgraph
)

echo Building...
echo.

pyinstaller --onefile --windowed --name LigEdit --clean ^
    --hidden-import=PyQt5 --hidden-import=PyQt5.QtCore --hidden-import=PyQt5.QtGui --hidden-import=PyQt5.QtWidgets ^
    --hidden-import=pyqtgraph --hidden-import=numpy --hidden-import=scipy --hidden-import=scipy.signal ^
    --exclude-module=matplotlib --exclude-module=tkinter ^
    lig_editor.py

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
