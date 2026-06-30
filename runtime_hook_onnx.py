# PyInstaller runtime hook: 在 EXE 启动时预加载 onnxruntime
# 必须在 PyQt5 等其他 C++ 库之前加载，避免 CRT 冲突
import os, sys

if getattr(sys, 'frozen', False):
    ort_dir = os.path.join(sys._MEIPASS, 'onnxruntime', 'capi')
    if os.path.isdir(ort_dir):
        os.add_dll_directory(ort_dir)
    os.add_dll_directory(sys._MEIPASS)

    # 预加载 onnxruntime，必须赶在 PyQt5 之前
    try:
        import onnxruntime
    except Exception:
        pass  # 不阻塞启动，后续 classify_module 会再次尝试并报详细错误
