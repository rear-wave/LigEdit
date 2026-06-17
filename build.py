#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LigEdit — Build EXE / .app 打包脚本
支持 Windows (.exe) 和 macOS (.app)
"""

import os
import sys
import shutil
import subprocess
import platform

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))


def check_deps():
    """检查 PyInstaller 是否安装"""
    try:
        import PyInstaller
        print("✓ PyInstaller 已安装")
    except ImportError:
        print("正在安装 PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def get_extra_args():
    """获取平台特定的打包参数"""
    is_win = platform.system() == "Windows"
    return ["--windowed"] if not is_win else []


def get_add_data():
    """获取需要打包的数据文件和模块"""
    if platform.system() == "Windows":
        sep = ";"
    else:
        sep = ":"

    data = [
        ("LigHead.lig", "."),
        ("Limitbyt", "."),
        ("站点经纬度.txt", "."),
    ]
    return [f"--add-data={src}{sep}{dst}" for src, dst in data]


def get_hidden_imports():
    """必须的 hidden-imports 列表"""
    return [
        "--hidden-import=PyQt5",
        "--hidden-import=PyQt5.QtCore",
        "--hidden-import=PyQt5.QtGui",
        "--hidden-import=PyQt5.QtWidgets",
        "--hidden-import=PyQt5.sip",
        "--hidden-import=pyqtgraph",
        "--hidden-import=numpy",
        "--hidden-import=scipy",
        "--hidden-import=scipy.signal",
        "--hidden-import=scipy.stats",
        "--hidden-import=scipy.ndimage",
        "--hidden-import=scipy.spatial",
        "--hidden-import=scipy.spatial._ckdtree",
        "--hidden-import=scipy.spatial.kdtree",
        "--hidden-import=pandas",
        "--hidden-import=pandas._libs",
        "--hidden-import=pandas._libs.tslibs",
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.cluster",
        "--hidden-import=sklearn.mixture",
        "--hidden-import=sklearn.manifold",
        "--hidden-import=sklearn.preprocessing",
        "--hidden-import=sklearn.decomposition",
        "--hidden-import=sklearn.neighbors",
        "--hidden-import=sklearn.metrics",
        "--hidden-import=sklearn.utils._cython_blas",
        # LigEdit modules
        "--hidden-import=lig_parser",
        "--hidden-import=pipeline",
        "--hidden-import=pipeline_dialog",
        "--hidden-import=waveform_widget",
        "--hidden-import=main_window",
        # Analytics package
        "--hidden-import=analytics",
        "--hidden-import=analytics.trace_core",
        "--hidden-import=analytics.trace_dialog",
        "--hidden-import=analytics.cluster_core",
        "--hidden-import=analytics.cluster_dialog",
        "--hidden-import=analytics.analyse_core",
        "--hidden-import=analytics.analyse_dialog",
    ]


def get_excludes():
    """排除不必要的模块以减小体积"""
    return [
        "--exclude-module=matplotlib",
        "--exclude-module=tkinter",
        "--exclude-module=IPython",
        "--exclude-module=jupyter",
        "--exclude-module=PIL",
        "--exclude-module=cv2",
    ]


def build():
    os.chdir(ROOT)
    check_deps()

    name = "LigEdit"
    if platform.system() == "Darwin":
        name += "_macOS"
    elif platform.system() == "Windows":
        name += ".exe"

    cmd = [sys.executable, "-m", "PyInstaller", "--clean",
           "--onefile",
           "--name", name,
           *get_extra_args(),
           *get_hidden_imports(),
           *get_add_data(),
           *get_excludes(),
           "main_window.py"]

    print("=" * 55)
    print(f"  LigEdit 打包 (PyInstaller)")
    print(f"  平台: {platform.system()} {platform.machine()}")
    print(f"  Python: {sys.executable}")
    print("=" * 55)
    print()
    print("执行命令:")
    print(f"  {' '.join(cmd)}")
    print()

    # 清理旧构建
    for d in ["build", "dist"]:
        p = os.path.join(ROOT, d)
        if os.path.exists(p):
            shutil.rmtree(p)

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print()
        print("=" * 55)
        print("  ✅ 构建成功!")
        print("=" * 55)
        dist_dir = os.path.join(ROOT, "dist")
        for f in os.listdir(dist_dir):
            size = os.path.getsize(os.path.join(dist_dir, f)) / 1024 / 1024
            print(f"  📦 {f}  ({size:.1f} MB)")
        print()
    else:
        print()
        print("  ❌ 构建失败，请检查上方错误信息")
        print()
        sys.exit(1)


if __name__ == "__main__":
    build()