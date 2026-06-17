#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理对话框 - 按距离分类 / 按昼夜分类
"""

import os

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QProgressBar, QSpinBox, QDoubleSpinBox, QComboBox,
    QMessageBox, QFileDialog,
)

from lig_parser import _resource_path
_DEFAULT_LIG_HEAD = _resource_path("LigHead.lig")
_DEFAULT_LIMITBYT = _resource_path("Limitbyt")
_STATION_FILE = _resource_path("站点经纬度.txt")


def _load_stations():
    """从站点经纬度.txt读取站点列表，返回 [(name, lat, lon), ...]"""
    stations = []
    if not os.path.exists(_STATION_FILE):
        return stations
    try:
        with open(_STATION_FILE, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        i = 0
        while i + 1 < len(lines):
            name = lines[i]
            parts = lines[i + 1].split()
            if len(parts) >= 2:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    stations.append((name, lat, lon))
                except ValueError:
                    pass
            i += 2
    except Exception:
        pass
    return stations


# ============================================================================
#                          后台工作线程
# ============================================================================

class _Worker(QThread):
    """通用后台工作线程"""
    progress = pyqtSignal(str, int)   # message, percent
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(self, func, kwargs):
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(**self.kwargs, progress_cb=self._progress, log_cb=self._log)
            self.finished_ok.emit(result or "处理完成！")
        except Exception as e:
            self.finished_err.emit(f"执行出错: {e}")

    def _progress(self, step, msg, pct):
        self.progress.emit(msg, pct)

    def _log(self, msg):
        self.log.emit(msg)


# ============================================================================
#                          按距离分类对话框
# ============================================================================

class DistanceClassifyDialog(QDialog):
    """按距离分类：提取时间戳→WWLLN匹配→距离筛选→波形提取"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("按距离分类")
        self.setMinimumWidth(620)
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---- 输入 ----
        input_group = QGroupBox("输入")
        form = QFormLayout(input_group)

        self.edit_lig_dir = self._add_dir_row(form, "lig文件目录:")
        self.edit_wwlln_folder = self._add_dir_row(form, "WWLLN数据目录:")

        self.combo_station = QComboBox()
        self._stations = _load_stations()
        for name, lat, lon in self._stations:
            self.combo_station.addItem(f"{name} ({lat}, {lon})")
        self.combo_station.addItem("自定义")
        self.combo_station.currentIndexChanged.connect(self._on_station_changed)
        form.addRow("站点:", self.combo_station)

        self.edit_station_lat = QDoubleSpinBox()
        self.edit_station_lat.setRange(-90, 90)
        self.edit_station_lat.setDecimals(6)
        self.edit_station_lat.setValue(23.568582)
        form.addRow("纬度:", self.edit_station_lat)

        self.edit_station_lon = QDoubleSpinBox()
        self.edit_station_lon.setRange(-180, 180)
        self.edit_station_lon.setDecimals(6)
        self.edit_station_lon.setValue(113.61469)
        form.addRow("经度:", self.edit_station_lon)

        layout.addWidget(input_group)

        # ---- 距离区间 ----
        dist_group = QGroupBox("距离区间")
        dist_form = QFormLayout(dist_group)

        self.edit_start_dist = QSpinBox()
        self.edit_start_dist.setRange(0, 10000)
        self.edit_start_dist.setValue(0)
        self.edit_start_dist.setSuffix(" km")
        dist_form.addRow("起始距离:", self.edit_start_dist)

        self.edit_end_dist = QSpinBox()
        self.edit_end_dist.setRange(0, 10000)
        self.edit_end_dist.setValue(3500)
        self.edit_end_dist.setSuffix(" km")
        dist_form.addRow("结束距离:", self.edit_end_dist)

        self.edit_step_dist = QSpinBox()
        self.edit_step_dist.setRange(10, 1000)
        self.edit_step_dist.setValue(100)
        self.edit_step_dist.setSuffix(" km")
        dist_form.addRow("距离步长:", self.edit_step_dist)

        layout.addWidget(dist_group)

        # ---- 输出 ----
        output_group = QGroupBox("输出")
        out_form = QFormLayout(output_group)
        self.edit_output_dir = self._add_dir_row(out_form, "输出目录:")
        layout.addWidget(output_group)

        # ---- 进度与日志 ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)

        # ---- 按钮 ----
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("执行")
        self.btn_run.setStyleSheet(
            "background-color: #0078d7; color: white; padding: 8px 24px; font-weight: bold;")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    # ---- 辅助 ----
    def _add_dir_row(self, form, label):
        layout = QHBoxLayout()
        edit = QLineEdit()
        btn = QPushButton("浏览...")
        btn.clicked.connect(lambda: self._browse_dir(edit))
        layout.addWidget(edit, stretch=1)
        layout.addWidget(btn)
        form.addRow(label, layout)
        return edit

    def _browse_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if d:
            edit.setText(d)

    def _on_station_changed(self, idx):
        if 0 <= idx < len(self._stations):
            _, lat, lon = self._stations[idx]
            self.edit_station_lat.setValue(lat)
            self.edit_station_lon.setValue(lon)

    # ---- 运行 ----
    def _on_run(self):
        lig_dir = self.edit_lig_dir.text().strip()
        wwlln_folder = self.edit_wwlln_folder.text().strip()
        if not lig_dir:
            QMessageBox.warning(self, "提示", "请选择lig文件目录")
            return
        if not wwlln_folder:
            QMessageBox.warning(self, "提示", "请选择WWLLN数据目录")
            return

        # 默认输出目录
        output_dir = self.edit_output_dir.text().strip() or os.path.join(lig_dir, "distance_classified")

        # 选择头文件
        lig_head_path = _DEFAULT_LIMITBYT if os.path.exists(_DEFAULT_LIMITBYT) else _DEFAULT_LIG_HEAD
        lig_file_head_path = _DEFAULT_LIG_HEAD if os.path.exists(_DEFAULT_LIG_HEAD) else _DEFAULT_LIMITBYT

        from pipeline import classify_by_distance
        kwargs = {
            'lig_dir': lig_dir,
            'wwlln_folder': wwlln_folder,
            'output_dir': output_dir,
            'station_lat': self.edit_station_lat.value(),
            'station_lon': self.edit_station_lon.value(),
            'start_dist': self.edit_start_dist.value(),
            'end_dist': self.edit_end_dist.value(),
            'step_dist': self.edit_step_dist.value(),
            'lig_head_path': lig_head_path,
            'lig_file_head_path': lig_file_head_path,
        }

        self.btn_run.setEnabled(False)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("正在执行...")

        self.worker = _Worker(classify_by_distance, kwargs)
        self.worker.progress.connect(lambda msg, pct: (
            self.status_label.setText(msg), self.progress_bar.setValue(pct)))
        self.worker.log.connect(self.log_text.append)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_err.connect(self._on_finished_err)
        self.worker.start()

    def _on_finished_ok(self, msg):
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText(msg)
        QMessageBox.information(self, "完成", msg)

    def _on_finished_err(self, msg):
        self.btn_run.setEnabled(True)
        self.status_label.setText(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)


# ============================================================================
#                          按昼夜分类对话框
# ============================================================================

class DayNightClassifyDialog(QDialog):
    """按昼夜分类：读取lig文件，按白天/夜晚分组输出"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("按昼夜分类")
        self.setMinimumWidth(520)
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---- 输入 ----
        input_group = QGroupBox("输入")
        form = QFormLayout(input_group)
        self.edit_lig_dir = self._add_dir_row(form, "lig文件目录:")
        layout.addWidget(input_group)

        # ---- 输出 ----
        output_group = QGroupBox("输出")
        out_form = QFormLayout(output_group)
        self.edit_output_dir = self._add_dir_row(out_form, "输出目录:")
        layout.addWidget(output_group)

        # ---- 进度与日志 ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)

        # ---- 按钮 ----
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("执行")
        self.btn_run.setStyleSheet(
            "background-color: #0078d7; color: white; padding: 8px 24px; font-weight: bold;")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    # ---- 辅助 ----
    def _add_dir_row(self, form, label):
        layout = QHBoxLayout()
        edit = QLineEdit()
        btn = QPushButton("浏览...")
        btn.clicked.connect(lambda: self._browse_dir(edit))
        layout.addWidget(edit, stretch=1)
        layout.addWidget(btn)
        form.addRow(label, layout)
        return edit

    def _browse_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if d:
            edit.setText(d)

    # ---- 运行 ----
    def _on_run(self):
        lig_dir = self.edit_lig_dir.text().strip()
        if not lig_dir:
            QMessageBox.warning(self, "提示", "请选择lig文件目录")
            return

        output_dir = self.edit_output_dir.text().strip() or os.path.join(lig_dir, "daynight_classified")
        lig_head_path = _DEFAULT_LIMITBYT if os.path.exists(_DEFAULT_LIMITBYT) else _DEFAULT_LIG_HEAD
        lig_file_head_path = _DEFAULT_LIG_HEAD if os.path.exists(_DEFAULT_LIG_HEAD) else _DEFAULT_LIMITBYT

        from pipeline import classify_by_daynight
        kwargs = {
            'lig_dir': lig_dir,
            'output_dir': output_dir,
            'lig_head_path': lig_head_path,
            'lig_file_head_path': lig_file_head_path,
        }

        self.btn_run.setEnabled(False)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("正在执行...")

        self.worker = _Worker(classify_by_daynight, kwargs)
        self.worker.progress.connect(lambda msg, pct: (
            self.status_label.setText(msg), self.progress_bar.setValue(pct)))
        self.worker.log.connect(self.log_text.append)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_err.connect(self._on_finished_err)
        self.worker.start()

    def _on_finished_ok(self, msg):
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText(msg)
        QMessageBox.information(self, "完成", msg)

    def _on_finished_err(self, msg):
        self.btn_run.setEnabled(True)
        self.status_label.setText(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)
