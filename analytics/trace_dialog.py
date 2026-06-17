#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analytics / trace_dialog — 多站闪电事件匹配对话框

移植自 LigTrace 的 tkinter GUI，改用 LigEdit 的 QDialog + QThread 模式。
"""

import os
import json
import queue
import threading

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QProgressBar, QSpinBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QFileDialog, QTabWidget, QWidget,
    QHeaderView, QTableWidget, QTableWidgetItem,
    QFrame, QScrollArea,
)

from lig_parser import _resource_path


# ============================================================================
#                          日志队列处理器
# ============================================================================

class QueueLogHandler:
    def __init__(self, log_queue: queue.Queue):
        self.log_queue = log_queue

    def emit(self, msg: str):
        self.log_queue.put(msg)


# ============================================================================
#                          后台工作线程
# ============================================================================

class TraceWorker(QThread):
    progress = pyqtSignal(float, str)   # fraction, text
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._stop_flag = threading.Event()

    def stop(self):
        self._stop_flag.set()

    def run(self):
        try:
            from analytics.trace_core import run_trace_matching
            result = run_trace_matching(
                self.config['stations'],
                self.config['wwlln_dir'],
                self.config['output_dir'],
                min_stations=self.config['min_stations'],
                time_window_s=self.config['time_window'],
                lig_head_path=self.config['lig_head'],
                limitbyt_path=self.config['limitbyt'],
                progress_cb=self._progress_cb,
                log_cb=self._log_cb,
            )
            self.finished_ok.emit(result)
        except Exception as e:
            import traceback
            self.finished_err.emit(f"{e}\n{traceback.format_exc()}")

    def _progress_cb(self, fraction: float, text: str):
        self.progress.emit(fraction, text)

    def _log_cb(self, msg: str):
        self.log.emit(msg)


# ============================================================================
#                          对话框
# ============================================================================

class TraceDialog(QDialog):
    """多站闪电事件匹配对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("多站闪电事件匹配 (LigTrace)")
        self.setMinimumSize(800, 700)
        self.resize(900, 780)

        self.station_rows = []  # [{widgets, vars, row_frame}]
        self.worker = None
        self.log_queue = queue.Queue()

        self._build_ui()
        self._poll_log()

    # ------------------------------------------------------------------
    #  UI 搭建
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("⚡ 多站闪电事件匹配")
        title.setFont(QFont("", 14, QFont.Bold))
        layout.addWidget(title)

        # 参数区（使用可滚动区域）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # ---- 站点配置 ----
        station_group = QGroupBox("站点配置")
        station_layout = QVBoxLayout(station_group)

        # 工具按钮
        station_toolbar = QHBoxLayout()
        btn_add = QPushButton("+ 添加站点")
        btn_add.clicked.connect(self._add_station_row)
        station_toolbar.addWidget(btn_add)
        station_toolbar.addStretch()
        station_layout.addLayout(station_toolbar)

        self.stations_frame = QFrame()
        self.station_scroll_layout = QVBoxLayout(self.stations_frame)
        station_layout.addWidget(self.stations_frame)
        scroll_layout.addWidget(station_group)

        # ---- 路径配置 ----
        path_group = QGroupBox("数据路径")
        path_form = QFormLayout(path_group)
        self.var_wwlln_dir = self._add_browse_row(path_form, "WWLLN 目录:", 'dir')
        self.var_output_dir = self._add_browse_row(path_form, "输出目录:", 'dir')
        scroll_layout.addWidget(path_group)

        # ---- 匹配参数 ----
        param_group = QGroupBox("匹配参数")
        param_form = QFormLayout(param_group)

        h_min = QHBoxLayout()
        self.spin_min_stations = QSpinBox()
        self.spin_min_stations.setRange(1, 20)
        self.spin_min_stations.setValue(2)
        h_min.addWidget(self.spin_min_stations)
        h_min.addWidget(QLabel("个站点"))
        param_form.addRow("最小站数:", h_min)

        h_time = QHBoxLayout()
        self.spin_time_window = QDoubleSpinBox()
        self.spin_time_window.setRange(0.001, 1.0)
        self.spin_time_window.setValue(0.050)
        self.spin_time_window.setDecimals(3)
        self.spin_time_window.setSingleStep(0.005)
        h_time.addWidget(self.spin_time_window)
        h_time.addWidget(QLabel("秒"))
        param_form.addRow("时间窗口:", h_time)

        info = QLabel("时间窗口是预期信号到达时间（WWLLN时间+距离/光速）±容差。\n默认 50ms 考虑了电离层波导传播差异。")
        info.setStyleSheet("color: #888; font-size: 11px;")
        param_form.addRow(info)
        scroll_layout.addWidget(param_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # ---- 日志 ----
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)

        # ---- 进度条 ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # ---- 按钮 ----
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("▶ 开始匹配")
        self.btn_run.setStyleSheet(
            "QPushButton { background: #2e7d32; color: white; padding: 8px 24px; font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { background: #388e3c; }")
        self.btn_run.clicked.connect(self._run)
        btn_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        btn_layout.addWidget(self.btn_stop)

        btn_layout.addStretch()

        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

        # 添加默认空行
        for _ in range(3):
            self._add_station_row()

    def _add_browse_row(self, form, label, mode='dir'):
        h = QHBoxLayout()
        edit = QLineEdit()
        btn = QPushButton("浏览...")
        btn.clicked.connect(lambda: self._browse(edit, mode))
        h.addWidget(edit, 1)
        h.addWidget(btn)
        form.addRow(label, h)
        return edit

    def _browse(self, edit, mode):
        if mode == 'dir':
            d = QFileDialog.getExistingDirectory(self, "选择目录")
            if d:
                edit.setText(d)
        else:
            f, _ = QFileDialog.getOpenFileName(self, "选择文件")
            if f:
                edit.setText(f)

    def _add_station_row(self, name='', lat='', lon='', directory=''):
        row_frame = QFrame()
        row_layout = QHBoxLayout(row_frame)
        row_layout.setContentsMargins(0, 2, 0, 2)

        chk = QCheckBox()
        chk.setChecked(True)
        row_layout.addWidget(chk)

        e_name = QLineEdit(name)
        e_name.setPlaceholderText("站名")
        e_name.setMaximumWidth(60)
        row_layout.addWidget(e_name)

        e_lat = QLineEdit(lat)
        e_lat.setPlaceholderText("纬度")
        e_lat.setMaximumWidth(90)
        row_layout.addWidget(e_lat)

        e_lon = QLineEdit(lon)
        e_lon.setPlaceholderText("经度")
        e_lon.setMaximumWidth(90)
        row_layout.addWidget(e_lon)

        e_dir = QLineEdit(directory)
        e_dir.setPlaceholderText("LIG 数据目录")
        row_layout.addWidget(e_dir, 1)

        btn_browse = QPushButton("...")
        btn_browse.setMaximumWidth(30)
        btn_browse.clicked.connect(lambda: self._browse(e_dir, 'dir'))
        row_layout.addWidget(btn_browse)

        btn_remove = QPushButton("✕")
        btn_remove.setMaximumWidth(30)
        btn_remove.clicked.connect(lambda: self._remove_station_row(row_frame))
        row_layout.addWidget(btn_remove)

        self.station_scroll_layout.addWidget(row_frame)
        self.station_rows.append({
            'chk': chk, 'name': e_name, 'lat': e_lat, 'lon': e_lon, 'dir': e_dir,
            'frame': row_frame,
        })

    def _remove_station_row(self, frame):
        for entry in self.station_rows:
            if entry['frame'] is frame:
                self.station_rows.remove(entry)
                frame.deleteLater()
                break

    # ------------------------------------------------------------------
    #  日志轮询
    # ------------------------------------------------------------------

    def _poll_log(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
                self.log_text.append(msg)
            except Exception:
                break
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(200, self._poll_log)

    # ------------------------------------------------------------------
    #  执行
    # ------------------------------------------------------------------

    def _run(self):
        # 收集站点
        stations = []
        for entry in self.station_rows:
            if not entry['chk'].isChecked():
                continue
            name = entry['name'].text().strip()
            if not name:
                continue
            try:
                lat = float(entry['lat'].text())
                lon = float(entry['lon'].text())
            except ValueError:
                QMessageBox.warning(self, "输入错误", f"站点 {name} 的经纬度格式不正确")
                return
            d = entry['dir'].text().strip()
            if not d or not os.path.isdir(d):
                QMessageBox.warning(self, "目录错误", f"站点 {name} 的目录不存在: {d}")
                return
            stations.append({'name': name, 'lat': lat, 'lon': lon, 'dir': d})

        if not stations:
            QMessageBox.warning(self, "无站点", "请添加至少一个启用且有数据的站点")
            return

        wwlln_dir = self.var_wwlln_dir.text().strip()
        if not wwlln_dir or not os.path.isdir(wwlln_dir):
            QMessageBox.warning(self, "目录错误", f"WWLLN 目录不存在:\n{wwlln_dir}")
            return

        output_dir = self.var_output_dir.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "输出目录", "请指定输出目录")
            return

        config = {
            'stations': stations,
            'wwlln_dir': wwlln_dir,
            'output_dir': output_dir,
            'min_stations': self.spin_min_stations.value(),
            'time_window': self.spin_time_window.value(),
            'lig_head': _resource_path('LigHead.lig'),
            'limitbyt': _resource_path('Limitbyt'),
        }

        self.log_text.clear()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)

        self.worker = TraceWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self.log_text.append)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_err.connect(self._on_finished_err)
        self.worker.start()

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log_text.append("用户中断")

    def _on_progress(self, fraction, text):
        self.progress_bar.setValue(int(fraction * 100))

    def _on_finished_ok(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.log_text.append(f"\n✓ {msg}")
        QMessageBox.information(self, "完成", msg)

    def _on_finished_err(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log_text.append(f"\n✗ {msg}")
        QMessageBox.critical(self, "错误", msg[:500])