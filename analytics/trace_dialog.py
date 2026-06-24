#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analytics / trace_dialog — 多站闪电事件匹配对话框

移植自 LigTrace 的 tkinter GUI，改用 LigEdit 的 QDialog + QThread 模式。
增强：设置持久化、站点配置导入、彩色日志、进度文本、关闭保护。
"""

import os
import json
import queue
import threading

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QTextCharFormat
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QProgressBar, QSpinBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QFileDialog, QFrame,
)

from lig_parser import _resource_path, load_station_coords


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
                target_day=self.config.get('target_day'),
                lig_head_path=self.config['lig_head'],
                limitbyt_path=self.config['limitbyt'],
                stop_flag=self._stop_flag,
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
        self.setMinimumSize(850, 720)
        self.resize(950, 820)

        self.station_rows = []  # [{chk, name, lat, lon, dir, frame}]
        self.worker = None
        self.log_queue = queue.Queue()
        self.settings = self._load_settings()

        self._build_ui()
        self._poll_log()
        self._apply_settings()

    # ------------------------------------------------------------------
    #  设置持久化
    # ------------------------------------------------------------------

    def _load_settings(self) -> dict:
        settings_path = os.path.expanduser('~/.ligedit_trace_settings.json')
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_settings(self):
        settings_path = os.path.expanduser('~/.ligedit_trace_settings.json')
        s = {
            'wwlln_dir': self.var_wwlln_dir.text(),
            'output_dir': self.var_output_dir.text(),
            'min_stations': self.spin_min_stations.value(),
            'time_window': self.spin_time_window.value(),
            'stations': [
                {
                    'name': e['name'].text(),
                    'lat': e['lat'].text(),
                    'lon': e['lon'].text(),
                    'dir': e['dir'].text(),
                    'enabled': e['chk'].isChecked(),
                }
                for e in self.station_rows
            ],
        }
        try:
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(s, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _apply_settings(self):
        s = self.settings
        if s.get('wwlln_dir'):
            self.var_wwlln_dir.setText(s['wwlln_dir'])
        if s.get('output_dir'):
            self.var_output_dir.setText(s['output_dir'])
        if s.get('min_stations'):
            self.spin_min_stations.setValue(s['min_stations'])
        if s.get('time_window'):
            self.spin_time_window.setValue(s['time_window'])
        if s.get('stations'):
            # 替换默认空行
            for entry in list(self.station_rows):
                self._remove_station_row(entry['frame'])
            for sta in s['stations']:
                self._add_station_row(
                    name=sta.get('name', ''),
                    lat=str(sta.get('lat', '')),
                    lon=str(sta.get('lon', '')),
                    directory=sta.get('dir', ''),
                )
                if self.station_rows:
                    self.station_rows[-1]['chk'].setChecked(bool(sta.get('enabled', True)))

    # ------------------------------------------------------------------
    #  UI 搭建
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("⚡ 多站闪电事件匹配")
        title.setFont(QFont("", 14, QFont.Bold))
        layout.addWidget(title)

        # ---- 站点配置 ----
        station_group = QGroupBox("站点配置")
        station_layout = QVBoxLayout(station_group)

        station_toolbar = QHBoxLayout()
        btn_add = QPushButton("+ 添加站点")
        btn_add.clicked.connect(lambda: self._add_station_row())
        station_toolbar.addWidget(btn_add)

        btn_import = QPushButton("\U0001f4c4 从配置文件导入")
        btn_import.clicked.connect(self._import_stations)
        station_toolbar.addWidget(btn_import)

        station_toolbar.addStretch()
        station_layout.addLayout(station_toolbar)

        self.stations_frame = QFrame()
        self.station_scroll_layout = QVBoxLayout(self.stations_frame)
        station_layout.addWidget(self.stations_frame)
        layout.addWidget(station_group)

        # ---- 路径配置 ----
        path_group = QGroupBox("数据路径")
        path_form = QFormLayout(path_group)
        self.var_wwlln_dir = self._add_browse_row(path_form, "WWLLN 目录:", 'dir')
        self.var_output_dir = self._add_browse_row(path_form, "输出目录:", 'dir')
        layout.addWidget(path_group)

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
        layout.addWidget(param_group)

        # ---- 日志 ----
        log_toolbar = QHBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))

        btn_clear = QPushButton("✕ 清除日志")
        btn_clear.clicked.connect(self.log_text.clear)
        log_toolbar.addWidget(QLabel("日志:"))
        log_toolbar.addStretch()
        log_toolbar.addWidget(btn_clear)
        layout.addLayout(log_toolbar)
        layout.addWidget(self.log_text)

        # ---- 进度条 + 进度文本 ----
        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_row.addWidget(self.progress_bar, 1)
        self.progress_label = QLabel("就绪")
        self.progress_label.setStyleSheet("color: #666; font-size: 11px;")
        progress_row.addWidget(self.progress_label)
        layout.addLayout(progress_row)

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

    # ------------------------------------------------------------------
    #  站点操作
    # ------------------------------------------------------------------

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

    def _import_stations(self):
        """从 站点经纬度.txt 导入站点配置"""
        coord_file, _ = QFileDialog.getOpenFileName(
            self, "选择站点配置文件",
            _resource_path('站点经纬度.txt'),
            "文本文件 (*.txt);;所有文件 (*)")
        if not coord_file:
            return

        try:
            coords = load_station_coords(coord_file)
            if not coords:
                QMessageBox.warning(self, "提示", "配置文件中没有有效的站点数据")
                return

            imported = 0
            for name, (lat, lon) in coords.items():
                self._add_station_row(name=name, lat=str(lat), lon=str(lon))
                imported += 1

            QMessageBox.information(self, "导入完成", f"已导入 {imported} 个站点\n请为每个站点指定 LIG 数据目录")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    # ------------------------------------------------------------------
    #  日志轮询 + 彩色
    # ------------------------------------------------------------------

    def _poll_log(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
                self._append_colored_log(msg)
            except Exception:
                break
        QTimer.singleShot(200, self._poll_log)

    def _append_colored_log(self, msg: str):
        """根据日志级别设置颜色"""
        cursor = self.log_text.textCursor()
        if 'ERROR' in msg:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor('#ef5350'))
            cursor.setCharFormat(fmt)
        elif 'WARN' in msg:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor('#ffa726'))
            cursor.setCharFormat(fmt)
        elif 'INFO' in msg:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor('#66bb6a'))
            cursor.setCharFormat(fmt)
        else:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor('#e0e0e0'))
            cursor.setCharFormat(fmt)
        cursor.insertText(msg + '\n')
        self.log_text.setTextCursor(cursor)

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
        self.progress_label.setText("正在启动...")

        self.worker = TraceWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(lambda msg: self.log_queue.put(msg))
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_err.connect(self._on_finished_err)
        self.worker.start()

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.progress_label.setText("正在停止...")
            self.log_queue.put("用户请求停止（可能需要几秒钟才能生效）")

    def _on_progress(self, fraction, text):
        self.progress_bar.setValue(int(fraction * 100))
        self.progress_label.setText(text)

    def _on_finished_ok(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_label.setText("完成")
        self.log_queue.put(f"\n✓ {msg}")
        QMessageBox.information(self, "完成", msg)

    def _on_finished_err(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_label.setText("错误")
        self.log_queue.put(f"\n✗ {msg}")
        QMessageBox.critical(self, "错误", msg[:500])

    # ------------------------------------------------------------------
    #  关闭保护
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "确认退出",
                "匹配正在进行中，是否停止并退出？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self._save_settings()
                event.accept()
            else:
                event.ignore()
        else:
            self._save_settings()
            event.accept()