#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analytics / analyse_dialog — 闪电数据分析对话框

移植自 LigAnalyse 的 MainWindow，缩减为 QDialog。
"""

import os
import csv

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QProgressBar, QSpinBox, QDoubleSpinBox,
    QMessageBox, QFileDialog, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QSplitter, QFrame,
)

import pyqtgraph as pg

CATEGORY_COLORS = [
    ('#4fc3f7', '浅蓝'), ('#66bb6a', '绿色'), ('#ffa726', '橙色'),
    ('#ef5350', '红色'), ('#ab47bc', '紫色'), ('#26c6da', '青色'),
    ('#ffca28', '黄色'), ('#8d6e63', '棕色'),
]


# ============================================================================
#                          后台工作线程
# ============================================================================

class AnalyseWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, lig_dir, lightning_dir, nbe_loc_file,
                 station_lat, station_lon, time_window_ms, distance_window_km):
        super().__init__()
        self.lig_dir = lig_dir
        self.lightning_dir = lightning_dir
        self.nbe_loc_file = nbe_loc_file
        self.station_lat = station_lat
        self.station_lon = station_lon
        self.time_window_ms = time_window_ms
        self.distance_window_km = distance_window_km

    def run(self):
        try:
            from analytics.analyse_core import run_full_analysis
            results = run_full_analysis(
                self.lig_dir, self.lightning_dir, self.nbe_loc_file,
                station_lat=self.station_lat, station_lon=self.station_lon,
                time_window_ms=self.time_window_ms,
                distance_window_km=self.distance_window_km,
                progress_cb=self._progress,
            )
            self.finished.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _progress(self, msg, pct):
        self.progress.emit(msg, pct)


# ============================================================================
#                          饼图控件
# ============================================================================

class PieChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = {}
        self.setMinimumSize(280, 280)

    def set_data(self, data):
        self.data = data
        self.update()

    def paintEvent(self, event):
        if not self.data:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        total = sum(self.data.values())
        if total == 0:
            return

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2 - 40

        start_angle = 0
        for idx, (label, value) in enumerate(self.data.items()):
            color = QColor(CATEGORY_COLORS[idx % len(CATEGORY_COLORS)][0])
            span = int(value / total * 360 * 16)

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor('#1e1e2e'), 2))
            painter.drawPie(cx - radius, cy - radius, radius * 2, radius * 2,
                            -start_angle, -span)

            mid_angle = start_angle / 16 + (span / 16) / 2
            mid_rad = np.radians(mid_angle)
            lx = cx + int(radius * 0.65 * np.cos(mid_rad))
            ly = cy - int(radius * 0.65 * np.sin(mid_rad))

            pct = value / total * 100
            if pct > 5:
                painter.setPen(QPen(QColor('white')))
                font = QFont("Microsoft YaHei", 9, QFont.Bold)
                painter.setFont(font)
                painter.drawText(lx - 40, ly - 8, 80, 16, Qt.AlignCenter, f"{pct:.1f}%")

            start_angle += span

        # 图例
        painter.setPen(QPen(QColor('#e0e0e0')))
        font = QFont("Microsoft YaHei", 9)
        painter.setFont(font)
        legend_y = h - len(self.data) * 20 - 10
        for idx, (label, value) in enumerate(self.data.items()):
            color = QColor(CATEGORY_COLORS[idx % len(CATEGORY_COLORS)][0])
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor('#1e1e2e'), 1))
            painter.drawRect(10, legend_y + idx * 20, 12, 12)
            painter.setPen(QPen(QColor('#e0e0e0')))
            pct = value / total * 100
            painter.drawText(28, legend_y + idx * 20 + 12,
                             f"{label} ({value}, {pct:.1f}%)")

        painter.end()


# ============================================================================
#                          数据分析对话框
# ============================================================================

class AnalyseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("闪电数据分析 (LigAnalyse)")
        self.setMinimumSize(1100, 800)
        self.resize(1200, 850)

        self.analysis_results = None
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 输入参数区
        input_group = QGroupBox("分析参数")
        input_form = QVBoxLayout(input_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("LIG 数据目录:"))
        self.lig_dir_edit = QLineEdit()
        self.lig_dir_edit.setPlaceholderText("选择包含子文件夹分类的 lig 目录...")
        row1.addWidget(self.lig_dir_edit, 1)
        btn_lig = QPushButton("浏览...")
        btn_lig.clicked.connect(lambda: self._browse_dir(self.lig_dir_edit))
        row1.addWidget(btn_lig)
        input_form.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("闪电定位目录:"))
        self.match_dir_edit = QLineEdit()
        row2.addWidget(self.match_dir_edit, 1)
        btn_match = QPushButton("浏览...")
        btn_match.clicked.connect(lambda: self._browse_dir(self.match_dir_edit))
        row2.addWidget(btn_match)
        input_form.addLayout(row2)

        row2b = QHBoxLayout()
        row2b.addWidget(QLabel("NBE 定位文件:"))
        self.nbe_loc_edit = QLineEdit()
        row2b.addWidget(self.nbe_loc_edit, 1)
        btn_nbe = QPushButton("浏览...")
        btn_nbe.clicked.connect(lambda: self._browse_file(self.nbe_loc_edit))
        row2b.addWidget(btn_nbe)
        input_form.addLayout(row2b)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("时间窗口 (ms):"))
        self.time_window_spin = QSpinBox()
        self.time_window_spin.setRange(1, 10000)
        self.time_window_spin.setValue(660)
        row3.addWidget(self.time_window_spin)

        row3.addSpacing(15)
        row3.addWidget(QLabel("距离窗口 (km):"))
        self.dist_window_spin = QDoubleSpinBox()
        self.dist_window_spin.setRange(0.1, 1000)
        self.dist_window_spin.setValue(10)
        self.dist_window_spin.setDecimals(1)
        row3.addWidget(self.dist_window_spin)

        row3.addStretch()
        input_form.addLayout(row3)

        row4 = QHBoxLayout()
        self.run_btn = QPushButton("开始分析")
        self.run_btn.setStyleSheet(
            "QPushButton { background: #2e7d32; color: white; font-size: 14px; padding: 8px 32px; }"
            "QPushButton:hover { background: #388e3c; }")
        self.run_btn.clicked.connect(self._run_analysis)
        row4.addWidget(self.run_btn)

        self.export_btn = QPushButton("导出结果")
        self.export_btn.setStyleSheet(
            "QPushButton { background: #1565c0; color: white; padding: 8px 24px; }"
            "QPushButton:hover { background: #1976d2; }")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        row4.addWidget(self.export_btn)

        self.progress_bar = QProgressBar()
        row4.addWidget(self.progress_bar, 1)

        self.status_label = QLabel("就绪")
        row4.addWidget(self.status_label)

        input_form.addLayout(row4)
        layout.addWidget(input_group)

        # 结果 Tabs
        self.result_tabs = QTabWidget()

        # — 汇总
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.result_tabs.addTab(self.summary_text, "汇总统计")

        # — 分类分析（饼图 + 表格）
        cat_widget = QWidget()
        cat_layout = QHBoxLayout(cat_widget)
        self.pie_chart = PieChartWidget()
        cat_layout.addWidget(self.pie_chart)
        self.cat_table = QTableWidget()
        self.cat_table.setColumnCount(7)
        self.cat_table.setHorizontalHeaderLabels(
            ['分类', '数量', '占比(%)', '平均距离(km)', '平均电压(V)', '独立比例(%)', '独立/非独立'])
        self.cat_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cat_table.setAlternatingRowColors(True)
        self.cat_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        cat_layout.addWidget(self.cat_table)
        cat_layout.setStretchFactor(self.pie_chart, 1)
        cat_layout.setStretchFactor(self.cat_table, 2)
        self.result_tabs.addTab(cat_widget, "分类分析")

        # — 距离分布
        dist_widget = QWidget()
        dist_layout = QVBoxLayout(dist_widget)
        self.dist_plot = pg.PlotWidget()
        self.dist_plot.setBackground('#1e1e2e')
        self.dist_plot.showGrid(x=True, y=True, alpha=0.2)
        dist_layout.addWidget(self.dist_plot, 1)
        self.dist_table = QTableWidget()
        self.dist_table.setColumnCount(6)
        self.dist_table.setHorizontalHeaderLabels(['时间', '分类', '距离(km)', '残差(ms)', '时延(ms)', '闪电经纬度'])
        self.dist_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dist_table.setAlternatingRowColors(True)
        self.dist_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        dist_layout.addWidget(self.dist_table, 1)
        self.result_tabs.addTab(dist_widget, "距离分布")

        # — 电流强度
        curr_widget = QWidget()
        curr_layout = QVBoxLayout(curr_widget)
        self.current_plot = pg.PlotWidget()
        self.current_plot.setBackground('#1e1e2e')
        self.current_plot.showGrid(x=True, y=True, alpha=0.2)
        curr_layout.addWidget(self.current_plot, 1)
        self.current_table = QTableWidget()
        self.current_table.setColumnCount(3)
        self.current_table.setHorizontalHeaderLabels(['时间', '分类', '峰值电压(V)'])
        self.current_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.current_table.setAlternatingRowColors(True)
        self.current_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        curr_layout.addWidget(self.current_table, 1)
        self.result_tabs.addTab(curr_widget, "电流强度")

        # — 独立分布
        indep_widget = QWidget()
        indep_layout = QVBoxLayout(indep_widget)
        self.indep_plot = pg.PlotWidget()
        self.indep_plot.setBackground('#1e1e2e')
        self.indep_plot.showGrid(x=True, y=True, alpha=0.2)
        indep_layout.addWidget(self.indep_plot, 1)
        self.indep_table = QTableWidget()
        self.indep_table.setColumnCount(5)
        self.indep_table.setHorizontalHeaderLabels(['时间', '分类', '是否独立', '附近闪电数', '最近闪电'])
        self.indep_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.indep_table.setAlternatingRowColors(True)
        self.indep_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        indep_layout.addWidget(self.indep_table, 1)
        self.result_tabs.addTab(indep_widget, "独立分布")

        # — 日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.result_tabs.addTab(self.log_text, "日志")

        layout.addWidget(self.result_tabs, 1)

    def _browse_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if d:
            edit.setText(d)

    def _browse_file(self, edit):
        f, _ = QFileDialog.getOpenFileName(self, "选择文件")
        if f:
            edit.setText(f)

    def _run_analysis(self):
        lig_dir = self.lig_dir_edit.text().strip()
        lightning_dir = self.match_dir_edit.text().strip()
        nbe_loc_file = self.nbe_loc_edit.text().strip()

        if not os.path.isdir(lig_dir):
            QMessageBox.warning(self, "路径错误", f"LIG 目录不存在:\n{lig_dir}")
            return
        if not os.path.isdir(lightning_dir):
            QMessageBox.warning(self, "路径错误", f"闪电定位目录不存在:\n{lightning_dir}")
            return
        if nbe_loc_file and not os.path.isfile(nbe_loc_file):
            nbe_loc_file = ''

        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self.worker = AnalyseWorker(
            lig_dir, lightning_dir, nbe_loc_file,
            23.5686, 113.6147,  # GZ
            self.time_window_spin.value(),
            self.dist_window_spin.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg, pct):
        self.status_label.setText(msg[:50])
        if pct >= 0:
            self.progress_bar.setValue(min(pct, 100))
        self.log_text.append(msg)

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.status_label.setText("出错")
        self.log_text.append(f"\n[错误] {msg}")
        QMessageBox.critical(self, "错误", msg[:500])

    def _on_finished(self, results):
        self.analysis_results = results
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText("完成")
        self.log_text.append("\n=== 分析完成 ===")
        self._display_results(results)

    def _display_results(self, results):
        self._display_summary(results)
        self._display_category(results)
        self._display_distance(results)
        self._display_current(results)
        self._display_independent(results)

    def _display_summary(self, results):
        summary = results.get('summary', {})
        lines = ["=" * 60, "  LigAnalyse 分析报告", "=" * 60]

        station = summary.get('station', {})
        if station:
            lines.append(f"\n【站点】{station.get('name', 'N/A')} ({station.get('lat')}, {station.get('lon')})")

        cat = summary.get('category', {})
        if cat:
            lines.append(f"\n【分类】总数: {cat['total']}")
            for name, cnt in cat['counts'].items():
                lines.append(f"  {name}: {cnt} ({cat['ratios'][name]}%)")

        dist = summary.get('distance', {})
        if dist:
            lines.append(f"\n【距离】均{dist['mean']}km 中位{dist['median']}km {dist['min']}~{dist['max']}km")

        curr = summary.get('current', {})
        if curr:
            lines.append(f"\n【电压】均{curr['mean']}V 中位{curr['median']}V {curr['min']}~{curr['max']}V")

        indep = summary.get('independent', {})
        if indep:
            lines.append(f"\n【独立】{indep['independent_count']}/{indep['total']} ({indep['independent_ratio']}%)")

        lines.append("=" * 60)
        self.summary_text.setPlainText("\n".join(lines))

    def _display_category(self, results):
        summary = results.get('summary', {})
        cat = summary.get('category', {})
        cat_summary = summary.get('category_summary', {})
        if cat and 'counts' in cat:
            self.pie_chart.set_data(cat['counts'])

        self.cat_table.setRowCount(len(cat_summary))
        for idx, (cat_name, cs) in enumerate(cat_summary.items()):
            color = QColor(CATEGORY_COLORS[idx % len(CATEGORY_COLORS)][0])
            name_item = QTableWidgetItem(cat_name)
            name_item.setForeground(color)
            self.cat_table.setItem(idx, 0, name_item)
            self.cat_table.setItem(idx, 1, QTableWidgetItem(str(cs['count'])))
            self.cat_table.setItem(idx, 2, QTableWidgetItem(f"{cat['ratios'].get(cat_name, 0)}"))
            d = cs.get('distance', {})
            self.cat_table.setItem(idx, 3, QTableWidgetItem(f"{d.get('mean', '-')}"))
            c = cs.get('current', {})
            self.cat_table.setItem(idx, 4, QTableWidgetItem(f"{c.get('mean', '-')}"))
            ind = cs.get('independent', {})
            self.cat_table.setItem(idx, 5, QTableWidgetItem(f"{ind.get('independent_ratio', '-')}"))
            self.cat_table.setItem(idx, 6, QTableWidgetItem(
                f"{ind.get('independent_count', 0)}/{ind.get('dependent_count', 0)}"))

    def _display_distance(self, results):
        distance_results = results.get('distance_results', [])
        self.dist_plot.clear()
        self.dist_table.setRowCount(0)
        if not distance_results:
            return

        distances = [r['distance_km'] for r in distance_results]
        y, x = np.histogram(distances, bins=20)
        width = np.diff(x)[0] * 0.85
        brush = pg.mkBrush(100, 150, 255, 180)
        pen = pg.mkPen(QColor('#4fc3f7').lighter(130), width=1)
        bar = pg.BarGraphItem(x=x[:-1], height=y, width=width, brush=brush, pen=pen)
        self.dist_plot.addItem(bar)
        self.dist_plot.setLabel('bottom', '距离 (km)')
        self.dist_plot.setLabel('left', '数量')

        self.dist_table.setRowCount(len(distance_results))
        for i, r in enumerate(distance_results):
            self.dist_table.setItem(i, 0, QTableWidgetItem(r['final_time']))
            self.dist_table.setItem(i, 1, QTableWidgetItem(r.get('category', '')))
            self.dist_table.setItem(i, 2, QTableWidgetItem(f"{r['distance_km']:.3f}"))
            self.dist_table.setItem(i, 3, QTableWidgetItem(f"{r.get('time_diff_ms', 0):.3f}"))
            self.dist_table.setItem(i, 4, QTableWidgetItem(f"{r.get('propagation_ms', 0):.3f}"))
            ml = r.get('matched_lightning', {})
            self.dist_table.setItem(i, 5, QTableWidgetItem(f"{ml.get('lat', '')}, {ml.get('lon', '')}"))

    def _display_current(self, results):
        current_results = results.get('current_results', [])
        self.current_plot.clear()
        self.current_table.setRowCount(0)
        if not current_results:
            return

        voltages = [r['peak_voltage'] for r in current_results]
        y, x = np.histogram(voltages, bins=20)
        width = np.diff(x)[0] * 0.85
        brush = pg.mkBrush(102, 187, 106, 180)
        pen = pg.mkPen(QColor('#66bb6a').lighter(130), width=1)
        bar = pg.BarGraphItem(x=x[:-1], height=y, width=width, brush=brush, pen=pen)
        self.current_plot.addItem(bar)
        self.current_plot.setLabel('bottom', '峰值电压 (V)')
        self.current_plot.setLabel('left', '数量')

        self.current_table.setRowCount(len(current_results))
        for i, r in enumerate(current_results):
            self.current_table.setItem(i, 0, QTableWidgetItem(r['final_time']))
            self.current_table.setItem(i, 1, QTableWidgetItem(r.get('category', '')))
            self.current_table.setItem(i, 2, QTableWidgetItem(f"{r['peak_voltage']:.6f}"))

    def _display_independent(self, results):
        independent_results = results.get('independent_results', [])
        self.indep_plot.clear()
        self.indep_table.setRowCount(0)
        if not independent_results:
            return

        for cat_idx, (cat_name, cat_results) in enumerate(results.get('category_independent', {}).items()):
            color = QColor(CATEGORY_COLORS[cat_idx % len(CATEGORY_COLORS)][0])
            indep_x, indep_y = [], []
            dep_x, dep_y = [], []
            for r in cat_results:
                global_idx = independent_results.index(r)
                if r['is_independent']:
                    indep_x.append(global_idx)
                    indep_y.append(r['nearby_count'])
                else:
                    dep_x.append(global_idx)
                    dep_y.append(r['nearby_count'])

            if indep_x:
                self.indep_plot.addItem(pg.ScatterPlotItem(
                    x=indep_x, y=indep_y,
                    brush=pg.mkBrush(color.red(), color.green(), color.blue(), 200),
                    pen=pg.mkPen(None), size=8))
            if dep_x:
                darker = color.darker(150)
                self.indep_plot.addItem(pg.ScatterPlotItem(
                    x=dep_x, y=dep_y,
                    brush=pg.mkBrush(darker.red(), darker.green(), darker.blue(), 200),
                    pen=pg.mkPen(QColor('#ef5350'), width=1), size=8))

        self.indep_plot.setLabel('bottom', '事件索引')
        self.indep_plot.setLabel('left', '附近闪电数')

        self.indep_table.setRowCount(len(independent_results))
        for i, r in enumerate(independent_results):
            self.indep_table.setItem(i, 0, QTableWidgetItem(r['final_time']))
            self.indep_table.setItem(i, 1, QTableWidgetItem(r.get('category', '')))
            is_indep = QTableWidgetItem("是" if r['is_independent'] else "否")
            is_indep.setForeground(QColor('#66bb6a') if r['is_independent'] else QColor('#ef5350'))
            self.indep_table.setItem(i, 2, is_indep)
            self.indep_table.setItem(i, 3, QTableWidgetItem(str(r['nearby_count'])))
            nearby = r.get('nearby_lightnings', [])
            info = "; ".join([f"Δt={n['time_diff_ms']:.1f}ms, d={n['distance_km']:.1f}km" for n in nearby[:3]])
            if len(nearby) > 3:
                info += f" ...等{len(nearby)}个"
            self.indep_table.setItem(i, 4, QTableWidgetItem(info or "无"))

    def _export_results(self):
        if not self.analysis_results:
            QMessageBox.warning(self, "提示", "请先运行分析")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not output_dir:
            return

        results = self.analysis_results

        try:
            for name, key in [("距离分布.csv", "distance_results"),
                              ("电流强度分布.csv", "current_results"),
                              ("独立分布判断.csv", "independent_results")]:
                items = results.get(key, [])
                if not items:
                    continue
                if key == "distance_results":
                    with open(os.path.join(output_dir, name), 'w', newline='', encoding='utf-8-sig') as f:
                        w = csv.writer(f)
                        w.writerow(['时间', '分类', '距离(km)', '残差(ms)', '时延(ms)', '闪电纬度', '闪电经度'])
                        for r in items:
                            ml = r.get('matched_lightning', {})
                            w.writerow([r['final_time'], r.get('category', ''), r['distance_km'],
                                       r.get('time_diff_ms', ''), r.get('propagation_ms', ''),
                                       ml.get('lat', ''), ml.get('lon', '')])
                elif key == "current_results":
                    with open(os.path.join(output_dir, name), 'w', newline='', encoding='utf-8-sig') as f:
                        w = csv.writer(f)
                        w.writerow(['时间', '分类', '峰值电压(V)'])
                        for r in items:
                            w.writerow([r['final_time'], r.get('category', ''), r['peak_voltage']])
                elif key == "independent_results":
                    with open(os.path.join(output_dir, name), 'w', newline='', encoding='utf-8-sig') as f:
                        w = csv.writer(f)
                        w.writerow(['时间', '分类', '是否独立', '附近闪电数', '详情'])
                        for r in items:
                            nearby = r.get('nearby_lightnings', [])
                            detail = "; ".join([f"Δt={n['time_diff_ms']:.1f}ms, d={n['distance_km']:.1f}km" for n in nearby])
                            w.writerow([r['final_time'], r.get('category', ''),
                                       '是' if r['is_independent'] else '否', r['nearby_count'], detail])

            with open(os.path.join(output_dir, "汇总统计.txt"), 'w', encoding='utf-8') as f:
                f.write(self.summary_text.toPlainText())

            QMessageBox.information(self, "导出完成", f"结果已导出到:\n{output_dir}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))