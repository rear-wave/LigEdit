#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analytics / cluster_dialog — 闪电波形聚类对话框

移植自 LigCluster 的 MainWindow，缩减为 QDialog。
"""

import os
import csv

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QProgressBar, QSpinBox, QDoubleSpinBox, QComboBox,
    QMessageBox, QFileDialog, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QSplitter,
)

import pyqtgraph as pg

from lig_parser import (
    _resource_path, time_classifier_display, ButterFilter, CutPieceTo16000,
)

# 聚类配色
CLUSTER_COLORS = [
    '#4fc3f7', '#66bb6a', '#ffa726', '#ef5350',
    '#ab47bc', '#ffee58', '#26c6da', '#f06292',
    '#c6ff00', '#ff7043', '#5c6bc0', '#26a69a',
]


# ============================================================================
#                          后台工作线程
# ============================================================================

class LoadWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, lig_dir):
        super().__init__()
        self.lig_dir = lig_dir

    def run(self):
        try:
            from analytics.cluster_core import load_lig_pieces
            pieces = load_lig_pieces(self.lig_dir, progress_cb=self._progress)
            self.finished.emit(pieces)
        except Exception as e:
            self.error.emit(str(e))

    def _progress(self, msg, pct):
        self.progress.emit(msg, pct)


class ClusterWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            from analytics.cluster_core import run_full_clustering
            results = run_full_clustering(**self.config, progress_cb=self._progress)
            self.finished.emit(results or {})
        except Exception as e:
            self.error.emit(str(e))

    def _progress(self, msg, pct):
        self.progress.emit(msg, pct)


# ============================================================================
#                          聚类对话框
# ============================================================================

class ClusterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("波形聚类分析 (LigCluster)")
        self.setMinimumSize(1100, 800)
        self.resize(1200, 850)

        self.pieces = []
        self.cluster_results = None
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---- 参数区 ----
        params = QGroupBox("聚类参数")
        pgrid = QVBoxLayout(params)

        # 第1行: 目录
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("lig 数据目录:"))
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("选择 .lig 文件目录...")
        row1.addWidget(self.dir_edit, 1)
        btn_dir = QPushButton("浏览...")
        btn_dir.clicked.connect(lambda: self._browse_dir())
        row1.addWidget(btn_dir)
        self.load_btn = QPushButton("加载数据")
        self.load_btn.setStyleSheet(
            "QPushButton { background: #1565c0; color: white; padding: 6px 20px; }"
            "QPushButton:hover { background: #1976d2; }")
        self.load_btn.clicked.connect(self._load_data)
        row1.addWidget(self.load_btn)
        self.info_label = QLabel("")
        row1.addWidget(self.info_label)
        pgrid.addLayout(row1)

        # 第2行: 算法+特征参数
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("特征:"))
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(["handcraft", "raw", "combined"])
        row2.addWidget(self.feature_combo)
        row2.addSpacing(15)
        row2.addWidget(QLabel("算法:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["kmeans", "dbscan", "agglomerative", "gmm"])
        self.algo_combo.currentTextChanged.connect(self._on_algo_changed)
        row2.addWidget(self.algo_combo)
        row2.addSpacing(15)
        row2.addWidget(QLabel("聚类数 k:"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 50)
        self.k_spin.setValue(3)
        row2.addWidget(self.k_spin)
        row2.addSpacing(15)
        row2.addWidget(QLabel("降维:"))
        self.dim_combo = QComboBox()
        self.dim_combo.addItems(["tsne", "pca", "umap"])
        row2.addWidget(self.dim_combo)
        row2.addStretch()
        pgrid.addLayout(row2)

        # 第3行: 算法专属参数
        row3 = QHBoxLayout()
        self.dbscan_group = QHBoxLayout()
        self.dbscan_group.addWidget(QLabel("  eps:"))
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.01, 100)
        self.eps_spin.setValue(0.5)
        self.eps_spin.setSingleStep(0.1)
        self.dbscan_group.addWidget(self.eps_spin)
        self.dbscan_group.addWidget(QLabel("min_samples:"))
        self.min_samp_spin = QSpinBox()
        self.min_samp_spin.setRange(2, 100)
        self.min_samp_spin.setValue(5)
        self.dbscan_group.addWidget(self.min_samp_spin)
        self._dbscan_widget = QWidget()
        self._dbscan_widget.setLayout(self.dbscan_group)
        self._dbscan_widget.setVisible(False)
        row3.addWidget(self._dbscan_widget)

        row3.addWidget(QLabel("  滤波 (kHz):"))
        self.fc_spin = QDoubleSpinBox()
        self.fc_spin.setRange(10, 2500)
        self.fc_spin.setValue(300)
        self.fc_spin.setSingleStep(50)
        row3.addWidget(self.fc_spin)
        row3.addStretch()
        pgrid.addLayout(row3)

        # 第4行: 运行
        row4 = QHBoxLayout()
        self.run_btn = QPushButton("开始聚类")
        self.run_btn.setStyleSheet(
            "QPushButton { background: #2e7d32; color: white; font-size: 14px; padding: 8px 32px; }"
            "QPushButton:hover { background: #388e3c; }")
        self.run_btn.clicked.connect(self._run_clustering)
        self.run_btn.setEnabled(False)
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
        pgrid.addLayout(row4)

        layout.addWidget(params)

        # ---- 结果 Tabs ----
        tabs = QTabWidget()

        # 可视化 Tab
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 0, 0, 0)

        self.scatter_plot = pg.PlotWidget()
        self.scatter_plot.setBackground('#1e1e2e')
        self.scatter_plot.setLabel('bottom', '维度 1')
        self.scatter_plot.setLabel('left', '维度 2')
        self.scatter_plot.showGrid(x=True, y=True, alpha=0.2)
        viz_layout.addWidget(self.scatter_plot, 1)

        # 波形预览
        splitter = QSplitter(Qt.Horizontal)
        self.cluster_table = QTableWidget()
        self.cluster_table.setColumnCount(5)
        self.cluster_table.setHorizontalHeaderLabels(['聚类', '时间', '昼夜', '峰值(V)', '预览'])
        self.cluster_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cluster_table.setAlternatingRowColors(True)
        self.cluster_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cluster_table.currentCellChanged.connect(self._on_table_select)
        splitter.addWidget(self.cluster_table)

        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setBackground('#0a0a14')
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.25)
        self.waveform_plot.setLabel('left', '', units='V')
        self.waveform_plot.setLabel('bottom', '时间', units='ms')
        self.waveform_plot.setMaximumWidth(450)
        self.waveform_curve = self.waveform_plot.plot(
            pen=pg.mkPen(color='#ffffff', width=1.2),
            autoDownsample=True, clipToView=True)
        splitter.addWidget(self.waveform_plot)
        splitter.setSizes([800, 400])
        viz_layout.addWidget(splitter)
        tabs.addTab(viz_widget, "聚类可视化")

        # 评估 Tab
        self.eval_text = QTextEdit()
        self.eval_text.setReadOnly(True)
        tabs.addTab(self.eval_text, "聚类评估")

        # 统计 Tab
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(3)
        self.stats_table.setHorizontalHeaderLabels(['聚类', '样本数', '占比(%)'])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tabs.addTab(self.stats_table, "统计")

        # 日志 Tab
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        tabs.addTab(self.log_text, "日志")

        layout.addWidget(tabs, 1)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择 lig 数据目录")
        if d:
            self.dir_edit.setText(d)

    def _on_algo_changed(self, algo):
        self._dbscan_widget.setVisible(algo == 'dbscan')
        self.k_spin.setEnabled(algo != 'dbscan')

    # ---- 加载 ----
    def _load_data(self):
        lig_dir = self.dir_edit.text().strip()
        if not lig_dir:
            QMessageBox.warning(self, "提示", "请选择 lig 数据目录")
            return

        self.load_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self.worker = LoadWorker(lig_dir)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_load_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_load_finished(self, pieces):
        self.pieces = pieces
        self.load_btn.setEnabled(True)
        self.run_btn.setEnabled(len(pieces) > 0)
        self.progress_bar.setValue(100)
        self.info_label.setText(f"已加载 {len(pieces)} 条波形")
        self.log_text.append(f"数据加载完成: {len(pieces)} 条波形")

    # ---- 聚类 ----
    def _run_clustering(self):
        if not self.pieces:
            QMessageBox.warning(self, "无数据", "请先加载数据")
            return

        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        config = {
            'pieces': self.pieces,
            'feature_mode': self.feature_combo.currentText(),
            'algorithm': self.algo_combo.currentText(),
            'n_clusters': self.k_spin.value(),
            'dbscan_eps': self.eps_spin.value(),
            'dbscan_min_samples': self.min_samp_spin.value(),
            'filter_fc': self.fc_spin.value() * 1000,
            'dim_reduction': self.dim_combo.currentText(),
            'export_dir': None,
            'lig_head_path': _resource_path('Limitbyt'),
            'lig_file_head_path': _resource_path('LigHead.lig'),
        }

        self.worker = ClusterWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_clustering_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_clustering_finished(self, results):
        self.cluster_results = results
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.log_text.append("\n=== 聚类完成 ===")
        if results:
            self._display_results(results)

    def _on_progress(self, msg, pct):
        self.status_label.setText(msg)
        if pct >= 0:
            self.progress_bar.setValue(pct)
        self.log_text.append(msg)

    def _on_error(self, err_msg):
        self.run_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.status_label.setText("出错")
        self.log_text.append(f"\n[错误] {err_msg}")

    # ---- 结果显示 ----
    def _display_results(self, results):
        self._display_scatter(results)
        self._display_eval(results)
        self._display_stats(results)
        self._display_table(results)

    def _display_scatter(self, results):
        self.scatter_plot.clear()
        embedding = results.get('embedding')
        labels = results.get('labels')
        if embedding is None or labels is None:
            return

        unique_labels = sorted(set(labels))
        for label in unique_labels:
            mask = labels == label
            x = embedding[mask, 0]
            y = embedding[mask, 1]
            if label == -1:
                color, name, size = '#666666', 'noise', 4
            else:
                color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
                name = f'cluster_{label}'
                size = 7

            scatter = pg.ScatterPlotItem(
                x=x, y=y, brush=pg.mkBrush(color),
                pen=pg.mkPen(None), size=size, name=name)
            self.scatter_plot.addItem(scatter)

        self.scatter_plot.addLegend(offset=(10, 10))

    def _display_eval(self, results):
        eval_data = results.get('evaluation', {})
        lines = []
        lines.append("=" * 50)
        lines.append("  聚类评估报告")
        lines.append("=" * 50)
        lines.append(f"算法: {self.algo_combo.currentText()}")
        lines.append(f"特征: {self.feature_combo.currentText()}")
        lines.append(f"有效样本: {len(results.get('valid_indices', []))}")
        lines.append("")
        lines.append(f"聚类数: {eval_data.get('n_clusters', 'N/A')}")
        lines.append(f"噪声: {eval_data.get('n_noise', 0)}")
        lines.append("")
        sil = eval_data.get('silhouette')
        ch = eval_data.get('calinski_harabasz')
        db = eval_data.get('davies_bouldin')
        lines.append(f"轮廓系数: {f'{sil:.4f}' if sil is not None else 'N/A'}")
        lines.append(f"CH 指数: {f'{ch:.2f}' if ch is not None else 'N/A'}")
        lines.append(f"DB 指数: {f'{db:.4f}' if db is not None else 'N/A'}")
        lines.append("")
        for name, size in sorted(eval_data.get('cluster_sizes', {}).items()):
            lines.append(f"  {name}: {size}")
        lines.append("=" * 50)
        self.eval_text.setPlainText("\n".join(lines))

    def _display_stats(self, results):
        eval_data = results.get('evaluation', {})
        cluster_sizes = eval_data.get('cluster_sizes', {})
        total = sum(cluster_sizes.values()) if cluster_sizes else 0

        self.stats_table.setRowCount(len(cluster_sizes))
        for i, (name, size) in enumerate(sorted(cluster_sizes.items())):
            self.stats_table.setItem(i, 0, QTableWidgetItem(name))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(size)))
            pct = f"{size / max(total, 1) * 100:.1f}"
            self.stats_table.setItem(i, 2, QTableWidgetItem(pct))

    def _display_table(self, results):
        labels = results.get('labels', [])
        valid_indices = results.get('valid_indices', [])
        self.cluster_table.setRowCount(len(valid_indices))

        for i, idx in enumerate(valid_indices):
            label = labels[i]
            piece = self.pieces[idx]

            cluster_name = "noise" if label == -1 else f"cluster_{label}"
            color = QColor('#666666') if label == -1 else QColor(CLUSTER_COLORS[label % len(CLUSTER_COLORS)])

            item = QTableWidgetItem(cluster_name)
            item.setForeground(color)
            self.cluster_table.setItem(i, 0, item)
            self.cluster_table.setItem(i, 1, QTableWidgetItem(piece.get('time_key', '')))

            self.cluster_table.setItem(i, 2, QTableWidgetItem(time_classifier_display(piece.get('time_key', ''))))

            peak_v = piece.get('peak_voltage')
            self.cluster_table.setItem(i, 3, QTableWidgetItem(f"{peak_v:.4f}" if peak_v else "N/A"))
            self.cluster_table.setItem(i, 4, QTableWidgetItem("双击预览"))

    def _on_table_select(self, row, col, prev_row, prev_col):
        if row < 0 or not self.cluster_results:
            return
        labels = self.cluster_results.get('labels', [])
        valid_indices = self.cluster_results.get('valid_indices', [])
        if row >= len(valid_indices):
            return

        idx = valid_indices[row]
        piece = self.pieces[idx]
        voltage = piece.get('voltage')
        if voltage is None:
            return

        v_centered = voltage - np.mean(voltage)
        try:
            v_filtered = ButterFilter(v_centered)
        except Exception:
            v_filtered = v_centered
        v_cut = CutPieceTo16000(v_filtered)

        fs = 5000000
        time_array = np.arange(len(v_cut)) / fs * 1000

        label = labels[row]
        color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)] if label >= 0 else '#666666'
        self.waveform_curve.setData(time_array, v_cut)
        self.waveform_curve.setPen(pg.mkPen(color=color, width=1.2))

    # ---- 导出 ----
    def _export_results(self):
        if not self.cluster_results:
            QMessageBox.warning(self, "提示", "请先运行聚类")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not output_dir:
            return

        from analytics.cluster_core import export_clusters_to_lig, export_cluster_timestamps

        try:
            labels = self.cluster_results['labels']
            valid_indices = self.cluster_results['valid_indices']

            export_clusters_to_lig(
                self.pieces, labels, valid_indices, output_dir,
                lig_head_path=_resource_path('Limitbyt'),
                lig_file_head_path=_resource_path('LigHead.lig'))

            export_cluster_timestamps(self.pieces, labels, valid_indices, output_dir)

            # 评估报告
            with open(os.path.join(output_dir, "聚类评估报告.txt"), 'w', encoding='utf-8') as f:
                f.write(self.eval_text.toPlainText())

            # CSV 详情
            csv_path = os.path.join(output_dir, "聚类详情.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['时间戳', '聚类', '昼夜', '峰值电压(V)'])
                for i, idx in enumerate(valid_indices):
                    piece = self.pieces[idx]
                    label = labels[i]
                    cluster_name = 'noise' if label == -1 else f'cluster_{label}'
                    writer.writerow([
                        piece['time_key'], cluster_name,
                        time_classifier_display(piece['time_key']),
                        piece.get('peak_voltage', ''),
                    ])

            QMessageBox.information(self, "导出完成", f"结果已导出到:\n{output_dir}")

        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))