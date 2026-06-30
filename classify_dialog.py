#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波形分类对话框 - 选择文件夹，批量对所有 .lig 文件进行 ResNet1D 分类
"""

import os

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QProgressBar, QSpinBox, QFileDialog, QMessageBox,
)


# ============================================================================
#                          后台工作线程
# ============================================================================

class _ClassifyWorker(QThread):
    progress = pyqtSignal(str, int)   # message, percent
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(self, input_dir, output_dir, batch_size, max_pieces):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_pieces = max_pieces

    def run(self):
        try:
            from classify_module import classify_folder
            counts = classify_folder(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                batch_size=self.batch_size,
                max_pieces=self.max_pieces,
                progress_cb=self._progress,
                log_cb=self._log,
            )
            total = sum(counts.values())
            if total == 0:
                self.finished_ok.emit("未找到可分类的波形数据")
                return
            from classify_module import _class_names
            lines = [f"分类完成！共 {total} 条波形"]
            for c in (_class_names or []):
                n = counts.get(c, 0)
                lines.append(f"  {c}: {n} 条 ({n/max(total,1)*100:.1f}%)")
            lines.append(f"\n结果保存至: {os.path.join(self.output_dir, 'summary.csv')}")
            self.finished_ok.emit("\n".join(lines))
        except Exception as e:
            self.finished_err.emit(f"执行出错: {e}")

    def _progress(self, step, msg, pct):
        self.progress.emit(msg, pct)

    def _log(self, msg):
        self.log.emit(msg)


# ============================================================================
#                          波形分类对话框
# ============================================================================

class WaveformClassifyDialog(QDialog):
    """波形分类：读取文件夹下所有 .lig 文件，用 ResNet1D 模型分类"""

    def __init__(self, parent=None, default_dir=""):
        super().__init__(parent)
        self.setWindowTitle("波形分类")
        self.setMinimumWidth(560)
        self.worker = None
        self._default_dir = default_dir
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---- 输入 ----
        input_group = QGroupBox("输入")
        form = QFormLayout(input_group)

        self.edit_input_dir = self._add_dir_row(form, "lig文件目录:")
        if self._default_dir:
            self.edit_input_dir.setText(self._default_dir)

        layout.addWidget(input_group)

        # ---- 参数 ----
        param_group = QGroupBox("参数")
        param_form = QFormLayout(param_group)

        self.edit_batch_size = QSpinBox()
        self.edit_batch_size.setRange(1, 2048)
        self.edit_batch_size.setValue(256)
        param_form.addRow("批次大小:", self.edit_batch_size)

        self.edit_max_pieces = QSpinBox()
        self.edit_max_pieces.setRange(0, 9999999)
        self.edit_max_pieces.setValue(0)
        self.edit_max_pieces.setSpecialValueText("不限制")
        param_form.addRow("最大片段数:", self.edit_max_pieces)

        layout.addWidget(param_group)

        # ---- 输出 ----
        output_group = QGroupBox("输出")
        out_form = QFormLayout(output_group)
        self.edit_output_dir = self._add_dir_row(out_form, "输出目录:")
        layout.addWidget(output_group)

        # ---- 模型信息 ----
        self.model_info_label = QLabel("")
        self.model_info_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.model_info_label)
        self._refresh_model_info()

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
        self.btn_run = QPushButton("执行分类")
        self.btn_run.setStyleSheet(
            "background-color: #0078d7; color: white; padding: 8px 24px; font-weight: bold;")
        self.btn_run.clicked.connect(self._on_run)
        btn_cancel = QPushButton("关闭")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    # ---- 辅助 ----

    def _add_dir_row(self, form, label):
        row_layout = QHBoxLayout()
        edit = QLineEdit()
        btn = QPushButton("浏览...")
        btn.clicked.connect(lambda: self._browse_dir(edit))
        row_layout.addWidget(edit, stretch=1)
        row_layout.addWidget(btn)
        form.addRow(label, row_layout)
        return edit

    def _browse_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if d:
            edit.setText(d)

    def _browse_model(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择模型检查点", "",
            "PyTorch模型 (*.pt *.pth);;所有文件 (*.*)"
        )
        if fp:
            self.edit_model_path.setText(fp)
            self._refresh_model_info()

    def _refresh_model_info(self):
        """更新模型信息标签"""
        try:
            from classify_module import get_model_info
            info = get_model_info()
            if info:
                classes = info.get("class_names", [])
                self.model_info_label.setText(
                    f"模型类别: {', '.join(classes)} | "
                    f"总类别数: {len(classes)}"
                )
            else:
                self.model_info_label.setText("(未检测到模型)")
        except Exception as e:
            self.model_info_label.setText(f"(模型信息读取失败: {e})")

    # ---- 运行 ----

    def _on_run(self):
        input_dir = self.edit_input_dir.text().strip()
        if not input_dir:
            QMessageBox.warning(self, "提示", "请选择 lig 文件目录")
            return
        if not os.path.isdir(input_dir):
            QMessageBox.warning(self, "提示", "lig 文件目录不存在")
            return

        output_dir = self.edit_output_dir.text().strip()
        if not output_dir:
            output_dir = os.path.join(input_dir, "classified")

        batch_size = self.edit_batch_size.value()
        max_pieces = self.edit_max_pieces.value()
        if max_pieces == 0:
            max_pieces = None

        self.btn_run.setEnabled(False)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("正在加载模型...")

        self.worker = _ClassifyWorker(input_dir, output_dir,
                                       batch_size, max_pieces)
        self.worker.progress.connect(
            lambda msg, pct: (self.status_label.setText(msg),
                              self.progress_bar.setValue(pct)))
        self.worker.log.connect(self.log_text.append)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_err.connect(self._on_finished_err)
        self.worker.start()

    def _on_finished_ok(self, msg):
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText("完成")
        QMessageBox.information(self, "分类完成", msg)

    def _on_finished_err(self, msg):
        self.btn_run.setEnabled(True)
        self.status_label.setText(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)
