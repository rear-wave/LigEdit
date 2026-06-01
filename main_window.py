#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MainWindow - LigEdit PyQt5 主窗口
替代原 tkinter LigEditorApp，功能完整迁移
"""

import os
import sys
import shutil

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QKeySequence
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QMenuBar, QMenu, QAction,
    QStatusBar, QFileDialog, QMessageBox, QLabel, QInputDialog,
    QFrame, QApplication, QShortcut, QAbstractItemView
)

from waveform_widget import WaveformWidget, SCOPE_STYLE
from lig_editor import (
    ReadLigFileWithOffsets, SaveLigFile, ButterFilter,
    load_station_coords, match_station_name,
    format_time_display, time_classifier_display,
)


# ============================================================================
#                          主窗口
# ============================================================================

class MainWindow(QMainWindow):
    """LigEdit 主窗口 - 雷电波形编辑器"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LigEdit - 雷电波形编辑器")
        self.setMinimumSize(1200, 700)
        self.showMaximized()

        # 数据模型
        self.file_data = {}
        self.deleted_sets = {}
        self.checked_sets = {}
        self.active_file = None
        self.selected_piece = None
        self.station_coords = load_station_coords()

        # 构建界面
        self._build_menubar()
        self._build_main_area()
        self._build_statusbar()
        self._build_shortcuts()

    # -------------------- 菜单栏 --------------------
    def _build_menubar(self):
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        file_menu.addAction("打开文件...", self.open_file, QKeySequence("Ctrl+O"))
        file_menu.addAction("打开文件夹...", self.open_folder)
        file_menu.addSeparator()
        file_menu.addAction("保存当前文件", self.save_file, QKeySequence("Ctrl+S"))
        file_menu.addAction("另存为...", self.save_as)
        file_menu.addSeparator()
        file_menu.addAction("关闭当前文件", self.close_current_file)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close, QKeySequence("Alt+F4"))

        # 编辑菜单
        edit_menu = menubar.addMenu("编辑(&E)")
        edit_menu.addAction("删除选中", self.delete_selected, QKeySequence("Delete"))
        edit_menu.addAction("删除全部勾选", self.delete_checked)
        edit_menu.addSeparator()
        edit_menu.addAction("撤销删除", self.undo_delete, QKeySequence("Ctrl+Z"))
        edit_menu.addAction("全部撤销删除", self.undo_all_delete)
        edit_menu.addSeparator()
        edit_menu.addAction("全勾选", self.check_all, QKeySequence("Ctrl+Shift+A"))
        edit_menu.addAction("全取消勾选", self.uncheck_all)

        # 筛选菜单
        filter_menu = menubar.addMenu("筛选(&L)")
        filter_menu.addAction("按关键字筛选...", self.filter_dialog)
        filter_menu.addAction("清除筛选", self.clear_filter)

        # 导出菜单
        export_menu = menubar.addMenu("导出(&X)")
        export_menu.addAction("导出勾选片段...", self.export_checked, QKeySequence("Ctrl+E"))
        export_menu.addAction("导出未删除片段...", self.export_not_deleted)
        export_menu.addSeparator()
        export_menu.addAction("按昼夜分类导出...", self.export_by_daynight)
        export_menu.addAction("导出时间戳列表...", self.export_timestamps)
        export_menu.addAction("导出勾选时间戳...", self.export_checked_timestamps)

        # 数据处理菜单
        data_menu = menubar.addMenu("数据处理(&D)")
        data_menu.addAction("按距离分类...", self.open_distance_classify)
        data_menu.addAction("按昼夜分类...", self.open_daynight_classify)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        help_menu.addAction("关于LigEdit", self.show_about)

    # -------------------- 主区域 --------------------
    def _build_main_area(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # 左侧: 文件树
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(4, 2, 4, 2)
        header_label = QLabel("波形列表")
        header_label.setFont(QFont("", 10, QFont.Bold))
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        self.list_count_label = QLabel("")
        header_layout.addWidget(self.list_count_label)
        left_layout.addWidget(header_frame)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["勾选", "发生时间", "昼夜", "状态"])
        self.tree.setColumnWidth(0, 55)
        self.tree.setColumnWidth(1, 220)
        self.tree.setColumnWidth(2, 60)
        self.tree.setColumnWidth(3, 70)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setAlternatingRowColors(True)
        self.tree.setIndentation(0)
        self.tree.setRootIsDecorated(False)

        # 样式
        self.tree.setStyleSheet("""
            QTreeWidget {
                font-size: 13px;
                selection-background-color: #0078d7;
                selection-color: #ffffff;
                outline: none;
            }
            QTreeWidget::item {
                padding: 3px 0px;
            }
            QTreeWidget::item:alternate {
                background-color: #f0f4ff;
            }
            QTreeWidget::item:selected {
                background-color: #0078d7;
                color: #ffffff;
            }
            QTreeWidget::item:hover {
                background-color: #c4d9f0;
            }
            QTreeWidget::item:selected:hover {
                background-color: #006abc;
            }
        """)

        left_layout.addWidget(self.tree)
        splitter.addWidget(left_widget)

        # 右侧: 波形预览
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 片段信息栏
        self.info_frame = QFrame()
        self.info_frame.setFrameShape(QFrame.StyledPanel)
        self.info_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a4a;
                border: 1px solid #333366;
                border-radius: 4px;
                padding: 4px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 12px;
            }
        """)
        info_layout = QVBoxLayout(self.info_frame)
        info_layout.setContentsMargins(8, 4, 8, 4)
        self.info_label = QLabel("请打开lig文件")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        right_layout.addWidget(self.info_frame)

        # 波形显示控件
        self.waveform = WaveformWidget()
        right_layout.addWidget(self.waveform, stretch=1)

        # 底部状态信息栏
        self.detail_frame = QFrame()
        self.detail_frame.setFrameShape(QFrame.StyledPanel)
        self.detail_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a4a;
                border: 1px solid #333366;
                border-radius: 4px;
                padding: 2px;
            }
            QLabel {
                color: #b0b0d0;
                font-size: 11px;
            }
        """)
        detail_layout = QHBoxLayout(self.detail_frame)
        detail_layout.setContentsMargins(8, 2, 8, 2)
        self.detail_label = QLabel("")
        detail_layout.addWidget(self.detail_label)
        right_layout.addWidget(self.detail_frame)

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])

        # 信号连接
        self.tree.currentItemChanged.connect(self.on_tree_select)
        self.tree.itemDoubleClicked.connect(self.on_tree_double_click)
        self.tree.customContextMenuRequested.connect(self.on_tree_right_click)

    # -------------------- 状态栏 --------------------
    def _build_statusbar(self):
        self.statusBar().showMessage("未打开文件")

        self.status_file_label = QLabel("未打开文件")
        self.status_files_label = QLabel("文件数: 0")
        self.status_total_label = QLabel("总片段: 0")
        self.status_deleted_label = QLabel("待删除: 0")
        self.status_checked_label = QLabel("已勾选: 0")

        for label in [self.status_file_label, self.status_files_label,
                      self.status_total_label, self.status_deleted_label,
                      self.status_checked_label]:
            self.statusBar().addPermanentWidget(label)

    # -------------------- 快捷键 --------------------
    def _build_shortcuts(self):
        pass  # 快捷键已通过 QKeySequence 在菜单中绑定

    # -------------------- 右键菜单 --------------------
    def _build_context_menu(self):
        menu = QMenu(self)
        menu.addAction("预览波形", self.ctx_preview)
        menu.addSeparator()
        menu.addAction("勾选", self.ctx_check)
        menu.addAction("取消勾选", self.ctx_uncheck)
        menu.addSeparator()
        menu.addAction("标记删除", self.ctx_delete)
        menu.addAction("撤销删除", self.ctx_undo_delete)
        return menu

    # -------------------- 状态更新 --------------------
    def update_status(self):
        total_pieces = sum(len(fd['pieces']) for fd in self.file_data.values())
        total_deleted = sum(len(s) for s in self.deleted_sets.values())
        total_checked = sum(len(s) for s in self.checked_sets.values())

        if self.active_file and self.active_file in self.file_data:
            self.status_file_label.setText(f"当前: {os.path.basename(self.active_file)}")
        else:
            self.status_file_label.setText("未打开文件")

        self.status_files_label.setText(f"文件数: {len(self.file_data)}")
        self.status_total_label.setText(f"总片段: {total_pieces}")
        self.status_deleted_label.setText(f"待删除: {total_deleted}")
        self.status_checked_label.setText(f"已勾选: {total_checked}")
        self.list_count_label.setText(f"共 {len(self.file_data)} 个文件, {total_pieces} 条")

    # -------------------- 解析选中项 --------------------
    def _parse_tree_item(self, item):
        """解析树节点为 (filepath, index_or_None)"""
        parent = item.parent()
        if parent is None:
            # 文件节点
            return item.data(0, Qt.UserRole), None
        else:
            filepath = parent.data(0, Qt.UserRole)
            index = item.data(0, Qt.UserRole)
            return filepath, index

    # -------------------- 文件操作 --------------------
    def open_file(self):
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "打开lig文件（可多选）", "",
            "LIG文件 (*.lig);;所有文件 (*.*)"
        )
        for fp in filepaths:
            self._load_file(fp)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择包含lig文件的文件夹")
        if not folder:
            return
        lig_files = []
        for root_dir, dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith('.lig'):
                    lig_files.append(os.path.join(root_dir, f))
        if not lig_files:
            QMessageBox.information(self, "提示", "该文件夹下未找到lig文件")
            return
        for fp in lig_files:
            self._load_file(fp)

    def close_current_file(self):
        if not self.active_file or self.active_file not in self.file_data:
            return
        self._remove_file(self.active_file)

    def _remove_file(self, filepath):
        if filepath in self.file_data:
            del self.file_data[filepath]
        if filepath in self.deleted_sets:
            del self.deleted_sets[filepath]
        if filepath in self.checked_sets:
            del self.checked_sets[filepath]

        # 删除树节点
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if item.data(0, Qt.UserRole) == filepath:
                root.removeChild(item)
                break

        self.active_file = next(iter(self.file_data.keys()), None)
        self.waveform.clear_waveform()
        self.update_status()

    def _load_file(self, filepath):
        if filepath in self.file_data:
            QMessageBox.information(self, "提示", f"文件已加载:\n{filepath}")
            return
        self._do_load_file(filepath)

    def _reload_file(self, filepath):
        """强制重载文件"""
        self._remove_file(filepath)
        self._do_load_file(filepath)

    def _do_load_file(self, filepath):
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            header, pieces, raw_data, piece_offsets, header_size = ReadLigFileWithOffsets(filepath)
            self.file_data[filepath] = {
                'header': header,
                'pieces': pieces,
                'raw_data': raw_data,
                'piece_offsets': piece_offsets,
                'header_size': header_size,
            }
            self.deleted_sets[filepath] = set()
            self.checked_sets[filepath] = set()

            self._add_file_to_tree(filepath)
            self.active_file = filepath
            self.update_status()
            QApplication.restoreOverrideCursor()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "加载失败", f"无法解析lig文件:\n{filepath}\n{e}")

    def _add_file_to_tree(self, filepath):
        fd = self.file_data[filepath]
        piece_count = len(fd['pieces'])

        # 文件节点
        file_item = QTreeWidgetItem()
        file_item.setText(0, "⚡")
        file_item.setText(1, os.path.basename(filepath))
        file_item.setText(3, f"{piece_count}条")
        file_item.setData(0, Qt.UserRole, filepath)
        file_item.setFont(1, QFont("", 9, QFont.Bold))
        file_item.setBackground(1, QColor("#E8F0FE"))
        self.tree.addTopLevelItem(file_item)
        file_item.setExpanded(True)

        # 时间戳子节点
        ds = self.deleted_sets.get(filepath, set())
        cs = self.checked_sets.get(filepath, set())
        for i, (time_key, _) in enumerate(fd['pieces']):
            display_time = format_time_display(time_key)
            daynight = time_classifier_display(time_key)
            status = "已删除" if i in ds else ""
            check_mark = "✓" if i in cs else ""

            child = QTreeWidgetItem()
            child.setText(0, check_mark)
            child.setText(1, display_time)
            child.setText(2, daynight)
            child.setText(3, status)
            child.setData(0, Qt.UserRole, i)
            child.setTextAlignment(0, Qt.AlignCenter)

            if i in ds:
                for col in range(4):
                    child.setForeground(col, QColor("red"))
            elif i in cs:
                child.setForeground(0, QColor("#008000"))
                child.setFont(0, QFont("Segoe UI Symbol", 11, QFont.Bold))
                for col in range(1, 4):
                    child.setForeground(col, QColor("#006400"))

            file_item.addChild(child)

    # -------------------- 树形事件 --------------------
    def on_tree_select(self, current, previous):
        if current is None:
            return
        filepath, idx = self._parse_tree_item(current)
        if idx is not None:
            self.active_file = filepath
            self.selected_piece = (filepath, idx)
            self._preview_piece(filepath, idx)
        elif filepath is not None:
            self.active_file = filepath
        self.update_status()

    def on_tree_double_click(self, item, column):
        filepath, idx = self._parse_tree_item(item)
        if idx is None or filepath is None:
            return
        cs = self.checked_sets.setdefault(filepath, set())
        if idx in cs:
            cs.discard(idx)
        else:
            cs.add(idx)
        self._update_tree_item(filepath, idx)
        self.update_status()
        # 刷新波形颜色
        self._preview_piece(filepath, idx)

    def on_tree_right_click(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        filepath, idx = self._parse_tree_item(item)
        if idx is not None:
            self.active_file = filepath
            self.selected_piece = (filepath, idx)
            menu = self._build_context_menu()
            menu.exec_(self.tree.viewport().mapToGlobal(pos))

    # -------------------- 右键菜单动作 --------------------
    def ctx_preview(self):
        if self.selected_piece:
            self._preview_piece(*self.selected_piece)

    def ctx_check(self):
        if self.selected_piece:
            fp, idx = self.selected_piece
            cs = self.checked_sets.setdefault(fp, set())
            cs.add(idx)
            self._update_tree_item(fp, idx)
            self.update_status()

    def ctx_uncheck(self):
        if self.selected_piece:
            fp, idx = self.selected_piece
            cs = self.checked_sets.get(fp, set())
            cs.discard(idx)
            self._update_tree_item(fp, idx)
            self.update_status()

    def ctx_delete(self):
        if self.selected_piece:
            fp, idx = self.selected_piece
            ds = self.deleted_sets.setdefault(fp, set())
            ds.add(idx)
            self._update_tree_item(fp, idx)
            self.update_status()
            self._preview_piece(fp, idx)

    def ctx_undo_delete(self):
        if self.selected_piece:
            fp, idx = self.selected_piece
            ds = self.deleted_sets.get(fp, set())
            ds.discard(idx)
            self._update_tree_item(fp, idx)
            self.update_status()
            self._preview_piece(fp, idx)

    # -------------------- 更新树节点 --------------------
    def _update_tree_item(self, filepath, idx):
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            file_item = root.child(i)
            if file_item.data(0, Qt.UserRole) == filepath:
                if idx < file_item.childCount():
                    child = file_item.child(idx)
                    fd = self.file_data[filepath]
                    time_key = fd['pieces'][idx][0]
                    display_time = format_time_display(time_key)
                    daynight = time_classifier_display(time_key)
                    ds = self.deleted_sets.get(filepath, set())
                    cs = self.checked_sets.get(filepath, set())

                    child.setText(0, "✓" if idx in cs else "")
                    child.setText(1, display_time)
                    child.setText(2, daynight)
                    child.setText(3, "已删除" if idx in ds else "")

                    if idx in ds:
                        color = QColor("red")
                        for col in range(4):
                            child.setForeground(col, color)
                    elif idx in cs:
                        child.setForeground(0, QColor("#008000"))
                        child.setFont(0, QFont("Segoe UI Symbol", 11, QFont.Bold))
                        for col in range(1, 4):
                            child.setForeground(col, QColor("#006400"))
                    else:
                        for col in range(4):
                            child.setForeground(col, QColor())
                break

    # -------------------- 波形预览 --------------------
    def _preview_piece(self, filepath, idx):
        fd = self.file_data.get(filepath)
        if not fd or idx < 0 or idx >= len(fd['pieces']):
            return

        time_key, piece_data = fd['pieces'][idx]

        try:
            raw_piece = np.array(piece_data['0'], dtype=np.float64)
        except (KeyError, IndexError):
            self.info_label.setText(f"片段 {idx+1}: 无法读取波形数据")
            return

        piece_centered = raw_piece - np.mean(raw_piece)
        try:
            filtered = ButterFilter(piece_centered)
        except Exception:
            filtered = piece_centered

        # 元数据
        version = piece_data.get('version', 'N/A')
        sampling_rate = piece_data.get('m_samplingRate', 'N/A')
        num_data = piece_data.get('m_numOfData', 'N/A')
        num_channel = piece_data.get('m_numOfChannel', 'N/A')
        station_id = piece_data.get('m_stationID', 'N/A')
        lat = piece_data.get('m_GPSCurrentLocationLat', 'N/A')
        lon = piece_data.get('m_GPSCurrentLocationLon', 'N/A')

        # 时间段检测
        is_daytime = time_classifier_display(time_key) == "白天"

        # 站点匹配
        station_name = "UNKNOWN"
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            if lat and lon and self.station_coords:
                station_name = match_station_name(lat, lon, self.station_coords)

        # 删除/勾选状态
        ds = self.deleted_sets.get(filepath, set())
        cs = self.checked_sets.get(filepath, set())
        is_deleted = idx in ds
        is_checked = idx in cs

        # 更新信息栏
        period_icon = "☀" if is_daytime else "🌙"
        period_text = "白天" if is_daytime else "夜晚"
        info_text = (
            f"[{os.path.basename(filepath)}] 片段 {idx+1}/{len(fd['pieces'])} | "
            f"时间: {format_time_display(time_key)} | "
            f"站点: {station_name} | "
            f"{period_icon} {period_text}"
        )
        if is_deleted:
            info_text += " | [已标记删除]"
        self.info_label.setText(info_text)

        # 更新底部详情
        detail_text = (
            f"版本: {version} | 采样率: {sampling_rate} | "
            f"数据点: {num_data} | 通道: {num_channel} | "
            f"站点ID: {station_id} | "
            f"纬度: {lat} | 经度: {lon}"
        )
        self.detail_label.setText(detail_text)

        # 生成时间轴
        fs = 5000000
        if isinstance(sampling_rate, (int, float)) and sampling_rate > 0:
            fs = sampling_rate
        time_array = np.arange(len(filtered)) / fs * 1000

        # 设置波形(同时显示原始+滤波)
        self.waveform.set_waveform(
            data=filtered,
            raw_data=piece_centered,
            time_array=time_array,
            is_daytime=is_daytime,
            is_deleted=is_deleted,
            is_checked=is_checked,
        )

    # -------------------- 编辑操作 --------------------
    def delete_selected(self):
        item = self.tree.currentItem()
        if item is None:
            return
        filepath, idx = self._parse_tree_item(item)
        if idx is None:
            return
        ds = self.deleted_sets.setdefault(filepath, set())
        if idx not in ds:
            ds.add(idx)
            self._update_tree_item(filepath, idx)
            self.update_status()
            self._preview_piece(filepath, idx)

    def delete_checked(self):
        count = 0
        for fp, cs in self.checked_sets.items():
            ds = self.deleted_sets.setdefault(fp, set())
            for idx in list(cs):
                if idx not in ds:
                    ds.add(idx)
                    self._update_tree_item(fp, idx)
                    count += 1
        self.update_status()
        if count > 0 and self.selected_piece:
            self._preview_piece(*self.selected_piece)

    def undo_delete(self):
        items = self.tree.selectedItems()
        for item in items:
            filepath, idx = self._parse_tree_item(item)
            if idx is None:
                continue
            ds = self.deleted_sets.get(filepath, set())
            if idx in ds:
                ds.discard(idx)
                self._update_tree_item(filepath, idx)
        self.update_status()
        if self.selected_piece:
            self._preview_piece(*self.selected_piece)

    def undo_all_delete(self):
        if not any(self.deleted_sets.values()):
            QMessageBox.information(self, "提示", "没有标记删除的片段")
            return
        total = sum(len(s) for s in self.deleted_sets.values())
        reply = QMessageBox.question(
            self, "确认", f"撤销全部 {total} 个删除标记？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        for fp, ds in self.deleted_sets.items():
            for idx in list(ds):
                ds.discard(idx)
                self._update_tree_item(fp, idx)
        self.update_status()
        if self.selected_piece:
            self._preview_piece(*self.selected_piece)

    # -------------------- 勾选操作 --------------------
    def check_all(self):
        for fp, fd in self.file_data.items():
            cs = self.checked_sets.setdefault(fp, set())
            for i in range(len(fd['pieces'])):
                cs.add(i)
                self._update_tree_item(fp, i)
        self.update_status()

    def uncheck_all(self):
        for fp, cs in list(self.checked_sets.items()):
            for idx in list(cs):
                cs.discard(idx)
                self._update_tree_item(fp, idx)
        self.update_status()

    # -------------------- 保存操作 --------------------
    def _get_default_filename(self, filepath=None):
        if filepath is None:
            filepath = self.active_file
        if not filepath or filepath not in self.file_data:
            return "output.lig"
        fd = self.file_data[filepath]
        first_time = fd['pieces'][0][0] if fd['pieces'] else ""
        station_name = "UNKNOWN"
        if fd['pieces']:
            _, piece_data = fd['pieces'][0]
            lat = piece_data.get('m_GPSCurrentLocationLat', 0)
            lon = piece_data.get('m_GPSCurrentLocationLon', 0)
            if lat and lon and self.station_coords:
                station_name = match_station_name(lat, lon, self.station_coords)
        return f"{station_name}_{first_time}.lig"

    def save_file(self):
        if not self.active_file or self.active_file not in self.file_data:
            QMessageBox.information(self, "提示", "请先打开文件")
            return
        ds = self.deleted_sets.get(self.active_file, set())
        if ds:
            self._save_with_deletions(self.active_file, self.active_file)
        else:
            QMessageBox.information(self, "提示", "没有修改，无需保存")

    def save_as(self):
        if not self.active_file or self.active_file not in self.file_data:
            QMessageBox.information(self, "提示", "请先打开文件")
            return
        default_name = self._get_default_filename()
        filepath, _ = QFileDialog.getSaveFileName(
            self, "另存为", default_name, "LIG文件 (*.lig)"
        )
        if not filepath:
            return
        ds = self.deleted_sets.get(self.active_file, set())
        if ds:
            self._save_with_deletions(self.active_file, filepath)
        else:
            self._save_without_deletions(self.active_file, filepath)

    def _save_without_deletions(self, src_filepath, output_path):
        try:
            shutil.copy2(src_filepath, output_path)
            if output_path == src_filepath:
                self._reload_file(output_path)
            else:
                self._remove_file(src_filepath)
                self._do_load_file(output_path)
            QMessageBox.information(self, "保存成功", f"文件已保存到:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存文件时出错:\n{e}")

    def _save_with_deletions(self, src_filepath, output_path):
        fd = self.file_data[src_filepath]
        ds = sorted(self.deleted_sets.get(src_filepath, set()))
        try:
            SaveLigFile(output_path, fd['raw_data'], fd['header_size'],
                        fd['piece_offsets'], ds)
            deleted_count = len(ds)
            if output_path == src_filepath:
                self._reload_file(output_path)
            else:
                self._remove_file(src_filepath)
                self._do_load_file(output_path)
            QMessageBox.information(self, "保存成功",
                                    f"已删除 {deleted_count} 个片段\n"
                                    f"保存到: {output_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存文件时出错:\n{e}")

    # -------------------- 导出操作 --------------------
    def _export_for_file(self, filepath, keep_indices):
        fd = self.file_data[filepath]
        all_indices = set(range(len(fd['piece_offsets'])))
        deleted = sorted(all_indices - keep_indices)
        default_name = self._get_default_filename(filepath)
        out, _ = QFileDialog.getSaveFileName(
            self, f"导出 - {os.path.basename(filepath)}",
            default_name, "LIG文件 (*.lig)"
        )
        if not out:
            return False
        SaveLigFile(out, fd['raw_data'], fd['header_size'],
                    fd['piece_offsets'], deleted)
        return True

    def export_checked(self):
        has_any = any(cs for cs in self.checked_sets.values())
        if not has_any:
            QMessageBox.information(self, "提示", "没有勾选的片段，请双击行进行勾选")
            return
        for fp, cs in self.checked_sets.items():
            if cs:
                if self._export_for_file(fp, cs):
                    QMessageBox.information(self, "导出成功",
                                            f"{os.path.basename(fp)}: 已导出 {len(cs)} 个勾选片段")

    def export_not_deleted(self):
        for fp, fd in self.file_data.items():
            ds = self.deleted_sets.get(fp, set())
            if ds:
                keep = set(i for i in range(len(fd['pieces'])) if i not in ds)
                if self._export_for_file(fp, keep):
                    QMessageBox.information(self, "导出成功",
                                            f"{os.path.basename(fp)}: 已导出 {len(keep)} 个未删除片段")

    def export_by_daynight(self):
        folder = QFileDialog.getExistingDirectory(self, "选择昼夜分类导出目录")
        if not folder:
            return
        for fp, fd in self.file_data.items():
            base_name = self._get_default_filename(fp).replace(".lig", "")
            ds = self.deleted_sets.get(fp, set())
            day_idx = []
            night_idx = []
            for i, (tk_, _) in enumerate(fd['pieces']):
                if i in ds:
                    continue
                dn = time_classifier_display(tk_)
                if dn == "白天":
                    day_idx.append(i)
                else:
                    night_idx.append(i)
            try:
                if day_idx:
                    dp = os.path.join(folder, f"{base_name}_day.lig")
                    dd = [x for x in range(len(fd['piece_offsets'])) if x not in set(day_idx)]
                    SaveLigFile(dp, fd['raw_data'], fd['header_size'],
                                fd['piece_offsets'], dd)
                if night_idx:
                    np_ = os.path.join(folder, f"{base_name}_night.lig")
                    nd = [x for x in range(len(fd['piece_offsets'])) if x not in set(night_idx)]
                    SaveLigFile(np_, fd['raw_data'], fd['header_size'],
                                fd['piece_offsets'], nd)
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出时出错:\n{e}")

        QMessageBox.information(self, "导出成功", f"导出目录: {folder}")

    def export_timestamps(self):
        if not self.file_data:
            QMessageBox.information(self, "提示", "请先打开文件")
            return
        default_name = self._get_default_filename().replace(".lig", ".txt")
        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出时间戳", default_name, "文本文件 (*.txt)"
        )
        if not filepath:
            return
        with open(filepath, 'w', encoding='utf-8') as f:
            for fp, fd in self.file_data.items():
                ds = self.deleted_sets.get(fp, set())
                for i, (tk_, _) in enumerate(fd['pieces']):
                    if i not in ds:
                        f.write(f"{tk_}\n")
        QMessageBox.information(self, "导出成功", f"已导出到:\n{filepath}")

    def export_checked_timestamps(self):
        has_any = any(cs for cs in self.checked_sets.values())
        if not has_any:
            QMessageBox.information(self, "提示", "没有勾选的片段")
            return
        default_name = self._get_default_filename().replace(".lig", ".txt")
        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出勾选时间戳", default_name, "文本文件 (*.txt)"
        )
        if not filepath:
            return
        with open(filepath, 'w', encoding='utf-8') as f:
            for fp, cs in self.checked_sets.items():
                fd = self.file_data[fp]
                for i in sorted(cs):
                    f.write(f"{fd['pieces'][i][0]}\n")
        QMessageBox.information(self, "导出成功", f"已导出到:\n{filepath}")

    # -------------------- 筛选 --------------------
    def filter_dialog(self):
        text, ok = QInputDialog.getText(
            self, "筛选", "输入时间戳关键字:"
        )
        if ok and text.strip():
            self._apply_filter(text.strip())

    def _apply_filter(self, filter_text):
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            file_item = root.child(i)
            filepath = file_item.data(0, Qt.UserRole)
            if filepath not in self.file_data:
                continue
            fd = self.file_data[filepath]
            for j in range(file_item.childCount()):
                child = file_item.child(j)
                idx = child.data(0, Qt.UserRole)
                time_key = fd['pieces'][idx][0]
                if filter_text and filter_text not in time_key:
                    child.setHidden(True)
                else:
                    child.setHidden(False)

    def clear_filter(self):
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            file_item = root.child(i)
            for j in range(file_item.childCount()):
                file_item.child(j).setHidden(False)

    # -------------------- 帮助 --------------------
    def show_about(self):
        QMessageBox.about(
            self, "关于 LigEdit",
            "LigEdit - 雷电波形编辑器\n\n"
            "功能：\n"
            "  - 同时加载多个lig文件\n"
            "  - 树形展示文件→时间戳\n"
            "  - 单击预览波形（示波器风格）\n"
            "  - 双击勾选/取消勾选\n"
            "  - 右键快捷操作（预览/勾选/删除）\n"
            "  - Delete 删除选中, Ctrl+Z 撤销\n"
            "  - 滚轮缩放幅值, 拖拽平移\n"
            "  - 多种导出方式\n\n"
            "快捷键：\n"
            "  Ctrl+O     打开文件\n"
            "  Ctrl+S     保存\n"
            "  Delete     删除选中\n"
            "  Ctrl+Z     撤销删除\n"
            "  Ctrl+E     导出勾选\n"
            "  Ctrl+Shift+A 全勾选\n"
            "  双击       勾选/取消勾选\n"
            "  右键       快捷菜单\n"
            "  滚轮       幅值缩放\n"
            "  左/右键    水平平移\n"
            "  Home       重置视图\n\n"
            "  All rights reserved by Shensi Wang\n"
            "  (shensiwang74@gmail.com, 18356054196)"
        )

    # -------------------- 数据处理 --------------------
    def open_distance_classify(self):
        from pipeline_dialog import DistanceClassifyDialog
        dlg = DistanceClassifyDialog(self)
        dlg.exec_()

    def open_daynight_classify(self):
        from pipeline_dialog import DayNightClassifyDialog
        dlg = DayNightClassifyDialog(self)
        dlg.exec_()

    # -------------------- 自动加载命令行文件 --------------------
    def auto_load_files(self):
        args = sys.argv[1:]
        for f in args:
            if f.lower().endswith('.lig') and os.path.isfile(f):
                self._load_file(f)
