#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WaveformWidget - 示波器风格双面板波形显示控件
上图: 细节视图(滚轮Y幅值缩放/拖拽X平移)
下图: 全局视图(完整波形+可见区域框+滚轮X拉伸压缩)
性能优化: 降采样 + 节流联动 + 关闭抗锯齿
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QPen, QColor, QBrush, QKeySequence, QFont
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QShortcut, QLabel)


# ============================================================================
#                          示波器风格主题配置
# ============================================================================

SCOPE_STYLE = {
    'bg_color': '#0a0a14',
    'grid_color': '#1a1a2e',
    'raw_color': '#ff1493',          # 原始波形: 荧光粉
    'filtered_color': '#ffffff',     # 滤波波形: 白色
    'wave_color_deleted': '#ff4444',
    'wave_color_checked': '#00d4ff',
    'text_color': '#cccccc',
    'axis_color': '#333355',
    'region_color': QColor(100, 150, 255, 60),
    'region_border_color': QColor(80, 120, 220, 180),
}


def setup_scope_theme():
    pg.setConfigOptions(
        antialias=False,          # 关闭抗锯齿，大幅提升渲染性能
        background=SCOPE_STYLE['bg_color'],
        foreground=SCOPE_STYLE['text_color'],
    )


# ============================================================================
#                          自定义PlotWidget: 拦截滚轮事件
# ============================================================================

class DetailPlotWidget(pg.PlotWidget):
    """细节视图: 滚轮控制Y幅值缩放"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._on_zoom_callback = None

    def set_zoom_callback(self, cb):
        self._on_zoom_callback = cb

    def wheelEvent(self, event):
        if self._on_zoom_callback:
            delta = event.angleDelta().y()
            if delta > 0:
                self._on_zoom_callback('in')
            elif delta < 0:
                self._on_zoom_callback('out')
            event.accept()
            return
        super().wheelEvent(event)


class OverviewPlotWidget(pg.PlotWidget):
    """全局视图: 滚轮控制上图X拉伸压缩"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._on_stretch_callback = None

    def set_stretch_callback(self, cb):
        self._on_stretch_callback = cb

    def wheelEvent(self, event):
        if self._on_stretch_callback:
            delta = event.angleDelta().y()
            if abs(delta) >= 30:
                direction = 'in' if delta > 0 else 'out'
                self._on_stretch_callback(direction)
            event.accept()
            return
        super().wheelEvent(event)


# ============================================================================
#                          双面板波形显示控件
# ============================================================================

class WaveformWidget(QWidget):
    """示波器风格双面板波形控件

    上图(细节视图):
      - 显示当前选中区域的放大细节
      - 鼠标滚轮: Y轴幅值缩放（对数级进）
      - 左键拖拽: X轴水平平移
      - 右键拖拽: X轴水平缩放

    下图(全局视图):
      - 显示完整波形
      - 半透明高亮框标记上图对应的区域
      - 可拖拽高亮框改变上图查看位置
      - 鼠标滚轮: 控制上图X轴水平拉伸/压缩
    """

    Y_ZOOM_LEVELS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._time_array = None
        self._y_range = (0, 1)
        self._x_full_range = (0, 1)
        self._zoom_index = 4  # 默认 ×1.0
        self._is_daytime = False

        # 联动节流: 避免拖拽时频繁更新region导致卡顿
        self._region_update_pending = False
        self._region_update_timer = QTimer(self)
        self._region_update_timer.setSingleShot(True)
        self._region_update_timer.setInterval(30)  # ~33fps
        self._region_update_timer.timeout.connect(self._do_update_region)

        self._linkage_enabled = True

        setup_scope_theme()
        self._build_ui()

        # 光标竖线 + 时间戳标签（必须在 _build_ui 之后，因为 detail_plot 在那里创建）
        self._cursor_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(color='#cccccc', width=1.5, style=Qt.SolidLine),
            movable=False,
        )
        self.detail_plot.addItem(self._cursor_line)
        self._cursor_line.setVisible(False)

        # 用 QLabel 作为时间戳覆盖层（像素坐标，不受 ViewBox 裁剪）
        self._cursor_label = QLabel('')
        self._cursor_label.setFont(QFont('Consolas', 10))
        self._cursor_label.setStyleSheet('color: #cccccc; background: transparent; padding: 2px 4px;')
        self._cursor_label.setParent(self.detail_plot)
        self._cursor_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._cursor_label.hide()

        # 鼠标跟踪
        self.detail_plot.setMouseTracking(True)
        self.detail_plot.viewport().setMouseTracking(True)
        self.detail_plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.detail_plot.viewport().installEventFilter(self)

        self._setup_linkage()
        self._setup_shortcuts()

    # -------------------- UI构建 --------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        # ---- 上图: 细节视图 ----
        self.detail_plot = DetailPlotWidget()
        self.detail_plot.showGrid(x=True, y=True, alpha=0.25)
        self.detail_plot.setLabel('left', '', units='V')
        self.detail_plot.hideAxis('bottom')
        dp = self.detail_plot.getPlotItem()
        dp.getViewBox().setMouseEnabled(x=True, y=False)
        dp.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        for axis in ['left']:
            ax = dp.getAxis(axis)
            ax.setPen(QPen(QColor(SCOPE_STYLE['axis_color']), 1))
            ax.setStyle(tickFont=QFont('Consolas', 9))

        # 原始波形(粉色, 先绘制, 在底层)
        self.detail_raw_curve = self.detail_plot.plot(
            pen=pg.mkPen(color=SCOPE_STYLE['raw_color'], width=1.0),
            autoDownsample=True,
            clipToView=True,
        )
        # 滤波波形(白色, 后绘制, 在顶层)
        self.detail_curve = self.detail_plot.plot(
            pen=pg.mkPen(color=SCOPE_STYLE['filtered_color'], width=1.2),
            autoDownsample=True,
            clipToView=True,
        )
        layout.addWidget(self.detail_plot, stretch=3)

        # ---- 下图: 全局视图 ----
        self.overview_plot = OverviewPlotWidget()
        self.overview_plot.showGrid(x=True, y=True, alpha=0.15)
        self.overview_plot.setMaximumHeight(140)
        self.overview_plot.setMinimumHeight(80)
        op = self.overview_plot.getPlotItem()
        op.getViewBox().setMouseEnabled(x=False, y=False)
        for axis in ['left', 'bottom']:
            ax = op.getAxis(axis)
            ax.setPen(QPen(QColor(SCOPE_STYLE['axis_color']), 1))
            ax.setStyle(tickFont=QFont('Consolas', 8))
        self.overview_plot.setLabel('left', '')

        # 原始波形(粉色)
        self.overview_raw_curve = self.overview_plot.plot(
            pen=pg.mkPen(color=SCOPE_STYLE['raw_color'], width=0.6),
            autoDownsample=True,
        )
        # 滤波波形(白色)
        self.overview_curve = self.overview_plot.plot(
            pen=pg.mkPen(color=SCOPE_STYLE['filtered_color'], width=0.8),
            autoDownsample=True,
        )
        layout.addWidget(self.overview_plot, stretch=1)

        # 可见区域高亮框
        self.region_item = pg.LinearRegionItem(
            movable=True,
            brush=QBrush(SCOPE_STYLE['region_color']),
            pen=pg.mkPen(color=SCOPE_STYLE['region_border_color'], width=1, style=Qt.DashLine),
        )
        self.overview_plot.addItem(self.region_item)

        # 绑定回调
        self.detail_plot.set_zoom_callback(self._handle_detail_wheel)
        self.overview_plot.set_stretch_callback(self._handle_overview_wheel)

    # -------------------- 联动机制(节流) --------------------

    def _setup_linkage(self):
        self.region_item.sigRegionChanged.connect(self._on_region_changed)
        self.detail_plot.getPlotItem().getViewBox().sigXRangeChanged.connect(
            self._on_detail_xrange_changed
        )
        self.detail_plot.getPlotItem().getViewBox().sigRangeChangedManually.connect(
            self._on_detail_range_manual
        )

    def _on_region_changed(self):
        """下图区域框拖拽 → 更新上图(直接，因为用户主动拖框)"""
        if not self._linkage_enabled:
            return
        rgn_min, rgn_max = self.region_item.getRegion()
        if self._data is not None and rgn_max > rgn_min:
            vb = self.detail_plot.getPlotItem().getViewBox()
            current_yr = vb.viewRange()[1]
            self._linkage_enabled = False
            vb.setXRange(rgn_min, rgn_max, padding=0)
            vb.setYRange(current_yr[0], current_yr[1], padding=0)
            self._linkage_enabled = True

    def _on_detail_xrange_changed(self, vb, range):
        """上图X范围变化 → 节流更新下图区域框"""
        if not self._linkage_enabled or self._data is None:
            return
        # 节流: 不立即更新，等30ms无新事件再更新
        if not self._region_update_pending:
            self._region_update_pending = True
            self._region_update_timer.start()

    def _on_detail_range_manual(self, vb):
        """上图手动拖拽结束 → 立即同步区域框"""
        self._do_update_region()

    def _do_update_region(self):
        """实际更新下图区域框"""
        self._region_update_pending = False
        if self._data is None:
            return
        vb = self.detail_plot.getPlotItem().getViewBox()
        xr = vb.viewRange()[0]
        try:
            self.region_item.blockSignals(True)
            self.region_item.setRegion((xr[0], xr[1]))
            self.region_item.blockSignals(False)
        except Exception:
            pass

    # -------------------- 快捷键 --------------------

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Home), self, self.reset_view)
        QShortcut(QKeySequence(Qt.Key_Left), self, self._pan_left)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._pan_right)

    # -------------------- 光标竖线 + 时间戳 --------------------

    def eventFilter(self, obj, event):
        if obj == self.detail_plot.viewport():
            if event.type() == event.Leave:
                self._cursor_line.setVisible(False)
                self._cursor_label.hide()
        return super().eventFilter(obj, event)

    def _on_mouse_moved(self, pos):
        if self._data is None:
            return
        vb = self.detail_plot.getPlotItem().getViewBox()
        mouse_point = vb.mapSceneToView(pos)
        x_val = mouse_point.x()

        # 检查是否在数据范围内
        x_min, x_max = self._x_full_range
        if x_val < x_min or x_val > x_max:
            self._cursor_line.setVisible(False)
            self._cursor_label.hide()
            return

        self._cursor_line.setPos(x_val)
        self._cursor_line.setVisible(True)

        # 计算时间戳文本
        time_str = self._format_cursor_time(x_val)
        self._cursor_label.setText(time_str)

        # 场景坐标 → 像素坐标，标签跟随鼠标位置（右上方偏移）
        vp_pos = self.detail_plot.mapFromScene(pos)
        self._cursor_label.move(vp_pos + QPoint(8, -8))
        self._cursor_label.show()

    def _format_cursor_time(self, x_ms):
        """将毫秒位置格式化为两行: GPS时间(=原始+偏移) + 相对时间"""
        if self._time_array is None or len(self._time_array) == 0:
            return f"{x_ms:.3f} ms"

        dt = x_ms - self._time_array[0] if len(self._time_array) > 0 else 0
        total_us = dt * 1000

        if total_us < 1000:
            rel_str = f"{total_us:.1f} μs"
        elif total_us < 1000000:
            rel_str = f"{dt:.3f} ms"
        else:
            rel_str = f"{dt / 1000:.6f} s"

        # 第一行 = 原始GPS时间 + 偏移量（随鼠标变化）
        gps_str = ""
        if self._gps_time_key:
            gps_str = self._add_offset_to_gps(self._gps_time_key, dt) + "\n"

        return f"{gps_str}{rel_str}"

    @staticmethod
    def _add_offset_to_gps(time_str, offset_ms):
        """在格式化的GPS时间上加上毫秒偏移，返回新字符串"""
        try:
            from datetime import datetime, timedelta
            s = time_str.strip()
            # format_time_display 可能输出7位小数(如 .6170650)，%f只接受6位
            if '.' in s:
                base, frac = s.rsplit('.', 1)
                frac = (frac + '000000')[:6]  # 补齐或截断到6位微秒
                s = base + '.' + frac
            t = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
            t2 = t + timedelta(milliseconds=offset_ms)
            return t2.strftime("%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            return time_str

    # -------------------- 公开接口 --------------------

    def set_waveform(self, data, raw_data=None, time_array=None, is_daytime=False,
                     is_deleted=False, is_checked=False, gps_time_key=None):
        if data is None or len(data) == 0:
            self.clear_waveform()
            return

        self._data = np.asarray(data, dtype=np.float64)
        self._is_daytime = is_daytime
        self._gps_time_key = gps_time_key

        if time_array is not None:
            self._time_array = np.asarray(time_array, dtype=np.float64)
        else:
            self._time_array = np.arange(len(data)) / 5000000 * 1000

        # 颜色: 根据状态变色
        if is_deleted:
            raw_color = SCOPE_STYLE['wave_color_deleted']
            filt_color = SCOPE_STYLE['wave_color_deleted']
        elif is_checked:
            raw_color = SCOPE_STYLE['raw_color']
            filt_color = SCOPE_STYLE['wave_color_checked']
        else:
            raw_color = SCOPE_STYLE['raw_color']
            filt_color = SCOPE_STYLE['filtered_color']

        # 原始波形
        if raw_data is not None and len(raw_data) > 0:
            raw_arr = np.asarray(raw_data, dtype=np.float64)
            self.detail_raw_curve.setPen(pg.mkPen(color=raw_color, width=1.0))
            self.overview_raw_curve.setPen(pg.mkPen(color=raw_color, width=0.6))
            self.detail_raw_curve.setData(self._time_array[:len(raw_arr)], raw_arr)
            self.overview_raw_curve.setData(self._time_array[:len(raw_arr)], raw_arr)
        else:
            self.detail_raw_curve.setData([], [])
            self.overview_raw_curve.setData([], [])

        # 滤波波形
        self.detail_curve.setPen(pg.mkPen(color=filt_color, width=1.2))
        self.overview_curve.setPen(pg.mkPen(color=filt_color, width=0.8))
        self.detail_curve.setData(self._time_array, self._data)
        self.overview_curve.setData(self._time_array, self._data)

        self._x_full_range = (self._time_array[0], self._time_array[-1])
        if len(self._data) > 0:
            y_min, y_max = np.min(self._data), np.max(self._data)
            margin = max(abs(y_max - y_min) * 0.15, 0.5)
            self._y_range = (y_min - margin, y_max + margin)

        self.reset_view()

    def clear_waveform(self):
        self._data = None
        self._time_array = None
        self._gps_time_key = None
        self.detail_raw_curve.setData([], [])
        self.detail_curve.setData([], [])
        self.overview_raw_curve.setData([], [])
        self.overview_curve.setData([], [])
        self._cursor_line.setVisible(False)
        self._cursor_label.hide()

    def reset_view(self):
        if self._data is None:
            return

        x_min, x_max = self._x_full_range
        self.detail_plot.setXRange(x_min, x_max, padding=0)

        # Y轴0点固定，围绕0对称缩放
        zoom = self.Y_ZOOM_LEVELS[self._zoom_index]
        yh = self._y_range[1] / zoom
        self.detail_plot.setYRange(-yh, yh, padding=0)

        x_span = x_max - x_min
        self.overview_plot.setXRange(x_min - x_span * 0.01, x_max + x_span * 0.01, padding=0)
        ov_yh = self._y_range[1] * 3
        self.overview_plot.setYRange(-ov_yh, ov_yh, padding=0)

        self.region_item.blockSignals(True)
        self.region_item.setRegion((x_min, x_max))
        self.region_item.blockSignals(False)

    def set_daytime(self, is_daytime):
        self._is_daytime = is_daytime

    def get_zoom_level(self):
        return self.Y_ZOOM_LEVELS[self._zoom_index]

    # -------------------- 上图: Y幅值缩放 --------------------

    def _handle_detail_wheel(self, direction):
        if direction == 'in':
            if self._zoom_index < len(self.Y_ZOOM_LEVELS) - 1:
                self._zoom_index += 1
        else:
            if self._zoom_index > 0:
                self._zoom_index -= 1
        self._apply_y_zoom()

    def _apply_y_zoom(self):
        if self._data is None:
            return
        zoom = self.Y_ZOOM_LEVELS[self._zoom_index]
        yh = self._y_range[1] / zoom
        self.detail_plot.setYRange(-yh, yh, padding=0)

    # -------------------- 下图滚轮 → 上图X轴拉伸压缩 --------------------

    def _handle_overview_wheel(self, direction):
        """下图滚轮控制上图(细节视图)的X轴水平拉伸/压缩"""
        if self._data is None:
            return
        vb = self.detail_plot.getPlotItem().getViewBox()
        xr = vb.viewRange()[0]
        center = (xr[0] + xr[1]) / 2
        half_span = (xr[1] - xr[0]) / 2
        if direction == 'in':
            half_span /= 1.3
        else:
            half_span *= 1.3
        x_min, x_max = self._x_full_range
        new_left = max(center - half_span, x_min)
        new_right = min(center + half_span, x_max)
        if new_right - new_left < (x_max - x_min) * 0.001:
            return
        vb.setXRange(new_left, new_right, padding=0)

    # -------------------- 键盘平移 --------------------

    def _pan_left(self):
        if self._data is None:
            return
        vr = self.detail_plot.getPlotItem().getViewBox().viewRange()
        step = (vr[0][1] - vr[0][0]) * 0.1
        self.detail_plot.getPlotItem().getViewBox().translateBy(x=-step)

    def _pan_right(self):
        if self._data is None:
            return
        vr = self.detail_plot.getPlotItem().getViewBox().viewRange()
        step = (vr[0][1] - vr[0][0]) * 0.1
        self.detail_plot.getPlotItem().getViewBox().translateBy(x=step)
