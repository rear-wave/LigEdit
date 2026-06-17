#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LigEdit - 雷电波形编辑器 - 后端模块
解析/滤波/站点匹配/时间工具已移至 lig_parser.py
此处保留 SaveLigFile / MergeLigFiles 和 main() 入口
"""

import os
import sys
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import struct

from lig_parser import (
    ReadLigFileWithOffsets, ButterFilter, load_station_coords, match_station_name,
    format_time_display, time_classifier_display, ReadLigFile, CutPieceTo16000,
    compute_final_time, repacklig, _resource_path,
)


# ============================================================================
#                          文件写入/合并（仅在此保留）
# ============================================================================

def SaveLigFile(output_path, raw_data, header_size, piece_offsets, deleted_indices):
    total = len(piece_offsets)
    new_num = total - len(deleted_indices)
    new_data = bytearray(raw_data[:header_size])
    struct.pack_into('i', new_data, 4, new_num)
    deleted_set = set(deleted_indices)
    for i, (start, end) in enumerate(piece_offsets):
        if i not in deleted_set:
            new_data.extend(raw_data[start:end])
    with open(output_path, 'wb') as fp:
        fp.write(new_data)


def MergeLigFiles(filepaths, output_path):
    """合并多个lig文件为一个，按时间排序"""
    all_pieces = []
    first_raw_data = None
    header_size = 0

    for filepath in filepaths:
        header, pieces, raw_data, piece_offsets, hdr_size = ReadLigFileWithOffsets(filepath)
        if first_raw_data is None:
            first_raw_data = raw_data
            header_size = hdr_size
        for i, (time_key, piece_data) in enumerate(pieces):
            start, end = piece_offsets[i]
            all_pieces.append((time_key, raw_data[start:end]))

    all_pieces.sort(key=lambda x: x[0])

    new_data = bytearray(first_raw_data[:header_size])
    struct.pack_into('i', new_data, 4, len(all_pieces))

    for _, piece_bytes in all_pieces:
        new_data.extend(piece_bytes)

    with open(output_path, 'wb') as fp:
        fp.write(new_data)

    return len(all_pieces)


# ============================================================================
#                          入口 (PyQt5)
# ============================================================================

def main():
    from PyQt5.QtWidgets import QApplication
    from main_window import MainWindow

    app = QApplication(sys.argv)

    # 全局样式
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    # 自动加载命令行传入的lig文件
    window.auto_load_files()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
