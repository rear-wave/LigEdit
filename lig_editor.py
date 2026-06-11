#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LigEdit - 雷电波形编辑器 - 后端模块
功能：lig文件解析、滤波、站点匹配、时间工具
GUI层见 main_window.py / waveform_widget.py
"""

import os
import sys
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import struct

import numpy as np
from scipy.signal import butter, filtfilt


# ============================================================================
#                          站点经纬度匹配
# ============================================================================

def _resource_path(relative_path):
    """获取资源文件的绝对路径，兼容PyInstaller打包环境"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def load_station_coords(filepath=None):
    if filepath is None:
        filepath = _resource_path('站点经纬度.txt')
    stations = {}
    if not os.path.exists(filepath):
        return stations
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i + 1 < len(lines):
        name = lines[i]
        parts = lines[i + 1].split()
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                stations[name] = (lat, lon)
            except ValueError:
                pass
        i += 2
    return stations


def match_station_name(lat, lon, stations, tolerance=0.02):
    best_name = "UNKNOWN"
    best_dist = tolerance
    for name, (s_lat, s_lon) in stations.items():
        d = max(abs(lat - s_lat), abs(lon - s_lon))
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


# ============================================================================
#                          lig文件解析模块
# ============================================================================

def ReadGPSTimeFromLig(fp):
    GpsTime = {}
    GpsTime['isTimeValid'] = struct.unpack('b', fp.read(1))[0]
    GpsTime['isTimeConfirm'] = struct.unpack('b', fp.read(1))[0]
    fp.read(2)
    GpsTime['Year'] = struct.unpack('i', fp.read(4))[0]
    GpsTime['Month'] = struct.unpack('i', fp.read(4))[0]
    GpsTime['Day'] = struct.unpack('i', fp.read(4))[0]
    GpsTime['Hour'] = struct.unpack('i', fp.read(4))[0]
    GpsTime['Min'] = struct.unpack('i', fp.read(4))[0]
    GpsTime['Sec'] = struct.unpack('i', fp.read(4))[0]
    fp.read(4)
    GpsTime['ActPointSec'] = struct.unpack('d', fp.read(8))[0]
    Time = ('%02d%02d%02d%02d%02d%010.7f' % (
        GpsTime['Year'], GpsTime['Month'], GpsTime['Day'],
        GpsTime['Hour'], GpsTime['Min'],
        GpsTime['Sec'] + GpsTime['ActPointSec']))
    return str(Time)


def ReadPerLigPieceFromLig(fp, version):
    S = {}
    S['version'] = struct.unpack('i', fp.read(4))[0]
    S['m_numOfDataOfPerCache'] = struct.unpack('L', fp.read(4))[0]
    S['m_samplingRate'] = struct.unpack('d', fp.read(8))[0]
    S['m_preTriggerNum'] = struct.unpack('i', fp.read(4))[0]
    S['m_numOfData'] = struct.unpack('i', fp.read(4))[0]
    S['m_numOfChannel'] = struct.unpack('i', fp.read(4))[0]
    S['m_samplingEventID'] = struct.unpack('i', fp.read(4))[0]
    S['m_stationName'] = struct.unpack('32c', fp.read(32))[0]
    S['m_stationID'] = struct.unpack('i', fp.read(4))[0]
    S['m_isCompress'] = struct.unpack('b', fp.read(1))[0]
    fp.read(3)
    S['m_dataSize'] = struct.unpack('i', fp.read(4))[0]
    S['m_cacheCount'] = struct.unpack('i', fp.read(4))[0]
    S['m_cachePlace'] = struct.unpack('i', fp.read(4))[0]
    S['m_meanLevel'] = struct.unpack('i', fp.read(4))[0]
    S['m_trigLevelUp'] = struct.unpack('i', fp.read(4))[0]
    S['m_trigLevelDown'] = struct.unpack('i', fp.read(4))[0]
    S['m_matchFileCacheCount'] = struct.unpack('i', fp.read(4))[0]
    fp.read(4)
    S['m_FirstPointTime'] = ReadGPSTimeFromLig(fp)
    S['m_CurrentGPSTime'] = ReadGPSTimeFromLig(fp)
    S['m_GPSCurrentLocationLat'] = struct.unpack('d', fp.read(8))[0]
    S['m_GPSCurrentLocationLon'] = struct.unpack('d', fp.read(8))[0]
    S['m_state'] = struct.unpack('i', fp.read(4))[0]

    if version == 1001:
        S['offset'] = 2048
        S['m_Digital_Range'] = 2048
        S['m_Range'] = 5
    elif version == 2001:
        S['offset'] = struct.unpack('i', fp.read(4))[0]
        S['m_LightningLocationLat'] = struct.unpack('d', fp.read(8))[0]
        S['m_LightningLocationLon'] = struct.unpack('d', fp.read(8))[0]
        S['m_Range'] = struct.unpack('d', fp.read(8))[0]
        S['m_Digital_Range'] = struct.unpack('i', fp.read(4))[0]
        S['Reserved'] = struct.unpack('224b', fp.read(224))
        S['offset'] = 0
        S['m_Range'] = 10
        S['m_Digital_Range'] = 8192
    elif version == 3001:
        S['offset'] = struct.unpack('i', fp.read(4))[0]
        S['m_LightningLocationLat'] = struct.unpack('d', fp.read(8))[0]
        S['m_LightningLocationLon'] = struct.unpack('d', fp.read(8))[0]
        S['m_Range'] = struct.unpack('d', fp.read(8))[0]
        S['m_Digital_Range'] = struct.unpack('i', fp.read(4))[0]
        S['Reserved'] = struct.unpack('224b', fp.read(224))

    fp.read(4)
    for i in range(S['m_numOfChannel']):
        cnt = S['m_numOfData']
        S[str(i)] = struct.unpack(f'{cnt}H', fp.read(cnt * 2))
    return S


def ReadLigFileWithOffsets(FileName):
    with open(FileName, 'rb') as fp:
        raw_data = fp.read()

    with open(FileName, 'rb') as fp:
        header = {}
        header['version'] = struct.unpack('i', fp.read(4))[0]
        header['NumOfPiece'] = struct.unpack('i', fp.read(4))[0]
        header['firstPieceCacheCount'] = struct.unpack('L', fp.read(4))[0]
        header['lastPieceCacheCount'] = struct.unpack('L', fp.read(4))[0]
        header['FirstPieceCachePlace'] = struct.unpack('L', fp.read(4))[0]
        header['LastPieceCachePlace'] = struct.unpack('L', fp.read(4))[0]
        header['SamplingEventID'] = struct.unpack('i', fp.read(4))[0]
        header['StationID'] = struct.unpack('i', fp.read(4))[0]
        header['m_firstPieceTime'] = ReadGPSTimeFromLig(fp)
        header['m_lastPieceTime'] = ReadGPSTimeFromLig(fp)

        header_size = fp.tell()

        pieces = []
        piece_offsets = []
        for i in range(header['NumOfPiece']):
            offset_start = fp.tell()
            try:
                piece = ReadPerLigPieceFromLig(fp, header['version'])
                offset_end = fp.tell()
                time_key = str(piece['m_FirstPointTime'])
                pieces.append((time_key, piece))
                piece_offsets.append((offset_start, offset_end))
            except Exception:
                pass

    return header, pieces, raw_data, piece_offsets, header_size


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
    all_pieces = []  # [(time_key, raw_bytes)]
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

    # 按时间排序
    all_pieces.sort(key=lambda x: x[0])

    # 构建新文件: 使用第一个文件的文件头作为模板，更新片段数
    new_data = bytearray(first_raw_data[:header_size])
    struct.pack_into('i', new_data, 4, len(all_pieces))

    for _, piece_bytes in all_pieces:
        new_data.extend(piece_bytes)

    with open(output_path, 'wb') as fp:
        fp.write(new_data)

    return len(all_pieces)


def ButterFilter(piece):
    fc = 300000
    fs = 5000000
    order = 3
    fc_normalized = fc / (fs / 2)
    b, a = butter(order, fc_normalized, btype='low')
    return filtfilt(b, a, piece)


def format_time_display(time_str):
    try:
        if len(time_str) < 13:
            return time_str
        yy = int(time_str[0:2])
        year = 2000 + yy if yy < 50 else 1900 + yy
        month = time_str[2:4]
        day = time_str[4:6]
        hour = time_str[6:8]
        minute = time_str[8:10]
        sec_part = time_str[10:]
        return f"{year}-{month}-{day} {hour}:{minute}:{sec_part}"
    except Exception:
        return time_str


def time_classifier_display(time_str):
    try:
        utc_hour = int(time_str[6:8])
        utc_minute = int(time_str[8:10])
        utc_second = float(time_str[10:])
        utc_total_hour = utc_hour + utc_minute / 60 + utc_second / 3600
        beijing_total_hour = utc_total_hour + 8
        if beijing_total_hour >= 24:
            beijing_total_hour -= 24
        if 5.5 <= beijing_total_hour < 19:
            return "白天"
        else:
            return "夜晚"
    except Exception:
        return "未知"


# ============================================================================
#                          流水线公共函数
# ============================================================================

def ReadLigFile(FileName):
    """读取lig文件全量数据（用于流水线），返回 {time_str: piece_data}"""
    with open(FileName, 'rb') as fp:
        Data = {}
        version = struct.unpack('i', fp.read(4))[0]
        NumOfPiece = struct.unpack('i', fp.read(4))[0]
        fp.read(4)  # firstPieceCacheCount
        fp.read(4)  # lastPieceCacheCount
        fp.read(4)  # FirstPieceCachePlace
        fp.read(4)  # LastPieceCachePlace
        fp.read(4)  # SamplingEventID
        fp.read(4)  # StationID
        ReadGPSTimeFromLig(fp)  # firstPieceTime
        ReadGPSTimeFromLig(fp)  # lastPieceTime
        for i in range(NumOfPiece):
            try:
                piece = ReadPerLigPieceFromLig(fp, version)
                time_key = str(piece['m_FirstPointTime'])
                Data[time_key] = piece
            except Exception:
                pass
    return Data


def CutPieceTo16000(piece):
    """截取波形峰值前后共16000点"""
    index_max = np.where(piece == piece.max())[0][0]
    if index_max - 4000 < 0:
        begin = 0
        end = min(begin + 16000, len(piece))
    elif index_max + 12000 > len(piece):
        end = len(piece)
        begin = max(end - 16000, 0)
    else:
        begin = index_max - 4000
        end = begin + 16000
    return piece[begin:end]


def compute_final_time(lig_time, piece):
    """从lig时间和波形数据计算精确时间"""
    from decimal import Decimal
    y = CutPieceTo16000(piece - np.mean(piece))
    y = ButterFilter(y)
    y_abs = np.abs(y)
    peak_index = np.argmax(y_abs)
    time_right = Decimal(lig_time)
    time_int = Decimal(10000)
    Time1 = time_right % time_int
    trigger_time = peak_index * 0.0002
    real_time1 = Time1 + Decimal(str(trigger_time)) * Decimal('0.001')
    real_time = f"{real_time1:.7f}"
    final_time = f"{lig_time.split('.')[0]}.{real_time.split('.')[1]}"
    return final_time


def repacklig(pulse, time_str, lig_head_path):
    """重新打包lig片段（用于流水线步骤4/5输出）"""
    try:
        with open(lig_head_path, 'rb') as f:
            lig_head = f.read()
        YMDHMS = [int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6]),
                   int(time_str[6:8]), int(time_str[8:10]), int(float(time_str[10:12]))]
        Sec = float(time_str[12:])
        PieceYMDHMS = struct.pack('6i4x', *YMDHMS)
        PieceSec = struct.pack('d', Sec)
        piece = pulse.tolist()
        pieceFile = lig_head + struct.pack('16000H', *piece)
        pieceFile = pieceFile[:108] + PieceYMDHMS + PieceSec + pieceFile[108 + 36:]
        return pieceFile
    except Exception:
        return None


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
