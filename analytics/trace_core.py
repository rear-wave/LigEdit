#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analytics / trace_core — 多站闪电事件匹配核心逻辑

移植自 LigTrace 项目 (ligtrace.py + lig_utils.py)
功能：将多个站点的 LIG 波形与 WWLLN 定位数据进行交叉匹配，
      识别由 >= N 个站同时检测到的闪电事件。
"""

import os
import struct
import math
import time as time_module
import logging
from datetime import datetime
from decimal import Decimal
from bisect import bisect_left
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from lig_parser import (
    ReadLigFile, compute_final_time, repacklig, _resource_path,
)


# ============================================================================
#                          物理常数
# ============================================================================

SPEED_OF_LIGHT_KM_S = Decimal("299792.458")
EARTH_RADIUS_KM = 6371.0


# ============================================================================
#                          地理工具
# ============================================================================

def deg2rad(deg: float) -> float:
    return deg * (math.pi / 180.0)


def spherical_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """球面距离 (km)，使用球面余弦定理"""
    lat1_r, lon1_r = deg2rad(lat1), deg2rad(lon1)
    lat2_r, lon2_r = deg2rad(lat2), deg2rad(lon2)
    cos_angle = (math.sin(lat1_r) * math.sin(lat2_r)
                 + math.cos(lat1_r) * math.cos(lat2_r)
                 * math.cos(lon1_r - lon2_r))
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return EARTH_RADIUS_KM * math.acos(cos_angle)


# ============================================================================
#                          WWLLN 数据加载
# ============================================================================

def parse_wwlln_time_to_decimal(date_str: str, time_str: str) -> Optional[Decimal]:
    """解析 WWLLN 时间到 YYMMDDhhmmss.ffffff 格式 Decimal"""
    try:
        date_str = date_str.strip().rstrip(',')
        time_str = time_str.strip().rstrip(',')
        parts = date_str.split('/')
        if len(parts) != 3:
            return None
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        time_parts = time_str.split(':')
        if len(time_parts) < 3:
            return None
        hour, minute = int(time_parts[0]), int(time_parts[1])
        sec_part = time_parts[2]
        if '.' in sec_part:
            sec_str, usec_str = sec_part.split('.')
            second = int(sec_str)
            usec_str = usec_str[:6].ljust(6, '0')
            decimal_part = f".{usec_str}"
        else:
            second = int(sec_part)
            decimal_part = ".000000"
        year_short = year % 100
        return Decimal(f"{year_short:02d}{month:02d}{day:02d}"
                       f"{hour:02d}{minute:02d}{second:02d}{decimal_part}")
    except Exception:
        return None


def load_wwlln_data(wwlln_folder: str, target_day: str = None) -> pd.DataFrame:
    """读取所有 .loc 文件，返回 DataFrame (Time_fmt, Latitude, Longitude, Stations, Energy)

    target_day: 可选，格式 "YYMMDD"，仅加载指定日期的数据
    """
    wwlln_files = sorted([f for f in os.listdir(wwlln_folder) if f.endswith('.loc')])
    all_data = []
    col_names = ["Date", "Time", "Latitude", "Longitude", "Error",
                 "Stations", "Energy", "Energy2", "Stations2"]
    for wwlln_file in wwlln_files:
        wwlln_file_path = os.path.join(wwlln_folder, wwlln_file)
        try:
            df = pd.read_csv(wwlln_file_path, header=None, names=col_names,
                             sep=',', dtype=str, skipinitialspace=True)
            for col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.rstrip(',')
            df = df[df['Date'].notna() & df['Time'].notna()]
            for col in ["Latitude", "Longitude", "Stations", "Energy"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['Time_fmt'] = df.apply(
                lambda row: parse_wwlln_time_to_decimal(row['Date'], row['Time']), axis=1)
            df = df.dropna(subset=['Time_fmt'])
            if target_day:
                df['day_str'] = df['Time_fmt'].apply(
                    lambda x: str(x)[:6] if x is not None else None)
                df = df[df['day_str'] == target_day]
            all_data.append(df)
        except Exception:
            continue
    if not all_data:
        return pd.DataFrame(columns=["Time_fmt", "Latitude", "Longitude", "Stations", "Energy"])
    combined = pd.concat(all_data, ignore_index=True)
    combined.sort_values(by='Time_fmt', inplace=True, ignore_index=True)
    return combined


def load_wwlln_events(wwlln_dir: str, target_day: str = None) -> List[dict]:
    """加载所有 WWLLN 事件，返回按时间排序的 dict 列表

    target_day: 可选，格式 "YYMMDD"，仅加载指定日期的数据
    """
    df = load_wwlln_data(wwlln_dir, target_day=target_day)
    if df.empty:
        return []
    events = []
    for _, row in df.iterrows():
        events.append({
            'time': row['Time_fmt'],
            'lat': float(row['Latitude']),
            'lon': float(row['Longitude']),
            'stations': int(row['Stations']) if not np.isnan(row['Stations']) else 0,
            'energy': float(row['Energy']) if not np.isnan(row['Energy']) else 0.0,
        })
    return events


# ============================================================================
#                          站点的波形时间线加载
# ============================================================================

def load_station_timeline(station_dir: str, station_name: str,
                          logger: logging.Logger) -> Tuple[List[Decimal], List[tuple]]:
    """扫描 station_dir 中所有 .lig 文件，计算每段的 final_time

    返回:
      times:   排序后的 Decimal(final_time) 列表
      entries: [(filtered_f64, lig_time_str, raw_uint16), ...]
    """
    lig_files = []
    for root, dirs, files in os.walk(station_dir):
        for f in files:
            if f.lower().endswith('.lig'):
                lig_files.append(os.path.join(root, f))

    if not lig_files:
        logger.warning("  [%s] No .lig files found in %s", station_name, station_dir)
        return [], []

    raw_entries = []
    for fpath in lig_files:
        try:
            lig_data = ReadLigFile(fpath)
        except Exception:
            logger.warning("  [%s] Failed to read %s", station_name, fpath)
            continue

        for lig_time_str, piece_data in lig_data.items():
            try:
                raw_uint16 = np.array(piece_data['0'], dtype=np.uint16)
                piece_f64 = raw_uint16.astype(np.float64)
            except KeyError:
                continue
            # compute_final_time 内部已做 demean → CutPieceTo16000 → ButterFilter
            # 返回 (final_time_dec, filtered_piece, lig_time_str)
            final_time_dec, v_filtered, _ = compute_final_time(piece_f64, lig_time_str)
            raw_entries.append((final_time_dec, v_filtered, lig_time_str, raw_uint16))

    raw_entries.sort(key=lambda x: x[0])
    times = [e[0] for e in raw_entries]
    entries = [(e[1], e[2], e[3]) for e in raw_entries]

    logger.info("  [%s] %d pieces loaded, time range %s → %s",
                station_name, len(entries),
                str(times[0]) if times else 'N/A',
                str(times[-1]) if times else 'N/A')
    return times, entries


# ============================================================================
#                          匹配引擎
# ============================================================================

def compute_expected_arrival(wwlln_time: Decimal, distance_km: float) -> Decimal:
    """预期信号到达时间 = WWLLN 时间 + 距离/光速"""
    return wwlln_time + Decimal(str(distance_km)) / SPEED_OF_LIGHT_KM_S


def find_closest_piece(times: List[Decimal], used: set,
                       target: Decimal, window: Decimal
                       ) -> Optional[Tuple[int, Decimal]]:
    """二分搜索在 times 中最接近 target 且未使用且满足 ±window 的项"""
    idx = bisect_left(times, target)
    best_idx = None
    best_delta = None
    if idx < len(times):
        delta = abs(times[idx] - target)
        if delta <= window and idx not in used:
            best_idx = idx
            best_delta = delta
    if idx > 0:
        delta = abs(times[idx - 1] - target)
        if delta <= window and (idx - 1) not in used:
            if best_idx is None or delta < best_delta:
                best_idx = idx - 1
                best_delta = delta
    if best_idx is not None:
        return best_idx, best_delta
    return None


def match_events(wwlln_events: List[dict],
                 station_data: dict,
                 time_window: Decimal,
                 min_stations: int,
                 stop_flag,
                 logger: logging.Logger):
    """生成器，产出 (event_idx, event, station_matches)

    station_matches 按距离升序排列。
    """
    used_per_station = {name: set() for name in station_data}
    total = len(wwlln_events)
    report_interval = max(1, total // 20)

    for event_idx, event in enumerate(wwlln_events):
        if stop_flag and stop_flag.is_set():
            logger.info("  匹配被用户中断，已处理 %d/%d 事件", event_idx, total)
            break

        wwlln_time = event['time']
        wwlln_lat = event['lat']
        wwlln_lon = event['lon']

        station_matches = []
        for sta_name, sdata in station_data.items():
            sta_lat, sta_lon = sdata['lat'], sdata['lon']
            dist = spherical_distance(wwlln_lat, wwlln_lon, sta_lat, sta_lon)
            expected = compute_expected_arrival(wwlln_time, dist)

            result = find_closest_piece(
                sdata['times'], used_per_station[sta_name],
                expected, time_window)

            if result is not None:
                idx, delta = result
                entries = sdata['entries'][idx]
                # entries 现在是 (filtered_piece, lig_time_str, raw_uint16) 三元组
                piece_data = entries[0]
                lig_time_str = entries[1]
                raw_uint16 = entries[2]
                station_matches.append({
                    'station_name': sta_name,
                    'distance_km': round(dist, 3),
                    'reception_time': sdata['times'][idx],
                    'expected_time': expected,
                    'delta_t_s': delta,
                    'piece_data': piece_data,
                    'lig_time_str': lig_time_str,
                    'piece_index': idx,
                    'raw_uint16': raw_uint16,
                })

        if station_matches:
            station_matches.sort(key=lambda m: m['distance_km'])
            if len(station_matches) >= min_stations:
                for m in station_matches:
                    used_per_station[m['station_name']].add(m['piece_index'])
                yield (event_idx, event, station_matches)

        if (event_idx + 1) % report_interval == 0:
            logger.info("  Progress: %d/%d events scanned", event_idx + 1, total)


def format_time_for_filename(time_str: str) -> str:
    """规范化时间字符串用于文件名"""
    if '.' in time_str:
        integer_part, decimal_part = time_str.split('.')
    else:
        integer_part = time_str
        decimal_part = ''
    if len(integer_part) < 12:
        integer_part = integer_part.zfill(12)
    elif len(integer_part) > 12:
        integer_part = integer_part[:12]
    if decimal_part:
        decimal_part = decimal_part.ljust(7, '0')[:7]
    else:
        decimal_part = '0000000'
    return f"{integer_part}.{decimal_part}"


def write_event_output(event_idx: int, event: dict,
                       station_matches: list, output_dir: str,
                       lig_head_path: str, limitbyt_path: str,
                       logger: logging.Logger):
    """将一个闪电事件写出为 .lig + .txt 文件"""
    formatted_time = format_time_for_filename(str(event['time']))
    base_name = f"EVENT_{formatted_time}"

    lig_path = os.path.join(output_dir, f"{base_name}.lig")
    txt_path = os.path.join(output_dir, f"{base_name}.txt")

    counter = 1
    while os.path.exists(lig_path) or os.path.exists(txt_path):
        lig_path = os.path.join(output_dir, f"{base_name}_{counter}.lig")
        txt_path = os.path.join(output_dir, f"{base_name}_{counter}.txt")
        counter += 1

    # 构建 .lig 文件
    with open(lig_head_path, 'rb') as fh:
        global_header = bytearray(fh.read())
    struct.pack_into('i', global_header, 4, len(station_matches))

    pieces_bytes = []
    for match in station_matches:
        # 使用原始 uint16 波形数据写入，不用滤波后的 float 数据
        packed = repacklig(match['raw_uint16'], match['lig_time_str'], limitbyt_path)
        if packed is not None:
            pieces_bytes.append(packed)
        else:
            logger.warning("  repacklig failed for station %s", match['station_name'])

    with open(lig_path, 'wb') as fout:
        fout.write(bytes(global_header))
        for pb in pieces_bytes:
            fout.write(pb)

    # 写入 .txt 伴侣文件
    with open(txt_path, 'w', encoding='utf-8') as ftxt:
        ftxt.write(f"# WWLLN Event: {formatted_time}  "
                   f"Lat={event['lat']:.4f}  Lon={event['lon']:.4f}  "
                   f"Energy={event['energy']}  WWLLN_Stations={event['stations']}\n")
        ftxt.write("Station\tDistance(km)\tReception_Time\tExpected_Time\tDelta_T(s)\n")
        for match in station_matches:
            ftxt.write(f"{match['station_name']}\t"
                       f"{match['distance_km']:.3f}\t"
                       f"{match['reception_time']}\t"
                       f"{match['expected_time']}\t"
                       f"{match['delta_t_s']}\n")

    logger.info("  Wrote %s: %d stations", base_name, len(station_matches))


# ============================================================================
#                          一站完成 API（供对话框调用）
# ============================================================================

def run_trace_matching(stations: list, wwlln_dir: str, output_dir: str,
                       min_stations: int = 2, time_window_s: float = 0.050,
                       lig_head_path: str = None, limitbyt_path: str = None,
                       target_day: str = None, stop_flag=None,
                       progress_cb=None, log_cb=None) -> str:
    """运行多站匹配并输出结果

    参数:
        stations: [{name, lat, lon, dir}, ...]
        wwlln_dir: WWLLN 数据目录
        output_dir: 输出目录
        min_stations: 最小站数
        time_window_s: 时间窗口（秒）
        target_day: 可选，格式 "YYMMDD"，仅加载指定日期的 WWLLN 数据
        stop_flag: 可选，threading.Event，设置后匹配循环中断
        progress_cb: func(fraction, text)
        log_cb: func(text)

    返回: 结果描述字符串
    """
    if lig_head_path is None:
        lig_head_path = _resource_path('LigHead.lig')
    if limitbyt_path is None:
        limitbyt_path = _resource_path('Limitbyt')

    if not os.path.exists(lig_head_path):
        return f"错误: LigHead.lig 未找到: {lig_head_path}"
    if not os.path.exists(limitbyt_path):
        return f"错误: Limitbyt 未找到: {limitbyt_path}"

    os.makedirs(output_dir, exist_ok=True)

    # 设置日志
    logger = logging.getLogger('LigTraceWorker')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if log_cb:
        class CallbackHandler(logging.Handler):
            def emit(self_, record):
                log_cb(self_.format(record))
        handler = CallbackHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s  %(levelname)-7s  %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    if progress_cb:
        progress_cb(0.02, '正在加载 WWLLN 数据...')

    logger.info("━" * 50)
    logger.info("LigTrace — 多站闪电事件匹配")
    logger.info("━" * 50)
    logger.info("活动站点 (%d): %s", len(stations), [s['name'] for s in stations])
    logger.info("参数: min_stations=%d, time_window=%s s", min_stations, time_window_s)

    # 加载 WWLLN
    logger.info("正在加载 WWLLN 数据: %s ...", wwlln_dir)
    if not os.path.isdir(wwlln_dir):
        return f"错误: WWLLN 目录不存在: {wwlln_dir}"

    wwlln_events = load_wwlln_events(wwlln_dir, target_day=target_day)
    if not wwlln_events:
        return "错误: 未加载到 WWLLN 事件"
    logger.info("已加载 %d 个 WWLLN 事件", len(wwlln_events))
    logger.info("  时间范围: %s → %s", wwlln_events[0]['time'], wwlln_events[-1]['time'])

    if progress_cb:
        progress_cb(0.05, '正在加载站点波形数据...')

    # 加载站点数据
    station_data = {}
    for idx, sta in enumerate(stations):
        name = sta['name']
        sta_dir = sta['dir']
        if not sta_dir or not os.path.isdir(sta_dir):
            logger.warning("  [%s] 目录不存在: %s", name, sta_dir)
            continue

        times, entries = load_station_timeline(sta_dir, name, logger)
        if not times:
            continue
        station_data[name] = {
            'lat': sta['lat'], 'lon': sta['lon'],
            'times': times, 'entries': entries,
        }
        if progress_cb:
            progress_cb(0.05 + 0.25 * (idx + 1) / max(len(stations), 1),
                        f'已加载: {name} ({len(times)} 条)')

    if len(station_data) < min_stations:
        return f"错误: 需 ≥{min_stations} 个站点有数据，当前 {len(station_data)}"

    if progress_cb:
        progress_cb(0.30, '正在匹配事件...')

    # 执行匹配
    time_window = Decimal(str(time_window_s))
    total_wwlln = len(wwlln_events)
    total_output = 0
    start_time = time_module.time()

    for event_idx, event, station_matches in match_events(
            wwlln_events, station_data, time_window, min_stations, stop_flag, logger):
        write_event_output(event_idx, event, station_matches, output_dir,
                           lig_head_path, limitbyt_path, logger)
        total_output += 1

        if progress_cb:
            frac = 0.30 + 0.65 * ((event_idx + 1) / total_wwlln)
            progress_cb(min(0.95, frac),
                        f'匹配中: {event_idx+1}/{total_wwlln} 事件, {total_output} 已匹配')

    elapsed = time_module.time() - start_time

    # 写入日志文件到输出目录
    log_path = os.path.join(output_dir, 'ligtrace.log')
    try:
        with open(log_path, 'w', encoding='utf-8') as lf:
            lf.write("LigTrace 多站闪电事件匹配日志\n")
            lf.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            lf.write(f"{'='*50}\n")
            lf.write(f"活动站点 ({len(station_data)}): {list(station_data.keys())}\n")
            lf.write(f"min_stations={min_stations}, time_window={time_window_s} s\n")
            lf.write(f"扫描 WWLLN 事件: {total_wwlln}\n")
            lf.write(f"多站事件输出: {total_output}\n")
            lf.write(f"耗时: {elapsed:.1f} 秒\n")
    except Exception:
        pass

    logger.info("━" * 50)
    logger.info("匹配完成!")
    logger.info("  扫描 WWLLN 事件: %d", total_wwlln)
    logger.info("  多站事件输出: %d", total_output)
    logger.info("  耗时: %.1f 秒", elapsed)
    logger.info("  输出目录: %s", output_dir)
    logger.info("━" * 50)

    if progress_cb:
        progress_cb(1.0, '完成')

    was_interrupted = stop_flag and stop_flag.is_set()
    return f"匹配完成！共匹配 {total_output} 个多站闪电事件（耗时 {elapsed:.1f}s）\n输出目录: {output_dir}"