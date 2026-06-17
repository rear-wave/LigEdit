#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline - 闪电数据处理5步流水线后端
步骤1：时间戳提取  步骤2：WWLLN匹配  步骤3：距离筛选
步骤4：波形提取    步骤5：昼夜分类
"""

import os
import bisect
import math
from decimal import Decimal, getcontext
from os.path import getsize

import numpy as np
import pandas as pd

from lig_parser import (
    ReadLigFile, ButterFilter, compute_final_time, repacklig,
    time_classifier_display, load_station_coords, match_station_name,
)

getcontext().prec = 40
EARTH_RADIUS = 6371.0
C_KM_S = Decimal("299792.458")


# ============================================================================
#                          工具函数
# ============================================================================

def deg2rad(deg):
    return deg * (math.pi / 180.0)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine公式计算球面距离（km）"""
    lat1_r, lon1_r = deg2rad(lat1), deg2rad(lon1)
    lat2_r, lon2_r = deg2rad(lat2), deg2rad(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    return EARTH_RADIUS * 2 * math.asin(math.sqrt(min(a, 1.0)))


def format_txt_time(txt_time_str):
    """将TXT时间统一为 'YYMMDDhhmmss.fffffff'（7位小数）"""
    s = str(txt_time_str).strip()
    if '.' in s:
        ip, fp = s.split('.', 1)
    else:
        ip, fp = s, ''
    if len(ip) < 12:
        ip = ip.zfill(12)
    elif len(ip) > 12:
        ip = ip[:12]
    fp = (fp + '0' * 7)[:7]
    return f"{ip}.{fp}"


def parse_wwlln_time_to_decimal(date_str, time_str):
    """解析WWLLN时间并转换为Decimal格式"""
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
            sec_str, usec_str = sec_part.split('.', 1)
            second = int(sec_str)
            usec_str = usec_str[:6].ljust(6, '0')
            decimal_part = f".{usec_str}"
        else:
            second = int(sec_part)
            decimal_part = ".000000"
        year_short = year % 100
        return Decimal(f"{year_short:02d}{month:02d}{day:02d}{hour:02d}{minute:02d}{second:02d}{decimal_part}")
    except Exception:
        return None


def check_distance_tolerance(actual_dist, time_diff, tolerance_ratio=0.1):
    """检查实际距离与时间差对应距离是否符合容差要求"""
    time_diff_dist = float(time_diff * C_KM_S)
    tolerance = actual_dist * tolerance_ratio
    is_ok = abs(actual_dist - time_diff_dist) <= tolerance
    return round(time_diff_dist, 3), is_ok


class _PieceWriter:
    """lig片段写入器：512条自动分卷，文件名格式 站点名_首条时间戳.lig"""

    def __init__(self, output_dir, station_name, lig_file_head_path):
        self.output_dir = output_dir
        self.station_name = station_name
        self.lig_file_head_path = lig_file_head_path
        self.current_path = None
        self.current_fp = None
        self.piece_count = 0
        self.root_time = None
        self.file_index = 1
        self.total_written = 0

    def write(self, piece_data, time_key):
        if self.piece_count >= 512 or self.current_fp is None:
            self._start_new_file(time_key)
        self.current_fp.write(piece_data)
        self.piece_count += 1
        self.total_written += 1

    def _start_new_file(self, time_key):
        self.close()
        if self.root_time is None:
            self.root_time = time_key
        if self.file_index == 1:
            filename = f"{self.station_name}_{self.root_time}.lig"
        else:
            filename = f"{self.station_name}_{self.root_time}_{self.file_index}.lig"
        self.current_path = os.path.join(self.output_dir, filename)
        self.current_fp = open(self.current_path, 'wb')
        if os.path.exists(self.lig_file_head_path):
            with open(self.lig_file_head_path, 'rb') as hf:
                self.current_fp.write(hf.read())
        self.piece_count = 0
        self.file_index += 1

    def close(self):
        if self.current_fp is not None:
            self.current_fp.close()
            self.current_fp = None


def _extract_txt_time_column(input_file, output_file):
    """提取txt文件中第一列（TXT_Time）数据"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        is_first = True
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            if is_first:
                is_first = False
                continue
            parts = line.split('\t')
            if parts:
                f_out.write(parts[0] + '\n')


# ============================================================================
#                          步骤1：时间戳提取
# ============================================================================

def step1_extract_timestamps(maindir, output_txt, progress_cb=None, log_cb=None):
    """从lig文件提取闪电波形时间戳，排序后输出到txt"""
    all_times = []
    file_count = 0

    lig_files = []
    for root, dirs, files in os.walk(maindir):
        for f in files:
            if f.lower().endswith('.lig'):
                lig_files.append(os.path.join(root, f))

    for idx, ligpath in enumerate(lig_files):
        file_count += 1
        if progress_cb:
            progress_cb(1, f"步骤1: 处理文件 {file_count}/{len(lig_files)}",
                        int(idx / max(len(lig_files), 1) * 100))
        try:
            lig_data = ReadLigFile(ligpath)
        except Exception as e:
            if log_cb:
                log_cb(f"[错误] 读取 {ligpath}: {e}")
            continue
        for time_key in lig_data:
            try:
                piece = np.array(lig_data[time_key]['0'])
                final_time = compute_final_time(time_key, piece)
                all_times.append(final_time)
            except Exception as e:
                if log_cb:
                    log_cb(f"[错误] 处理片段 {time_key}: {e}")

    sorted_times = sorted(all_times, key=lambda x: Decimal(x))
    with open(output_txt, 'w', encoding='utf-8') as f:
        for t in sorted_times:
            f.write(f"{t}\n")

    if progress_cb:
        progress_cb(1, f"步骤1完成: {len(sorted_times)} 个时间点", 100)
    return output_txt


# ============================================================================
#                          步骤2：WWLLN匹配
# ============================================================================

def step2_match_wwlln(txt_file_path, wwlln_folder, output_dir,
                      station_lat, station_lon,
                      max_time_diff_s=Decimal("0.011"),
                      max_distance_km=3500.0,
                      tolerance_ratio=0.1,
                      progress_cb=None, log_cb=None):
    """将txt时间与WWLLN定位数据匹配"""
    os.makedirs(output_dir, exist_ok=True)
    out_data = os.path.join(output_dir, "matched_WWLLN.txt")
    out_log = os.path.join(output_dir, "match_log_WWLLN.txt")

    # 读取并分组TXT时间
    with open(txt_file_path, "r", encoding="utf-8") as f:
        raw_times = [ln.strip() for ln in f if ln.strip()]
    by_day = {}
    for t in raw_times:
        t_fmt = format_txt_time(t)
        day_id = t_fmt[:6]
        by_day.setdefault(day_id, []).append(t_fmt)
    for k in by_day:
        by_day[k].sort(key=lambda x: Decimal(x))

    if not by_day:
        if log_cb:
            log_cb("TXT文件无有效时间")
        return None

    # 写表头
    with open(out_data, "w", encoding="utf-8") as f:
        f.write("TXT_Time\tWWLLN_Time\tLatitude\tLongitude\tStations\tEnergy\tDistance_to_Station(km)\n")
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(f"===== WWLLN匹配日志 =====\n")
        f.write(f"站点经纬度：({station_lat}, {station_lon})\n\n")

    total_hits = 0
    day_list = sorted(by_day.keys())

    for day_idx, day_id in enumerate(day_list):
        txt_times = by_day[day_id]
        if progress_cb:
            progress_cb(2, f"步骤2: 匹配 {day_id} ({day_idx+1}/{len(day_list)})",
                        int((day_idx + 1) / max(len(day_list), 1) * 100))

        # 读取WWLLN数据
        wwlln_files = sorted([f for f in os.listdir(wwlln_folder) if f.endswith('.loc')])
        all_dfs = []
        for wf in wwlln_files:
            try:
                col_names = ["Date", "Time", "Latitude", "Longitude", "Error",
                             "Stations", "Energy", "Energy2", "Stations2"]
                df = pd.read_csv(os.path.join(wwlln_folder, wf), header=None, names=col_names,
                                 sep=',', dtype=str, skipinitialspace=True)
                for col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.rstrip(',')
                df = df[df['Date'].notna() & df['Time'].notna()]
                for col in ["Latitude", "Longitude", "Stations", "Energy"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['Time_fmt'] = df.apply(
                    lambda r: parse_wwlln_time_to_decimal(r['Date'], r['Time']), axis=1)
                df = df.dropna(subset=['Time_fmt'])
                df['day_str'] = df['Time_fmt'].apply(lambda x: str(x)[:6] if x else None)
                df = df[df['day_str'] == day_id]
                all_dfs.append(df)
            except Exception as e:
                if log_cb:
                    log_cb(f"[错误] 读取WWLLN文件 {wf}: {e}")
        if not all_dfs:
            continue
        wwlln_df = pd.concat(all_dfs, ignore_index=True).sort_values(by='Time_fmt', ignore_index=True)

        # 双指针匹配
        i, j = 0, 0
        n_txt, n_wwlln = len(txt_times), len(wwlln_df)
        txt_vals = [Decimal(t) for t in txt_times]
        matched_rows = []

        while i < n_txt and j < n_wwlln:
            txt_val = txt_vals[i]
            wwlln_val = wwlln_df.at[j, "Time_fmt"]
            dt = txt_val - wwlln_val
            if dt > 0 and dt <= max_time_diff_s:
                try:
                    lat = float(wwlln_df.at[j, "Latitude"])
                    lon = float(wwlln_df.at[j, "Longitude"])
                    dist = haversine_distance(lat, lon, station_lat, station_lon)
                    dist_rounded = round(dist, 3)
                except Exception:
                    i += 1; j += 1; continue
                if dist_rounded <= max_distance_km:
                    _, is_ok = check_distance_tolerance(dist_rounded, dt, tolerance_ratio)
                    if is_ok:
                        matched_rows.append([
                            txt_times[i], str(wwlln_val), lat, lon,
                            wwlln_df.at[j, "Stations"], wwlln_df.at[j, "Energy"], dist_rounded])
                i += 1; j += 1
            elif txt_val <= wwlln_val:
                i += 1
            else:
                j += 1

        total_hits += len(matched_rows)
        with open(out_data, "a", encoding="utf-8") as f:
            for row in matched_rows:
                f.write("\t".join(str(x) for x in row) + "\n")
        with open(out_log, "a", encoding="utf-8") as f:
            f.write(f"===== {day_id} =====\n匹配成功：{len(matched_rows)}条\n\n")

    if progress_cb:
        progress_cb(2, f"步骤2完成: 匹配 {total_hits} 条", 100)
    return out_data


# ============================================================================
#                          步骤3：按距离筛选
# ============================================================================

def step3_distance_select(matched_file, start_dist, end_dist, step_dist,
                          prefix="lightning", progress_cb=None, log_cb=None):
    """按距离区间筛选，输出不同距离的txt文件"""
    df = pd.read_csv(matched_file, sep='\t', dtype=str)
    df['Distance_to_Station(km)'] = df['Distance_to_Station(km)'].astype(float)
    output_dir = os.path.dirname(matched_file)
    output_files = []
    current_start = start_dist
    while current_start + step_dist <= end_dist:
        current_end = current_start + step_dist
        df_range = df[(df['Distance_to_Station(km)'] >= current_start) &
                      (df['Distance_to_Station(km)'] < current_end)]
        if len(df_range) > 0:
            output_file = os.path.join(output_dir, f"{prefix}_distance_{current_start}_{current_end}.txt")
            df_range.to_csv(output_file, sep='\t', index=False)
            output_files.append((current_start, current_end, output_file))
            if log_cb:
                log_cb(f"区间{current_start}-{current_end}km: {len(df_range)}行")
        current_start += step_dist
    if progress_cb:
        progress_cb(3, f"步骤3完成: {len(output_files)} 个区间文件", 100)
    return output_files


# ============================================================================
#                          步骤4：提取lig波形
# ============================================================================

def step4_extract_lig(distance_files, lig_maindir, lig_head_path, lig_file_head_path,
                      station_coords=None, progress_cb=None, log_cb=None):
    """根据距离txt文件从lig文件中提取对应波形
    
    lig_head_path: Limitbyt路径（用于repacklig打包）
    lig_file_head_path: LigHead.lig路径（用于创建新lig文件头）
    station_coords: 站点经纬度字典（用于匹配站点名）
    """
    if not distance_files:
        if log_cb:
            log_cb("无距离区间文件，跳过步骤4")
        return []

    if station_coords is None:
        station_coords = load_station_coords()

    # 准备区间信息
    range_info = []
    total_txt = 0
    for start_dist, end_dist, dist_txt_file in distance_files:
        dist_tag = f"{start_dist}-{end_dist}km"
        base_dir = os.path.dirname(dist_txt_file)
        temp_txt = os.path.join(base_dir, "_temp_times.txt")
        _extract_txt_time_column(dist_txt_file, temp_txt)
        output_dir = os.path.join(base_dir, dist_tag)
        os.makedirs(output_dir, exist_ok=True)
        with open(temp_txt, 'r') as f:
            txt_times = sorted([Decimal(line.strip()) for line in f if line.strip()])
        if os.path.exists(temp_txt):
            os.remove(temp_txt)
        txt_set = set(txt_times)
        range_info.append({
            'dist_tag': dist_tag, 'txt_times': txt_times, 'txt_set': txt_set,
            'output_dir': output_dir, 'match_count': 0, 'txt_matched': set(),
            'writers': {},
        })
        total_txt += len(txt_times)

    # 遍历lig文件
    lig_files = []
    for root, dirs, files in os.walk(lig_maindir):
        for f in files:
            if f.lower().endswith('.lig'):
                lig_files.append(os.path.join(root, f))

    total_matched = 0
    for file_idx, lig_file in enumerate(lig_files):
        if progress_cb:
            progress_cb(4, f"步骤4: 处理lig文件 {file_idx+1}/{len(lig_files)}",
                        int((file_idx + 1) / max(len(lig_files), 1) * 100))
        try:
            lig_data = ReadLigFile(lig_file)
        except Exception as e:
            if log_cb:
                log_cb(f"[错误] 加载 {lig_file}: {e}")
            continue
        for time_key, piece_data in lig_data.items():
            try:
                if '0' not in piece_data:
                    continue
                piece = np.array(piece_data['0'])
                final_time = compute_final_time(time_key, piece)
                final_dec = Decimal(final_time)
            except Exception:
                continue
            for rinfo in range_info:
                if final_dec in rinfo['txt_set']:
                    pos = bisect.bisect_left(rinfo['txt_times'], final_dec)
                    if (pos < len(rinfo['txt_times']) and
                            rinfo['txt_times'][pos] == final_dec and
                            pos not in rinfo['txt_matched']):
                        event = repacklig(piece, time_key, lig_head_path)
                        if event is not None:
                            lat = piece_data.get('m_GPSCurrentLocationLat', 0)
                            lon = piece_data.get('m_GPSCurrentLocationLon', 0)
                            sname = match_station_name(lat, lon, station_coords) if (lat and lon) else "UNKNOWN"
                            if sname not in rinfo['writers']:
                                rinfo['writers'][sname] = _PieceWriter(
                                    rinfo['output_dir'], sname, lig_file_head_path)
                            rinfo['writers'][sname].write(event, time_key)
                            rinfo['match_count'] += 1
                            rinfo['txt_matched'].add(pos)
                            total_matched += 1

    for rinfo in range_info:
        for w in rinfo['writers'].values():
            w.close()

    output_folders = [r['output_dir'] for r in range_info]
    if progress_cb:
        progress_cb(4, f"步骤4完成: 匹配 {total_matched}/{total_txt}", 100)
    return output_folders


# ============================================================================
#                          步骤5：昼夜分类
# ============================================================================

def step5_day_night_classifier(input_folders, lig_head_path, lig_file_head_path,
                               station_coords=None, progress_cb=None, log_cb=None):
    """对不同距离的lig波形文件进行白天/夜晚分类
    
    lig_head_path: Limitbyt路径（用于repacklig打包）
    lig_file_head_path: LigHead.lig路径（用于创建新lig文件头）
    station_coords: 站点经纬度字典（用于匹配站点名）
    """
    if not input_folders:
        return

    if station_coords is None:
        station_coords = load_station_coords()

    for folder_idx, input_folder in enumerate(input_folders):
        if not os.path.exists(input_folder):
            continue
        distance_range = os.path.basename(input_folder.strip(os.sep))
        if progress_cb:
            progress_cb(5, f"步骤5: 分类 {distance_range} ({folder_idx+1}/{len(input_folders)})",
                        int((folder_idx + 1) / max(len(input_folders), 1) * 100))

        lig_files_found = []
        for root, dirs, files in os.walk(input_folder):
            for f in files:
                if f.lower().endswith('.lig'):
                    lig_files_found.append(os.path.join(root, f))
        if not lig_files_found:
            continue

        pieces = []
        for lf in lig_files_found:
            try:
                lig_data = ReadLigFile(lf)
                for time_key, piece_data in lig_data.items():
                    if '0' in piece_data:
                        pieces.append((time_key, piece_data))
            except Exception:
                pass
        pieces.sort(key=lambda x: x[0])

        output_subfolders = {
            "day": os.path.join(input_folder, f"day_{distance_range}"),
            "night": os.path.join(input_folder, f"night_{distance_range}")
        }
        for folder in output_subfolders.values():
            os.makedirs(folder, exist_ok=True)

        writers = {}
        for time_key, piece_data in pieces:
            try:
                piece = np.array(piece_data['0'])
                period = time_classifier_display(time_key)
                period_en = "day" if period == "白天" else "night"
                packed = repacklig(piece, time_key, lig_head_path)
                if packed is None:
                    continue
                lat = piece_data.get('m_GPSCurrentLocationLat', 0)
                lon = piece_data.get('m_GPSCurrentLocationLon', 0)
                sname = match_station_name(lat, lon, station_coords) if (lat and lon) else "UNKNOWN"
                writer_key = (period_en, sname)
                if writer_key not in writers:
                    writers[writer_key] = _PieceWriter(
                        output_subfolders[period_en], sname, lig_file_head_path)
                writers[writer_key].write(packed, time_key)
            except Exception as e:
                if log_cb:
                    log_cb(f"[错误] 处理片段 {time_key}: {e}")

        for w in writers.values():
            w.close()

    if progress_cb:
        progress_cb(5, "步骤5完成", 100)


# ============================================================================
#                          一键运行
# ============================================================================

def run_full_pipeline(config, progress_cb=None, log_cb=None):
    """运行完整5步流水线，config为dict包含所有参数"""
    # 步骤1
    if progress_cb:
        progress_cb(1, "步骤1: 提取时间戳...", 0)
    txt_file = step1_extract_timestamps(
        config['lig_dir'], config['output_txt'], progress_cb, log_cb)
    config['_txt_file'] = txt_file

    # 步骤2
    if progress_cb:
        progress_cb(2, "步骤2: WWLLN匹配...", 0)
    matched_file = step2_match_wwlln(
        txt_file, config['wwlln_folder'], config['step2_output_dir'],
        config['station_lat'], config['station_lon'],
        config.get('max_time_diff_s', Decimal("0.011")),
        config.get('max_distance_km', 3500.0),
        config.get('tolerance_ratio', 0.1),
        progress_cb, log_cb)
    if not matched_file:
        if log_cb:
            log_cb("[终止] 步骤2未生成匹配文件")
        return

    # 步骤3
    if progress_cb:
        progress_cb(3, "步骤3: 距离筛选...", 0)
    distance_files = step3_distance_select(
        matched_file, config['start_dist'], config['end_dist'], config['step_dist'],
        config.get('prefix', 'lightning'), progress_cb, log_cb)
    if not distance_files:
        if log_cb:
            log_cb("[终止] 步骤3未生成距离区间文件")
        return

    # 步骤4
    if progress_cb:
        progress_cb(4, "步骤4: 提取lig波形...", 0)
    output_folders = step4_extract_lig(
        distance_files, config['lig_dir'], config['lig_head_path'],
        config.get('lig_file_head_path', config['lig_head_path']),
        station_coords=None, progress_cb=progress_cb, log_cb=log_cb)

    # 步骤5
    if progress_cb:
        progress_cb(5, "步骤5: 昼夜分类...", 0)
    step5_day_night_classifier(
        output_folders, config['lig_head_path'],
        config.get('lig_file_head_path', config['lig_head_path']),
        station_coords=None, progress_cb=progress_cb, log_cb=log_cb)

    if progress_cb:
        progress_cb(0, "全部完成！", 100)


# ============================================================================
#                          组合函数（供对话框直接调用）
# ============================================================================

def classify_by_distance(lig_dir, wwlln_folder, output_dir,
                         station_lat, station_lon,
                         start_dist, end_dist, step_dist,
                         lig_head_path, lig_file_head_path,
                         max_time_diff_s=Decimal("0.011"),
                         max_distance_km=3500.0,
                         tolerance_ratio=0.1,
                         progress_cb=None, log_cb=None):
    """按距离分类：提取时间戳→WWLLN匹配→距离筛选→波形提取，一键完成
    
    lig_head_path: Limitbyt路径（用于repacklig打包）
    lig_file_head_path: LigHead.lig路径（用于创建新lig文件头）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 提取时间戳
    if progress_cb:
        progress_cb(1, "提取时间戳...", 0)
    txt_file = os.path.join(output_dir, "timestamps.txt")
    step1_extract_timestamps(lig_dir, txt_file, progress_cb, log_cb)

    # 2. WWLLN匹配
    if progress_cb:
        progress_cb(2, "WWLLN匹配...", 0)
    step2_dir = os.path.join(output_dir, "wwlln_matched")
    matched_file = step2_match_wwlln(
        txt_file, wwlln_folder, step2_dir,
        station_lat, station_lon,
        max_time_diff_s, max_distance_km, tolerance_ratio,
        progress_cb, log_cb)
    if not matched_file:
        if log_cb:
            log_cb("[终止] WWLLN匹配未生成结果")
        return "WWLLN匹配未生成结果"

    # 3. 距离筛选
    if progress_cb:
        progress_cb(3, "距离筛选...", 0)
    distance_files = step3_distance_select(
        matched_file, start_dist, end_dist, step_dist,
        "lightning", progress_cb, log_cb)
    if not distance_files:
        if log_cb:
            log_cb("[终止] 距离筛选未生成区间文件")
        return "距离筛选未生成区间文件"

    # 4. 提取lig波形
    if progress_cb:
        progress_cb(4, "提取lig波形...", 0)
    output_folders = step4_extract_lig(
        distance_files, lig_dir, lig_head_path, lig_file_head_path,
        station_coords=None, progress_cb=progress_cb, log_cb=log_cb)

    if progress_cb:
        progress_cb(0, "按距离分类完成！", 100)
    return f"按距离分类完成！输出目录: {output_dir}"


def classify_by_daynight(lig_dir, output_dir, lig_head_path, lig_file_head_path,
                         progress_cb=None, log_cb=None):
    """按昼夜分类：读取lig文件，按白天/夜晚分组输出
    
    lig_head_path: Limitbyt路径（用于repacklig打包）
    lig_file_head_path: LigHead.lig路径（用于创建新lig文件头）
    """
    os.makedirs(output_dir, exist_ok=True)
    station_coords = load_station_coords()

    # 收集lig文件
    lig_files = []
    for root, dirs, files in os.walk(lig_dir):
        for f in files:
            if f.lower().endswith('.lig'):
                lig_files.append(os.path.join(root, f))

    if not lig_files:
        if log_cb:
            log_cb("[终止] 未找到lig文件")
        return "未找到lig文件"

    # 读取所有片段
    if progress_cb:
        progress_cb(0, "读取lig文件...", 0)
    all_pieces = []
    for file_idx, lig_file in enumerate(lig_files):
        if progress_cb:
            progress_cb(0, f"读取lig文件 {file_idx+1}/{len(lig_files)}",
                        int((file_idx + 1) / max(len(lig_files), 1) * 50))
        try:
            lig_data = ReadLigFile(lig_file)
            for time_key, piece_data in lig_data.items():
                if '0' in piece_data:
                    all_pieces.append((time_key, piece_data))
        except Exception as e:
            if log_cb:
                log_cb(f"[错误] 读取 {lig_file}: {e}")

    all_pieces.sort(key=lambda x: x[0])

    # 创建输出目录
    day_dir = os.path.join(output_dir, "day")
    night_dir = os.path.join(output_dir, "night")
    os.makedirs(day_dir, exist_ok=True)
    os.makedirs(night_dir, exist_ok=True)

    # 分类写入
    writers = {}
    day_count = 0
    night_count = 0
    for idx, (time_key, piece_data) in enumerate(all_pieces):
        if progress_cb:
            progress_cb(0, f"分类片段 {idx+1}/{len(all_pieces)}",
                        50 + int((idx + 1) / max(len(all_pieces), 1) * 50))
        try:
            piece = np.array(piece_data['0'])
            period = time_classifier_display(time_key)
            packed = repacklig(piece, time_key, lig_head_path)
            if packed is None:
                continue
            period_en = "day" if period == "白天" else "night"
            if period_en == "day":
                day_count += 1
            else:
                night_count += 1
            lat = piece_data.get('m_GPSCurrentLocationLat', 0)
            lon = piece_data.get('m_GPSCurrentLocationLon', 0)
            sname = match_station_name(lat, lon, station_coords) if (lat and lon) else "UNKNOWN"
            writer_key = (period_en, sname)
            if writer_key not in writers:
                out_dir = day_dir if period_en == "day" else night_dir
                writers[writer_key] = _PieceWriter(out_dir, sname, lig_file_head_path)
            writers[writer_key].write(packed, time_key)
        except Exception as e:
            if log_cb:
                log_cb(f"[错误] 处理片段 {time_key}: {e}")

    for w in writers.values():
        w.close()

    if progress_cb:
        progress_cb(0, "按昼夜分类完成！", 100)
    return f"按昼夜分类完成！白天: {day_count}条, 夜晚: {night_count}条\n输出目录: {output_dir}"
