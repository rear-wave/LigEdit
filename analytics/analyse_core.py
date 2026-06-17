#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analytics / analyse_core — 闪电数据分析核心逻辑

移植自 LigAnalyse 项目的 analyse_core.py
功能：距离分布、电流强度分布、独立分布判断、按文件夹分类分析。
"""

import os
import gc
import bisect
from decimal import Decimal, getcontext

import numpy as np

from lig_parser import (
    ReadLigFile, compute_final_time, compute_peak_voltage,
    haversine_distance, format_txt_time,
    ButterFilter, CutPieceTo16000,
    load_station_coords, match_station_name,
)

getcontext().prec = 40

GZ_LAT = 23.5686
GZ_LON = 113.6147


# ============================================================================
#                          数据加载
# ============================================================================

def load_lig_pieces(lig_dir, progress_cb=None, skip_waveform=False, category=None):
    """从目录加载所有 lig 文件片段"""
    lig_files = []
    for root, dirs, files in os.walk(lig_dir):
        for f in files:
            if f.lower().endswith('.lig'):
                lig_files.append(os.path.join(root, f))

    all_pieces = []
    total_files = len(lig_files)
    station_coords = load_station_coords()

    for file_idx, ligpath in enumerate(lig_files):
        if progress_cb:
            progress_cb(f"加载 {os.path.basename(ligpath)} ({file_idx+1}/{total_files})",
                        int((file_idx + 1) / max(total_files, 1) * 100))
        try:
            lig_data = ReadLigFile(ligpath, skip_waveform=skip_waveform)
        except Exception as e:
            if progress_cb:
                progress_cb(f"[错误] {ligpath}: {e}", -1)
            continue

        for time_key, piece_data in lig_data.items():
            try:
                if not skip_waveform and '0' not in piece_data:
                    continue

                station_lat = piece_data.get('m_GPSCurrentLocationLat', 0)
                station_lon = piece_data.get('m_GPSCurrentLocationLon', 0)
                station_name = match_station_name(station_lat, station_lon, station_coords)

                if skip_waveform:
                    final_time = time_key
                    peak_v = None
                else:
                    piece = np.array(piece_data['0'])
                    final_time = compute_final_time(time_key, piece)
                    peak_v = compute_peak_voltage(piece_data)

                all_pieces.append({
                    'time_key': time_key,
                    'final_time': final_time,
                    'final_time_dec': Decimal(final_time),
                    'peak_voltage': peak_v,
                    'station_lat': station_lat,
                    'station_lon': station_lon,
                    'station_name': station_name,
                    'lightning_lat': piece_data.get('m_LightningLocationLat', None),
                    'lightning_lon': piece_data.get('m_LightningLocationLon', None),
                    'category': category,
                })
            except Exception:
                pass

        del lig_data

    all_pieces.sort(key=lambda x: x['final_time_dec'])

    if progress_cb:
        progress_cb(f"加载完成: {len(all_pieces)} 个片段", 100)
    return all_pieces


def load_lig_pieces_by_category(lig_dir, progress_cb=None):
    """按子文件夹分类加载 lig 数据

    返回: (categories_dict, all_pieces_list)
    """
    categories = {}
    subdirs = []
    for entry in sorted(os.listdir(lig_dir)):
        full_path = os.path.join(lig_dir, entry)
        if os.path.isdir(full_path):
            subdirs.append((entry, full_path))

    has_root_lig = any(
        f.lower().endswith('.lig')
        for f in os.listdir(lig_dir)
        if os.path.isfile(os.path.join(lig_dir, f))
    )

    total_steps = len(subdirs) + (1 if has_root_lig else 0)
    step = 0

    for cat_name, cat_dir in subdirs:
        step += 1
        if progress_cb:
            progress_cb(f"加载 [{cat_name}] ({step}/{total_steps})...",
                        int(step / max(total_steps, 1) * 100))
        pieces = load_lig_pieces(cat_dir, skip_waveform=False, category=cat_name)
        if pieces:
            categories[cat_name] = pieces

    if has_root_lig:
        step += 1
        if progress_cb:
            progress_cb(f"加载 [未分类] ({step}/{total_steps})...",
                        int(step / max(total_steps, 1) * 100))
        pieces = load_lig_pieces(lig_dir, skip_waveform=False, category="未分类")
        if pieces:
            categories["未分类"] = pieces

    all_pieces = []
    for cat_pieces in categories.values():
        all_pieces.extend(cat_pieces)
    all_pieces.sort(key=lambda x: x['final_time_dec'])

    if progress_cb:
        cat_info = ", ".join(f"{k}={len(v)}" for k, v in categories.items())
        progress_cb(f"分类加载完成: 共{len(all_pieces)}个 ({cat_info})", 100)

    return categories, all_pieces


# ============================================================================
#                          闪电定位数据加载
# ============================================================================

def parse_wwlln_time(date_str, time_str):
    """解析 WWLLN 时间到 YYMMDDhhmmss.ffffff（7位小数）"""
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
        return f"{year_short:02d}{month:02d}{day:02d}{hour:02d}{minute:02d}{second:02d}{decimal_part}"
    except Exception:
        return None


def load_wwlln_loc(loc_file, lat_range=None, lon_range=None, progress_cb=None):
    """加载 WWLLN .loc 格式闪电定位文件"""
    lightnings = []
    with open(loc_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if l.strip()]
    total_lines = len(lines)

    for i, line in enumerate(lines):
        if progress_cb and i % 10000 == 0:
            progress_cb(f"解析 .loc {i}/{total_lines}", int(i / max(total_lines, 1) * 100))
        parts = line.split(',')
        if len(parts) < 4:
            continue
        try:
            date_str = parts[0].strip()
            time_str = parts[1].strip()
            lat = float(parts[2].strip())
            lon = float(parts[3].strip())
            if lat_range and not (lat_range[0] <= lat <= lat_range[1]):
                continue
            if lon_range and not (lon_range[0] <= lon <= lon_range[1]):
                continue
            time_fmt = parse_wwlln_time(date_str, time_str)
            if time_fmt is None:
                continue
            lightnings.append({
                'time': time_fmt,
                'time_dec': Decimal(time_fmt),
                'lat': lat, 'lon': lon,
                'error': float(parts[4]) if len(parts) > 4 else 0,
                'stations': int(parts[5]) if len(parts) > 5 else 0,
            })
        except Exception:
            pass

    lightnings.sort(key=lambda x: x['time_dec'])
    return lightnings


def load_lightning_data(match_file, progress_cb=None):
    """加载闪电定位数据，自动识别格式（.loc / .txt）"""
    if match_file.lower().endswith('.loc'):
        return load_wwlln_loc(match_file, progress_cb=progress_cb)

    lightnings = []
    with open(match_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return lightnings

    has_header = 'Time' in lines[0] or 'Latitude' in lines[0] or 'time' in lines[0].lower()
    start_idx = 1 if has_header else 0

    for i in range(start_idx, len(lines)):
        parts = lines[i].split('\t')
        if len(parts) < 3:
            parts = lines[i].split()
        if len(parts) < 3:
            continue
        try:
            time_str = format_txt_time(parts[0].strip())
            lat = float(parts[2].strip()) if len(parts) >= 4 else float(parts[1].strip())
            lon = float(parts[3].strip()) if len(parts) >= 4 else float(parts[2].strip())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                lightnings.append({
                    'time': time_str,
                    'time_dec': Decimal(time_str),
                    'lat': lat, 'lon': lon,
                })
        except Exception:
            pass

    lightnings.sort(key=lambda x: x['time_dec'])
    return lightnings


def load_xlsx_lightning_dir(xlsx_dir, progress_cb=None, utc_hour_range=None):
    """加载 xlsx 格式闪电定位数据

    xlsx 中的 DDATETIME 为北京时间 (UTC+8)，自动转 UTC。
    """
    import glob as glob_mod
    from datetime import datetime, timedelta

    xlsx_files = sorted(glob_mod.glob(os.path.join(xlsx_dir, '**', '*.xlsx'), recursive=True))
    if not xlsx_files:
        return []

    # 按 UTC 小时范围预筛选文件
    if utc_hour_range:
        utc_min, utc_max = utc_hour_range
        bj_hours_needed = set()
        for h in range(utc_min, utc_max):
            bj_hours_needed.add((h + 8) % 24)
        filtered = []
        for f in xlsx_files:
            basename = os.path.basename(f)
            try:
                if int(basename.split('时')[0]) in bj_hours_needed:
                    filtered.append(f)
            except ValueError:
                pass
        xlsx_files = filtered

    try:
        import openpyxl
    except ImportError:
        if progress_cb:
            progress_cb("需要 openpyxl: pip install openpyxl", -1)
        return []

    all_lightnings = []

    for fi, xlsx_file in enumerate(xlsx_files):
        if progress_cb:
            progress_cb(f"加载 xlsx {fi+1}/{len(xlsx_files)}: {os.path.basename(xlsx_file)}",
                        int((fi + 1) / len(xlsx_files) * 100))
        try:
            wb = openpyxl.load_workbook(xlsx_file, read_only=True)
            ws = wb.active
            header = None
            for row in ws.iter_rows(values_only=True):
                if header is None:
                    header = [str(h).strip() if h else '' for h in row]
                    continue
                try:
                    row_dict = {}
                    for i, val in enumerate(row):
                        if i < len(header):
                            row_dict[header[i]] = val

                    ddatetime = row_dict.get('DDATETIME', '')
                    nano = row_dict.get('NANO', 0)
                    lat = row_dict.get('LATITUDE', 0)
                    lon = row_dict.get('LONGITUDE', 0)
                    if not ddatetime or lat is None or lon is None:
                        continue

                    lat, lon = float(lat), float(lon)

                    if isinstance(ddatetime, datetime):
                        bj_dt = ddatetime
                    else:
                        parts = str(ddatetime).replace('-', ' ').replace(':', ' ').split()
                        if len(parts) < 6:
                            continue
                        bj_dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                                         int(parts[3]), int(parts[4]), int(parts[5]))

                    utc_dt = bj_dt - timedelta(hours=8)
                    nano_val = int(nano) if nano else 0
                    frac_sec = nano_val / 1e9
                    frac_str = f"{frac_sec:.7f}"[2:]

                    time_str = (f"{utc_dt.year % 100:02d}{utc_dt.month:02d}{utc_dt.day:02d}"
                                f"{utc_dt.hour:02d}{utc_dt.minute:02d}{utc_dt.second:02d}"
                                f".{frac_str}")

                    if utc_hour_range:
                        if not (utc_hour_range[0] <= utc_dt.hour < utc_hour_range[1]):
                            continue

                    all_lightnings.append({
                        'time': time_str,
                        'time_dec': Decimal(time_str),
                        'lat': lat, 'lon': lon,
                        'signal': float(row_dict.get('SIGNAL', 0)) if row_dict.get('SIGNAL') is not None else 0,
                        'multi': int(row_dict.get('MULTI', 0)) if row_dict.get('MULTI') is not None else 0,
                    })
                except Exception:
                    pass
            wb.close()
        except Exception:
            pass

    all_lightnings.sort(key=lambda x: x['time_dec'])
    return all_lightnings


def load_nbe_loc_file(loc_file, progress_cb=None):
    """加载 NBE 定位文件（如 2020_05_20-22_负极性NBEs.txt）"""
    locations = []
    with open(loc_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if l.strip()]
    total_lines = len(lines)

    for i, line in enumerate(lines):
        if progress_cb and i % 5000 == 0:
            progress_cb(f"解析 NBE {i}/{total_lines}", int(i / max(total_lines, 1) * 100))
        if line.startswith('Time'):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            time_str = parts[0].strip()
            lat = float(parts[2].strip())
            lon = float(parts[3].strip())
            time_dec = Decimal(time_str)
            locations.append({
                'time': time_str,
                'time_dec': time_dec,
                'lat': lat, 'lon': lon,
                'signal': float(parts[4]) if len(parts) > 4 and parts[4] != 'NaN' else 0,
                'flags': parts[5] if len(parts) > 5 else '',
                'multi': int(parts[6]) if len(parts) > 6 and parts[6] not in ('NaN', '') else 0,
            })
        except Exception:
            pass

    locations.sort(key=lambda x: x['time_dec'])
    return locations


# ============================================================================
#                          距离分布分析
# ============================================================================

SPEED_OF_LIGHT = 299792.458


def analyse_distance_distribution(lig_pieces, lightnings, nbe_locations=None,
                                  station_lat=None, station_lon=None,
                                  max_time_diff_s=0.011, max_distance_km=3500,
                                  progress_cb=None):
    """计算每个 lig 片段到匹配闪电的距离分布

    优先 NBE 数据，其次 xlsx 闪电定位数据。
    """
    if not lightnings and not nbe_locations:
        if progress_cb:
            progress_cb("无闪电定位数据", -1)
        return []

    s_lat = station_lat or GZ_LAT
    s_lon = station_lon or GZ_LON
    c_dec = Decimal(str(SPEED_OF_LIGHT))

    if nbe_locations:
        return _match_distance_precise(lig_pieces, nbe_locations, s_lat, s_lon, c_dec, progress_cb, "NBE")
    return _match_distance_precise(lig_pieces, lightnings, s_lat, s_lon, c_dec, progress_cb, "闪电定位")


def _match_distance_precise(lig_pieces, loc_data, s_lat, s_lon, c_dec,
                            progress_cb=None, source_name="定位"):
    loc_times = [l['time_dec'] for l in loc_data]
    match_tolerance = Decimal('0.001')

    results = []
    total = len(lig_pieces)

    for idx, piece in enumerate(lig_pieces):
        if progress_cb and idx % 100 == 0:
            progress_cb(f"距离分析 ({source_name}) {idx+1}/{total}",
                        int((idx + 1) / max(total, 1) * 100))

        piece_time = piece['final_time_dec']
        pos = bisect.bisect_left(loc_times, piece_time - match_tolerance)
        best_match = None
        best_dt = None

        for j in range(pos, min(pos + 10, len(loc_data))):
            dt = abs(piece_time - loc_data[j]['time_dec'])
            if dt > match_tolerance:
                if loc_data[j]['time_dec'] > piece_time + match_tolerance:
                    break
                continue
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_match = loc_data[j]

        if best_match is not None:
            dist = haversine_distance(best_match['lat'], best_match['lon'], s_lat, s_lon)
            propagation_s = Decimal(str(dist)) / c_dec
            results.append({
                'final_time': piece['final_time'],
                'distance_km': round(dist, 3),
                'matched_lightning': best_match,
                'time_diff_ms': round(float(best_dt) * 1000, 3),
                'propagation_ms': round(float(propagation_s) * 1000, 3),
                'category': piece.get('category', ''),
            })

    if progress_cb:
        progress_cb(f"距离分析完成 ({source_name}): {len(results)}/{total} 匹配", 100)
    return results


# ============================================================================
#                          电流强度分布
# ============================================================================

def analyse_current_distribution(lig_pieces, progress_cb=None):
    results = []
    total = len(lig_pieces)
    for idx, piece in enumerate(lig_pieces):
        if progress_cb and idx % 100 == 0:
            progress_cb(f"电流分析 {idx+1}/{total}", int((idx + 1) / max(total, 1) * 100))
        peak_v = piece.get('peak_voltage')
        if peak_v is not None:
            results.append({
                'final_time': piece['final_time'],
                'peak_voltage': round(peak_v, 6),
                'category': piece.get('category', ''),
            })
    if progress_cb:
        progress_cb(f"电流分析完成: {len(results)} 条", 100)
    return results


# ============================================================================
#                          独立分布判断
# ============================================================================

def analyse_independent_distribution(lig_pieces, lightnings, nbe_locations=None,
                                     time_window_ms=660, distance_window_km=10,
                                     progress_cb=None):
    """判断每个 lig 片段是否为独立闪电事件"""
    if not lightnings and not nbe_locations:
        return []

    time_window_s = Decimal(str(time_window_ms)) / Decimal('1000')
    match_tolerance = Decimal('0.001')

    nbe_times = [l['time_dec'] for l in nbe_locations] if nbe_locations else []
    lightning_times = [l['time_dec'] for l in lightnings] if lightnings else []

    # NBE 时间集合（±5ms）
    nbe_time_ms_set = set()
    if nbe_locations:
        for nb in nbe_locations:
            t_ms = int(nb['time_dec'] * 1000)
            for delta in range(-5, 6):
                nbe_time_ms_set.add(t_ms + delta)

    results = []
    total = len(lig_pieces)

    for idx, piece in enumerate(lig_pieces):
        if progress_cb and idx % 50 == 0:
            progress_cb(f"独立判断 {idx+1}/{total}", int((idx + 1) / max(total, 1) * 100))

        piece_time = piece['final_time_dec']

        # 匹配参考闪电
        ref_lat, ref_lon, ref_time = None, None, None

        if nbe_locations:
            pos = bisect.bisect_left(nbe_times, piece_time - match_tolerance)
            best_match = None
            best_dt = None
            for j in range(pos, min(pos + 10, len(nbe_locations))):
                dt = abs(piece_time - nbe_locations[j]['time_dec'])
                if dt > match_tolerance:
                    if nbe_locations[j]['time_dec'] > piece_time + match_tolerance:
                        break
                    continue
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_match = nbe_locations[j]
            if best_match:
                ref_lat, ref_lon = best_match['lat'], best_match['lon']
                ref_time = best_match['time_dec']

        if ref_lat is None and lightnings:
            pos = bisect.bisect_left(lightning_times, piece_time - match_tolerance)
            best_match = None
            best_dt = None
            for j in range(pos, min(pos + 10, len(lightnings))):
                dt = abs(piece_time - lightnings[j]['time_dec'])
                if dt > match_tolerance:
                    if lightnings[j]['time_dec'] > piece_time + match_tolerance:
                        break
                    continue
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_match = lightnings[j]
            if best_match:
                ref_lat, ref_lon = best_match['lat'], best_match['lon']
                ref_time = best_match['time_dec']

        if ref_lat is None:
            results.append({
                'final_time': piece['final_time'],
                'is_independent': False, 'nearby_count': -1,
                'nearby_lightnings': [], 'category': piece.get('category', ''),
            })
            continue

        # 在时间/距离窗口内搜索
        nearby = []
        if lightnings:
            t_min = ref_time - time_window_s
            t_max = ref_time + time_window_s
            p_min = bisect.bisect_left(lightning_times, t_min)
            p_max = bisect.bisect_right(lightning_times, t_max)
            for j in range(p_min, p_max):
                lw = lightnings[j]
                dist = haversine_distance(lw['lat'], lw['lon'], ref_lat, ref_lon)
                time_diff_ms = abs(float(lw['time_dec'] - ref_time) * 1000)

                if dist < 0.5 and time_diff_ms < 5.0:
                    continue  # 自身

                lw_time_ms = int(lw['time_dec'] * 1000)
                if lw_time_ms in nbe_time_ms_set:
                    continue  # 排除 NNBE

                if dist <= distance_window_km:
                    nearby.append({
                        'final_time': lw['time'],
                        'distance_km': round(dist, 3),
                        'time_diff_ms': round(time_diff_ms, 3),
                        'source': '闪电定位',
                    })

        is_independent = len(nearby) == 0
        results.append({
            'final_time': piece['final_time'],
            'is_independent': is_independent,
            'nearby_count': len(nearby),
            'nearby_lightnings': nearby,
            'category': piece.get('category', ''),
        })

    if progress_cb:
        indep_cnt = sum(1 for r in results if r['is_independent'])
        progress_cb(f"独立判断完成: {indep_cnt}/{len(results)} 独立", 100)
    return results


# ============================================================================
#                          综合分析
# ============================================================================

def run_full_analysis(lig_dir, lightning_dir, nbe_loc_file=None,
                      station_lat=None, station_lon=None,
                      time_window_ms=660, distance_window_km=10,
                      progress_cb=None):
    """运行完整分析流程"""
    results = {
        'lig_pieces': [], 'categories': {}, 'lightnings': [],
        'nbe_locations': [], 'distance_results': [], 'current_results': [],
        'independent_results': [], 'category_distance': {},
        'category_current': {}, 'category_independent': {}, 'summary': {},
    }

    # 1. 按分类加载
    if progress_cb:
        progress_cb("步骤 1: 加载 lig 数据...", 0)
    categories, all_pieces = load_lig_pieces_by_category(lig_dir, progress_cb)
    results['lig_pieces'] = all_pieces
    results['categories'] = categories

    if station_lat is None and all_pieces:
        station_lat = all_pieces[0]['station_lat']
        station_lon = all_pieces[0]['station_lon']

    # 2. 加载闪电定位数据
    if progress_cb:
        progress_cb("步骤 2: 加载闪电定位数据...", 0)
    lightnings = load_xlsx_lightning_dir(lightning_dir, progress_cb, utc_hour_range=(15, 20))
    results['lightnings'] = lightnings

    nbe_locations = None
    if nbe_loc_file:
        if progress_cb:
            progress_cb("步骤 2.5: 加载 NBE 定位数据...", 0)
        nbe_locations = load_nbe_loc_file(nbe_loc_file, progress_cb)
        results['nbe_locations'] = nbe_locations

    # 3-5 各项分析
    if progress_cb:
        progress_cb("步骤 3: 距离分布分析...", 0)
    results['distance_results'] = analyse_distance_distribution(
        all_pieces, lightnings, nbe_locations, station_lat, station_lon, 0.011, 3500, progress_cb)

    if progress_cb:
        progress_cb("步骤 4: 电流强度分析...", 0)
    results['current_results'] = analyse_current_distribution(all_pieces, progress_cb)

    if progress_cb:
        progress_cb("步骤 5: 独立分布判断...", 0)
    results['independent_results'] = analyse_independent_distribution(
        all_pieces, lightnings, nbe_locations, time_window_ms, distance_window_km, progress_cb)

    # 6. 分类统计
    for cat_name in categories:
        results['category_distance'][cat_name] = [
            r for r in results['distance_results'] if r.get('category') == cat_name]
        results['category_current'][cat_name] = [
            r for r in results['current_results'] if r.get('category') == cat_name]
        results['category_independent'][cat_name] = [
            r for r in results['independent_results'] if r.get('category') == cat_name]

    # 7. 汇总统计
    summary = {}
    category_counts = {name: len(pieces) for name, pieces in categories.items()}
    total_count = sum(category_counts.values())
    summary['category'] = {
        'counts': category_counts, 'total': total_count,
        'ratios': {n: round(c / max(total_count, 1) * 100, 2) for n, c in category_counts.items()},
    }

    if results['distance_results']:
        dists = [r['distance_km'] for r in results['distance_results']]
        summary['distance'] = {
            'count': len(dists), 'min': round(min(dists), 3), 'max': round(max(dists), 3),
            'mean': round(np.mean(dists), 3), 'median': round(np.median(dists), 3),
            'std': round(np.std(dists), 3),
        }

    if results['current_results']:
        volts = [r['peak_voltage'] for r in results['current_results']]
        summary['current'] = {
            'count': len(volts), 'min': round(min(volts), 6), 'max': round(max(volts), 6),
            'mean': round(np.mean(volts), 6), 'median': round(np.median(volts), 6),
            'std': round(np.std(volts), 6),
        }

    if results['independent_results']:
        indep_cnt = sum(1 for r in results['independent_results'] if r['is_independent'])
        summary['independent'] = {
            'total': len(results['independent_results']),
            'independent_count': indep_cnt,
            'dependent_count': len(results['independent_results']) - indep_cnt,
            'independent_ratio': round(indep_cnt / max(len(results['independent_results']), 1) * 100, 2),
        }

    # 分类详细
    category_summary = {}
    for cat_name in categories:
        cs = {'count': category_counts[cat_name]}

        cat_dist = results['category_distance'].get(cat_name, [])
        if cat_dist:
            d = [r['distance_km'] for r in cat_dist]
            cs['distance'] = {
                'mean': round(np.mean(d), 3), 'median': round(np.median(d), 3),
                'std': round(np.std(d), 3),
            }

        cat_curr = results['category_current'].get(cat_name, [])
        if cat_curr:
            v = [r['peak_voltage'] for r in cat_curr]
            cs['current'] = {
                'mean': round(np.mean(v), 6), 'median': round(np.median(v), 6),
                'std': round(np.std(v), 6),
            }

        cat_indep = results['category_independent'].get(cat_name, [])
        if cat_indep:
            ic = sum(1 for r in cat_indep if r['is_independent'])
            cs['independent'] = {
                'independent_count': ic,
                'dependent_count': len(cat_indep) - ic,
                'independent_ratio': round(ic / max(len(cat_indep), 1) * 100, 2),
            }

        category_summary[cat_name] = cs

    summary['category_summary'] = category_summary

    if all_pieces:
        summary['station'] = {
            'name': all_pieces[0]['station_name'], 'lat': station_lat, 'lon': station_lon,
        }

    results['summary'] = summary

    if progress_cb:
        progress_cb("分析完成！", 100)

    return results