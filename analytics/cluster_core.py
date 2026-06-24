#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analytics / cluster_core — 闪电波形聚类核心逻辑

移植自 LigCluster 项目的 cluster_core.py
功能：波形特征提取、无监督聚类 (K-Means/DBSCAN/层次/GMM)、
      聚类评估、降维可视化 (t-SNE/PCA/UMAP)、聚类结果导出。
"""

import os
from decimal import Decimal

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from lig_parser import (
    ReadLigFile, ButterFilter, CutPieceTo16000, repacklig, PieceWriter,
    compute_final_time, compute_peak_voltage, voltage_from_piece,
    format_time_display, time_classifier_display,
    _resource_path,
)


# ============================================================================
#                          数据加载
# ============================================================================

def load_lig_pieces(lig_dir, progress_cb=None):
    """从目录加载所有 lig 文件的波形片段

    返回: list of dict，每个包含:
        - time_key, final_time, piece_data, peak_voltage, voltage
        - station_lat, station_lon
    """
    lig_files = []
    for root, dirs, files in os.walk(lig_dir):
        for f in files:
            if f.lower().endswith('.lig'):
                lig_files.append(os.path.join(root, f))

    all_pieces = []
    total_files = len(lig_files)

    for file_idx, ligpath in enumerate(lig_files):
        if progress_cb:
            progress_cb(f"加载 {os.path.basename(ligpath)} ({file_idx+1}/{total_files})",
                        int((file_idx + 1) / max(total_files, 1) * 100))
        try:
            lig_data = ReadLigFile(ligpath)
        except Exception as e:
            if progress_cb:
                progress_cb(f"[错误] {ligpath}: {e}", -1)
            continue

        for time_key, piece_data in lig_data.items():
            try:
                if '0' not in piece_data:
                    continue
                piece = np.array(piece_data['0'])
                final_time, _, _ = compute_final_time(time_key, piece)
                peak_v = compute_peak_voltage(piece_data)
                voltage = voltage_from_piece(piece_data)

                all_pieces.append({
                    'time_key': time_key,
                    'final_time': final_time,
                    'piece_data': piece_data,
                    'peak_voltage': peak_v,
                    'voltage': voltage,
                    'station_lat': piece_data.get('m_GPSCurrentLocationLat', 0),
                    'station_lon': piece_data.get('m_GPSCurrentLocationLon', 0),
                })
            except Exception as e:
                if progress_cb:
                    progress_cb(f"[警告] 片段 {time_key}: {e}", -1)

    all_pieces.sort(key=lambda x: Decimal(x['final_time']))
    return all_pieces


# ============================================================================
#                          波形特征提取
# ============================================================================

def extract_waveform_features(voltage, fs=5000000):
    """从电压波形提取 16 维手工特征"""
    v = np.asarray(voltage, dtype=np.float64)
    v_abs = np.abs(v)

    peak_pos = np.max(v)
    peak_neg = np.min(v)
    peak_abs = np.max(v_abs)
    mean_v = np.mean(v)
    std_v = np.std(v)
    peak_idx = np.argmax(v_abs)

    # 上升/下降时间
    rise_time = 0.0
    fall_time = 0.0
    try:
        if peak_pos > -peak_neg:
            threshold_10 = peak_pos * 0.1
            threshold_90 = peak_pos * 0.9
            rise_indices = np.where(v[:peak_idx+1] >= threshold_10)[0]
            rise_90_indices = np.where(v[:peak_idx+1] >= threshold_90)[0]
            if len(rise_indices) > 0 and len(rise_90_indices) > 0:
                rise_time = (rise_90_indices[0] - rise_indices[0]) / fs * 1e6
            fall_indices = np.where(v[peak_idx:] <= threshold_10)[0]
            fall_90_indices = np.where(v[peak_idx:] <= threshold_90)[0]
            if len(fall_indices) > 0 and len(fall_90_indices) > 0:
                fall_time = (fall_indices[0] - fall_90_indices[0]) / fs * 1e6
        else:
            threshold_10 = peak_neg * 0.1
            threshold_90 = peak_neg * 0.9
            rise_indices = np.where(v[:peak_idx+1] <= threshold_10)[0]
            rise_90_indices = np.where(v[:peak_idx+1] <= threshold_90)[0]
            if len(rise_indices) > 0 and len(rise_90_indices) > 0:
                rise_time = (rise_90_indices[0] - rise_indices[0]) / fs * 1e6
            fall_indices = np.where(v[peak_idx:] >= threshold_10)[0]
            fall_90_indices = np.where(v[peak_idx:] >= threshold_90)[0]
            if len(fall_indices) > 0 and len(fall_90_indices) > 0:
                fall_time = (fall_indices[0] - fall_90_indices[0]) / fs * 1e6
    except Exception:
        pass

    # 脉宽 (FWHM)
    pulse_width = 0.0
    try:
        half_max = peak_abs / 2
        above_half = np.where(v_abs >= half_max)[0]
        if len(above_half) > 0:
            pulse_width = (above_half[-1] - above_half[0]) / fs * 1e6
    except Exception:
        pass

    zero_crossings = np.sum(np.diff(np.sign(v)) != 0)

    kurtosis = 0.0
    skewness = 0.0
    try:
        from scipy.stats import kurtosis as _kurtosis, skew as _skew
        kurtosis = _kurtosis(v)
        skewness = _skew(v)
    except Exception:
        pass

    energy = np.sum(v ** 2)

    # 频谱特征
    spec_centroid = 0.0
    spec_bandwidth = 0.0
    try:
        fft_vals = np.abs(np.fft.rfft(v))
        freqs = np.fft.rfftfreq(len(v), 1.0 / fs)
        total_power = np.sum(fft_vals ** 2)
        if total_power > 0:
            spec_centroid = np.sum(freqs * fft_vals ** 2) / total_power
            spec_bandwidth = np.sqrt(np.sum(((freqs - spec_centroid) ** 2) * (fft_vals ** 2)) / total_power)
    except Exception:
        pass

    polarity = 1.0 if peak_pos > -peak_neg else -1.0

    overshoot_ratio = 0.0
    try:
        if polarity > 0:
            after_peak = v[peak_idx:]
            neg_overshoot = np.min(after_peak) if len(after_peak) > 0 else 0
            if peak_pos > 0:
                overshoot_ratio = abs(neg_overshoot) / peak_pos
        else:
            after_peak = v[peak_idx:]
            pos_overshoot = np.max(after_peak) if len(after_peak) > 0 else 0
            if abs(peak_neg) > 0:
                overshoot_ratio = pos_overshoot / abs(peak_neg)
    except Exception:
        pass

    return {
        'peak_pos': peak_pos, 'peak_neg': peak_neg, 'peak_abs': peak_abs,
        'std': std_v,
        'rise_time': rise_time, 'fall_time': fall_time, 'pulse_width': pulse_width,
        'zero_crossings': zero_crossings,
        'kurtosis': kurtosis, 'skewness': skewness,
        'energy': energy,
        'spec_centroid': spec_centroid, 'spec_bandwidth': spec_bandwidth,
        'polarity': polarity, 'overshoot_ratio': overshoot_ratio,
    }


def build_feature_matrix(pieces, feature_mode='handcraft', filter_fc=100000, fs=5000000,
                         progress_cb=None):
    """构建特征矩阵

    返回: (feature_matrix, valid_indices)
    """
    valid_indices = []
    handcraft_features = []
    raw_waveforms = []

    total = len(pieces)

    for idx, piece in enumerate(pieces):
        if progress_cb and idx % 50 == 0:
            progress_cb(f"特征提取 {idx+1}/{total}",
                        int((idx + 1) / max(total, 1) * 100))

        voltage = piece.get('voltage')
        if voltage is None:
            continue

        try:
            v_centered = voltage - np.mean(voltage)
            if filter_fc < fs / 2:
                fc_norm = filter_fc / (fs / 2)
                b, a = butter(4, fc_norm, btype='low')
                v_filtered = filtfilt(b, a, v_centered)
            else:
                v_filtered = v_centered

            v_cut = CutPieceTo16000(v_filtered)
            v_max = np.max(np.abs(v_cut))
            if v_max > 0:
                v_norm = v_cut / v_max
            else:
                continue

            if feature_mode in ('handcraft', 'combined'):
                feat = extract_waveform_features(v_cut, fs)
                handcraft_features.append(list(feat.values()))

            if feature_mode in ('raw', 'combined'):
                downsampled = v_norm[::len(v_norm) // 800][:800]
                if len(downsampled) < 800:
                    downsampled = np.pad(downsampled, (0, 800 - len(downsampled)))
                raw_waveforms.append(downsampled)

            valid_indices.append(idx)
        except Exception:
            continue

    if not valid_indices:
        return np.array([]), []

    if feature_mode == 'handcraft':
        feature_matrix = np.array(handcraft_features, dtype=np.float64)
    elif feature_mode == 'raw':
        raw_matrix = np.array(raw_waveforms, dtype=np.float64)
        n_components = min(50, raw_matrix.shape[0] - 1, raw_matrix.shape[1])
        pca = PCA(n_components=n_components)
        feature_matrix = pca.fit_transform(raw_matrix)
    elif feature_mode == 'combined':
        raw_matrix = np.array(raw_waveforms, dtype=np.float64)
        n_components = min(30, raw_matrix.shape[0] - 1, raw_matrix.shape[1])
        pca = PCA(n_components=n_components)
        raw_features = pca.fit_transform(raw_matrix)
        hc_matrix = np.array(handcraft_features, dtype=np.float64)
        feature_matrix = np.hstack([hc_matrix, raw_features])
    else:
        feature_matrix = np.array(handcraft_features, dtype=np.float64)

    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_matrix, valid_indices


# ============================================================================
#                          聚类算法
# ============================================================================

def run_kmeans(feature_matrix, n_clusters, **kwargs):
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42, **kwargs)
    labels = model.fit_predict(feature_matrix)
    return labels, model


def run_dbscan(feature_matrix, eps=0.5, min_samples=5, **kwargs):
    model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    labels = model.fit_predict(feature_matrix)
    return labels, model


def run_agglomerative(feature_matrix, n_clusters, linkage='ward', **kwargs):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **kwargs)
    labels = model.fit_predict(feature_matrix)
    return labels, model


def run_gmm(feature_matrix, n_clusters, **kwargs):
    model = GaussianMixture(n_components=n_clusters, random_state=42, **kwargs)
    labels = model.fit_predict(feature_matrix)
    return labels, model


def run_clustering(feature_matrix, algorithm='kmeans', n_clusters=3,
                   dbscan_eps=0.5, dbscan_min_samples=5,
                   agglomerative_linkage='ward',
                   progress_cb=None):
    if progress_cb:
        progress_cb(f"执行 {algorithm} 聚类...", 50)

    if algorithm == 'kmeans':
        labels, model = run_kmeans(feature_matrix, n_clusters)
    elif algorithm == 'dbscan':
        labels, model = run_dbscan(feature_matrix, eps=dbscan_eps, min_samples=dbscan_min_samples)
    elif algorithm == 'agglomerative':
        labels, model = run_agglomerative(feature_matrix, n_clusters, linkage=agglomerative_linkage)
    elif algorithm == 'gmm':
        labels, model = run_gmm(feature_matrix, n_clusters)
    else:
        labels, model = run_kmeans(feature_matrix, n_clusters)

    if progress_cb:
        progress_cb("聚类完成", 100)
    return labels, model


# ============================================================================
#                          聚类评估
# ============================================================================

def evaluate_clustering(feature_matrix, labels):
    unique_labels = set(labels)
    n_noise = np.sum(labels == -1)
    non_noise_labels = labels[labels != -1]
    n_clusters = len(set(non_noise_labels))

    result = {
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'cluster_sizes': {},
    }

    for label in sorted(unique_labels):
        if label == -1:
            result['cluster_sizes']['noise'] = int(np.sum(labels == -1))
        else:
            result['cluster_sizes'][f'cluster_{label}'] = int(np.sum(labels == label))

    if n_clusters >= 2:
        mask = labels != -1
        if np.sum(mask) >= n_clusters + 1:
            try:
                result['silhouette'] = float(silhouette_score(feature_matrix[mask], non_noise_labels))
            except Exception:
                result['silhouette'] = None
            try:
                result['calinski_harabasz'] = float(calinski_harabasz_score(feature_matrix[mask], non_noise_labels))
            except Exception:
                result['calinski_harabasz'] = None
            try:
                result['davies_bouldin'] = float(davies_bouldin_score(feature_matrix[mask], non_noise_labels))
            except Exception:
                result['davies_bouldin'] = None
    else:
        result['silhouette'] = None
        result['calinski_harabasz'] = None
        result['davies_bouldin'] = None

    return result


def find_optimal_k(feature_matrix, k_range=range(2, 11)):
    """基于轮廓系数搜索最优 k 值"""
    scores = {}
    for k in k_range:
        if k > feature_matrix.shape[0] - 1:
            break
        try:
            labels, _ = run_kmeans(feature_matrix, k)
            score = silhouette_score(feature_matrix, labels)
            scores[k] = score
        except Exception:
            scores[k] = None

    valid = {k: s for k, s in scores.items() if s is not None}
    best_k = max(valid, key=valid.get) if valid else 3
    return best_k, scores


# ============================================================================
#                          降维可视化
# ============================================================================

def compute_tsne(feature_matrix, perplexity=30, n_iter=1000, random_state=42, progress_cb=None):
    if progress_cb:
        progress_cb("计算 t-SNE 降维...", 30)
    n_samples = feature_matrix.shape[0]
    perp = min(perplexity, max(1, (n_samples - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perp, max_iter=n_iter, random_state=random_state)
    embedding = tsne.fit_transform(feature_matrix)
    if progress_cb:
        progress_cb("t-SNE 完成", 100)
    return embedding


def compute_pca_2d(feature_matrix, progress_cb=None):
    if progress_cb:
        progress_cb("计算 PCA 降维...", 50)
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(feature_matrix)
    if progress_cb:
        progress_cb("PCA 完成", 100)
    return embedding


def compute_umap_2d(feature_matrix, n_neighbors=15, min_dist=0.1, random_state=42, progress_cb=None):
    if progress_cb:
        progress_cb("计算 UMAP 降维...", 30)
    try:
        import umap
        n_samples = feature_matrix.shape[0]
        nn = min(n_neighbors, max(2, n_samples - 1))
        reducer = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=min_dist, random_state=random_state)
        embedding = reducer.fit_transform(feature_matrix)
        if progress_cb:
            progress_cb("UMAP 完成", 100)
        return embedding
    except ImportError:
        if progress_cb:
            progress_cb("UMAP 未安装，回退到 PCA", -1)
        return compute_pca_2d(feature_matrix, progress_cb)


# ============================================================================
#                          聚类结果导出
# ============================================================================

def export_clusters_to_lig(pieces, labels, valid_indices, output_dir,
                           lig_head_path=None, lig_file_head_path=None,
                           progress_cb=None):
    if lig_head_path is None:
        lig_head_path = _resource_path('Limitbyt')
    if lig_file_head_path is None:
        lig_file_head_path = _resource_path('LigHead.lig')

    os.makedirs(output_dir, exist_ok=True)

    cluster_groups = {}
    for i, idx in enumerate(valid_indices):
        label = labels[i]
        cluster_name = 'noise' if label == -1 else f'cluster_{label}'
        cluster_groups.setdefault(cluster_name, []).append(idx)

    stats = {}
    total = sum(len(v) for v in cluster_groups.values())
    processed = 0

    for cluster_name, indices in sorted(cluster_groups.items()):
        cluster_dir = os.path.join(output_dir, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)

        writer = PieceWriter(cluster_dir, '', lig_file_head_path)
        success_count = 0

        for idx in indices:
            processed += 1
            if progress_cb and processed % 20 == 0:
                progress_cb(f"导出 {cluster_name}: {processed}/{total}",
                            int(processed / max(total, 1) * 100))

            piece = pieces[idx]
            piece_data = piece['piece_data']
            time_key = piece['time_key']
            if '0' not in piece_data:
                continue

            pulse = np.array(piece_data['0'])
            packed = repacklig(pulse, time_key, lig_head_path)
            if packed is not None:
                writer.write(packed, time_key)
                success_count += 1

        writer.close()
        stats[cluster_name] = success_count

    if progress_cb:
        progress_cb(f"导出完成: {len(stats)} 类, {sum(stats.values())} 条", 100)
    return stats


def export_cluster_timestamps(pieces, labels, valid_indices, output_dir, progress_cb=None):
    os.makedirs(output_dir, exist_ok=True)

    cluster_groups = {}
    for i, idx in enumerate(valid_indices):
        label = labels[i]
        cluster_name = 'noise' if label == -1 else f'cluster_{label}'
        cluster_groups.setdefault(cluster_name, []).append(pieces[idx]['time_key'])

    for cluster_name, timestamps in sorted(cluster_groups.items()):
        with open(os.path.join(output_dir, f'{cluster_name}.txt'), 'w', encoding='utf-8') as f:
            for ts in sorted(timestamps):
                f.write(f"{ts}\n")

    if progress_cb:
        progress_cb(f"时间戳导出完成: {len(cluster_groups)} 类", 100)
    return cluster_groups


# ============================================================================
#                          完整聚类流程
# ============================================================================

def run_full_clustering(pieces, feature_mode='handcraft', algorithm='kmeans',
                        n_clusters=3, dbscan_eps=0.5, dbscan_min_samples=5,
                        agglomerative_linkage='ward', filter_fc=100000,
                        dim_reduction='tsne', export_dir=None,
                        lig_head_path=None, lig_file_head_path=None,
                        progress_cb=None):
    """执行完整聚类流程（含特征提取→聚类→评估→降维→导出）"""
    results = {}

    if progress_cb:
        progress_cb("步骤 1: 特征提取...", 0)
    feature_matrix, valid_indices = build_feature_matrix(
        pieces, feature_mode=feature_mode, filter_fc=filter_fc, progress_cb=progress_cb)
    if len(valid_indices) == 0:
        if progress_cb:
            progress_cb("无有效波形数据", -1)
        return None

    results['feature_matrix'] = feature_matrix
    results['valid_indices'] = valid_indices

    if progress_cb:
        progress_cb(f"特征提取: {len(valid_indices)} 条, {feature_matrix.shape[1]} 维", 30)

    if progress_cb:
        progress_cb("步骤 2: 执行聚类...", 40)
    labels, model = run_clustering(
        feature_matrix, algorithm=algorithm, n_clusters=n_clusters,
        dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
        agglomerative_linkage=agglomerative_linkage, progress_cb=progress_cb)
    results['labels'] = labels
    results['model'] = model

    if progress_cb:
        progress_cb("步骤 3: 评估质量...", 60)
    evaluation = evaluate_clustering(feature_matrix, labels)
    results['evaluation'] = evaluation

    if progress_cb:
        sil = evaluation.get('silhouette')
        sil_str = f"{sil:.4f}" if sil is not None else "N/A"
        progress_cb(f"聚类: {evaluation['n_clusters']} 类, 轮廓={sil_str}", 70)

    if progress_cb:
        progress_cb("步骤 4: 降维可视化...", 75)
    if dim_reduction == 'tsne':
        embedding = compute_tsne(feature_matrix, progress_cb=progress_cb)
    elif dim_reduction == 'umap':
        embedding = compute_umap_2d(feature_matrix, progress_cb=progress_cb)
    else:
        embedding = compute_pca_2d(feature_matrix, progress_cb=progress_cb)
    results['embedding'] = embedding

    if export_dir:
        if progress_cb:
            progress_cb("步骤 5: 导出结果...", 85)
        export_stats = export_clusters_to_lig(
            pieces, labels, valid_indices, export_dir,
            lig_head_path=lig_head_path, lig_file_head_path=lig_file_head_path,
            progress_cb=progress_cb)
        export_cluster_timestamps(pieces, labels, valid_indices, export_dir, progress_cb)
        results['export_stats'] = export_stats

    if progress_cb:
        progress_cb("聚类流程完成！", 100)
    return results