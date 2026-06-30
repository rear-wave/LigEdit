#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_module — 波形分类模块 (ONNX Runtime 版)
整合 ligClassify 的 ResNet1D 模型和预处理管线，为 LigEdit 提供：
  1. 单片段分类推理 (classify_single)
  2. 批量文件夹分类 (classify_folder)

使用 ONNX Runtime 推理，无需安装 PyTorch，包体积 ~30MB。
"""

import os
import sys
import csv
import logging

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
#                          预处理函数 (从 ligClassify 迁移)
# ============================================================================

def butterworth_filter(piece, fc=120000, fs=5000000, order=2):
    """Butterworth 低通滤波 (sos 形式，数值稳定)"""
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(order, fc / (fs / 2), btype='low', output='sos')
        return sosfiltfilt(sos, piece).astype(np.float32)
    except ImportError:
        return piece.astype(np.float32)


def cut_around_peak(piece, before=2000, after=6000, target_length=8000):
    """以最大值为中心裁剪波形"""
    index_max = int(np.argmax(piece))
    begin = index_max - before
    end = index_max + after
    if begin < 0:
        begin = 0
        end = target_length
    elif end > len(piece):
        end = len(piece)
        begin = end - target_length
    cut = piece[begin:end]
    if len(cut) < target_length:
        cut = np.pad(cut, (0, target_length - len(cut)), 'constant')
    elif len(cut) > target_length:
        cut = cut[:target_length]
    return cut


def normalize_minmax(piece):
    """MinMax 归一化: (x - mean) / (max - min)"""
    pmin, pmax = piece.min(), piece.max()
    if pmax - pmin < 1e-8:
        return (piece - piece.mean()).astype(np.float32)
    return ((piece - piece.mean()) / (pmax - pmin)).astype(np.float32)


def preprocess_waveform(piece, use_filter=True, cut_peak=True,
                        target_length=8000, normalize_mode='minmax'):
    """完整预处理流水线: 滤波 → 峰值裁剪 → 归一化"""
    piece = piece.astype(np.float32)
    if use_filter:
        piece = butterworth_filter(piece)
    if cut_peak:
        piece = cut_around_peak(piece, target_length=target_length)
    if normalize_mode == 'minmax':
        piece = normalize_minmax(piece)
    elif normalize_mode == 'zscore':
        std = piece.std()
        if std > 1e-8:
            piece = ((piece - piece.mean()) / std).astype(np.float32)
    return piece


# ============================================================================
#                          Batch 预处理 (批量加速)
# ============================================================================

def butterworth_filter_batch(pieces, fc=120000, fs=5000000, order=2):
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(order, fc / (fs / 2), btype='low', output='sos')
        return sosfiltfilt(sos, pieces, axis=-1).astype(np.float32)
    except ImportError:
        return pieces.astype(np.float32)


def cut_around_peak_batch(pieces, before=2000, after=6000, target_length=8000):
    N, T = pieces.shape
    peak_indices = np.argmax(pieces, axis=1).astype(np.int64)
    begins = peak_indices - before
    ends = peak_indices + after
    clamp_begin = (begins < 0)
    begins[clamp_begin] = 0
    ends[clamp_begin] = target_length
    clamp_end = (ends > T)
    ends[clamp_end] = T
    begins[clamp_end] = ends[clamp_end] - target_length
    result = np.empty((N, target_length), dtype=np.float32)
    for i in range(N):
        seg = pieces[i, begins[i]:ends[i]]
        L = len(seg)
        if L < target_length:
            result[i, :L] = seg
            result[i, L:] = 0.0
        else:
            result[i] = seg[:target_length]
    return result


def normalize_minmax_batch(pieces):
    pmin = pieces.min(axis=1, keepdims=True)
    pmax = pieces.max(axis=1, keepdims=True)
    pmean = pieces.mean(axis=1, keepdims=True)
    denom = pmax - pmin
    denom[denom < 1e-8] = 1.0
    return ((pieces - pmean) / denom).astype(np.float32)


def preprocess_batch(pieces, use_filter=True, cut_peak=True,
                     target_length=8000, normalize_mode='minmax'):
    if pieces.ndim == 1:
        pieces = pieces.reshape(1, -1)
    if use_filter:
        pieces = butterworth_filter_batch(pieces)
    if cut_peak:
        pieces = cut_around_peak_batch(pieces, target_length=target_length)
    if normalize_mode == 'minmax':
        pieces = normalize_minmax_batch(pieces)
    return pieces


# ============================================================================
#                          模型加载 (ONNX Runtime 懒加载单例)
# ============================================================================

_session = None
_class_names = None

# 默认类别名 (硬编码，与 checkpoint 中一致)
_DEFAULT_CLASS_NAMES = ['IC', 'NCG', 'NNBE', 'PCG', 'PNBE']


def _get_resource_path(relative_path):
    """获取资源文件路径，兼容 PyInstaller 打包"""
    import sys
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base_path, relative_path))


def load_model(checkpoint_path=None):
    """
    懒加载分类模型 (ONNX Runtime 单例)。

    Args:
        checkpoint_path: ONNX 模型文件路径 (.onnx)，默认使用 checkpoints/resnet.onnx

    Returns:
        (session, class_names, 'cpu')
    """
    global _session, _class_names

    if _session is not None:
        return _session, _class_names, 'cpu'

    try:
        # PyInstaller 打包后，.pyd 加载时需要能找到同级目录下的 DLL
        if getattr(sys, 'frozen', False):
            ort_capi_dir = os.path.join(sys._MEIPASS, 'onnxruntime', 'capi')
            if os.path.isdir(ort_capi_dir):
                os.add_dll_directory(ort_capi_dir)
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(f"onnxruntime 未安装，请 pip install onnxruntime (详情: {e})")

    if checkpoint_path is None:
        checkpoint_path = _get_resource_path("checkpoints/resnet.onnx")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"ONNX 模型不存在: {checkpoint_path}")

    logger.info(f"加载 ONNX 模型: {checkpoint_path}")
    _session = ort.InferenceSession(checkpoint_path, providers=['CPUExecutionProvider'])
    _class_names = _DEFAULT_CLASS_NAMES

    logger.info(f"ONNX 模型加载完成: {len(_class_names)} 类 {_class_names}")
    return _session, _class_names, 'cpu'


def _ensure_model(checkpoint_path=None):
    """确保模型已加载"""
    if _session is None:
        return load_model(checkpoint_path)
    return _session, _class_names, 'cpu'


def is_model_loaded():
    """检查模型是否已加载"""
    return _session is not None


# ============================================================================
#                          Softmax 工具 (纯 numpy)
# ============================================================================

def _softmax(logits):
    """稳定 softmax: exp(x-max) / sum(exp(x-max))"""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# ============================================================================
#                          单片段分类
# ============================================================================

def classify_single(waveform, checkpoint_path=None):
    """
    对单个波形片段进行分类。

    Args:
        waveform: (T,) float64/float32 原始波形数据
        checkpoint_path: ONNX 模型路径（首次调用时使用）

    Returns:
        (class_name, confidence): 预测类别名和置信度 (0~1)
        例如: ("NCG", 0.8723)
    """
    session, class_names, _ = _ensure_model(checkpoint_path)

    wf = preprocess_waveform(waveform, normalize_mode='minmax')
    x = wf.reshape(1, 1, -1).astype(np.float32)
    logits = session.run(None, {'waveform': x})[0]
    probs = _softmax(logits)[0]

    pred_idx = int(probs.argmax())
    return class_names[pred_idx], float(probs[pred_idx])


def classify_single_with_probs(waveform, checkpoint_path=None):
    """
    对单个波形片段进行分类，返回完整概率分布。

    Returns:
        (class_name, confidence, {class: prob})
    """
    session, class_names, _ = _ensure_model(checkpoint_path)

    wf = preprocess_waveform(waveform, normalize_mode='minmax')
    x = wf.reshape(1, 1, -1).astype(np.float32)
    logits = session.run(None, {'waveform': x})[0]
    probs = _softmax(logits)[0]

    pred_idx = int(probs.argmax())
    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return class_names[pred_idx], float(probs[pred_idx]), prob_dict


# ============================================================================
#                          批量文件夹分类
# ============================================================================

def classify_folder(input_dir, output_dir=None, checkpoint_path=None,
                    batch_size=256, max_pieces=None,
                    progress_cb=None, log_cb=None):
    """
    对文件夹中所有 .lig 文件的波形片段进行分类。

    Args:
        input_dir:      输入目录（递归搜索 .lig 文件）
        output_dir:     输出目录（默认: input_dir/classified/），保存 summary.csv
        checkpoint_path:ONNX 模型路径
        batch_size:     推理批次大小
        max_pieces:     最大处理片段数
        progress_cb:    进度回调 (step, message, percent)
        log_cb:         日志回调 (message)

    Returns:
        dict: 分类汇总
    """
    from lig_parser import ReadLigFile

    session, class_names, _ = _ensure_model(checkpoint_path)

    if output_dir is None:
        output_dir = os.path.join(input_dir, "classified")
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有 .lig 文件
    lig_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.lig'):
                lig_files.append(os.path.join(root, f))

    if not lig_files:
        if log_cb:
            log_cb("[终止] 未找到 .lig 文件")
        return {}

    logger.info(f"找到 {len(lig_files)} 个 .lig 文件")

    # 收集所有波形片段
    if progress_cb:
        progress_cb(0, "读取 .lig 文件...", 0)

    all_waveforms = []
    all_meta = []

    for file_idx, lig_file in enumerate(lig_files):
        if progress_cb:
            progress_cb(0, f"读取文件 {file_idx+1}/{len(lig_files)}",
                        int((file_idx + 1) / max(len(lig_files), 1) * 30))
        try:
            lig_data = ReadLigFile(lig_file)
            for time_key, piece_data in lig_data.items():
                if '0' not in piece_data:
                    continue
                wf = np.array(piece_data['0'], dtype=np.float64)
                all_waveforms.append(wf)
                all_meta.append((os.path.basename(lig_file), -1, time_key))
                if max_pieces and len(all_waveforms) >= max_pieces:
                    break
        except Exception as e:
            if log_cb:
                log_cb(f"[错误] 读取 {lig_file}: {e}")
        if max_pieces and len(all_waveforms) >= max_pieces:
            break

    total = len(all_waveforms)
    if total == 0:
        if log_cb:
            log_cb("[终止] 未读取到任何波形数据")
        return {}

    logger.info(f"共读取 {total} 个波形片段")

    if progress_cb:
        progress_cb(0, f"共 {total} 个片段，开始分类...", 30)

    csv_path = os.path.join(output_dir, "summary.csv")
    counts = {c: 0 for c in class_names}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "piece_index", "time", "predicted_class",
                     "confidence"] + [f"prob_{c}" for c in class_names])

        for start in range(0, total, 5000):
            end = min(start + 5000, total)
            chunk_waveforms = all_waveforms[start:end]

            max_len = max(len(wf) for wf in chunk_waveforms)
            wf_array = np.zeros((len(chunk_waveforms), max_len), dtype=np.float32)
            for i, wf in enumerate(chunk_waveforms):
                L = min(len(wf), max_len)
                wf_array[i, :L] = wf[:L]

            wf_proc = preprocess_batch(wf_array, normalize_mode='minmax')

            # ONNX Runtime 批量推理
            for s in range(0, len(wf_proc), batch_size):
                e = min(s + batch_size, len(wf_proc))
                x = wf_proc[s:e].reshape(-1, 1, 8000).astype(np.float32)
                logits = session.run(None, {'waveform': x})[0]
                probs = _softmax(logits)

                for i, gi in enumerate(range(start + s, start + e)):
                    cls = class_names[probs[i].argmax()]
                    counts[cls] += 1
                    fname, piece_idx, time_key = all_meta[gi]
                    w.writerow(
                        [fname, gi, time_key, cls,
                         f"{probs[i].max():.4f}"] +
                        [f"{probs[i, j]:.4f}" for j in range(len(class_names))]
                    )

            del wf_array, wf_proc
            pct = 30 + int((end / max(total, 1)) * 70)
            if progress_cb:
                progress_cb(0, f"分类中... {end}/{total}", pct)

    summary_lines = []
    summary_lines.append(f"{'='*50}")
    for c in class_names:
        n = counts.get(c, 0)
        pct_val = n / max(sum(counts.values()), 1) * 100
        summary_lines.append(f"  {c:<8}: {n:>7d}  ({pct_val:5.1f}%)")
    summary_lines.append(f"{'='*50}")
    summary_lines.append(f"结果保存至: {csv_path}")

    for line in summary_lines:
        logger.info(line)
        if log_cb:
            log_cb(line)

    if progress_cb:
        progress_cb(0, "分类完成！", 100)

    return counts


# ============================================================================
#                          批量原始波形分类 (用于文件加载时预分类)
# ============================================================================

def classify_batch_raw(waveforms, checkpoint_path=None, batch_size=256):
    """
    批量分类原始波形数据 — 适用于加载文件后一次性预分类所有片段。

    Args:
        waveforms: list of (T,) float64 原始波形数组
        checkpoint_path: ONNX 模型路径
        batch_size: 推理批次大小

    Returns:
        list of (class_name, confidence)
    """
    session, class_names, _ = _ensure_model(checkpoint_path)

    if not waveforms:
        return []

    # Padding 到相同长度
    max_len = max(len(wf) for wf in waveforms)
    wf_array = np.zeros((len(waveforms), max_len), dtype=np.float32)
    for i, wf in enumerate(waveforms):
        L = min(len(wf), max_len)
        wf_array[i, :L] = wf[:L]

    wf_proc = preprocess_batch(wf_array, normalize_mode='minmax')

    results = []
    N = len(wf_proc)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        x = wf_proc[s:e].reshape(-1, 1, 8000).astype(np.float32)
        logits = session.run(None, {'waveform': x})[0]
        probs = _softmax(logits)
        for i in range(e - s):
            cls = class_names[probs[i].argmax()]
            conf = float(probs[i].max())
            results.append((cls, conf))

    return results


# ============================================================================
#                          模型信息查询
# ============================================================================

def get_model_info(checkpoint_path=None):
    """获取模型信息（类别名等），不强制加载"""
    if _class_names is not None:
        return {"class_names": _class_names, "device": "cpu"}

    try:
        import onnx
        if checkpoint_path is None:
            checkpoint_path = _get_resource_path("checkpoints/resnet.onnx")
        if os.path.exists(checkpoint_path):
            return {"class_names": _DEFAULT_CLASS_NAMES, "device": "cpu"}
    except ImportError:
        pass

    return {"class_names": _DEFAULT_CLASS_NAMES, "device": "cpu"}
