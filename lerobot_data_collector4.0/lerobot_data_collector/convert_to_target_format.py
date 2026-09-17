#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量数据格式转换脚本
将原始采集数据转换为目标格式结构
"""

import os
import subprocess
import glob
import time
import shutil
import uuid
import json
import pandas as pd
from typing import List, Dict, Optional

# 导入对齐功能
try:
    from copasi_data_preprocess.align_multimodal import (
        align_streams, StreamTimestamps, read_timestamp_file, 
        read_parquet_timestamps, nearest_index
    )
    ALIGN_AVAILABLE = True
except ImportError:
    ALIGN_AVAILABLE = False
    print("⚠️  对齐功能不可用：copasi_data_preprocess 模块未找到")
import h5py
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
import argparse
import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from typing import Tuple, Optional, Dict
import contextlib

# --- CUDA 环境自修复（为 CuPy 提供 NVRTC 等库路径） ---
def _ensure_cuda_env_vars():
    try:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix and os.path.isdir(conda_prefix):
            if "CUDA_PATH" not in os.environ:
                os.environ["CUDA_PATH"] = conda_prefix
            lib_dir = os.path.join(conda_prefix, "lib")
            if os.path.isdir(lib_dir):
                ld = os.environ.get("LD_LIBRARY_PATH", "")
                parts = [p for p in ld.split(":") if p]
                if lib_dir not in parts:
                    os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + ld if ld else "")
    except Exception:
        pass
import wave

# 尝试导入cv2，如果不存在则跳过深度数据检查
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("⚠️  cv2未安装，将跳过深度数据检查")
    CV2_AVAILABLE = False
# 导入配置文件
from config import *

# 当前实际处理的数据集根目录（批量模式下为匹配到的子目录；单次模式为 DATASET_ROOT）
CURRENT_DATASET_ROOT = DATASET_ROOT
# 当前转换报告文件路径（用于权限问题时的备用路径）
CURRENT_REPORT_PATH = None
# 可选：外部注入的 episode_index -> uuid 预设映射（用于强制复用UUID，例如从低清报告/标注导入）
INJECTED_PRESET_UUID_MAP: Optional[Dict[int, str]] = None
# 可选：外部注入的标注目录列表（优先从这些目录查找 task_info 下的标注JSON）
INJECTED_ANNOTATION_DIRS: Optional[list[str]] = None

def _extract_cutoff_frames(anno_obj: object) -> Optional[int]:
    """从标注对象中尽可能鲁棒地提取最后一条动作的 end_frame。
    支持的数据形态：
    - dict: { label_info: { action_config: [ {end_frame:int}, ... ] } }
    - dict: { actions: [ {end_frame:int}, ... ] }
    - list: [ 上述dict, ... ] → 取最后一个中可解析到的 end_frame；若都可解析取最大 end_frame
    返回 N = end_frame + 1（包含截止帧）；解析失败返回 None。
    """
    try:
        def _from_dict(d: dict) -> Optional[int]:
            # label_info.action_config
            try:
                acts = d.get('label_info', {}).get('action_config', [])
                if isinstance(acts, list) and acts:
                    end_f = acts[-1].get('end_frame')
                    if end_f is not None:
                        return int(end_f) + 1
            except Exception:
                pass
            # actions
            try:
                acts2 = d.get('actions', [])
                if isinstance(acts2, list) and acts2:
                    end_f = acts2[-1].get('end_frame')
                    if end_f is not None:
                        return int(end_f) + 1
            except Exception:
                pass
            return None

        # dict 形态
        if isinstance(anno_obj, dict):
            return _from_dict(anno_obj)
        # list 形态
        if isinstance(anno_obj, list):
            candidates: list[int] = []
            for item in anno_obj:
                if isinstance(item, dict):
                    n = _from_dict(item)
                    if isinstance(n, int):
                        candidates.append(n)
            if candidates:
                return max(candidates)
        return None
    except Exception:
        return None

def check_align_availability(episode_dir: str) -> bool:
    """检查episode是否支持多模态对齐"""
    if not ALIGN_AVAILABLE:
        return False
    
    from config import ALIGN_ENABLED
    if not ALIGN_ENABLED:
        return False
    
    # 检查三路时间戳文件是否存在
    images_dir = os.path.join(episode_dir, "images")
    if not os.path.exists(images_dir):
        return False
    
    required_files = [
        "hand_l_rgb/hand_l_timestamps.txt",
        "hand_r_rgb/hand_r_timestamps.txt", 
        "headf_rgbd_color/headf_rgbd_timestamps.txt"
    ]
    
    for file_path in required_files:
        full_path = os.path.join(images_dir, file_path)
        if not os.path.exists(full_path):
            return False
    
    return True

def generate_aligned_frame_indices(episode_dir: str, parquet_path: str) -> Optional[List[dict]]:
    """生成对齐后的帧索引列表"""
    if not check_align_availability(episode_dir):
        return None
    
    try:
        from config import ALIGN_REF, ALIGN_MAX_DELTA_MS, ALIGN_MIN_RATIO
        
        images_dir = os.path.join(episode_dir, "images")
        
        # 读取三路时间戳
        from pathlib import Path
        hand_l_vals = read_timestamp_file(Path(os.path.join(images_dir, "hand_l_rgb", "hand_l_timestamps.txt")))
        hand_r_vals = read_timestamp_file(Path(os.path.join(images_dir, "hand_r_rgb", "hand_r_timestamps.txt")))
        headf_vals = read_timestamp_file(Path(os.path.join(images_dir, "headf_rgbd_color", "headf_rgbd_timestamps.txt")))
        
        # 创建StreamTimestamps对象
        hand_l = StreamTimestamps(name="hand_l", timestamps_ns=hand_l_vals)
        hand_r = StreamTimestamps(name="hand_r", timestamps_ns=hand_r_vals)
        headf = StreamTimestamps(name="headf", timestamps_ns=headf_vals)
        
        max_delta_ns = int(ALIGN_MAX_DELTA_MS * 1_000_000)
        
        if ALIGN_REF == 'parquet':
            # 以parquet为参考
            if not os.path.exists(parquet_path):
                print(f"⚠️  Parquet文件不存在，回退到原始方法: {parquet_path}")
                return None
            
            from pathlib import Path
            parquet_ts = read_parquet_timestamps(Path(parquet_path), "timestamps", strict_drop=False)
            parquet_stream = StreamTimestamps(name="parquet", timestamps_ns=parquet_ts)
            rows = align_streams(parquet_stream, [hand_l, hand_r, headf], max_delta_ns)
        else:
            # 以某个相机为参考
            if ALIGN_REF == 'headf':
                ref_stream = headf
                others = [hand_l, hand_r]
            elif ALIGN_REF == 'hand_l':
                ref_stream = hand_l
                others = [hand_r, headf]
            elif ALIGN_REF == 'hand_r':
                ref_stream = hand_r
                others = [hand_l, headf]
            else:
                print(f"⚠️  未知的参考流: {ALIGN_REF}，回退到原始方法")
                return None
            
            rows = align_streams(ref_stream, others, max_delta_ns)
        
        # 检查对齐效果
        if not rows:
            print("⚠️  对齐结果为空，回退到原始方法")
            return None
        
        # 计算保留率
        if ALIGN_REF == 'parquet':
            total_ref = len(parquet_ts)
        else:
            total_ref = len(ref_stream.timestamps_ns)
        
        ratio = len(rows) / total_ref
        if ratio < ALIGN_MIN_RATIO:
            print(f"⚠️  对齐保留率过低 ({ratio:.1%} < {ALIGN_MIN_RATIO:.1%})，回退到原始方法")
            return None
        
        print(f"✅ 多模态对齐成功: {len(rows)}/{total_ref} = {ratio:.1%}")
        return rows
        
    except Exception as e:
        print(f"⚠️  对齐过程出错，回退到原始方法: {e}")
        return None

def _extract_cutoff_frames_for_uuid(anno_obj: object, episode_uuid: str) -> Optional[int]:
    """从标注对象中为指定 uuid 提取其专属的截止帧 N（end_frame+1）。
    支持：
    - 顶层为 list，元素为 dict，包含 'episode_id' 与动作列表（与示例一致）
    - 顶层为 dict 且就是该 uuid 的对象（兼容单条写法）
    若找不到匹配 uuid，则返回 None。
    """
    try:
        def last_end_plus_one(d: dict) -> Optional[int]:
            acts = d.get('label_info', {}).get('action_config', [])
            if isinstance(acts, list) and acts:
                ef = acts[-1].get('end_frame')
                if ef is not None:
                    return int(ef) + 1
            return None

        # list 顶层：查找 episode_id 匹配项
        if isinstance(anno_obj, list):
            for item in anno_obj:
                if isinstance(item, dict) and str(item.get('episode_id')) == str(episode_uuid):
                    return last_end_plus_one(item)
            return None
        # dict 顶层：若 episode_id 匹配则解析
        if isinstance(anno_obj, dict):
            if str(anno_obj.get('episode_id')) == str(episode_uuid):
                return last_end_plus_one(anno_obj)
            return None
        return None
    except Exception:
        return None

def setup_dynamic_output_paths(dataset_root: str):
    """根据配置动态设置输出路径"""
    global OUTPUT_ROOT_PATH, OUTPUT_LOWRES_ROOT_PATH
    
    # 重新导入配置以确保获取最新值
    from config import USE_SOURCE_FILENAME_AS_OUTPUT, USE_ABSOLUTE_OUTPUT_PATHS
    from config import OUTPUT_ROOT_PATH_ABSOLUTE, OUTPUT_LOWRES_ROOT_PATH_ABSOLUTE
    from config import OUTPUT_DIR_NAME, OUTPUT_LOWRES_DIR_NAME
    
    if USE_SOURCE_FILENAME_AS_OUTPUT:
        # 使用源文件名作为输出文件夹名
        _ds_name = os.path.basename(dataset_root.rstrip('/'))
        
        if USE_ABSOLUTE_OUTPUT_PATHS:
            # 使用绝对路径配置，但替换文件夹名为源文件名
            # 高清使用 0_ 前缀，低清使用 1_ 前缀
            base_root = os.path.dirname(OUTPUT_ROOT_PATH_ABSOLUTE.rstrip('/'))
            OUTPUT_ROOT_PATH = os.path.join(base_root, '0_' + _ds_name)
            OUTPUT_LOWRES_ROOT_PATH = os.path.join(base_root, '1_' + _ds_name)
        else:
            # 基于数据集根目录
            OUTPUT_ROOT_PATH = os.path.join(dataset_root, '0_' + _ds_name)
            OUTPUT_LOWRES_ROOT_PATH = os.path.join(dataset_root, '1_' + _ds_name)
    else:
        # 使用固定文件夹名
        if USE_ABSOLUTE_OUTPUT_PATHS:
            OUTPUT_ROOT_PATH = OUTPUT_ROOT_PATH_ABSOLUTE
            OUTPUT_LOWRES_ROOT_PATH = OUTPUT_LOWRES_ROOT_PATH_ABSOLUTE
        else:
            OUTPUT_ROOT_PATH = os.path.join(dataset_root, OUTPUT_DIR_NAME)
            OUTPUT_LOWRES_ROOT_PATH = os.path.join(dataset_root, OUTPUT_LOWRES_DIR_NAME)

def _uuid_map_path(dataset_root: str) -> str:
    """返回源数据集根目录的UUID映射文件路径。"""
    return os.path.join(dataset_root, "uuid_mapping.json")

def load_uuid_mapping(dataset_root: str) -> Dict[int, str]:
    """从数据集根目录加载 episode_index -> uuid 的映射。"""
    try:
        path = _uuid_map_path(dataset_root)
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 键转为int
            return {int(k): str(v) for k, v in data.items()}
    except Exception as e:
        print(f"⚠️  读取UUID映射失败: {e}")
    return {}

def save_uuid_mapping(dataset_root: str, mapping: Dict[int, str]) -> None:
    """保存 episode_index -> uuid 的映射到数据集根目录。"""
    try:
        path = _uuid_map_path(dataset_root)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)
        print(f"📝 已保存UUID映射: {path}")
    except Exception as e:
        print(f"⚠️  保存UUID映射失败: {e}")

def copy_task_info_from_source(src_root: str, dst_logistics_root: str) -> None:
    """将源数据根目录下的 task_info 目录完整复制到输出的 Logistics/task_info。
    - 若目标已存在，直接覆盖（保留同名文件的最新内容）。
    - 支持携带标注JSON与异常报告txt。
    """
    try:
        import shutil as _shutil
        src_task = os.path.join(src_root, 'task_info')
        dst_task = os.path.join(dst_logistics_root, 'task_info')
        if not os.path.isdir(src_task):
            print(f"ℹ️  源 task_info 不存在，跳过替换: {src_task}")
            return
        os.makedirs(dst_task, exist_ok=True)
        # 覆盖复制
        for root, dirs, files in os.walk(src_task):
            rel = os.path.relpath(root, src_task)
            target_dir = os.path.join(dst_task, rel) if rel != '.' else dst_task
            os.makedirs(target_dir, exist_ok=True)
            for d in dirs:
                os.makedirs(os.path.join(target_dir, d), exist_ok=True)
            for f in files:
                _shutil.copy2(os.path.join(root, f), os.path.join(target_dir, f))
        print(f"✅ 已用源 task_info 覆盖输出: {dst_task}")
    except Exception as e:
        print(f"⚠️  替换 task_info 失败: {e}")

def load_uuid_mapping_from_report(dataset_root: str) -> Dict[int, str]:
    """从源目录中的低清转换报告提取episode->uuid映射（作为uuid_mapping.json缺失时的后备）。"""
    mapping: Dict[int, str] = {}
    try:
        report_path = os.path.join(dataset_root, "conversion_report_low_only.txt")
        if not os.path.isfile(report_path):
            return mapping
        with open(report_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 解析形如: episode_000123 | uuid=xxxxxxxx-... |
                if line.startswith('episode_') and '| uuid=' in line:
                    try:
                        left, _rest = line.split('| uuid=', 1)
                        uuid_part = _rest.split('|', 1)[0].strip()
                        ep_idx_str = left.replace('episode_', '').strip()
                        ep_idx = int(ep_idx_str)
                        mapping[ep_idx] = uuid_part
                    except Exception:
                        continue
                # 解析尾部的 "N -> uuid" 列表
                elif '->' in line and line.replace(' ', '').replace('->', '').replace('-', '').replace('_', '').isalnum():
                    try:
                        k, v = line.split('->', 1)
                        ep_idx = int(k.strip())
                        mapping[ep_idx] = v.strip()
                    except Exception:
                        continue
    except Exception as e:
        print(f"⚠️  从报告解析UUID映射失败: {e}")
    return mapping

def detect_param_profile_from_path(path: str) -> str | None:
    """从路径中检测设备标识，并映射到参数文件夹
    
    新版本：优先匹配顶层目录格式 {ROBOT_NAME}_lerobot
    兼容旧版本：支持复杂的正则表达式匹配
    
    支持的顶层目录格式：
    - T008_lerobot -> parameters_T008
    - T009_lerobot -> parameters_T009  
    - T010_lerobot -> parameters_T010
    - P53_lerobot -> parameters_P053
    - P55_lerobot -> parameters_P055
    - P64_lerobot -> parameters_P064
    
    返回参数文件夹绝对路径；若未匹配，返回 None
    """
    import re
    
    # 新版本：优先检查顶层目录格式 {ROBOT_NAME}_lerobot
    # 提取路径中的顶层目录名
    path_parts = path.split(os.sep)
    if path_parts:
        top_level_dir = path_parts[-1] if path_parts else ""
        
        # 检查是否为 {ROBOT_NAME}_lerobot 格式
        lerobot_patterns = {
            'T008_lerobot': 'parameters_T008',
            'T009_lerobot': 'parameters_T009', 
            'T010_lerobot': 'parameters_T010',
            'P53_lerobot': 'parameters_P053',
            'P55_lerobot': 'parameters_P055',
            'P64_lerobot': 'parameters_P064'
        }
        
        if top_level_dir in lerobot_patterns:
            folder = lerobot_patterns[top_level_dir]
            return os.path.join(CURRENT_WORKSPACE, folder)
    
    # 旧版本兼容：复杂的正则表达式匹配
    s = path
    # 先匹配规范形式 pNNN/tNNN（包含分隔符与大小写、两位自动补零）
    m = re.search(r'([pPtT])[\-_]?(\d{2,3})', s)
    if m:
        letter = m.group(1).lower()
        digits = m.group(2)
        if len(digits) == 2:
            digits = digits.zfill(3)
        key = f"{letter}{digits}"
        folder = PARAM_PROFILE_TO_FOLDER.get(key)
        if folder:
            return os.path.join(CURRENT_WORKSPACE, folder)
    
    # 兼容：连续4个字符窗口内，只要同时出现字母(p/t)和编号关键字即可
    # 规则映射：p -> {053,055,064,065}；t -> {008,010}
    s_low = s.lower()
    for i in range(len(s_low) - 3):
        win = s_low[i:i+4]
        if 'p' in win:
            for d in ['053','055','064','065']:
                if any(x in win for x in [d, d[-2:]]):  # 命中 3位或2位形式
                    key = f"p{d}"
                    folder = PARAM_PROFILE_TO_FOLDER.get(key)
                    if folder:
                        return os.path.join(CURRENT_WORKSPACE, folder)
        if 't' in win:
            for d in ['008','010']:
                if any(x in win for x in [d, d[-2:]]):
                    key = f"t{d}"
                    folder = PARAM_PROFILE_TO_FOLDER.get(key)
                    if folder:
                        return os.path.join(CURRENT_WORKSPACE, folder)
    
    return None

# 尝试导入pinocchio，如果不存在或版本不兼容则跳过运动学计算
try:
    import pinocchio
    PINOCCHIO_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"⚠️  pinocchio不可用（{e}），将跳过运动学计算")
    PINOCCHIO_AVAILABLE = False

# 记录本次运行的预检结果
LAST_PRECHECK_FAILED_EPISODES = set()
LAST_PRECHECK_RAN = False

def check_depth_data_from_images(png_files, sample_count=5):
    """检查深度图像数据的数值分布和物理意义"""
    # 检查cv2是否可用
    if not CV2_AVAILABLE:
        print("⚠️  cv2未安装，跳过深度数据检查")
        return True
    
    print(f"🔬 分析深度数据分布 (采样{sample_count}张图像)...")
    
    if not png_files:
        print("❌ 没有找到深度图像文件")
        return False
    
    # 采样分析
    sample_files = png_files[:sample_count] if len(png_files) > sample_count else png_files
    depth_values = []
    frame_stats = []
    
    print(f"📊 开始分析 {len(sample_files)} 张图像...")
    
    for i, png_file in enumerate(sample_files):
        try:
            # 读取图像
            img = cv2.imread(png_file, cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"⚠️  无法读取图像: {png_file}")
                continue
            
            # 转换为灰度图（如果是彩色深度图）
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 统计有效深度值（排除0值）
            valid_pixels = img[img > 0]
            if len(valid_pixels) > 0:
                depth_values.extend(valid_pixels.tolist())
                
                frame_min = np.min(valid_pixels)
                frame_max = np.max(valid_pixels)
                frame_mean = np.mean(valid_pixels)
                frame_std = np.std(valid_pixels)
                
                frame_stats.append({
                    'frame': i+1,
                    'min': frame_min,
                    'max': frame_max,
                    'mean': frame_mean,
                    'std': frame_std,
                    'valid_pixels': len(valid_pixels),
                    'total_pixels': img.size
                })
                
                # 每10帧显示一次进度
                if (i + 1) % 10 == 0:
                    print(f"   已分析 {i+1}/{len(sample_files)} 帧")
            else:
                print(f"   图像 {i+1}: 无有效深度数据")
                
        except Exception as e:
            print(f"⚠️  处理图像 {png_file} 时出错: {e}")
            continue
    
    if not depth_values:
        print("❌ 没有找到有效的深度数据")
        return False
    
    # 分析深度值分布
    depth_values = np.array(depth_values)
    min_val = np.min(depth_values)
    max_val = np.max(depth_values)
    mean_val = np.mean(depth_values)
    median_val = np.median(depth_values)
    std_val = np.std(depth_values)
    
    print(f"\n📊 深度数据统计 (基于{len(depth_values)}个有效像素):")
    print(f"   - 数值范围: [{min_val}, {max_val}]")
    print(f"   - 平均值: {mean_val:.2f}")
    print(f"   - 中位数: {median_val:.2f}")
    print(f"   - 标准差: {std_val:.2f}")
    
    # 分析无效深度值（65535）的比例
    invalid_count = np.sum(np.array(depth_values) == 65535)
    invalid_ratio = invalid_count / len(depth_values) * 100 if len(depth_values) > 0 else 0
    
    print(f"   - 无效深度值(65535): {invalid_count} 个 ({invalid_ratio:.2f}%)")
    
    # 分析有效深度值的分布
    valid_values = [v for v in depth_values if v != 65535 and v > 0]
    if valid_values:
        valid_values = np.array(valid_values)
        print(f"   - 有效深度值范围: [{np.min(valid_values)}, {np.max(valid_values)}]")
        print(f"   - 有效深度平均值: {np.mean(valid_values):.2f}")
        
        # 分析不同深度区间的分布
        print(f"\n📈 深度值分布分析:")
        ranges = [
            (0, 1000, "近距 (0-1m)"),
            (1000, 3000, "中距 (1-3m)"),
            (3000, 10000, "远距 (3-10m)"),
            (10000, 65535, "超远距 (>10m)"),
            (65535, 65535, "无效值 (=65535)")
        ]
        
        for min_range, max_range, label in ranges:
            if min_range == 65535:
                count = np.sum(np.array(depth_values) == 65535)
            else:
                count = np.sum((np.array(depth_values) >= min_range) & (np.array(depth_values) < max_range))
            ratio = count / len(depth_values) * 100 if len(depth_values) > 0 else 0
            print(f"   - {label}: {count} 个 ({ratio:.2f}%)")
    
    # 单位分析
    print(f"\n🔍 单位分析:")
    
    # 假设不同单位的情况
    units_analysis = [
        ("毫米 (mm)", 1, "正常范围: 100-10000mm"),
        ("厘米 (cm)", 10, "正常范围: 10-1000cm"),
        ("分米 (dm)", 100, "正常范围: 1-100dm"),
        ("米 (m)", 1000, "正常范围: 0.1-10m")
    ]
    
    issues_found = []
    
    for unit_name, multiplier, normal_range in units_analysis:
        converted_min = min_val / multiplier
        converted_max = max_val / multiplier
        converted_mean = mean_val / multiplier
        
        print(f"   {unit_name}:")
        print(f"     范围: [{converted_min:.2f}, {converted_max:.2f}]")
        print(f"     平均值: {converted_mean:.2f}")
        
        # 判断是否合理
        if unit_name == "毫米 (mm)":
            if 50 <= converted_min <= 200 and 5000 <= converted_max <= 15000:
                print(f"     ✅ 看起来合理")
            elif converted_max > 50000:
                print(f"     ⚠️  数值过大，可能需要转换")
                issues_found.append(f"数值过大 ({converted_max:.1f}mm)，建议转换为米")
            elif converted_max < 1000:
                print(f"     ⚠️  数值过小，可能需要转换")
                issues_found.append(f"数值过小 ({converted_max:.1f}mm)，建议检查单位")
        elif unit_name == "米 (m)":
            if 0.05 <= converted_min <= 0.2 and 0.5 <= converted_max <= 15:
                print(f"     ✅ 看起来合理")
            elif converted_max > 50:
                print(f"     ⚠️  数值过大，可能需要转换")
                issues_found.append(f"数值过大 ({converted_max:.1f}m)，建议检查单位")
            elif converted_max < 0.1:
                print(f"     ⚠️  数值过小，可能需要转换")
                issues_found.append(f"数值过小 ({converted_max:.1f}m)，建议转换为毫米")
        
        print(f"     评估: {normal_range}")
        print()
    
    # 输出建议
    if issues_found:
        print(f"🔧 发现的问题:")
        for issue in issues_found:
            print(f"   - {issue}")
        
        print(f"\n💡 修复建议:")
        if max_val > 50000:
            print(f"   - 如果当前是毫米单位，建议转换为米 (除以1000)")
            print(f"   - 如果当前是厘米单位，建议转换为毫米 (乘以10)")
        elif max_val < 1000:
            print(f"   - 如果当前是米单位，建议转换为毫米 (乘以1000)")
            print(f"   - 如果当前是分米单位，建议转换为毫米 (乘以100)")
        else:
            print(f"   - 数值范围看起来合理，但建议确认单位")
        
        # 基于100帧分析的具体建议
        if invalid_ratio > 10:
            print(f"   - 无效深度值比例较高 ({invalid_ratio:.1f}%)，建议检查传感器设置")
        if valid_values is not None and len(valid_values) > 0:
            valid_max = np.max(valid_values)
            if valid_max > 10000:
                print(f"   - 有效深度值过大 ({valid_max})，建议进行单位转换")
            elif valid_max < 1000:
                print(f"   - 有效深度值过小 ({valid_max})，建议检查单位")
        
        return False
    else:
        print(f"✅ 深度数据看起来正常")
        return True

def scan_depth_episodes(input_root_path):
    """扫描指定数据集的所有episode深度PNG，产出异常episode列表并写入FAILED_EPISODES_FILE。
    仅在cv2可用且DEPTH_CHECK_BEHAVIOR为skip/warn时有效。
    返回: set[str] (异常episode的绝对路径集合)
    """
    global LAST_PRECHECK_FAILED_EPISODES, LAST_PRECHECK_RAN
    failed = set()
    if not CV2_AVAILABLE:
        print("⚠️  cv2未安装，跳过深度数据扫描")
        return failed
    if DEPTH_CHECK_BEHAVIOR not in ("skip", "warn"):
        return failed

    try:
        episode_dirs = sorted([d for d in os.listdir(input_root_path)
                              if os.path.isdir(os.path.join(input_root_path, d)) and d.startswith('episode_')])
        if not episode_dirs:
            return failed

        print(f"🔎 深度监测：开始扫描 {len(episode_dirs)} 个episode ...")
        for ep in episode_dirs:
            ep_path = os.path.join(input_root_path, ep)
            images_dir = os.path.join(ep_path, "images")
            if not os.path.isdir(images_dir):
                continue

            # 针对每个配置的深度源目录进行检测（任一失败则标记该episode异常）
            episode_ok = True
            for src_dir in DEPTH_MAPPINGS.keys():
                src_path = os.path.join(images_dir, src_dir)
                if not os.path.isdir(src_path):
                    # 没有该深度源则跳过该源，不判失败
                    continue
                png_files = sorted(glob.glob(os.path.join(src_path, "*.png")))
                sample_files = png_files[:10] if len(png_files) > 10 else png_files
                if not check_depth_data_from_images(sample_files, sample_count=min(5, len(sample_files))):
                    episode_ok = False
                    break

            if not episode_ok:
                failed.add(ep_path)

        # 写入（或覆盖）异常列表文件
        try:
            with open(FAILED_EPISODES_FILE, 'w') as f:
                for item in sorted(failed):
                    f.write(item + "\n")
            print(f"📝 已更新异常episode列表: {FAILED_EPISODES_FILE} (共 {len(failed)} 条)")
        except Exception as e:
            print(f"⚠️  写入异常episode列表失败: {e}")

        # 记录本次结果
        LAST_PRECHECK_FAILED_EPISODES = set(failed)
        LAST_PRECHECK_RAN = True
        return failed
    except Exception as e:
        print(f"⚠️  深度监测扫描失败: {e}")
        LAST_PRECHECK_FAILED_EPISODES = set()
        LAST_PRECHECK_RAN = True
        return failed

# 添加ArmKinematics类定义
class ArmKinematics:
    def __init__(self, urdf_path, ee_frame_name="link_tcp_l"):
        self.urdf_path = urdf_path
        self.model = pinocchio.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        self.ee_frame_name = ee_frame_name
        self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)
        if self.ee_frame_id >= len(self.model.frames):
            print(f"URDF文件 {self.urdf_path} 中所有frame名称如下：")
            for i, f in enumerate(self.model.frames):
                print(f"  id={i}, name={f.name}, type={f.type}")
            raise ValueError(f"Frame '{self.ee_frame_name}' 不存在于URDF: {self.urdf_path}")
        self.end_effector_pos = np.zeros(3)
        self.end_effector_rot = np.eye(3)
        self.end_effector_quat = np.zeros(4)

    def forward_kinematics(self, qpos):
        pinocchio.forwardKinematics(self.model, self.data, qpos)
        pinocchio.updateFramePlacements(self.model, self.data)
        ee_pose = self.data.oMf[self.ee_frame_id]
        self.end_effector_pos = ee_pose.translation.copy()
        self.end_effector_rot = ee_pose.rotation.copy()
        self.end_effector_quat = self.rot_to_quat(ee_pose.rotation)

    def rot_to_quat(self, rot_matrix):
        qw = 0.5 * np.sqrt(1 + rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2])
        qx = (rot_matrix[2, 1] - rot_matrix[1, 2]) / (4 * qw)
        qy = (rot_matrix[0, 2] - rot_matrix[2, 0]) / (4 * qw)
        qz = (rot_matrix[1, 0] - rot_matrix[0, 1]) / (4 * qw)
        return np.array([qw, qx, qy, qz])

    def get_end_effector_pos(self):
        return self.end_effector_pos.copy()

    def get_end_effector_quat(self):
        return self.end_effector_quat.copy()

def compute_kinematics_batch(arm_kin, joint_positions, description=""):
    """批量计算运动学，带进度显示"""
    positions = []
    quaternions = []
    
    total_frames = len(joint_positions)
   # print(f"   {description}: 计算 {total_frames} 帧运动学...")
    
    # 每100帧显示一次进度
    for i, q in enumerate(joint_positions):
        arm_kin.forward_kinematics(q)
        positions.append(arm_kin.get_end_effector_pos().copy())
        quat = arm_kin.get_end_effector_quat()
        # 转为[x, y, z, w]顺序
        quaternions.append([quat[1], quat[2], quat[3], quat[0]])
        
        if (i + 1) % 100 == 0 or (i + 1) == total_frames:
            progress = (i + 1) / total_frames * 100
            #print(f"     进度: {i+1}/{total_frames} ({progress:.1f}%)")
    
    return np.array(positions), np.array(quaternions)

def compute_kinematics_with_velocity_batch(arm_kin, joint_positions, joint_velocities, description=""):
    """批量计算运动学和速度，带进度显示"""
    positions = []
    quaternions = []
    linear_velocities = []
    angular_velocities = []
    
    total_frames = len(joint_positions)
    #print(f"   {description}: 计算 {total_frames} 帧运动学和速度...")
    
    for i, (q, dq) in enumerate(zip(joint_positions, joint_velocities)):
        arm_kin.forward_kinematics(q)
        positions.append(arm_kin.get_end_effector_pos().copy())
        quat = arm_kin.get_end_effector_quat()
        quaternions.append([quat[1], quat[2], quat[3], quat[0]])
        
        # 计算末端执行器速度 - 需要先计算雅可比矩阵
        J = pinocchio.computeFrameJacobian(arm_kin.model, arm_kin.data, q, arm_kin.ee_frame_id, pinocchio.ReferenceFrame.LOCAL)
        # 使用雅可比矩阵计算末端速度
        end_vel = J @ dq
        linear_velocities.append(end_vel[:3].copy())  # 线性速度 (前3个元素)
        angular_velocities.append(end_vel[3:].copy())  # 角速度 (后3个元素)
        
        if (i + 1) % 100 == 0 or (i + 1) == total_frames:
            progress = (i + 1) / total_frames * 100
            #print(f"     进度: {i+1}/{total_frames} ({progress:.1f}%)")
    
    return np.array(positions), np.array(quaternions), np.array(linear_velocities), np.array(angular_velocities)

def check_gpu_encoder():
    """检查GPU编码器是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        if 'h264_nvenc' in result.stdout:
            # 检查CUDA库是否可用
            try:
                subprocess.run(['nvidia-smi'], capture_output=True, check=True)
                return 'h264_nvenc'
            except:
                print("⚠️  NVIDIA GPU不可用，使用CPU编码")
                return None
        elif 'h264_amf' in result.stdout:
            return 'h264_amf'
        elif 'h264_qsv' in result.stdout:
            return 'h264_qsv'
        else:
            return None
    except:
        return None

def _apply_round(x: np.ndarray) -> np.ndarray:
    """应用取整模式"""
    if DEPTH_ROUND_MODE == 'floor':
        return np.floor(x)
    if DEPTH_ROUND_MODE == 'ceil':
        return np.ceil(x)
    # 默认 round
    return np.round(x)

def scale_depth_chunk_uint16(chunk: bytes, width: int, height: int, scale: float) -> bytes:
    """缩放深度数据块"""
    if not chunk:
        return b''
    # Each frame is width*height uint16
    frame_size_bytes = width * height * 2
    if len(chunk) % frame_size_bytes != 0:
        # trim partial trailing bytes to avoid reshape errors
        usable = (len(chunk) // frame_size_bytes) * frame_size_bytes
        chunk = chunk[:usable]
    if not chunk:
        return b''
    arr = np.frombuffer(chunk, dtype=np.uint16)
    num_frames = arr.size // (width * height)
    try:
        arr = arr.reshape((num_frames, height, width))
    except Exception:
        return b''
    arr = arr.copy()
    if DEPTH_NON_ZERO_ONLY:
        mask = arr > 0
        if mask.any():
            arr[mask] = _apply_round(arr[mask].astype(np.float64) * scale).astype(np.uint16)
    else:
        arr[...] = _apply_round(arr.astype(np.float64) * scale).astype(np.uint16)
    return arr.tobytes()

def ffprobe_stream_info(video_path: str) -> Optional[dict]:
    """获取视频流信息"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,pix_fmt,codec_name',
        '-of', 'json',
        video_path
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
        streams = data.get('streams', [])
        if not streams:
            return None
        return streams[0]
    except Exception:
        return None

def parse_fps(r_frame_rate: str) -> float:
    """解析帧率"""
    if not r_frame_rate:
        return 30.0
    if '/' in r_frame_rate:
        num, den = r_frame_rate.split('/')
        try:
            num_f = float(num)
            den_f = float(den) if float(den) != 0 else 1.0
            return num_f / den_f
        except Exception:
            return 30.0
    try:
        return float(r_frame_rate)
    except Exception:
        return 30.0

def process_hand_depth_video_scaling(video_path: str, batch_frames: int = DEPTH_BATCH_FRAMES, scale: float = DEPTH_SCALE_FACTOR) -> Tuple[bool, str]:
    """处理手部深度视频缩放"""
    print(f"🔧 开始处理手部深度视频缩放: {os.path.basename(video_path)}")
    
    info = ffprobe_stream_info(video_path)
    if not info:
        return False, 'ffprobe失败'
    
    width = int(info.get('width', 0) or 0)
    height = int(info.get('height', 0) or 0)
    pix_fmt = info.get('pix_fmt', '')
    fps = parse_fps(info.get('r_frame_rate', ''))
    
    if width <= 0 or height <= 0:
        return False, '无效的分辨率'
    
    print(f"📊 视频信息: {width}x{height}, {fps:.2f}fps, {pix_fmt}")
    
    # 解码进程：输出 raw gray16le 到 stdout
    dec_cmd = [
        'ffmpeg',
        '-v', 'error',
        '-fflags', '+genpts+discardcorrupt',
        '-err_detect', 'ignore_err',
        '-analyzeduration', '0',
        '-probesize', '64M',
        '-i', video_path,
        '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-'
    ]
    
    # 编码进程：从 stdin 接收 raw gray16le，编码为 ffv1 gray16le
    temp_output = video_path.replace('.mkv', '.tmp.mkv')
    enc_cmd = [
        'ffmpeg', '-v', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'gray16le',
        '-s', f'{width}x{height}', '-r', f'{fps:.6f}',
        '-i', '-',
        '-c:v', 'ffv1', '-pix_fmt', 'gray16le',
        '-y', temp_output
    ]
    
    frame_bytes = width * height * 2
    chunk_bytes = frame_bytes * max(1, batch_frames)
    
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
    
    ok = True
    msg = 'ok'
    frames_done = 0
    
    try:
        assert dec.stdout is not None
        assert enc.stdin is not None
        while True:
            chunk = dec.stdout.read(chunk_bytes)
            if not chunk:
                break
            out_chunk = scale_depth_chunk_uint16(chunk, width, height, scale)
            if not out_chunk:
                continue
            enc.stdin.write(out_chunk)
            # 进度：按输出字节估算帧数
            frames_written = len(out_chunk) // (width * height * 2)
            frames_done += frames_written
            if frames_done % 100 == 0:  # 每100帧显示一次进度
                print(f"    已处理帧: {frames_done}")
        
        # close writer to flush
        enc.stdin.close()
        dec_ret = dec.wait()
        enc_ret = enc.wait()
        
        if dec_ret != 0:
            ok = False
            msg = f'decoder退出码{dec_ret}'
        if enc_ret != 0:
            ok = False
            msg = f'encoder退出码{enc_ret}'
            
    except Exception as e:
        ok = False
        msg = f'异常: {e}'
    finally:
        # ensure processes are terminated
        if dec.poll() is None:
            with contextlib.suppress(Exception):
                dec.terminate()
        if enc.poll() is None:
            with contextlib.suppress(Exception):
                enc.terminate()
    
    if not ok:
        # cleanup temp
        if os.path.exists(temp_output):
            with contextlib.suppress(Exception):
                os.unlink(temp_output)
        return False, msg
    
    # 替换原文件（不保留原始）
    try:
        # 保证写入完成后原子替换
        tmp_final = video_path.replace('.mkv', '.mkv.replacing')
        if os.path.exists(tmp_final):
            os.unlink(tmp_final)
        os.rename(temp_output, tmp_final)
        # On same filesystem, atomic rename
        os.rename(tmp_final, video_path)
        print(f"✅ 手部深度视频缩放完成: {frames_done} 帧")
    except Exception as e:
        return False, f'替换失败: {e}'
    
    return True, 'ok'

def convert_rgb_images_to_video(input_dir, output_path, fps=30, gpu_encoder=None, is_head_camera=False, target_width=None, target_height=None, max_frames: Optional[int] = None, force_concat: bool = True, skip_head_crop: bool = False, aligned_indices: Optional[List[int]] = None):
    """将RGB图像序列转换为MP4视频，保持原始分辨率"""
    
    # 检查是否有图片文件
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not image_files:
        image_files = glob.glob(os.path.join(input_dir, "*.png"))
        if not image_files:
            print(f"在 {input_dir} 中没有找到jpg或png文件")
            return False
        else:
            file_ext = "png"
    else:
        file_ext = "jpg"
        
    # 如果提供了对齐索引，按索引选择文件
    if aligned_indices is not None:
        # 创建数字ID到文件路径的映射
        id_to_file = {}
        for file_path in image_files:
            filename = os.path.basename(file_path)
            try:
                # 提取文件名中的数字ID
                file_id = int(os.path.splitext(filename)[0])
                id_to_file[file_id] = file_path
            except ValueError:
                continue
        
        # 按对齐索引选择文件
        selected_files = []
        for idx in aligned_indices:
            if idx in id_to_file:
                selected_files.append(id_to_file[idx])
        
        if not selected_files:
            print(f"❌ 对齐索引无匹配文件: {input_dir}")
            return False
        
        image_files = selected_files
        print(f"🎯 使用对齐索引: {len(image_files)} 个文件")
    
    print(f"找到 {len(image_files)} 个{file_ext}文件")

    # 读取顺序IO批处理配置
    try:
        from config import SEQ_IO_BATCH_ENABLED, SEQ_IO_BATCH_SIZE
    except Exception:
        SEQ_IO_BATCH_ENABLED, SEQ_IO_BATCH_SIZE = False, 256

    # 若启用顺序IO批处理：通过rawvideo管道喂给ffmpeg（减少小文件风暴）
    # 注意：JPG 由 Python 解码往往比 ffmpeg 慢，默认仅对 PNG 更有收益；对 JPG 继续走 concat 路径
    if SEQ_IO_BATCH_ENABLED and not aligned_indices and file_ext.lower() == 'png':
        try:
            files_sorted = sorted(image_files, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]) if os.path.splitext(os.path.basename(p))[0].isdigit() else p)
        except Exception:
            files_sorted = sorted(image_files)

        if isinstance(max_frames, int) and max_frames > 0:
            files_sorted = files_sorted[:max_frames]

        if not files_sorted:
            print(f"在 {input_dir} 中没有可用图像")
            return False

        first_img = cv2.imread(files_sorted[0])
        if first_img is None:
            print(f"❌ 无法读取首帧: {files_sorted[0]}")
            return False

        h0, w0 = first_img.shape[:2]
        out_w = int(target_width) if target_width else w0
        out_h = int(target_height) if target_height else h0

        # 构建ffmpeg命令：rawvideo(bgr24) 从stdin输入
        cmd = [
            'ffmpeg', '-y',
            '-fflags', '+genpts',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{out_w}x{out_h}', '-r', str(fps),
            '-i', '-',
        ]

        # 选择编码器参数
        if gpu_encoder and (out_w >= 1280 and out_h >= 720):
            cmd += ['-c:v', gpu_encoder, '-preset', 'p1', '-tune', 'hq', '-rc', 'vbr', '-cq', '23', '-b:v', '5M', '-maxrate', '10M', '-bufsize', '10M']
        else:
            cmd += ['-c:v', 'libx264', '-preset', X264_PRESET, '-crf', X264_CRF]

        # NVENC路径不额外传 -threads，避免无效线程竞争；CPU路径才设置 threads
        if gpu_encoder and (out_w >= 1280 and out_h >= 720):
            pass
        else:
            cmd += ['-threads', str(FFMPEG_THREADS)]
        cmd += ['-pix_fmt', 'yuv420p', output_path]

        print(f"🔄 顺序IO批处理启用（batch={SEQ_IO_BATCH_SIZE}）→ rawvideo 管道编码")
        print(f"🔍 FFmpeg命令: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        except Exception as e:
            print(f"❌ 启动ffmpeg失败，回退至concat路径: {e}")
        else:
            try:
                batch_cnt = 0
                total = 0
                skipped = 0
                for pth in files_sorted:
                    img = cv2.imread(pth)
                    if img is None:
                        skipped += 1
                        continue
                    if (img.shape[1] != out_w) or (img.shape[0] != out_h):
                        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    # 逐帧写入，避免大栈拷贝
                    proc.stdin.write(np.ascontiguousarray(img).tobytes())
                    total += 1
                    batch_cnt += 1
                    if batch_cnt >= SEQ_IO_BATCH_SIZE:
                        batch_cnt = 0  # 保持节奏，触发磁盘顺序读
                proc.stdin.close()
                ret = proc.wait()
                if ret == 0:
                    print(f"✅ 顺序IO批处理完成: 写入帧 {total}, 跳过 {skipped}")
                    return True
                else:
                    stderr = proc.stderr.read().decode('utf-8', errors='ignore') if proc.stderr else ''
                    print(f"❌ ffmpeg返回码 {ret}: {stderr[:500]}")
                    return False
            except Exception as e:
                print(f"❌ 顺序IO批处理异常: {e}")
                try:
                    if proc and proc.poll() is None:
                        proc.terminate()
                except Exception:
                    pass
                return False

    # 总是使用 concat 列表以严格帧顺序，避免历史帧覆盖/漂移（回退路径）
    list_file = None
    import re as _re
    def _num_key(p: str) -> tuple:
        name = os.path.basename(p)
        nums = _re.findall(r"\d+", name)
        return tuple(int(n) for n in nums) if nums else (name,)

    # 1) 排序
    files_sorted = sorted(image_files, key=_num_key)

    # 2) 提取序号并截断到首个连续区间，避免中间缺帧导致 ffmpeg 读取失败
    # 注意：当使用aligned_indices时，索引可能不连续（有重复），需要跳过连续性检查
    indices: list[int] = []
    base_dir = input_dir
    # 依据最后一个数字段作为帧号（如 image_1057.jpg）
    for p in files_sorted:
        nums = _re.findall(r"(\d+)", os.path.basename(p))
        if nums:
            try:
                indices.append(int(nums[-1]))
            except ValueError:
                indices.append(-1)
        else:
            indices.append(-1)
    
    # 计算从起点开始的最大连续窗口
    trimmed_files = []
    if aligned_indices is not None:
        # 使用对齐索引时，跳过连续性检查，直接使用所有文件
        # 因为对齐索引可能包含重复，这是正常的（用于时间同步）
        trimmed_files = files_sorted
        print(f"🎯 使用对齐索引，跳过连续性检查: {len(trimmed_files)} 个文件")
    elif indices and indices[0] >= 0:
        # 原始逻辑：检查连续性
        expected = indices[0]
        for i, idx in enumerate(indices):
            if idx != expected:
                break
            trimmed_files.append(files_sorted[i])
            expected += 1
    else:
        trimmed_files = files_sorted

    # 3) 应用 max_frames 限制
    if isinstance(max_frames, int) and max_frames > 0:
        trimmed_files = trimmed_files[:max_frames]

    # 4) 生成 concat 列表（写入前再次确认文件存在，避免生成后被删）
    valid_files = [p for p in trimmed_files if os.path.exists(p)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as lf:
        for p in valid_files:
            lf.write(f"file '{p}'\n")
        list_file = lf.name

    # 构建ffmpeg命令
    cmd = [
        'ffmpeg', '-y',
        '-fflags', '+genpts',
        '-f', 'concat', '-safe', '0',
        '-r', str(fps),
        '-i', list_file,
        '-vsync', 'cfr',
        '-pix_fmt', 'yuv420p',
        '-bufsize', '32M',
        '-maxrate', '20M',
    ]
    
    # 视频滤镜处理
    video_filters = []
    
    # 高清模式：不进行任何裁剪和缩放，保持原始分辨率
    # 低清模式：仅进行缩放，不进行裁剪
    if target_width and target_height:
        video_filters.append(f'scale={target_width}:{target_height}')
        print(f"🔧 缩放到目标分辨率: {target_width}x{target_height}")
    else:
        print(f"🔧 保持原始分辨率，不进行任何处理")
    
    # 应用视频滤镜
    if video_filters:
        cmd.extend(['-vf', ','.join(video_filters)])
    
    # 添加线程数配置
    cmd.extend(['-threads', str(FFMPEG_THREADS)])

    # 限制帧数（仅在非 concat 模式下，concat 已按列表截断）
    # 不再使用 -frames:v，列表已精确截断
    
    # 如果使用GPU编码器
    if gpu_encoder:
        # 检查分辨率是否满足GPU编码器要求
        # 对于低分辨率视频，CPU编码更快；对于高分辨率视频，GPU编码更快
        if target_width and target_height and (target_width < 1280 or target_height < 720):
            print(f"ℹ️  分辨率 {target_width}x{target_height} 较低，使用CPU编码（更快）")
            gpu_encoder = None  # 使用CPU编码
        else:
            print(f"🎮 高分辨率 {target_width}x{target_height}，使用GPU编码")
            cmd.extend([
                '-c:v', gpu_encoder,
                '-preset', 'p1',
                '-tune', 'hq',
                '-rc', 'vbr',
                '-cq', '23',
                '-b:v', '5M',
                '-maxrate', '10M',
                '-bufsize', '10M'
            ])
    
    if not gpu_encoder:
        # 使用CPU编码，优化速度
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', X264_PRESET,  # 使用配置的快速预设
            '-crf', X264_CRF
        ])
    
    # 若启用临时工作区：先写入临时盘，再原子搬迁
    final_output_path = output_path
    try:
        from config import TEMP_WORKDIR_ENABLED, TEMP_WORKDIR_PATH
    except Exception:
        TEMP_WORKDIR_ENABLED, TEMP_WORKDIR_PATH = False, None
    temp_out = None
    if TEMP_WORKDIR_ENABLED and TEMP_WORKDIR_PATH and os.path.isdir(TEMP_WORKDIR_PATH):
        try:
            os.makedirs(TEMP_WORKDIR_PATH, exist_ok=True)
            temp_out = os.path.join(TEMP_WORKDIR_PATH, os.path.basename(output_path))
            cmd.append(temp_out)
            print(f"📦 临时写出到: {temp_out}")
        except Exception:
            cmd.append(output_path)
    else:
        cmd.append(output_path)
    
    try:
        encoder_type = "GPU" if gpu_encoder else "CPU"
        print(f"使用{encoder_type}编码器: {gpu_encoder or 'libx264'}")
        print(f"🔍 FFmpeg命令: {' '.join(cmd)}")
        # 检查是否需要裁剪（头部相机且不是跳过裁剪模式）
        need_crop = is_head_camera and not skip_head_crop
        if is_head_camera and need_crop:
            print("应用头部相机裁剪: scale=1920:1440, crop=1920:1080:0:360")
        elif is_head_camera and not need_crop:
            print("头部相机图片分辨率已经是1920*1080，跳过裁剪")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode == 0:
            duration = end_time - start_time
            # 若使用了临时盘，搬迁到最终位置
            if temp_out and os.path.exists(temp_out):
                try:
                    # 若在同一文件系统，可原子rename；否则使用shutil.move
                    try:
                        os.rename(temp_out, final_output_path)
                    except Exception:
                        import shutil
                        shutil.move(temp_out, final_output_path)
                    print(f"🚚 已搬迁到最终位置: {final_output_path}")
                except Exception as _mv_e:
                    print(f"⚠️ 临时文件搬迁失败，仍保留于: {temp_out} ({_mv_e})")
            print(f"✅ RGB视频转换成功: {final_output_path}")
            print(f"⏱️  转换耗时: {duration:.2f} 秒")
            return True
        else:
            print(f"❌ GPU编码失败，尝试CPU编码...")
            print(result.stderr)
            # 如果GPU失败，尝试CPU编码
            if gpu_encoder:
                return convert_rgb_images_to_video(input_dir, output_path, fps, None, is_head_camera, target_width, target_height, max_frames)
            return False
            
    except Exception as e:
        print(f"❌ RGB视频转换出错: {e}")
        # 若因列表中途失效，重建一次列表重试（一次）
        try:
            if list_file and os.path.exists(list_file):
                os.unlink(list_file)
        except Exception:
            pass
        # 重新过滤仍存在的文件并生成列表
        current_files = sorted(glob.glob(os.path.join(input_dir, f"*.{file_ext}")), key=_num_key)
        current_files = [p for p in current_files if os.path.exists(p)]
        if not current_files:
            return False
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as lf:
            for p in current_files:
                lf.write(f"file '{p}'\n")
            list_file_retry = lf.name
        cmd_retry = cmd.copy()
        # 替换输入列表路径
        for i in range(len(cmd_retry)):
            if i > 0 and cmd_retry[i-1] == '-i':
                cmd_retry[i] = list_file_retry
                break
        try:
            result2 = subprocess.run(cmd_retry, capture_output=True, text=True)
            if result2.returncode == 0:
                print("✅ 重试成功（使用现存文件列表）")
                return True
            else:
                print(result2.stderr)
                return False
        except Exception as _e2:
            print(f"❌ 重试失败: {_e2}")
            return False

def convert_depth_images_to_mkv(input_dir, output_path, fps=30, is_hand_depth=False, max_frames: Optional[int] = None, aligned_indices: Optional[List[int]] = None):
    """将深度图像序列转换为无损MKV视频
    - 手部深度且开启预缩放时，改用流式PNG->raw->缩放->MKV，逐像素严格等价：仅非零×比例，四舍五入，uint16
    - 其他情况走常规路径
    """
    
    # 获取PNG文件
    png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    # 若提供对齐索引，则按索引筛选并排序
    if aligned_indices is not None:
        id_to_file = {}
        for p in png_files:
            name = os.path.splitext(os.path.basename(p))[0]
            try:
                fid = int(name)
                id_to_file[fid] = p
            except Exception:
                continue
        selected = [id_to_file[i] for i in aligned_indices if i in id_to_file]
        if selected:
            png_files = selected
            print(f"🎯 使用对齐索引深度帧: {len(png_files)} 个 -> {os.path.basename(output_path)}")
    if isinstance(max_frames, int) and max_frames > 0:
        png_files = png_files[:max_frames]
    
    if not png_files:
        print(f"在 {input_dir} 中没有找到PNG文件")
        return False
    
    print(f"找到 {len(png_files)} 个PNG深度文件")
    
    # 跳过深度数据检查，直接进行转换
    
    # 手部深度 + 预缩放：走流式方案，避免滤镜边界像素偏差
    if is_hand_depth and ENABLE_HAND_DEPTH_SCALING and DEPTH_SCALE_IN_FFMPEG and float(DEPTH_SCALE_FACTOR) > 0:
        try:
            print("🔧 使用流式PNG->raw->缩放->MKV(仅非零×比例、四舍五入、uint16)")
            # 确保 CUDA 环境变量在进程内可见（data_test 环境）
            _ensure_cuda_env_vars()
            # 探测分辨率
            first_png = png_files[0]
            with Image.open(first_png) as _im:
                width, height = _im.size
            frame_bytes = width * height * 2
            # 可配置批大小
            try:
                from config import DEPTH_SCALE_BACKEND, DEPTH_STREAM_CHUNK_FRAMES
            except Exception:
                DEPTH_SCALE_BACKEND, DEPTH_STREAM_CHUNK_FRAMES = 'numpy', 256
            chunk_frames = int(max(16, DEPTH_STREAM_CHUNK_FRAMES))
            chunk_bytes = frame_bytes * chunk_frames

            # 解码PNG为raw gray16le
            dec_cmd = [
                'ffmpeg','-v','error',
                '-framerate', str(fps),
                '-pattern_type','glob','-i', os.path.join(input_dir,'*.png'),
                # 限制解码帧数
                *(['-vframes', str(len(png_files))] if isinstance(max_frames, int) and max_frames > 0 else []),
                '-f','rawvideo','-pix_fmt','gray16le',
                '-bufsize', '32M',  # 增加缓冲区大小
                '-'
            ]
            # 编码raw gray16le为ffv1 gray16le
            enc_cmd = [
                'ffmpeg','-v','error',
                '-f','rawvideo','-pix_fmt','gray16le',
                '-s', f'{width}x{height}','-r', str(fps),
                '-i','-',
                '-c:v','ffv1','-level','3','-pix_fmt','gray16le','-g','1','-slicecrc','1','-slices','16',
                '-threads', str(FFMPEG_THREADS),
                '-bufsize', '64M',  # 增加缓冲区大小
                '-y', output_path
            ]

            dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE)
            enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
            frames_done = 0
            use_cupy = False
            if DEPTH_SCALE_BACKEND == 'cupy':
                try:
                    import cupy as cp  # type: ignore
                    use_cupy = True
                    print("🎮 深度缩放后端: cupy (GPU)")
                except Exception:
                    print("⚠️ cupy 不可用，回退 numpy")
                    use_cupy = False
            else:
                print("🧮 深度缩放后端: numpy (CPU)")
            try:
                assert dec.stdout is not None and enc.stdin is not None
                while True:
                    buf = dec.stdout.read(chunk_bytes)
                    if not buf:
                        break
                    usable = (len(buf)//frame_bytes)*frame_bytes
                    if usable == 0:
                        continue
                    raw = buf[:usable]
                    if use_cupy:
                        gpu_arr = cp.frombuffer(raw, dtype=cp.uint16)
                        nf = int(gpu_arr.size//(width*height))
                        gpu_arr = gpu_arr.reshape((nf, height, width))
                        if DEPTH_NON_ZERO_ONLY:
                            mask = gpu_arr > 0
                            if mask.any():
                                # 四舍五入：+0.5 后 floor
                                gpu_arr[mask] = cp.floor(gpu_arr[mask].astype(cp.float64) * float(DEPTH_SCALE_FACTOR) + 0.5).astype(cp.uint16)
                        else:
                            gpu_arr = cp.floor(gpu_arr.astype(cp.float64) * float(DEPTH_SCALE_FACTOR) + 0.5).astype(cp.uint16)
                        enc.stdin.write(cp.asnumpy(gpu_arr).tobytes())
                    else:
                        arr = np.frombuffer(raw, dtype=np.uint16)
                        nf = arr.size//(width*height)
                        arr = arr.reshape((nf, height, width))
                        # 写前复制，避免修改只读缓冲
                        arr = arr.copy()
                        if DEPTH_NON_ZERO_ONLY:
                            mask = arr > 0
                            if mask.any():
                                arr[mask] = np.floor(arr[mask].astype(np.float64) * float(DEPTH_SCALE_FACTOR) + 0.5).astype(np.uint16)
                        else:
                            arr = np.floor(arr.astype(np.float64) * float(DEPTH_SCALE_FACTOR) + 0.5).astype(np.uint16)
                        enc.stdin.write(arr.tobytes())
                    frames_done += nf
                    if frames_done % 200 == 0:
                        print(f"    已处理帧: {frames_done}")
                # flush/close
                enc.stdin.close()
                dr = dec.wait(); er = enc.wait()
                if dr != 0 or er != 0:
                    print(f"❌ 流式合成失败 dec={dr} enc={er}")
                    return False
                print(f"✅ 流式合成完成: {output_path}")
                return True
            finally:
                try:
                    if dec.poll() is None:
                        dec.terminate()
                except Exception:
                    pass
                try:
                    if enc.poll() is None:
                        enc.terminate()
                except Exception:
                    pass
        except Exception as e:
            print(f"❌ 流式预缩放合成异常: {e}")
            return False

    # ===== 常规路径（非手部或未启用预缩放） =====
    # 若使用 concat，需要列表文件；若使用 image2(glob) 则不需要
    list_file = None
    # 统一改为 image2(glob) 读取，避免 concat 的时间戳混乱
    use_image2_glob = True
    
    try:
        # 构建ffmpeg命令 - 使用FFV1编码器进行无损压缩
        if use_image2_glob:
            # 手部深度 + 预缩放：使用 image2(glob) 读取，避免 concat 的边界像素偏差
            cmd = [
                "ffmpeg",
                "-y",
                "-threads", str(FFMPEG_THREADS),
                "-framerate", str(fps),
                "-pattern_type", "glob",
                "-i", os.path.join(input_dir, "*.png"),
                # 限制帧数（仅当未强制列表文件的情况下）
                *(["-vframes", str(len(png_files))] if isinstance(max_frames, int) and max_frames > 0 else []),
                "-bufsize", "64M",  # 增加缓冲区大小
                "-maxrate", "50M",  # 限制最大码率
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",  # 覆盖输出文件
                "-threads", str(FFMPEG_THREADS),  # 线程数优化
                "-fflags", "+genpts",
                "-f", "concat",
                "-safe", "0",
                "-r", str(fps),
                "-i", list_file,
                "-bufsize", "64M",  # 增加缓冲区大小
                "-maxrate", "50M",  # 限制最大码率
            ]

        # 若为手部深度并启用在PNG->MKV阶段预缩放，则直接在滤镜中做比例缩放（仅非零，四舍五入，保持16位）
        if is_hand_depth and ENABLE_HAND_DEPTH_SCALING and DEPTH_SCALE_IN_FFMPEG and float(DEPTH_SCALE_FACTOR) > 0:
            vf_expr = (
                "format=gray16le,"
                f"geq=lum='if(gte(lum(X,Y),1),floor(lum(X,Y)*{DEPTH_SCALE_FACTOR}+0.5),0)'"
            )
            cmd.extend(["-vf", vf_expr])
            print(f"🔧 PNG->MKV阶段应用手部深度缩放：非零*{DEPTH_SCALE_FACTOR}并四舍五入，uint16保存")
        else:
            # 不再单独追加 -pix_fmt，统一在编码参数处指定，避免重复
            pass

        # 通用编码参数（可配置速度模式）
        try:
            from config import DEPTH_FFV1_SPEED_MODE, DEPTH_FFV1_LEVEL, DEPTH_FFV1_SLICES, DEPTH_FFV1_SLICECRC
        except Exception:
            DEPTH_FFV1_SPEED_MODE, DEPTH_FFV1_LEVEL, DEPTH_FFV1_SLICES, DEPTH_FFV1_SLICECRC = 'default', 3, 16, True
        ffv1_level = max(3, int(DEPTH_FFV1_LEVEL) if str(DEPTH_FFV1_SPEED_MODE).lower() == 'fast' else 3)
        ffv1_slices = int(DEPTH_FFV1_SLICES) if str(DEPTH_FFV1_SPEED_MODE).lower() == 'fast' else 16
        ffv1_slicecrc = 1 if (bool(DEPTH_FFV1_SLICECRC) if str(DEPTH_FFV1_SPEED_MODE).lower() == 'fast' else True) else 0
        cmd.extend([
            "-c:v", "ffv1",
            "-level", str(ffv1_level),
            "-pix_fmt", "gray16le",
            "-g", "1",
            "-slicecrc", str(ffv1_slicecrc),
            "-slices", str(ffv1_slices),
            "-an",
            "-loglevel", "warning",
            "-fps_mode", "cfr",
        ])

        # 临时工作区：优先写临时盘
        final_output_path = output_path
        temp_out = None
        try:
            from config import TEMP_WORKDIR_ENABLED, TEMP_WORKDIR_PATH
        except Exception:
            TEMP_WORKDIR_ENABLED, TEMP_WORKDIR_PATH = False, None
        if TEMP_WORKDIR_ENABLED and TEMP_WORKDIR_PATH and os.path.isdir(TEMP_WORKDIR_PATH):
            try:
                os.makedirs(TEMP_WORKDIR_PATH, exist_ok=True)
                temp_out = os.path.join(TEMP_WORKDIR_PATH, os.path.basename(output_path))
                cmd.append(temp_out)
                print(f"📦 深度临时写出到: {temp_out}")
            except Exception:
                cmd.append(output_path)
        else:
            cmd.append(output_path)

        # 若为手部深度并启用在PNG->MKV阶段预缩放，则直接在滤镜中做 *0.1（或配置的比例）
        try:
            if (
                is_hand_depth
                and ENABLE_HAND_DEPTH_SCALING
                and DEPTH_SCALE_IN_FFMPEG
                and float(DEPTH_SCALE_FACTOR) > 0
            ):
                # 仅对非零像素缩放；四舍五入取整；保持16位
                # geq 表达式：if(lum>=1, floor(lum*scale + 0.5), 0)
                vf_expr = (
                    "format=gray16le,"
                    f"geq=lum='if(gte(lum,1),floor(lum*{DEPTH_SCALE_FACTOR}+0.5),0)'"
                )
                # 在 output_path 之前插入 -vf 与表达式
                cmd.insert(-1, vf_expr)
                cmd.insert(-1, "-vf")
                print(f"🔧 已在PNG->MKV阶段应用手部深度缩放：非零*{DEPTH_SCALE_FACTOR}并四舍五入，uint16保存")
        except Exception as _:
            # 若出现异常，不影响后续流程
            pass
        
        print("正在执行深度视频转换...")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode == 0:
            duration = end_time - start_time
            # 临时盘搬迁
            if temp_out and os.path.exists(temp_out):
                try:
                    try:
                        os.rename(temp_out, final_output_path)
                    except Exception:
                        import shutil
                        shutil.move(temp_out, final_output_path)
                    print(f"🚚 深度视频已搬迁到最终位置: {final_output_path}")
                except Exception as _mv_e:
                    print(f"⚠️ 深度临时文件搬迁失败，仍保留于: {temp_out} ({_mv_e})")
            print(f"✅ 深度视频转换成功: {final_output_path}")
            print(f"⏱️  转换耗时: {duration:.2f} 秒")
            
            # 验证转换后的视频
            print("🔍 验证转换后的深度视频...")
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"📊 输出文件大小: {file_size:.2f} MB")
                
                # 快速验证视频文件
                if CV2_AVAILABLE:
                    try:
                        cap = cv2.VideoCapture(output_path)
                        if cap.isOpened():
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            print(f"📹 视频信息验证:")
                            print(f"   - 帧数: {frame_count}")
                            print(f"   - 帧率: {fps:.2f} fps")
                            print(f"   - 分辨率: {width}x{height}")
                            
                            # 读取第一帧验证数据
                            ret, frame = cap.read()
                            if ret:
                                if len(frame.shape) == 3:
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                
                                valid_pixels = frame[frame > 0]
                                if len(valid_pixels) > 0:
                                    frame_min = np.min(valid_pixels)
                                    frame_max = np.max(valid_pixels)
                                    frame_mean = np.mean(valid_pixels)
                                    print(f"   - 第一帧深度范围: [{frame_min}, {frame_max}], 平均{frame_mean:.1f}")
                                else:
                                    print(f"   - 第一帧: 无有效深度数据")
                            
                            cap.release()
                            print("✅ 深度视频验证完成")
                        else:
                            print("⚠️  无法打开转换后的视频文件进行验证")
                    except Exception as e:
                        print(f"⚠️  视频验证时出错: {e}")
                else:
                    print("⚠️  cv2未安装，跳过视频验证")
            
            # 如果已在PNG->MKV阶段做了预缩放，则跳过后处理的视频到视频缩放
            # 否则按原有流程进行后处理
            if is_hand_depth and ENABLE_HAND_DEPTH_SCALING and not DEPTH_SCALE_IN_FFMPEG:
                print(f"🔧 检测到手部深度视频，开始缩放处理...")
                scaling_success, scaling_msg = process_hand_depth_video_scaling(
                    output_path, 
                    batch_frames=DEPTH_BATCH_FRAMES, 
                    scale=DEPTH_SCALE_FACTOR
                )
                if scaling_success:
                    print(f"✅ 手部深度视频缩放处理完成")
                else:
                    print(f"⚠️  手部深度视频缩放处理失败: {scaling_msg}")
                    # 缩放失败不影响整体转换，继续执行
            
            return True
        else:
            print(f"❌ 深度视频转换失败:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 深度视频转换出错: {e}")
        return False
    finally:
        # 清理临时文件
        if list_file and os.path.exists(list_file):
            os.unlink(list_file)

def merge_audio_frames(audio_dir, output_path, target_duration=40.0, sample_rate=16000):
    """合并音频帧为完整音频文件"""
    
    # 查找音频文件
    audio_files = glob.glob(os.path.join(audio_dir, AUDIO_PATTERN))
    audio_files.sort()
    
    if not audio_files:
        print(f"在 {audio_dir} 中没有找到音频文件")
        return False
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    try:
        # 使用ffmpeg合并音频
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for audio_file in audio_files:
                f.write(f"file '{audio_file}'\n")
            list_file = f.name
        
        # 构建ffmpeg命令
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c:a", "pcm_s16le",  # 16位PCM
            "-ar", str(sample_rate),  # 采样率
            "-ac", "1",  # 单声道
            output_path
        ]
        
        print("正在合并音频...")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode == 0:
            duration = end_time - start_time
            print(f"✅ 音频合并成功: {output_path}")
            print(f"⏱️  合并耗时: {duration:.2f} 秒")
            return True
        else:
            print(f"❌ 音频合并失败:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 音频合并出错: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(list_file):
            os.unlink(list_file)

def copy_audio_file(audio_dir, output_path, max_seconds: Optional[float] = None):
    """复制或裁剪已合成的音频文件（当提供 max_seconds 时进行时间裁剪）。
    约定：audio_dir 内只有一个音频文件，不关心文件名，直接使用该文件。
    """
    
    # 在目录中寻找唯一的音频文件
    try:
        candidates = [
            os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
            if os.path.isfile(os.path.join(audio_dir, f))
        ]
        if not candidates:
            print(f"在 {audio_dir} 中没有找到音频文件")
            return False
        # 如果有多个，优先 .wav；否则取修改时间最新的
        wavs = [p for p in candidates if p.lower().endswith('.wav')]
        if len(wavs) == 1:
            audio_file = wavs[0]
        elif len(wavs) > 1:
            audio_file = sorted(wavs, key=lambda p: os.path.getmtime(p), reverse=True)[0]
        else:
            audio_file = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)[0]
        print(f"🔎 选定音频源: {audio_file}")
    except Exception as _e:
        print(f"⚠️  枚举音频失败: {_e}")
        return False
    
    try:
        # 直接从源音频裁剪到临时文件，再覆盖目标；强制采样率/声道，使用atrim保证精确切割
        if isinstance(max_seconds, (int, float)) and max_seconds > 0:
            print(f"🧮 开始硬裁剪音频，目标秒数: {max_seconds:.6f}")
            # 直接按样本数裁切PCM（完全绕过ffmpeg）
            try:
                with wave.open(audio_file, 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    n_frames_total = wf.getnframes()
                    target_frames = int(round(max_seconds * framerate))
                    target_frames = max(0, min(target_frames, n_frames_total))
                    print(f"📐 源参数: channels={n_channels}, sampwidth={sampwidth*8}bit, rate={framerate}Hz, total_frames={n_frames_total}, target_frames={target_frames}")
                    frames = wf.readframes(target_frames)
                # 保持源参数，避免时长因通道/采样率改变而失真
                with wave.open(output_path, 'wb') as out:
                    out.setnchannels(n_channels)
                    out.setsampwidth(sampwidth)
                    out.setframerate(framerate)
                    out.writeframes(frames)
                print(f"✅ 音频按样本数裁剪完成: {target_frames}/{n_frames_total} 帧 @ {framerate}Hz (≈{target_frames/framerate:.3f}s) -> {output_path}")
            except Exception as _e:
                print(f"⚠️  PCM样本裁剪失败，退化为复制: {_e}")
                shutil.copy2(audio_file, output_path)
        else:
            shutil.copy2(audio_file, output_path)
            print(f"✅ 音频文件复制成功: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 音频文件复制失败: {e}")
        return False

def convert_parquet_to_hdf5(parquet_file, output_path, max_frames: Optional[int] = None):
    """将parquet文件转换为HDF5格式并保存到指定路径"""
    """将parquet文件转换为HDF5格式 - 使用convert_all_new1.py的逻辑"""
    
    try:
        print(f"正在转换parquet文件: {parquet_file}")
        
        # 检查文件是否存在
        if not os.path.exists(parquet_file):
            print(f"❌ Parquet文件不存在: {parquet_file}")
            return False
        
        # 先尝试直接读取parquet文件
        try:
            print("尝试直接读取parquet文件...")
            # 尝试不同的引擎
            engines = ['pyarrow', 'fastparquet']
            df = None
            
            for engine in engines:
                try:
                    print(f"尝试使用 {engine} 引擎...")
                    df = pd.read_parquet(parquet_file, engine=engine)
                    print(f"✅ 使用 {engine} 引擎读取成功，形状: {df.shape}")
                    break
                except Exception as e:
                    print(f"使用 {engine} 引擎失败: {e}")
                    continue
            
            if df is None:
                raise Exception("所有引擎都失败了")
                
        except Exception as e:
            print(f"直接读取失败: {e}")
            print("尝试使用子进程读取...")
            
            # 使用子进程读取parquet文件
            try:
                # 创建临时脚本
                import tempfile
                temp_script = f'''
import pandas as pd
import numpy as np
import pickle
import sys

try:
    df = pd.read_parquet("{parquet_file}")
    # 保存到pickle文件
    with open("{parquet_file}.pkl", "wb") as f:
        pickle.dump(df, f)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
'''
                
                # 写入临时脚本文件
                script_path = f"/tmp/read_parquet_{os.getpid()}.py"
                with open(script_path, 'w') as f:
                    f.write(temp_script)
                
                # 先用当前环境运行
                result = subprocess.run([
                    'python3', script_path
                ], capture_output=True, text=True, cwd=os.path.dirname(parquet_file))
                
                if result.returncode == 0 and "SUCCESS" in result.stdout:
                    # 读取pickle文件
                    import pickle
                    with open(f"{parquet_file}.pkl", "rb") as f:
                        df = pickle.load(f)
                    print(f"✅ 子进程读取成功，形状: {df.shape}")
                    
                    # 清理临时文件
                    os.unlink(script_path)
                    os.unlink(f"{parquet_file}.pkl")
                else:
                    print(f"❌ 子进程读取失败: {result.stderr}")
                    print(f"尝试使用{CONDA_ENV_NAME}环境...")
                    
                    # 尝试使用配置的conda环境
                    try:
                        result = subprocess.run([
                            'conda', 'run', '-n', CONDA_ENV_NAME, 'python3', script_path
                        ], capture_output=True, text=True, cwd=os.path.dirname(parquet_file))
                        
                        if result.returncode == 0 and "SUCCESS" in result.stdout:
                            # 读取pickle文件
                            import pickle
                            with open(f"{parquet_file}.pkl", "rb") as f:
                                df = pickle.load(f)
                            print(f"✅ 使用{CONDA_ENV_NAME}环境成功读取parquet文件，形状: {df.shape}")
                            
                            # 清理临时文件
                            os.unlink(script_path)
                            os.unlink(f"{parquet_file}.pkl")
                        else:
                            print(f"❌ {CONDA_ENV_NAME}环境读取失败: {result.stderr}")
                            os.unlink(script_path)
                            return False
                    except Exception as e:
                        print(f"❌ 调用{CONDA_ENV_NAME}环境失败: {e}")
                        os.unlink(script_path)
                        return False
                    
            except Exception as e:
                print(f"❌ 调用子进程失败: {e}")
                return False
        
        print(f"✅ 成功读取parquet文件，形状: {df.shape}")

        # 若设置了最大帧数，先截断 DataFrame
        try:
            if isinstance(max_frames, int) and max_frames > 0:
                df = df.iloc[:max_frames]
                print(f"✂️  按标注截断 parquet 到前 {len(df)} 帧")
        except Exception as _e:
            print(f"⚠️  截断parquet失败，忽略: {_e}")

        # 提取各个数据到变量
        task_mode = df['task_mode'].values
        timestamp = df['timestamp'].values
        frame_index = df['frame_index'].values
        episode_index = df['episode_index'].values
        # 生成从1开始的连续帧索引
        index = np.arange(0, len(timestamp) , dtype=np.int32)
        task_index = df['task_index'].values
        
        # 使用时间戳格式
        # 使用pandas实现纳秒精度UTC时间戳转换
        if 'timestamps' in df.columns:
            unix_timestamps = df['timestamps'].values
        else:
            raise ValueError("找不到 'timestamps' 字段")
        
        # 注释掉unix转utc的转换操作，直接提取parquet里的utc时间
        # timestamps_utc = pd.to_datetime(unix_timestamps, unit='ns', utc=True).strftime('%Y-%m-%dT%H:%M:%S.%f%z')
        # timestamps_utc = np.array(timestamps_utc)
        
        # 直接从parquet文件中提取utc时间戳
        if 'timestamps_utc' in df.columns:
            timestamps_utc = df['timestamps_utc'].values
            print("✅ 直接使用parquet文件中的UTC时间戳")
        else:
            # 如果没有utc时间戳字段，使用unix转换（备用方案）
            print("⚠️  parquet文件中没有UTC时间戳字段，使用unix转换")
            timestamps_utc = pd.to_datetime(unix_timestamps, unit='ns', utc=True).strftime('%Y-%m-%dT%H:%M:%S.%f%z')
            timestamps_utc = np.array(timestamps_utc)
        
        # 定义N变量（数据长度）
        N = len(timestamp)

        # 手部数据
        hands_torque = np.stack(df['observation.hands.torque'].values)
        # 手部电流：优先使用数据列；如缺失且允许推导，则按比值由力矩推导
        if 'observation.hands.current' in df.columns:
            hands_current = np.stack(df['observation.hands.current'].values)
        else:
            if 'DERIVE_HAND_CURRENT_ENABLED' in globals() and DERIVE_HAND_CURRENT_ENABLED:
                ratio = HAND_FORCE_TO_CURRENT_RATIO if 'HAND_FORCE_TO_CURRENT_RATIO' in globals() else 6.8
                print(f"ℹ️  缺少 observation.hands.current，按比值推导：current = torque / {ratio}")
                hands_current = hands_torque / float(ratio)
            else:
                raise KeyError("observation.hands.current 缺失，且未启用推导（DERIVE_HAND_CURRENT_ENABLED=False）")
        hands_vel = np.stack(df['observation.hands.vel'].values)
        hands_state = np.stack(df['observation.hands.state'].values)
        hands_action = np.stack(df['observation.hands.action'].values)

        # 腿部数据
        legs_state_q = np.stack(df['observation.legs.state_q'].values)
        legs_state_qd = np.stack(df['observation.legs.state_qd'].values)
        legs_state_tau = np.stack(df['observation.legs.state_tau'].values)
        legs_cmd_q = np.stack(df['observation.legs.cmd_q'].values)
        legs_cmd_qd = np.stack(df['observation.legs.cmd_qd'].values)
        legs_cmd_tau = np.stack(df['observation.legs.cmd_tau'].values)

        # 拆分手部数据 - 只保留夹爪相关数据
        left_arm_action = hands_action[:, 0:7]
        right_arm_action = hands_action[:, 7:14]
        arm_action_all = hands_action[:, 0:14]
        # 夹爪数据 (索引14和20是左右夹爪)
        left_gripper_action = hands_action[:, 14:15] 
        right_gripper_action = hands_action[:, 20:21] 
        gripper_action = hands_action[:, [14, 20]]
        waist_action = hands_action[:, 26:28]
        head_action = hands_action[:, 28:30]

        left_arm_state = hands_state[:, 0:7]
        right_arm_state = hands_state[:, 7:14]
        arm_state_all = hands_state[:, 0:14]
        # 夹爪状态数据
        left_gripper_state = hands_state[:, 14:15] * 80
        right_gripper_state = hands_state[:, 20:21] * 80
        gripper_state = hands_state[:, [14, 20]]/80
        waist_state = hands_state[:, 26:28]
        head_state = hands_state[:, 28:30]

        left_arm_torque = hands_torque[:, 0:7]
        right_arm_torque = hands_torque[:, 7:14]
        arm_torque_all = hands_torque[:, 0:14]
        # 夹爪扭矩数据
        left_gripper_torque = hands_torque[:, 14:15]
        right_gripper_torque = hands_torque[:, 20:21]
        gripper_torque = hands_torque[:, [14, 20]]
        waist_torque = hands_torque[:, 26:28]
        head_torque = hands_torque[:, 28:30]

        # 拆分电流数据
        left_arm_current = hands_current[:, 0:7]
        right_arm_current = hands_current[:, 7:14]
        arm_current_all = hands_current[:, 0:14]
        # 夹爪电流数据
        left_gripper_current = hands_current[:, 14:15]
        right_gripper_current = hands_current[:, 20:21]
        gripper_current = hands_current[:, [14, 20]]
        waist_current = hands_current[:, 26:28]
        head_current = hands_current[:, 28:30]

        left_arm_vel = hands_vel[:, 0:7]
        right_arm_vel = hands_vel[:, 7:14]
        arm_vel_all = hands_vel[:, 0:14]
        # 夹爪速度数据
        left_gripper_vel = hands_vel[:, 14:15]
        right_gripper_vel = hands_vel[:, 20:21]
        gripper_vel = hands_vel[:, [14, 20]]
        waist_vel = hands_vel[:, 26:28]
        head_vel = hands_vel[:, 28:30]

        # 拆分腿部数据
        left_leg_position = legs_state_q[:, 0:6]
        right_leg_position = legs_state_q[:, 6:12]
        left_leg_velocity = legs_state_qd[:, 0:6]
        right_leg_velocity = legs_state_qd[:, 6:12]
        left_leg_torque = legs_state_tau[:, 0:6]
        right_leg_torque = legs_state_tau[:, 6:12]
        left_leg_cmd_position = legs_cmd_q[:, 0:6]
        right_leg_cmd_position = legs_cmd_q[:, 6:12]
        left_leg_cmd_velocity = legs_cmd_qd[:, 0:6]
        right_leg_cmd_velocity = legs_cmd_qd[:, 6:12]
        left_leg_cmd_torque = legs_cmd_tau[:, 0:6]
        right_leg_cmd_torque = legs_cmd_tau[:, 6:12]

        # === 初始化左右臂运动学模型 ===
        urdf_file_left = URDF_LEFT_ARM
        urdf_file_right = URDF_RIGHT_ARM
        
        # 检查URDF文件是否存在和运动学开关
        if os.path.exists(urdf_file_left) and os.path.exists(urdf_file_right) and PINOCCHIO_AVAILABLE and ENABLE_KINEMATICS:
            try:
                # 记录运动学计算开始时间
                kinematics_start_time = time.time()
               # print(f"🕐 运动学计算开始时间: {time.strftime('%H:%M:%S')} (总帧数: {N})")
                
                arm_left_kin = ArmKinematics(urdf_file_left, ee_frame_name="link_tcp_l")
                arm_right_kin = ArmKinematics(urdf_file_right, ee_frame_name="link_tcp_r")
                
                # === action部分：优化的批量运动学计算 ===
                action_start_time = time.time()
                print("📐 计算action末端位置和姿态...")
                
               # 批量计算左臂action运动学
                left_action_pos, left_action_quat = compute_kinematics_batch(
                    arm_left_kin, arm_action_all[:, 0:7], "左臂Action"
                )
                
                # 批量计算右臂action运动学  
                right_action_pos, right_action_quat = compute_kinematics_batch(
                    arm_right_kin, arm_action_all[:, 7:14], "右臂Action"
                )
                    
                action_end_position = np.stack([left_action_pos, right_action_pos], axis=1)  # (N,2,3)
                action_end_orientation = np.stack([left_action_quat, right_action_quat], axis=1)  # (N,2,4)
                
                action_end_time = time.time()
                action_duration = action_end_time - action_start_time
               # print(f"⏱️  Action运动学计算耗时: {action_duration:.3f}秒")

                # === state部分：优化的批量运动学和速度计算 ===
                state_start_time = time.time()
                #print("📐 计算state末端位置、速度和姿态...")
                
                # 批量计算左臂state运动学和速度
                left_state_pos, left_state_quat, left_state_linvel, left_state_angvel = compute_kinematics_with_velocity_batch(
                    arm_left_kin, arm_state_all[:, 0:7], arm_vel_all[:, 0:7], "左臂State"
                )
                
                # 批量计算右臂state运动学和速度
                right_state_pos, right_state_quat, right_state_linvel, right_state_angvel = compute_kinematics_with_velocity_batch(
                    arm_right_kin, arm_state_all[:, 7:14], arm_vel_all[:, 7:14], "右臂State"
                )
                    
                state_end_position = np.stack([left_state_pos, right_state_pos], axis=1)  # (N,2,3)
                state_end_orientation = np.stack([left_state_quat, right_state_quat], axis=1)  # (N,2,4)
                state_end_velocity = np.stack([left_state_linvel, right_state_linvel], axis=1)  # (N,2,3)
                state_end_angular = np.stack([left_state_angvel, right_state_angvel], axis=1)  # (N,2,3)
                
                state_end_time = time.time()
                state_duration = state_end_time - state_start_time
                total_kinematics_duration = state_end_time - kinematics_start_time
                
                print(f"⏱️  State运动学计算耗时: {state_duration:.3f}秒")
                print(f"⏱️  总运动学计算耗时: {total_kinematics_duration:.3f}秒")
                print(f"📊  Action计算占比: {action_duration/total_kinematics_duration*100:.1f}%")
                print(f"📊  State计算占比: {state_duration/total_kinematics_duration*100:.1f}%")
                print("✅ 运动学计算完成")
                
            except Exception as e:
                print(f"⚠️  运动学计算失败: {e}，使用零数据")
                N = len(timestamp)
                action_end_position = np.zeros((N, 2, 3), dtype=np.float32)
                action_end_orientation = np.zeros((N, 2, 4), dtype=np.float32)
                state_end_position = np.zeros((N, 2, 3), dtype=np.float32)
                state_end_orientation = np.zeros((N, 2, 4), dtype=np.float32)
                state_end_velocity = np.zeros((N, 2, 3), dtype=np.float32)
                state_end_angular = np.zeros((N, 2, 3), dtype=np.float32)
        else:
            print("⚠️  URDF文件不存在，跳过运动学计算，使用零数据")
            N = len(timestamp)
            action_end_position = np.zeros((N, 2, 3), dtype=np.float32)
            action_end_orientation = np.zeros((N, 2, 4), dtype=np.float32)
            state_end_position = np.zeros((N, 2, 3), dtype=np.float32)
            state_end_orientation = np.zeros((N, 2, 4), dtype=np.float32)
            state_end_velocity = np.zeros((N, 2, 3), dtype=np.float32)
            state_end_angular = np.zeros((N, 2, 3), dtype=np.float32)
        
        # === 写入HDF5文件 ===
        hdf5_filename = output_path

        # 创建一个空的HDF5文件并将数据写入
        with h5py.File(hdf5_filename, 'w') as f:
            # ====== 造缺失字段的简单数据 ======
            np.random.seed(RANDOM_SEED)  # 保证每次生成一致
            # 本体位置：原点+极小高斯噪声
            robot_position = 0.001 * np.random.randn(N, 3).astype(np.float32)
            # 本体姿态：单位四元数+极小z轴扰动
            drift_angle = 0.001 * np.random.randn(N)
            robot_orientation = np.stack([
                np.zeros(N),
                np.zeros(N),
                np.sin(drift_angle / 2),
                np.cos(drift_angle / 2)
            ], axis=1).astype(np.float32)
            # 位置漂移：极小高斯噪声
            robot_position_drift = 0.001 * np.random.randn(N, 3).astype(np.float32)
            # 姿态漂移：单位四元数+极小z轴扰动
            drift_angle2 = 0.001 * np.random.randn(N)
            robot_orientation_drift = np.stack([
                np.zeros(N),
                np.zeros(N),
                np.sin(drift_angle2 / 2),
                np.cos(drift_angle2 / 2)
            ], axis=1).astype(np.float32)
            # 速度：x方向有小波动，y方向更小
            robot_velocity = np.stack([
                0.01 + 0.001 * np.random.randn(N),  # x方向速度
                0.0005 * np.random.randn(N)         # y方向速度
            ], axis=1).astype(np.float32)
            # ====== END ======
            
            # 根目录下的文件级属性
            f.attrs['description'] = 'Robot simulation data'
            f.attrs['date_created'] = str(pd.Timestamp.now())

            # 添加时间相关的字符串信息
            start_time_utc = timestamps_utc[0] if len(timestamps_utc) > 0 else START_TIME_UTC
            f.attrs['start_time_utc'] = start_time_utc
            f.attrs['frame_rate'] = float(FPS)  # 每秒30帧
            
            # 保存UTC时间戳
            # 转换为字节串数组避免HDF5写入问题
            timestamps_utc_bytes = [ts.encode('utf-8') for ts in timestamps_utc]
            f.create_dataset('timestamps_utc', data=timestamps_utc_bytes)
            
            # 保存Unix时间戳
            f.create_dataset('timestamps', data=unix_timestamps)
            f.create_dataset('end_timestamps', data=unix_timestamps)
            # 添加时间戳字段 - 先检查parquet文件中的实际字段，然后创建对应的数据集
            print("🔍 检查parquet文件中的时间戳字段...")
            
            # 检查并添加RGB视频时间戳（只保留color时间戳，去掉depth时间戳）
            if 'headf_mp4_timestamps' in df.columns:
                headf_color_timestamps = df['headf_mp4_timestamps'].values
                f.create_dataset('head_front_color_timestamps', data=headf_color_timestamps)
                print(f"✅ 添加head_front_color_timestamps 时间戳数据")
            
            if 'hand_r_rgb_mp4_timestamps' in df.columns:
                hand_r_color_timestamps = df['hand_r_rgb_mp4_timestamps'].values
                f.create_dataset('hand_right_color_timestamps', data=hand_r_color_timestamps)
                print(f"✅ 添加 hand_right_color_timestamps 时间戳数据")
            
            if 'hand_l_rgb_mp4_timestamps' in df.columns:
                hand_l_color_timestamps = df['hand_l_rgb_mp4_timestamps'].values
                f.create_dataset('hand_left_color_timestamps', data=hand_l_color_timestamps)
                print(f"✅ 添加 hand_left_color_timestamps 时间戳数据")
            
            # 输出所有可用的时间戳字段
            available_timestamp_fields = [col for col in df.columns if 'timestamp' in col.lower()]
            print(f"📋 Parquet文件中可用的时间戳字段: {available_timestamp_fields}")

            # 创建根目录下的 `action` 组
            action_group = f.create_group('action')

            # 在 `action` 组下创建并填充 `effector` 子组
            effector_group = action_group.create_group('effector')
            effector_group.create_dataset('index', data=ensure_hdf5_compatible(index))
            effector_group.create_dataset('position', data=gripper_action)

            # 在 `action` 组下创建并填充 `end` 子组
            end_group = action_group.create_group('end')
            end_group.create_dataset('orientation', data=action_end_orientation)
            end_group.create_dataset('position', data=action_end_position.astype(np.float32))
            end_group.create_dataset('index', data=ensure_hdf5_compatible(index))
            
            # 创建并填充各个动作数据组
            head_group = action_group.create_group('head')
            head_group.create_dataset('position', data=head_action)
            head_group.create_dataset('index', data=ensure_hdf5_compatible(index))

            joint_group = action_group.create_group('joint')
            joint_group.create_dataset('position', data=arm_action_all)
            joint_group.create_dataset('index', data=ensure_hdf5_compatible(index))

            waist_group = action_group.create_group('waist')
            waist_group.create_dataset('position', data=waist_action)
            waist_group.create_dataset('index', data=ensure_hdf5_compatible(index))
            
            # 添加腿部动作组
            legs_group = action_group.create_group('legs')
            legs_group.create_dataset('position', data=legs_cmd_q)
            legs_group.create_dataset('velocity', data=legs_cmd_qd)
            legs_group.create_dataset('effort', data=legs_cmd_tau)
            legs_group.create_dataset('index', data=ensure_hdf5_compatible(index))

            # 创建根目录下的 `state` 组并填充数据
            state_group = f.create_group('state')

            # 在 `state` 组下创建 `effector` 子组并填充数据
            effector_state_group = state_group.create_group('effector')
            effector_state_group.create_dataset('force', data=gripper_torque)
           # effector_state_group.create_dataset('current', data=gripper_current)
            effector_state_group.create_dataset('position', data=gripper_state)
            effector_state_group.create_dataset('velocity', data=gripper_vel)  # 添加velocity字段

            # 创建并填充 `end`, `head`, `joint`, `robot`, `waist` 子组
            end_state_group = state_group.create_group('end')
            end_state_group.create_dataset('angular', data=ensure_hdf5_compatible(state_end_angular.astype(np.float32)))
            end_state_group.create_dataset('orientation', data=ensure_hdf5_compatible(state_end_orientation.astype(np.float32)))
            end_state_group.create_dataset('position', data=ensure_hdf5_compatible(state_end_position.astype(np.float32)))
            end_state_group.create_dataset('velocity', data=ensure_hdf5_compatible(state_end_velocity.astype(np.float32)))
            

            head_state_group = state_group.create_group('head')
            head_state_group.create_dataset('effort', data=ensure_hdf5_compatible(head_torque))
           # head_state_group.create_dataset('current', data=ensure_hdf5_compatible(head_current))
            head_state_group.create_dataset('position', data=ensure_hdf5_compatible(head_state))
            head_state_group.create_dataset('velocity', data=ensure_hdf5_compatible(head_vel))

            joint_state_group = state_group.create_group('joint')
            joint_state_group.create_dataset('effort', data=ensure_hdf5_compatible(arm_torque_all))
            joint_state_group.create_dataset('current_value', data=ensure_hdf5_compatible(arm_current_all))
            joint_state_group.create_dataset('position', data=ensure_hdf5_compatible(arm_state_all))
            joint_state_group.create_dataset('velocity', data=ensure_hdf5_compatible(arm_vel_all))
            
            # 添加关节名称属性
            joint_names = [
                "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4", 
                "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
                "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", 
                "right_arm_joint5", "right_arm_joint6", "right_arm_joint7"
            ]
            joint_state_group.attrs['joint_names'] = joint_names

            waist_state_group = state_group.create_group('waist')
            waist_state_group.create_dataset('effort', data=ensure_hdf5_compatible(waist_torque))
           # waist_state_group.create_dataset('current', data=ensure_hdf5_compatible(waist_current))
            waist_state_group.create_dataset('position', data=ensure_hdf5_compatible(waist_state))
            waist_state_group.create_dataset('velocity', data=ensure_hdf5_compatible(waist_vel))
            
            # 添加腿部状态组
            legs_state_group = state_group.create_group('legs')
            
            # 全部腿部状态
            legs_state_group.create_dataset('position', data=ensure_hdf5_compatible(legs_state_q))
            legs_state_group.create_dataset('velocity', data=ensure_hdf5_compatible(legs_state_qd))
            legs_state_group.create_dataset('effort', data=ensure_hdf5_compatible(legs_state_tau))

           
            # legs_state_group.attrs['joint_names'] = leg_joint_names
        
        print(f"✅ Parquet转HDF5成功: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Parquet转HDF5失败: {e}")
        return False

def ensure_hdf5_compatible(data, str_len=50):
    arr = np.asarray(data)
    if arr.dtype.kind in {'U', 'O'}:
        arr = np.array([str(x).encode('utf-8') for x in arr], dtype=f'S{str_len}')
    return arr

def copy_camera_parameters(src_dir, dst_dir):
    """从parameters文件夹复制相机参数文件"""
    
    try:
        # 确保目标目录存在
        os.makedirs(dst_dir, exist_ok=True)
        
        # 显示当前使用的参数目录信息
        print(f"📁 使用相机参数目录: {src_dir}")
        print(f"📁 目标参数目录: {dst_dir}")
        
        # 复制所有JSON文件
        json_files = glob.glob(os.path.join(src_dir, "*.json"))
        
        if not json_files:
            print(f"⚠️  在参数目录中没有找到JSON文件: {src_dir}")
            return False
        
        for json_file in json_files:
            filename = os.path.basename(json_file)
            dst_file = os.path.join(dst_dir, filename)
            shutil.copy2(json_file, dst_file)
            print(f"✅ 复制相机参数: {filename}")
        
        print(f"✅ 相机参数复制完成，共复制 {len(json_files)} 个文件")
        return True
        
    except Exception as e:
        print(f"❌ 相机参数复制失败: {e}")
        return False

def process_single_task(task_info):
    """处理单个任务（用于并发）"""
    task_type, args = task_info
    
    try:
        if task_type == "rgb_video":
            if len(args) == 10:  # 新版本，包含对齐索引
                src_path, dst_path, fps, gpu_encoder, is_head, src_dir, dst_file, cutoff_frames, force_concat, stream_indices = args
            else:  # 旧版本，兼容
                src_path, dst_path, fps, gpu_encoder, is_head, src_dir, dst_file, cutoff_frames, force_concat = args
                stream_indices = None
            print(f"🎬 处理RGB视频: {src_dir} -> {dst_file}")
            return convert_rgb_images_to_video(src_path, dst_path, fps, gpu_encoder, is_head, None, None, cutoff_frames, force_concat, stream_indices)
        elif task_type == "depth_video":
            # 兼容老参数个数
            if len(args) == 8:
                src_path, dst_path, fps, src_dir, dst_file, is_hand_depth, cutoff_frames, depth_indices = args
            else:
                src_path, dst_path, fps, src_dir, dst_file, is_hand_depth, cutoff_frames = args
                depth_indices = None
            print(f"🎯 处理深度视频: {src_dir} -> {dst_file}")
            return convert_depth_images_to_mkv(src_path, dst_path, fps, is_hand_depth, cutoff_frames, depth_indices)
        elif task_type == "hdf5":
            parquet_file, hdf5_output, cutoff_frames = args
            print(f"📊 处理parquet数据: {os.path.basename(parquet_file)}")
            return convert_parquet_to_hdf5(parquet_file, hdf5_output, cutoff_frames)
        elif task_type == "audio":
            episode_audio_dir, audio_output, audio_seconds = args
            print(f"🎵 处理音频文件")
            seconds = None
            # 这里第三个参数就是"秒数"，无需再按帧换算
            if isinstance(audio_seconds, (int, float)) and audio_seconds > 0:
                seconds = float(audio_seconds)
            return copy_audio_file(episode_audio_dir, audio_output, seconds)
        elif task_type == "parameters":
            src_dir, dst_dir = args
            print(f"📁 复制相机参数文件")
            return copy_camera_parameters(src_dir, dst_dir)
        elif task_type == "lowres_from_images":
            episode_dir, output_dir, gpu_encoder, aligned_indices = args
            print(f"🎬 直接从图像生成低清视频")
            return generate_lowres_color_videos_from_images(episode_dir, output_dir, gpu_encoder=gpu_encoder, aligned_indices=aligned_indices)
        else:
            print(f"❌ 未知任务类型: {task_type}")
            return False
    except Exception as e:
        print(f"❌ 任务 {task_type} 执行失败: {e}")
        return False

def generate_lowres_color_videos_from_images(episode_dir: str, output_dir: str, width: int = 320, height: int = 180, gpu_encoder=None, aligned_indices: Optional[Dict] = None) -> bool:
    """直接从图像生成三个低清color视频（hand_left, hand_right, head_front）
    - 输入: episode目录路径（包含images子目录）
    - 输出: 在output_dir下生成三个低分辨率mp4文件
    - 目标文件: hand_left_color_low.mp4, hand_right_color_low.mp4, head_front_color_low.mp4
    - 默认分辨率: 320x180
    - aligned_indices: 多模态对齐索引，确保三个相机视频同步
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义需要生成低清视频的相机映射
        lowres_mappings = {
            "hand_l_rgb": "hand_left_color_low.mp4",
            "hand_r_rgb": "hand_right_color_low.mp4", 
            "headf_rgbd_color": "head_front_color_low.mp4"
        }
        
        images_dir = os.path.join(episode_dir, "images")
        if not os.path.exists(images_dir):
            print(f"❌ 图像目录不存在: {images_dir}")
            return False
            
        success_count = 0
        for src_dir, dst_file in lowres_mappings.items():
            src_path = os.path.join(images_dir, src_dir)
            dst_path = os.path.join(output_dir, dst_file)
            
            if not os.path.exists(src_path):
                print(f"⚠️  跳过不存在的图像目录: {src_path}")
                continue
            
            # 获取对应的对齐索引
            stream_indices = None
            if aligned_indices:
                if src_dir == "hand_l_rgb":
                    stream_indices = aligned_indices.get('hand_l')
                elif src_dir == "hand_r_rgb":
                    stream_indices = aligned_indices.get('hand_r')
                elif src_dir == "headf_rgbd_color":
                    stream_indices = aligned_indices.get('headf')
                
            # 低清模式下缩放到指定分辨率，但不进行裁剪
            is_head_camera = False  # 强制设置为False，避免任何裁剪逻辑
            
            if stream_indices:
                print(f"🎬 生成低清视频: {src_dir} -> {dst_file} (缩放到{width}x{height}, 使用对齐索引)")
            else:
                print(f"🎬 生成低清视频: {src_dir} -> {dst_file} (缩放到{width}x{height}, 无对齐)")
                
            success = convert_rgb_images_to_video(
                src_path, dst_path, FPS, gpu_encoder, is_head_camera, 
                target_width=width, target_height=height, skip_head_crop=True, aligned_indices=stream_indices
            )
            
            if success:
                success_count += 1
                print(f"✅ 低清视频生成成功: {dst_file}")
            else:
                print(f"❌ 低清视频生成失败: {dst_file}")
        
        print(f"📊 低清视频生成完成: {success_count}/{len(lowres_mappings)} 成功")
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 生成低清视频异常: {e}")
        return False

def generate_lowres_color_videos(highres_video_dir: str, output_dir: str, width: int = 640, height: int = 360, create_subdir: bool = True, subdir_name: str = "vido_color") -> bool:
    """生成低清晰度的手部与头部前向color视频，供标注使用
    - 输入: 已生成的高分辨率 color 视频所在目录（camera/video）
    - 输出: 在 output_dir 下创建 vido_color/ 并生成低清晰度 mp4
    - 目标文件: hand_left_color_low.mp4, hand_right_color_low.mp4, head_front_color_low.mp4
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        target_dir = os.path.join(output_dir, subdir_name) if create_subdir else output_dir
        os.makedirs(target_dir, exist_ok=True)

        targets = [
            ("hand_left_color.mp4",  os.path.join(target_dir, "hand_left_color_low.mp4")),
            ("hand_right_color.mp4", os.path.join(target_dir, "hand_right_color_low.mp4")),
            ("head_front_color.mp4", os.path.join(target_dir, "head_front_color_low.mp4")),
        ]

        success_any = False
        for src_name, dst_path in targets:
            src_path = os.path.join(highres_video_dir, src_name)
            if not os.path.exists(src_path):
                continue
            cmd = [
                "ffmpeg",
                "-y",
                "-threads", str(FFMPEG_THREADS),
                "-i", src_path,
                "-vf", f"scale={LOWRES_WIDTH}:{LOWRES_HEIGHT}",
                "-c:v", "libx264",
                "-preset", X264_PRESET,
                "-bufsize", "16M",  # 增加缓冲区大小
                "-maxrate", "5M",   # 限制最大码率
                "-crf", str(LOWRES_CRF),
                "-pix_fmt", "yuv420p",
                dst_path
            ]
            print(f"🎞️  生成低清晰度标注视频: {os.path.basename(src_path)} -> {os.path.basename(dst_path)}")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                success_any = True
            else:
                print(f"⚠️  低清晰度转码失败: {src_name}\n{r.stderr}")
        return success_any
    except Exception as e:
        print(f"❌ 生成低清晰度标注视频异常: {e}")
        return False

def process_episode(episode_dir, output_base_dir, episode_index, gpu_encoder=None, preset_uuid: Optional[str] = None, uuid_to_cutoff: Optional[Dict[str, int]] = None):
    """处理单个episode - 使用并发优化"""
    
    print(f"\n{'='*60}")
    print(f"🚀 处理episode: {episode_dir}")
    print(f"{'='*60}")
    
    # 导入导出模式配置
    from config import EXPORT_MODE
    
    # 生成或复用UUID（若提供preset_uuid则复用）
    episode_uuid = preset_uuid or str(uuid.uuid4())
    # Persist UUID mapping immediately to avoid generating new UUID on resume
    try:
        current_map = load_uuid_mapping(CURRENT_DATASET_ROOT)
        if not isinstance(current_map, dict):
            current_map = {}
        need_save = (current_map.get(episode_index) != episode_uuid)
        if need_save:
            current_map[episode_index] = episode_uuid
            save_uuid_mapping(CURRENT_DATASET_ROOT, current_map)
            # Also mirror to output task_info for visibility (best-effort)
            try:
                out_task_info = os.path.join(OUTPUT_ROOT_PATH, 'Logistics', 'task_info')
                os.makedirs(out_task_info, exist_ok=True)
                mapping_txt_out = os.path.join(out_task_info, 'uuid_mapping.json')
                with open(mapping_txt_out, 'w', encoding='utf-8') as f:
                    json.dump({str(k): v for k, v in current_map.items()}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
    except Exception as _m_e:
        print(f"⚠️ UUID映射即时保存失败，不影响继续转换: {_m_e}")
    episode_output_dir = os.path.join(output_base_dir, episode_uuid)
    
    # 尝试生成对齐索引（所有模式都支持对齐）
    aligned_indices = None
    if EXPORT_MODE in [1, 2, 3]:  # both、low-only 或 high-only 模式
        parquet_file = os.path.join(PARQUET_PATH, f"episode_{episode_index:06d}.parquet")
        aligned_rows = generate_aligned_frame_indices(episode_dir, parquet_file)
        if aligned_rows:
            # 提取各流的索引
            aligned_indices = {
                'hand_l': [row['hand_l_idx'] for row in aligned_rows],
                'hand_r': [row['hand_r_idx'] for row in aligned_rows], 
                'headf': [row['headf_idx'] for row in aligned_rows]
            }
            print(f"🎯 使用多模态对齐: {len(aligned_rows)} 个对齐帧")
    
    # 创建目录结构
    camera_dir = os.path.join(episode_output_dir, "camera")
    video_dir = os.path.join(camera_dir, "video")
    depth_dir = os.path.join(camera_dir, "depth")
    audio_dir = os.path.join(episode_output_dir, "audio")
    parameters_dir = os.path.join(episode_output_dir, "parameters")
    proprio_stats_dir = os.path.join(episode_output_dir, "proprio_stats")
    
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(parameters_dir, exist_ok=True)
    os.makedirs(proprio_stats_dir, exist_ok=True)
    
    # 准备所有任务
    tasks = []
    # 截止帧默认值，避免未赋值引用
    cutoff_frames: Optional[int] = None

    # 从主线程预解析的映射中获取截止帧；未命中则不裁剪
    cutoff_frames = None
    if isinstance(uuid_to_cutoff, dict):
        cf_val = uuid_to_cutoff.get(episode_uuid)
        if isinstance(cf_val, int) and cf_val > 0:
            cutoff_frames = cf_val
            print(f"✂️  UUID专属截止帧N={cutoff_frames} (uuid={episode_uuid})")
        else:
            print(f"ℹ️  未在映射中找到匹配uuid={episode_uuid} 的条目，跳过裁剪")
    
    # 1. HDF5转换任务（优先级最高，独立执行）
    parquet_file = os.path.join(PARQUET_PATH, f"episode_{episode_index:06d}.parquet")
    if os.path.exists(parquet_file):
        hdf5_output = os.path.join(proprio_stats_dir, "proprio_stats.hdf5")
        print("📊 开始HDF5转换（高优先级）...")
        hdf5_start = time.time()
        # 传入截止帧（若有）
        cf = cutoff_frames if isinstance(cutoff_frames, int) and cutoff_frames > 0 else None
        print(f"📏 HDF5 裁剪到前 N={cf if cf else 'ALL'} 帧")
        hdf5_success = convert_parquet_to_hdf5(parquet_file, hdf5_output, cf)
        hdf5_duration = time.time() - hdf5_start
        print(f"✅ HDF5转换完成，耗时: {hdf5_duration:.2f}秒" if hdf5_success else "❌ HDF5转换失败")
        # Post-process: add *_names fields if needed
        if hdf5_success:
            try:
                # Prefer the unified tool under data_processing_tools
                from data_processing_tools.update_hdf5_names_fields import update_hdf5_file as _update_h5_names
            except Exception:
                try:
                    # Fallback to local script version if exists
                    from update_hdf5_names_fields import update_hdf5_file as _update_h5_names
                except Exception:
                    _update_h5_names = None
            if _update_h5_names:
                try:
                    print("🔧 更新HDF5 *_names 字段...")
                    _update_h5_names(hdf5_output)
                    print("✅ HDF5 *_names 字段更新完成")
                except Exception as _e:
                    print(f"⚠️ HDF5 *_names 字段更新失败: {_e}")
    else:
        print(f"⚠️  Parquet文件不存在: {parquet_file}")
    
    # 2. 参数复制任务（快速）
    tasks.append(("parameters", (PARAMETERS_PATH, parameters_dir)))
    
    # 3. 音频处理任务
    episode_audio_dir = os.path.join(episode_dir, "audio")
    if os.path.exists(episode_audio_dir):
        audio_output = os.path.join(audio_dir, "microphone.wav")
        seconds = None
        if isinstance(cutoff_frames, int) and cutoff_frames > 0:
            seconds = cutoff_frames / float(FPS)
        print(f"📏 音频裁剪时长 = {seconds if seconds else 'ALL'} 秒 (FPS={FPS})")
        tasks.append(("audio", (episode_audio_dir, audio_output, seconds)))
    
    # 截止帧已由主线程在uuid_to_cutoff中预解析，不再二次读取JSON

    # 4. RGB视频任务（根据导出模式决定）
    images_dir = os.path.join(episode_dir, "images")
    if os.path.exists(images_dir):
        # EXPORT_MODE: 1=both, 2=low-only, 3=high-only
        if EXPORT_MODE in [1, 3]:  # both 或 high-only 模式
            for src_dir, (dst_file, is_head) in CAMERA_MAPPINGS.items():
                src_path = os.path.join(images_dir, src_dir)
                dst_path = os.path.join(video_dir, dst_file)
                
                if os.path.exists(src_path):
                    print(f"📏 RGB {src_dir} 限制帧数 = {cutoff_frames if cutoff_frames else 'ALL'} (保持原始分辨率)")
                    # 获取对应的对齐索引
                    stream_indices = None
                    if aligned_indices:
                        if src_dir == "hand_l_rgb":
                            stream_indices = aligned_indices['hand_l']
                        elif src_dir == "hand_r_rgb":
                            stream_indices = aligned_indices['hand_r']
                        elif src_dir == "headf_rgbd_color":
                            stream_indices = aligned_indices['headf']
                    
                    # 所有相机都不再进行裁剪和缩放，保持原始分辨率
                    tasks.append(("rgb_video", (src_path, dst_path, FPS, gpu_encoder, False, src_dir, dst_file, cutoff_frames, True, stream_indices)))
        elif EXPORT_MODE == 2:  # low-only 模式
            # 直接生成低清视频到out1结构（不裁剪，用于标注）
            from config import OUTPUT_LOWRES_ROOT_PATH
            lowres_video_dir = os.path.join(OUTPUT_LOWRES_ROOT_PATH, "Logistics", "Materialtransfer", "Distribute_Parcels_To_Corresponding_Regions", episode_uuid, "camera", "video")
            os.makedirs(lowres_video_dir, exist_ok=True)
            print(f"📏 Low-only mode: 不裁剪，保持完整长度用于标注")
            tasks.append(("lowres_from_images", (episode_dir, lowres_video_dir, gpu_encoder, aligned_indices)))
    
    # 5. 深度视频任务（根据导出模式决定）
    if os.path.exists(images_dir) and EXPORT_MODE in [1, 3]:  # both 或 high-only 模式
        for src_dir, dst_file in DEPTH_MAPPINGS.items():
            src_path = os.path.join(images_dir, src_dir)
            dst_path = os.path.join(depth_dir, dst_file)
            
            if os.path.exists(src_path):
                # 判断是否为手部深度视频
                is_hand_depth = src_dir in ['hand_l_depth', 'hand_r_depth']
                print(f"📏 DEPTH {src_dir} 限制帧数 = {cutoff_frames if cutoff_frames else 'ALL'}")
                # 对齐索引：headf深度与headf_rgbd_color一致；手部深度与手部rgb一致
                depth_indices = None
                if aligned_indices:
                    if src_dir == 'headf_rgbd_depth':
                        depth_indices = aligned_indices.get('headf')
                    elif src_dir == 'hand_l_depth':
                        depth_indices = aligned_indices.get('hand_l')
                    elif src_dir == 'hand_r_depth':
                        depth_indices = aligned_indices.get('hand_r')
                tasks.append(("depth_video", (src_path, dst_path, FPS, src_dir, dst_file, is_hand_depth, cutoff_frames, depth_indices)))
    
    # 并发执行任务
    if tasks:
        print(f"🔄 并发执行 {len(tasks)} 个任务...")
        parallel_start = time.time()
        
        # 使用线程池并发执行任务
        try:
            with ThreadPoolExecutor(max_workers=MAX_TASKS_PER_EPISODE) as executor:
                # 提交所有任务
                future_to_task = {}
                for task_type, args in tasks:
                    # 将低清模式下的任务串行化，避免过多并发（可选）
                    future = executor.submit(process_single_task, (task_type, args))
                    future_to_task[future] = (task_type, args)
                
                # 等待所有任务完成
                for future in as_completed(future_to_task):
                    task_type, args = future_to_task[future]
                    try:
                        result = future.result()
                        if result:
                            print(f"✅ {task_type} 任务完成")
                        else:
                            print(f"❌ {task_type} 任务失败")
                    except Exception as e:
                        print(f"❌ {task_type} 任务异常: {e}")
        except KeyboardInterrupt:
            print("⛔ 收到中断信号，尝试优雅停止当前Episode任务...")
            # 线程池上下文会尽快收尾，直接继续到后续清理
        
        parallel_duration = time.time() - parallel_start
        print(f"✅ 并发任务完成，耗时: {parallel_duration:.2f}秒")
    
    print(f"🎉 Episode {episode_index} 处理完成: {episode_uuid}")
    return episode_uuid

def calculate_output_size(output_dir):
    """计算输出目录的总大小（GB）"""
    total_size_bytes = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            total_size_bytes += os.path.getsize(file_path)
    
    # 先转换为MB，再转换为GB
    total_size_mb = total_size_bytes / (1024**2)  # 字节转MB
    total_size_gb = total_size_mb / 1024  # MB转GB
    return round(total_size_gb, 1)  # 保留1位小数

def format_duration(seconds):
    """格式化时长（小时）"""
    hours = seconds / 3600
    return f"{hours:.2f}h".replace('.', 'p')

def format_size_gb(size_gb):
    """格式化大小（GB）"""
    if size_gb < 1:
        # 小于1GB，显示为0pXGB格式
        return f"0p{int(size_gb * 10)}GB"
    else:
        # 大于等于1GB，显示为XpYGB格式
        return f"{int(size_gb)}p{int((size_gb % 1) * 10)}GB"

def pre_generate_conversion_report(dataset_root: str, root_key: str):
    """预生成转换报告文件。
    - EXPORT_MODE==2（低清）：写入 data_diqing/conversion_report_lowres.txt
    - EXPORT_MODE==3（高清）：写入 data_gaoqing/conversion_report.txt
    """
    import datetime
    from config import EXPORT_MODE, get_output_paths, PRESERVE_ORIGINAL_STRUCTURE
    
    print("📝 预生成转换报告文件...")

    # 根据模式决定输出根与文件名
    if EXPORT_MODE == 3:
        # 高清
        if PRESERVE_ORIGINAL_STRUCTURE:
            highres_path, _ = get_output_paths(dataset_root, root_key)
        else:
            highres_path = os.path.join(OUTPUT_BASE_PATH, OUTPUT_HIGHRES_DIR, root_key)
        os.makedirs(highres_path, exist_ok=True)
        report_path = os.path.join(highres_path, "conversion_report.txt")
        mode_name = "高清"
        output_root_path = highres_path
    else:
        # 低清（默认）
        if PRESERVE_ORIGINAL_STRUCTURE:
            _, lowres_path = get_output_paths(dataset_root, root_key)
        else:
            lowres_path = os.path.join(OUTPUT_BASE_PATH, OUTPUT_LOWRES_DIR, root_key)
        os.makedirs(lowres_path, exist_ok=True)
        report_path = os.path.join(lowres_path, "conversion_report_lowres.txt")
        mode_name = "低清"
        output_root_path = lowres_path
    
    try:
        # 若报告已存在，则不覆盖，直接接续使用
        if os.path.exists(report_path):
            print(f"ℹ️ 检测到已有{mode_name}报告，接续写入: {report_path}")
        else:
            with open(report_path, 'w', encoding='utf-8') as rf:
                header = (
                    f"Conversion Time: {datetime.datetime.now().isoformat()}\n"
                    f"INPUT_ROOT_PATH: {os.path.join(dataset_root, CHUNK_DIR_NAME)}\n"
                    f"PARQUET_PATH: {os.path.join(dataset_root, PARQUET_DIR_REL)}\n"
                    f"OUTPUT_ROOT_PATH: {output_root_path}\n"
                    f"EXPORT_MODE: {mode_name}\n"
                    f"STATUS: 开始转换\n"
                    f"PROGRESS: 0/0\n"
                    "----------------------------------------\n"
                    "Episode UUID Mapping:\n"
                )
                rf.write(header)
            print(f"✅ 已预生成{mode_name}转换报告: {report_path}")
        # 设置全局报告路径
        global CURRENT_REPORT_PATH
        CURRENT_REPORT_PATH = report_path
    except Exception as e:
        print(f"⚠️ 预生成转换报告失败: {e}")

def update_conversion_progress(episode_idx: int, total_episodes: int, episode_uuid: str, status: str = "处理中", only_update_mapping: bool = False):
    """更新转换进度到报告文件"""
    try:
        from config import EXPORT_MODE
        
        # 优先使用全局报告路径（如果设置了）
        if 'CURRENT_REPORT_PATH' in globals() and CURRENT_REPORT_PATH:
            report_path = CURRENT_REPORT_PATH
        # 其次强制使用低清输出根目录（新规范：低清报告只在 data_diqing 下）
        elif 'OUTPUT_LOWRES_ROOT_PATH' in globals() and OUTPUT_LOWRES_ROOT_PATH:
            report_path = os.path.join(OUTPUT_LOWRES_ROOT_PATH, "conversion_report_lowres.txt")
        else:
            report_filename = "conversion_report_lowres.txt" if EXPORT_MODE == 2 else "conversion_report.txt"
            report_path = os.path.join(CURRENT_DATASET_ROOT, report_filename)
        
        # 诊断：打印一次目标报告路径（仅在首次进入时打印）
        try:
            if not hasattr(update_conversion_progress, "_printed_path"):
                print(f"🧭 报告目标路径: {report_path}", flush=True)
                update_conversion_progress._printed_path = True
        except Exception:
            pass

        # 若报告不存在，先初始化一份（避免 0/0 模板无法被更新的假象）
        if not os.path.exists(report_path):
            try:
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                base = (
                    f"STATUS: {status}\n"
                    f"PROGRESS: {episode_idx}/{total_episodes}\n"
                    "----------------------------------------\n"
                    "Episode UUID Mapping:\n"
                )
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(base)
                print(f"📝 报告初始化完成: {report_path}", flush=True)
            except Exception as e:
                print(f"⚠️ 报告初始化失败: {e}", flush=True)

        if os.path.exists(report_path):
            # 读取现有内容
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            import re
            # 更新进度与状态（除非只更新映射）
            if not only_update_mapping:
                content = re.sub(r'^STATUS: .*$', f'STATUS: {status}', content, flags=re.MULTILINE)
                # 兼容两种写法：PROGRESS: X/Y 与 PROGRESS: X/Y episodes
                content = re.sub(r'^PROGRESS: \d+/\d+(?:\s*episodes)?$', f'PROGRESS: {episode_idx}/{total_episodes}', content, flags=re.MULTILINE)
            
            # 使用原始 episode 名称格式写入（不依赖顺序），若已存在则就地替换，否则追加一行
            if episode_uuid:
                header_tag = "Episode UUID Mapping:\n"
                import re as _re
                # 目标行（兼容旧写法替换为新写法）
                line_new = f"episode_{episode_idx:06d}: {episode_uuid}\n"
                # 若已存在 header
                if header_tag in content:
                    # 先尝试替换旧格式行
                    pattern_old = rf"^Episode\s+{episode_idx}:\s+[0-9a-fA-F\-]{{36}}\s*$"
                    content2 = _re.sub(pattern_old, line_new.strip(), content, flags=_re.MULTILINE)
                    if content2 != content:
                        content = content2 if content2.endswith("\n") else content2 + "\n"
                    else:
                        # 再尝试替换新格式行
                        pattern_new = rf"^episode_{episode_idx:06d}:\s+[0-9a-fA-F\-]{{36}}\s*$"
                        content3 = _re.sub(pattern_new, line_new.strip(), content, flags=_re.MULTILINE)
                        if content3 != content:
                            content = content3 if content3.endswith("\n") else content3 + "\n"
                        else:
                            # 两种都不存在，则在 header 后面追加一行
                            pos = content.find(header_tag) + len(header_tag)
                            content = content[:pos] + line_new + content[pos:]
                else:
                    # 无 header，补上 header 与当前行
                    content += ("\n" if not content.endswith("\n") else "") + header_tag + line_new
            
            # 写回文件
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            # 追加一份侧写日志，便于确认写入
            try:
                dbg_path = os.path.join(os.path.dirname(report_path), 'progress_debug.log')
                with open(dbg_path, 'a', encoding='utf-8') as df:
                    df.write(f"PROGRESS {episode_idx}/{total_episodes} UUID {episode_uuid or '-'}\n")
            except Exception:
                pass
            try:
                msg_uuid = episode_uuid if episode_uuid else "-"
                print(f"📝 报告更新成功: {report_path} | PROGRESS {episode_idx}/{total_episodes} | UUID {msg_uuid}", flush=True)
            except Exception:
                pass
                
    except Exception as e:
        print(f"⚠️ 更新转换进度失败: {e}")

    
def _checkpoint_path() -> str:
    """Return checkpoint file path under output root."""
    try:
        return os.path.join(OUTPUT_ROOT_PATH, 'conversion_checkpoint.json')
    except Exception:
        return os.path.join(os.getcwd(), 'conversion_checkpoint.json')


def load_checkpoint() -> set:
    """Load completed episode indices from checkpoint file."""
    path = _checkpoint_path()
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return set(int(x) for x in data.get('completed_episode_indices', []))
    except Exception as e:
        print(f"⚠️ 加载断点文件失败: {e}")
    return set()


def save_checkpoint(completed_indices: set, last_started: int | None = None) -> None:
    """Persist checkpoint to file (atomic write)."""
    path = _checkpoint_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + '.tmp'
        payload = {
            'completed_episode_indices': sorted(list(int(i) for i in completed_indices)),
            'updated_at': datetime.datetime.now().isoformat()
        }
        if isinstance(last_started, int):
            payload['last_started_episode_index'] = int(last_started)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"⚠️ 保存断点文件失败: {e}")


def update_checkpoint_started(episode_index: int, completed_indices: set) -> None:
    """Record the latest started episode index, without marking it completed."""
    try:
        save_checkpoint(completed_indices, last_started=episode_index)
    except Exception as e:
        print(f"⚠️ 记录开始处理的episode失败: {e}")


def preflight_self_check() -> bool:
    """Basic environment self-check before conversion."""
    print("🔍 环境自检...")
    ok = True
    try:
        r = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            print("❌ 未检测到可用的 ffmpeg")
            ok = False
        else:
            print("✅ ffmpeg 可用")
    except Exception as e:
        print(f"❌ 调用 ffmpeg 失败: {e}")
        ok = False
    for mod in ['h5py', 'numpy']:
        try:
            __import__(mod)
        except Exception as e:
            print(f"❌ 缺少依赖: {mod} - {e}")
            ok = False
    try:
        out_root = OUTPUT_ROOT_PATH
        os.makedirs(out_root, exist_ok=True)
        test_file = os.path.join(out_root, '.write_test')
        with open(test_file, 'w') as f:
            f.write('ok')
        os.unlink(test_file)
        st = os.statvfs(out_root)
        free_gb = (st.f_frsize * st.f_bavail) / (1024**3)
        if free_gb < 5:
            print(f"⚠️ 输出盘可用空间较低: {free_gb:.2f} GB")
        else:
            print(f"✅ 输出盘可用空间: {free_gb:.2f} GB")
    except Exception as e:
        print(f"❌ 输出目录检查失败: {e}")
        ok = False
    return ok

def process_single_dataset(dataset_root: str, root_key: str):
    """处理单个数据集"""
    global INPUT_ROOT_PATH, PARQUET_PATH, OUTPUT_ROOT_PATH, OUTPUT_LOWRES_ROOT_PATH, DEPTH_CHECK_INPUT_PATH, PARAMETERS_PATH, CURRENT_DATASET_ROOT
    
    print(f"\n📦 处理数据集: {dataset_root}")
    print("=" * 50)
    
    # 设置当前数据集根目录
    CURRENT_DATASET_ROOT = dataset_root
    
    # 设置路径
    INPUT_ROOT_PATH = os.path.join(dataset_root, CHUNK_DIR_NAME)
    PARQUET_PATH = os.path.join(dataset_root, PARQUET_DIR_REL)
    
    # 设置自适应输出路径
    from config import get_output_paths, PRESERVE_ORIGINAL_STRUCTURE
    if PRESERVE_ORIGINAL_STRUCTURE:
        highres_path, lowres_path = get_output_paths(dataset_root, root_key)
        OUTPUT_ROOT_PATH = highres_path
        OUTPUT_LOWRES_ROOT_PATH = lowres_path
    else:
        # 使用原有逻辑
        setup_dynamic_output_paths(dataset_root)
    
    DEPTH_CHECK_INPUT_PATH = INPUT_ROOT_PATH
    
    # 自动匹配参数配置
    auto_param = detect_param_profile_from_path(dataset_root)
    if auto_param and os.path.isdir(auto_param):
        PARAMETERS_PATH = auto_param
        print(f"🧭 已自动匹配参数目录: {PARAMETERS_PATH}")
    else:
        print(f"⚠️ 未找到匹配的参数目录，使用默认: {PARAMETERS_PATH}")
    
    # 预生成转换报告文件
    pre_generate_conversion_report(dataset_root, root_key)
    
    # 执行转换
    if DEPTH_PRECHECK_ENABLED:
        scan_depth_episodes(INPUT_ROOT_PATH)
    run_conversion_core()

def main():
    """主函数：优先读取 config 中的多根目录批量扫描配置；若未启用则执行单次转换"""
    global PARAMETERS_PATH, OUTPUT_ROOT_PATH, CURRENT_DATASET_ROOT
    # 从 config 读取多根目录批量配置
    from config import MULTI_ROOT_BATCH_ENABLED, BATCH_ROOT_KEYS, BATCH_SUBDIR_KEYWORDS
    from config import BATCH_SCAN_SUBDIRS, BATCH_BASE_PATH, BATCH_SUBDIR_KEYWORD

    if MULTI_ROOT_BATCH_ENABLED and BATCH_ROOT_KEYS:
        # 多根目录批量处理模式
        print("🚀 多根目录批量转换模式")
        print("=" * 50)
        
        # 验证根目录路径
        from config import validate_root_paths
        root_validation = validate_root_paths()
        
        print("📁 根目录验证结果:")
        for key, info in root_validation.items():
            status = "✅" if info["exists"] else "❌"
            print(f"  {status} {key}: {info['path']} - {info['description']}")
        
        # 处理每个根目录
        for root_key in BATCH_ROOT_KEYS:
            if root_key not in root_validation:
                print(f"⚠️ 跳过未知根目录键: {root_key}")
                continue
                
            if not root_validation[root_key]["exists"]:
                print(f"⚠️ 跳过不存在的根目录: {root_key}")
                continue
            
            root_config = root_validation[root_key]
            root_path = root_config["path"]
            keyword = BATCH_SUBDIR_KEYWORDS.get(root_key, root_key)
            
            print(f"\n🔍 处理根目录: {root_key} ({root_config['description']})")
            print(f"   路径: {root_path}")
            print(f"   关键词: {keyword}")
            
            # 查找数据集目录
            if not os.path.isdir(root_path):
                print(f"❌ 根目录不存在: {root_path}")
                continue
            
            # 检查是否启用深层扫描
            deep_scan = root_validation[root_key].get("deep_scan", False)
            
            if deep_scan:
                print(f"   启用深层扫描模式...")
                from config import find_deep_datasets
                datasets = find_deep_datasets(root_path, keyword)
                print(f"   深层扫描找到 {len(datasets)} 个数据集")
            else:
                # 原有的浅层扫描
                subdirs = [d for d in os.listdir(root_path) 
                          if os.path.isdir(os.path.join(root_path, d)) and keyword in d]
                datasets = [os.path.join(root_path, d) for d in subdirs]
                print(f"   浅层扫描找到 {len(datasets)} 个子目录")
            
            if not datasets:
                print(f"⚠️ 未找到包含关键字 '{keyword}' 的数据集")
                continue
            
            # 处理每个数据集
            for dataset_root in datasets:
                dataset_name = os.path.basename(dataset_root)
                print(f"\n📂 处理数据集: {dataset_name}")
                print(f"   完整路径: {dataset_root}")
                
                # 标记当前数据集根目录（供报告与映射写入）
                global CURRENT_DATASET_ROOT
                CURRENT_DATASET_ROOT = dataset_root
                
                # 设置自适应输出路径
                from config import get_output_paths, PRESERVE_ORIGINAL_STRUCTURE
                if PRESERVE_ORIGINAL_STRUCTURE:
                    highres_path, lowres_path = get_output_paths(dataset_root, root_key)
                    print(f"   高清输出: {highres_path}")
                    print(f"   低清输出: {lowres_path}")
                
                # 执行转换
                try:
                    process_single_dataset(dataset_root, root_key)
                    print(f"✅ 完成转换: {dataset_name}")
                except Exception as e:
                    print(f"❌ 转换失败: {dataset_name} - {e}")
                    continue
    elif BATCH_SCAN_SUBDIRS and BATCH_SUBDIR_KEYWORD:
        # 兼容原有批量扫描模式
        base_path = BATCH_BASE_PATH if BATCH_BASE_PATH else os.path.dirname(DATASET_ROOT)
        keyword = BATCH_SUBDIR_KEYWORD
        if not os.path.isdir(base_path):
            print(f"❌ BATCH_BASE_PATH 不存在或不是目录: {base_path}")
            return
        subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and keyword in d]
        if not subdirs:
            print(f"⚠️ 未找到包含关键字 '{keyword}' 的子目录")
            return
        for d in sorted(subdirs):
            dataset_root = os.path.join(base_path, d)
            # 标记当前数据集根目录（供报告与映射写入）
            CURRENT_DATASET_ROOT = dataset_root
            print("\n============================================================")
            print(f"📦 批量模式：正在处理数据集: {dataset_root}")
            print("============================================================")
            # 重设全局路径，并按数据集路径自动匹配参数配置
            global INPUT_ROOT_PATH, PARQUET_PATH, OUTPUT_ROOT_PATH, OUTPUT_LOWRES_ROOT_PATH, DEPTH_CHECK_INPUT_PATH, PARAMETERS_PATH
            # 输入路径基于实际的数据集根目录
            INPUT_ROOT_PATH = os.path.join(dataset_root, CHUNK_DIR_NAME)
            PARQUET_PATH = os.path.join(dataset_root, PARQUET_DIR_REL)
            # 根据配置动态设置输出路径（基于实际的数据集根目录）
            setup_dynamic_output_paths(dataset_root)
            DEPTH_CHECK_INPUT_PATH = INPUT_ROOT_PATH
            auto_param = detect_param_profile_from_path(dataset_root)
            if auto_param and os.path.isdir(auto_param):
                PARAMETERS_PATH = auto_param
                print(f"🧭 已自动匹配参数目录: {PARAMETERS_PATH}")
            else:
                # 数据集匹配不上时默认使用 p064
                fallback = os.path.join(CURRENT_WORKSPACE, 'parameters_P064')
                if os.path.isdir(fallback):
                    PARAMETERS_PATH = fallback
                    print(f"🧭 未匹配到特定参数目录，使用默认: {PARAMETERS_PATH}")
                else:
                    print(f"🧭 未匹配到特定参数目录，且默认 p064 不存在，沿用: {PARAMETERS_PATH}")
            # 深度预检（可选）
            try:
                from config import DEPTH_PRECHECK_BATCH
            except Exception:
                DEPTH_PRECHECK_BATCH = False
            if DEPTH_PRECHECK_BATCH:
                scan_depth_episodes(INPUT_ROOT_PATH)
            # 执行一次转换
            run_conversion_core()
            # 若存在uuid映射，复制一份到 out 与 out1 的 task_info 便于对照
            try:
                mapping = load_uuid_mapping(dataset_root)
                if mapping:
                    # 保存UUID映射到原始数据根目录
                    mapping_txt = os.path.join(CURRENT_DATASET_ROOT, 'uuid_mapping.json')
                    with open(mapping_txt, 'w', encoding='utf-8') as f:
                        json.dump({str(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)
                    
                    # 同时保存到输出目录的task_info
                    mapping_txt_out = os.path.join(OUTPUT_ROOT_PATH, 'Logistics', 'task_info', 'uuid_mapping.json')
                    os.makedirs(os.path.dirname(mapping_txt_out), exist_ok=True)
                    with open(mapping_txt_out, 'w', encoding='utf-8') as f:
                        json.dump({str(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)
                    
                    if EXPORT_MODE == 1:  # both模式时也保存到低清目录
                        mapping_txt_low = os.path.join(OUTPUT_LOWRES_ROOT_PATH, 'Logistics', 'task_info', 'uuid_mapping.json')
                        os.makedirs(os.path.dirname(mapping_txt_low), exist_ok=True)
                        with open(mapping_txt_low, 'w', encoding='utf-8') as f:
                            json.dump({str(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)
                    print("📝 已复制UUID映射到 out 与 out1 的 task_info")
            except Exception as _e:
                print(f"⚠️  复制UUID映射失败: {_e}")
    else:
        # 单次转换：沿用当前 config 的 DATASET_ROOT，同时尝试按路径匹配参数
        # 单次模式预检（可选）
        try:
            from config import DEPTH_PRECHECK_ENABLED
        except Exception:
            DEPTH_PRECHECK_ENABLED = False
        try:
            auto_param = detect_param_profile_from_path(DATASET_ROOT)
            if auto_param and os.path.isdir(auto_param):
                PARAMETERS_PATH = auto_param
                print(f"🧭 已自动匹配参数目录: {PARAMETERS_PATH}")
            else:
                # 单数据集模式下匹配不上时默认使用 p064
                fallback = os.path.join(CURRENT_WORKSPACE, 'parameters_P064')
                if os.path.isdir(fallback):
                    PARAMETERS_PATH = fallback
                    print(f"🧭 未匹配到特定参数目录，使用默认: {PARAMETERS_PATH}")
                else:
                    print(f"🧭 未匹配到特定参数目录，且默认 p064 不存在，沿用: {PARAMETERS_PATH}")
        except Exception as _:
            pass
        # 单次模式：根据配置动态设置输出路径
        setup_dynamic_output_paths(DATASET_ROOT)
        if DEPTH_PRECHECK_ENABLED:
            scan_depth_episodes(INPUT_ROOT_PATH)
        run_conversion_core()

def run_conversion_core():
    """原 main 的单次转换核心逻辑"""
    global INPUT_ROOT_PATH, DEPTH_CHECK_INPUT_PATH, INJECTED_PRESET_UUID_MAP, INJECTED_ANNOTATION_DIRS
    print("🚀 开始批量数据格式转换")
    print(f"输入路径: {INPUT_ROOT_PATH}")
    print(f"输出路径: {OUTPUT_ROOT_PATH}")
    print(f"📁 使用相机参数目录: {PARAMETERS_PATH}")
    # Preflight self-check
    try:
        if not preflight_self_check():
            print("❌ 自检未通过，终止转换")
            return
    except Exception as _e:
        print(f"⚠️ 自检执行异常: {_e}")
    
    # 显示相机配置信息
    from config import PARAMETERS_FOLDER
    print(f"📁 使用相机参数文件夹: {PARAMETERS_FOLDER}")
    
    # 检查GPU编码器
    if USE_GPU_ENCODER:
        # 优先使用环境变量中的GPU编码器（来自GPU版本脚本）
        gpu_encoder = os.environ.get('GPU_ENCODER')
        if not gpu_encoder:
            gpu_encoder = check_gpu_encoder()
        
        if gpu_encoder:
            print(f"🎮 检测到GPU编码器: {gpu_encoder}")
        else:
            print("⚠️  未检测到GPU编码器，将使用CPU编码")
    else:
        gpu_encoder = None
        print("ℹ️  GPU编码已禁用，将使用CPU编码")
    
    # 检查异常episode列表
    from config import FAILED_EPISODES_FILE, DEPTH_CHECK_BEHAVIOR
    failed_episodes = set()
    if DEPTH_CHECK_BEHAVIOR == "skip" and os.path.exists(FAILED_EPISODES_FILE):
        with open(FAILED_EPISODES_FILE, 'r') as f:
            failed_episodes = set(line.strip() for line in f if line.strip())
        print(f"⚠️  跳过模式：将跳过 {len(failed_episodes)} 个异常episode")
    elif DEPTH_CHECK_BEHAVIOR == "warn" and os.path.exists(FAILED_EPISODES_FILE):
        print(f"⚠️  警告模式：检测到异常episode但继续处理全部数据")
    elif DEPTH_CHECK_BEHAVIOR == "off":
        print("ℹ️  深度检查已禁用，将处理全部episode")
    else:
        print("ℹ️  未检测到failed_episodes.txt，将处理全部episode")
    
    # 处理所有episode（健壮性：若输入目录不存在，尝试自动纠正）
    if not os.path.isdir(INPUT_ROOT_PATH):
        print(f"⚠️  输入帧目录不存在: {INPUT_ROOT_PATH}")
        # 尝试基于 DATASET_ROOT 自动修正
        try:
            candidate = os.path.join(DATASET_ROOT, CHUNK_DIR_NAME)
            if os.path.isdir(candidate):
                print(f"🧭 自动切换到: {candidate}")
                global DEPTH_CHECK_INPUT_PATH
                INPUT_ROOT_PATH = candidate
                DEPTH_CHECK_INPUT_PATH = INPUT_ROOT_PATH
            else:
                # 搜索 DATASET_ROOT 下的 chunk_* 目录
                import glob as _glob
                matches = sorted([p for p in _glob.glob(os.path.join(DATASET_ROOT, 'chunk_*')) if os.path.isdir(p)])
                if matches:
                    print(f"🧭 在 DATASET_ROOT 下发现候选输入目录: {matches[0]}")
                    INPUT_ROOT_PATH = matches[0]
                    DEPTH_CHECK_INPUT_PATH = INPUT_ROOT_PATH
                else:
                    print("❌ 无法找到可用的 chunk 目录，请检查配置中的 DATASET_ROOT/CHUNK_DIR_NAME")
                    return
        except Exception as e:
            print(f"❌ 自动修正输入路径失败: {e}")
            return

    episode_dirs = sorted([d for d in os.listdir(INPUT_ROOT_PATH)
                          if os.path.isdir(os.path.join(INPUT_ROOT_PATH, d)) and d.startswith('episode_')])
    print(f"找到 {len(episode_dirs)} 个episode")
    
    if len(episode_dirs) == 0:
        print("❌ 没有找到episode目录")
        return
    
    # 低清专用模式：仅导出低清三路视频与完整out1结构，跳过其它所有内容
    from config import EXPORT_MODE
    if EXPORT_MODE == 2:
        print("🟡 模式: low-only（仅导出低清三路视频与完整out1结构）")
        start_time = time.time()
        processed_uuids = []
        processed_mappings = []
        # 加载或初始化UUID映射（应基于当前数据集根目录，而非全局DATASET_ROOT）
        uuid_map = load_uuid_mapping(CURRENT_DATASET_ROOT)
        # 补充：从低清报告合并映射，支持上次中断但报告已写入的情况
        try:
            _report_map = load_uuid_mapping_from_report(CURRENT_DATASET_ROOT)
            if _report_map:
                # 报告为后备来源，不覆盖已有 uuid_map
                for _k, _v in _report_map.items():
                    uuid_map.setdefault(_k, _v)
        except Exception:
            pass

        # out1 目录
        alt_output_root = OUTPUT_LOWRES_ROOT_PATH
        alt_logistics_dir = os.path.join(alt_output_root, "Logistics")
        os.makedirs(alt_logistics_dir, exist_ok=True)

        # 生成 task_info（low-only不依赖out的meta，直接创建空meta）
        alt_task_info_dir = os.path.join(alt_logistics_dir, "task_info")
        os.makedirs(alt_task_info_dir, exist_ok=True)
        meta_filename = f"{MAIN_SCENE_NAME}-{SUB_SCENE_NAME}-{ACTION_NAME}.json"
        alt_meta_file_path = os.path.join(alt_task_info_dir, meta_filename)
        try:
            with open(alt_meta_file_path, 'w') as _f:
                pass
            print(f"✅ 创建 out1/task_info/{meta_filename}")
        except Exception as e:
            print(f"⚠️  创建 task_info 失败: {e}")

        # 创建场景三层结构（low-only 简化命名，不带体积/时长后缀）
        main_scene_dir_name = f"{MAIN_SCENE_NAME}"
        sub_scene_dir_name = f"{SUB_SCENE_NAME}"
        action_dir_name = f"{ACTION_NAME}"
        alt_main_scene_dir = os.path.join(alt_logistics_dir, main_scene_dir_name)
        alt_sub_scene_dir = os.path.join(alt_main_scene_dir, sub_scene_dir_name)
        alt_action_dir = os.path.join(alt_sub_scene_dir, action_dir_name)
        for d in [alt_main_scene_dir, alt_sub_scene_dir, alt_action_dir]:
            os.makedirs(d, exist_ok=True)

        # 进度统计（用于报告实时刷新）
        total_ep = len(episode_dirs)
        completed_ep = 0

        # 遍历 episodes，逐个生成低清视频到 out1
        for episode_dir in episode_dirs:
            episode_path = os.path.join(INPUT_ROOT_PATH, episode_dir)
            try:
                episode_number_str = episode_dir.replace('episode_', '')
                episode_index = int(episode_number_str)
            except Exception:
                print(f"⚠️  无法解析episode序号，跳过: {episode_dir}")
                continue

            # 若已有映射，先做“已完成三文件”预检测；全部存在则跳过
            existing_uuid = uuid_map.get(episode_index)
            if existing_uuid:
                _dst_uuid_dir = os.path.join(alt_action_dir, existing_uuid)
                _video_dir = os.path.join(_dst_uuid_dir, 'camera', 'video')
                _targets = [
                    os.path.join(_video_dir, 'hand_left_color_low.mp4'),
                    os.path.join(_video_dir, 'hand_right_color_low.mp4'),
                    os.path.join(_video_dir, 'head_front_color_low.mp4'),
                ]
                if all(os.path.exists(p) for p in _targets):
                    print(f"⏭️  低清已存在，跳过 episode_{episode_index:06d} (uuid={existing_uuid})")
                    processed_uuids.append(existing_uuid)
                    processed_mappings.append((episode_index, episode_path, existing_uuid))
                    completed_ep += 1
                    try:
                        update_conversion_progress(completed_ep, total_ep, existing_uuid, f"已存在跳过 Episode {episode_index}")
                    except Exception:
                        pass
                    continue

            # 复用或生成UUID（无法跳过时才需要）
            ep_uuid = existing_uuid or str(uuid.uuid4())
            uuid_map[episode_index] = ep_uuid
            dst_uuid_dir = os.path.join(alt_action_dir, ep_uuid)
            camera_dir = os.path.join(dst_uuid_dir, 'camera')
            video_dir = os.path.join(camera_dir, 'video')
            os.makedirs(video_dir, exist_ok=True)

            # 报告：开始处理该 episode
            try:
                update_conversion_progress(completed_ep, total_ep, "", f"处理Episode {episode_index}")
            except Exception:
                pass

            # 生成对齐索引（低清模式也支持对齐）
            aligned_indices = None
            parquet_file = os.path.join(PARQUET_PATH, f"episode_{episode_index:06d}.parquet")
            aligned_rows = generate_aligned_frame_indices(episode_path, parquet_file)
            if aligned_rows:
                # 提取各流的索引
                aligned_indices = {
                    'hand_l': [row['hand_l_idx'] for row in aligned_rows],
                    'hand_r': [row['hand_r_idx'] for row in aligned_rows], 
                    'headf': [row['headf_idx'] for row in aligned_rows]
                }
                print(f"🎯 使用多模态对齐: {len(aligned_rows)} 个对齐帧")

            ok = generate_lowres_color_videos_from_images(
                episode_path, video_dir, width=LOWRES_WIDTH, height=LOWRES_HEIGHT, gpu_encoder=gpu_encoder, aligned_indices=aligned_indices
            )
            if ok:
                processed_uuids.append(ep_uuid)
                processed_mappings.append((episode_index, episode_path, ep_uuid))
                completed_ep += 1
                # 报告：完成该 episode 并记录 UUID
                try:
                    update_conversion_progress(completed_ep, total_ep, ep_uuid, f"完成Episode {episode_index}")
                except Exception:
                    pass
            else:
                print(f"❌ 低清视频生成失败，跳过: {episode_dir}")

        # 汇总与报告（仅 out1）
        processed_count = len(processed_uuids)
        total_time = time.time() - start_time
        processed_duration_seconds = processed_count * 40
        processed_duration_str = format_duration(processed_duration_seconds)
        size_gb = calculate_output_size(alt_logistics_dir)
        size_str = format_size_gb(size_gb)

        # 报告：仅保存在低清输出根目录（data_diqing）
        report_path = os.path.join(OUTPUT_LOWRES_ROOT_PATH, "conversion_report_lowres.txt")
        try:
            # 报告：整体完成
            try:
                update_conversion_progress(completed_ep, total_ep, "", f"转换完成 ({completed_ep}/{total_ep})")
            except Exception:
                pass

            with open(report_path, 'w', encoding='utf-8') as rf:
                header = (
                    f"Conversion Time: {datetime.datetime.now().isoformat()}\n"
                    f"INPUT_ROOT_PATH: {INPUT_ROOT_PATH}\n"
                    f"PARQUET_PATH: {PARQUET_PATH}\n"
                    f"OUTPUT_ROOT_PATH: {OUTPUT_LOWRES_ROOT_PATH}\n"
                    f"FINAL_OUTPUT_DIR: {alt_action_dir}\n"
                    f"EXPORT_MODE: 低清\n"
                    "----------------------------------------\n"
                    "Episode UUID Mapping:\n"
                )
                rf.write(header)
                for ep_idx, ep_path, ep_uuid in sorted(processed_mappings, key=lambda x: x[0]):
                    rf.write(f"Episode {ep_idx}: {ep_uuid}\n")
            print(f"📝 已生成低清模式报告: {report_path}")
        except Exception as e:
            print(f"⚠️  生成报告失败: {e}")

        # 持久化UUID映射在当前源数据集根目录
        save_uuid_mapping(CURRENT_DATASET_ROOT, uuid_map)

        print(f"\n{'='*60}")
        print("🎉 低清模式转换完成！")
        print(f"📊 统计信息:")
        print(f"  - 总episode数: {len(episode_dirs)}")
        print(f"  - 成功处理: {processed_count}")
        print(f"  - 处理成功率: {processed_count/max(1, len(episode_dirs))*100:.1f}%")
        print(f"  - 总耗时: {total_time:.2f} 秒")
        print(f"  - 实际处理时长: {processed_duration_str}")
        print(f"  - 实际输出大小: {size_str}")
        print(f"📁 最终输出路径(out1): {alt_action_dir}")
        print(f"{'='*60}")
        return
    
    # 计算总时长（每个episode 40秒）
    total_duration_seconds = len(episode_dirs) * 40
    total_duration_str = format_duration(total_duration_seconds)
    
    # 先创建临时目录进行转换
    temp_output_dir = os.path.join(OUTPUT_ROOT_PATH, "temp_output")
    os.makedirs(temp_output_dir, exist_ok=True)
    # Load resume checkpoint
    completed_from_ckpt = load_checkpoint()
    if completed_from_ckpt:
        print(f"📋 断点恢复：已完成 {len(completed_from_ckpt)} 个episode，将跳过这些索引")
    # 确保在开始时就创建/刷新断点文件（即使为空），行为与预生成报告一致
    try:
        save_checkpoint(completed_from_ckpt)
    except Exception:
        pass

    # 预解析 task_info：uuid -> N 映射
    uuid_to_cutoff: Dict[str, int] = {}
    try:
        anno_dirs = [
            os.path.join(CURRENT_DATASET_ROOT, 'task_info'),
            os.path.join(CURRENT_DATASET_ROOT, 'Logistics', 'task_info')
        ]
        # 若外部注入了标注目录，则放到最前面优先查找
        try:
            if isinstance(INJECTED_ANNOTATION_DIRS, list):
                anno_dirs = INJECTED_ANNOTATION_DIRS + anno_dirs
        except Exception:
            pass
        anno_json = None
        for ad in anno_dirs:
            if os.path.isdir(ad):
                cand = sorted([os.path.join(ad, f) for f in os.listdir(ad) if f.endswith('.json')], key=lambda p: os.path.getmtime(p), reverse=True)
                if cand:
                    anno_json = cand[0]
                    break
        if anno_json:
            with open(anno_json, 'r', encoding='utf-8') as jf:
                anno = json.load(jf)
            # 获取预设UUID映射，用于将episode_id转换为实际UUID
            preset_uuid_map = {}
            try:
                if isinstance(INJECTED_PRESET_UUID_MAP, dict):
                    preset_uuid_map = INJECTED_PRESET_UUID_MAP
            except Exception:
                pass
            
            if isinstance(anno, list):
                for item in anno:
                    if isinstance(item, dict) and 'episode_id' in item:
                        episode_id = str(item.get('episode_id'))
                        n = _extract_cutoff_frames_for_uuid([item], episode_id)
                        if isinstance(n, int) and n > 0:
                            # 尝试将episode_id转换为实际UUID
                            try:
                                episode_index = int(episode_id.replace('episode_', ''))
                                actual_uuid = preset_uuid_map.get(episode_index)
                                if actual_uuid:
                                    uuid_to_cutoff[actual_uuid] = n
                                else:
                                    # 如果找不到对应UUID，仍使用episode_id作为键
                                    uuid_to_cutoff[episode_id] = n
                            except (ValueError, KeyError):
                                # 如果转换失败，仍使用episode_id作为键
                                uuid_to_cutoff[episode_id] = n
            elif isinstance(anno, dict) and 'episode_id' in anno:
                episode_id = str(anno.get('episode_id'))
                n = _extract_cutoff_frames_for_uuid(anno, episode_id)
                if isinstance(n, int) and n > 0:
                    # 尝试将episode_id转换为实际UUID
                    try:
                        episode_index = int(episode_id.replace('episode_', ''))
                        actual_uuid = preset_uuid_map.get(episode_index)
                        if actual_uuid:
                            uuid_to_cutoff[actual_uuid] = n
                        else:
                            # 如果找不到对应UUID，仍使用episode_id作为键
                            uuid_to_cutoff[episode_id] = n
                    except (ValueError, KeyError):
                        # 如果转换失败，仍使用episode_id作为键
                        uuid_to_cutoff[episode_id] = n
            print(f"🗺️  已解析UUID截止帧映射: {len(uuid_to_cutoff)} 条")
        else:
            print("ℹ️  未找到标注JSON，所有uuid不裁剪")
    except Exception as _e:
        print(f"⚠️  预解析 task_info 失败，所有uuid不裁剪: {_e}")
    
    # 性能配置显示
    print(f"⚙️  性能配置:")
    print(f"   - ffmpeg线程数: {FFMPEG_THREADS}")
    print(f"   - 最大并发episode: {MAX_PARALLEL_EPISODES}")
    print(f"   - 单episode内最大并发任务: {MAX_TASKS_PER_EPISODE}")
    print(f"   - 运动学计算: {'启用' if ENABLE_KINEMATICS else '禁用'}")
    print(f"   - 深度检查模式: {DEPTH_CHECK_BEHAVIOR}")
    print(f"   - x264预设: {X264_PRESET} (CRF: {X264_CRF})")
    
    start_time = time.time()
    processed_uuids = []
    processed_mappings = []  # (episode_index, episode_path, uuid)
    
    # 准备episode处理任务 - 修复episode索引对应问题
    episode_tasks = []
    for episode_dir in episode_dirs:
        episode_path = os.path.join(INPUT_ROOT_PATH, episode_dir)
        # 跳过异常episode
        if episode_path in failed_episodes:
            print(f"⏭️  跳过异常episode: {episode_path}")
            continue
        
        # 从episode目录名中提取真实的episode编号
        try:
            # 支持多种命名格式：episode_1, episode_001, episode_000001 等
            episode_number_str = episode_dir.replace('episode_', '')
            episode_index = int(episode_number_str)
            print(f"🔗 解析episode目录: {episode_dir} -> episode_index: {episode_index}")
        except ValueError as e:
            print(f"❌ 无法解析episode目录名: {episode_dir}, 错误: {e}")
            continue
        
        # Skip if checkpoint marks as completed
        if episode_index in completed_from_ckpt:
            print(f"⏭️  断点标记已完成，跳过 episode_{episode_index:06d}")
            continue
        episode_tasks.append((episode_path, temp_output_dir, episode_index, gpu_encoder, uuid_to_cutoff))
    
    print(f"🚀 开始处理 {len(episode_tasks)} 个episode（并发数: {MAX_PARALLEL_EPISODES}）...")
    
    # 验证episode目录和parquet文件的对应关系
    print("🔍 验证episode目录和parquet文件对应关系...")
    valid_tasks = []
    missing_parquets = []
    missing_episodes = []
    
    # 获取所有parquet文件列表
    import glob
    all_parquet_files = sorted(glob.glob(os.path.join(PARQUET_PATH, "episode_*.parquet")))
    parquet_indices = set()
    for parquet_file in all_parquet_files:
        basename = os.path.basename(parquet_file)
        try:
            # 从parquet文件名提取索引
            episode_number_str = basename.replace('episode_', '').replace('.parquet', '')
            parquet_index = int(episode_number_str)
            parquet_indices.add(parquet_index)
        except ValueError:
            print(f"❌ 无法解析parquet文件名: {basename}")
    
    # 检查每个episode目录
    episode_indices = set()
    for episode_path, temp_output_dir, episode_index, gpu_encoder, uuid_to_cutoff_map in episode_tasks:
        episode_indices.add(episode_index)
        parquet_file = os.path.join(PARQUET_PATH, f"episode_{episode_index:06d}.parquet")
        if os.path.exists(parquet_file):
            valid_tasks.append((episode_path, temp_output_dir, episode_index, gpu_encoder, uuid_to_cutoff_map))
            print(f"✅ 验证通过: chunk/{os.path.basename(episode_path)} <-> data/episode_{episode_index:06d}.parquet")
        else:
            missing_parquets.append((episode_index, os.path.basename(episode_path)))
            print(f"❌ 验证失败: chunk/{os.path.basename(episode_path)} 对应的 parquet文件不存在: episode_{episode_index:06d}.parquet")
    
    # 检查是否有parquet文件没有对应的episode目录
    for parquet_index in parquet_indices:
        if parquet_index not in episode_indices:
            missing_episodes.append(parquet_index)
            print(f"❌ 发现孤立的parquet文件: episode_{parquet_index:06d}.parquet 没有对应的episode目录")
    
    # 检查序号连续性
    if episode_indices:
        min_episode = min(episode_indices)
        max_episode = max(episode_indices)
        expected_episodes = set(range(min_episode, max_episode + 1))
        missing_sequential = expected_episodes - episode_indices
        
        if missing_sequential:
            print(f"❌ Episode序号不连续，缺失的序号: {sorted(missing_sequential)}")
    
    if parquet_indices:
        min_parquet = min(parquet_indices)
        max_parquet = max(parquet_indices)
        expected_parquets = set(range(min_parquet, max_parquet + 1))
        missing_sequential_parquets = expected_parquets - parquet_indices
        
        if missing_sequential_parquets:
            print(f"❌ Parquet序号不连续，缺失的序号: {sorted(missing_sequential_parquets)}")
    
    # 统计信息
    print(f"\n📊 验证统计:")
    print(f"  - chunk目录中的episode数量: {len(episode_indices)}")
    print(f"  - data目录中的parquet数量: {len(parquet_indices)}")
    print(f"  - 成功匹配的数量: {len(valid_tasks)}")
    print(f"  - 缺失parquet文件的episode数量: {len(missing_parquets)}")
    print(f"  - 缺失episode目录的parquet数量: {len(missing_episodes)}")
    
    # 根据用户要求：不再因此停止，打印告警并继续，仅对 valid_tasks 执行
    if len(episode_indices) != len(parquet_indices) or missing_parquets or missing_episodes or missing_sequential or missing_sequential_parquets:
        print(f"\n⚠️ 发现数据不一致，将跳过缺失项继续转换：")
        if len(episode_indices) != len(parquet_indices):
            print(f"  - 数量不一致: chunk有{len(episode_indices)}个episode，data有{len(parquet_indices)}个parquet")
        if missing_parquets:
            print(f"  - 有{len(missing_parquets)}个episode目录缺少对应的parquet文件")
            for episode_index, episode_name in missing_parquets:
                print(f"    · 跳过 {episode_name} (缺失: episode_{episode_index:06d}.parquet)")
        if missing_episodes:
            print(f"  - 有{len(missing_episodes)}个parquet文件缺少对应的episode目录")
            for parquet_index in missing_episodes:
                print(f"    · 跳过 parquet: episode_{parquet_index:06d}.parquet")
        if missing_sequential:
            print(f"  - episode序号不连续，缺失: {sorted(missing_sequential)}")
        if missing_sequential_parquets:
            print(f"  - parquet序号不连续，缺失: {sorted(missing_sequential_parquets)}")
    
    if not valid_tasks:
        print("❌ 没有找到有效的episode-parquet对应关系，请检查数据完整性")
        return
    
    print(f"✅ 验证完成，共找到 {len(valid_tasks)} 个有效的episode-parquet对应关系")
    
    # 使用线程池并发处理episode（IO密集型任务适合线程池）
    # 顺序逐UUID处理，避免跨UUID并发造成的N污染（episode内仍使用并发任务）
    # 读取uuid映射以复用UUID
    preset_map = {}
    try:
        # 优先使用外部注入的预设映射（例如 run_highres_from_annotations 注入）
        if isinstance(INJECTED_PRESET_UUID_MAP, dict) and INJECTED_PRESET_UUID_MAP:
            preset_map = {int(k): str(v) for k, v in INJECTED_PRESET_UUID_MAP.items()}
            print(f"🔗 使用注入的UUID映射: {len(preset_map)} 条")
        else:
            preset_map = load_uuid_mapping(CURRENT_DATASET_ROOT)
            if not preset_map:
                preset_map = load_uuid_mapping_from_report(CURRENT_DATASET_ROOT)
            print(f"🔗 使用源数据集UUID映射: {len(preset_map)} 条")
    except Exception as e:
        print(f"⚠️ UUID映射加载失败: {e}")
        preset_map = {}
    
    # 打印映射详情，便于核对
    if uuid_to_cutoff:
        print("📌 UUID→N 映射：")
        for u, n in uuid_to_cutoff.items():
            print(f"  - {u}: N={n}")

    completed = 0
    # 若校验后无有效任务，则以发现到的 episode 数作为进度的分母，避免报告停留在 0/0
    total_tasks = len(valid_tasks) if len(valid_tasks) > 0 else len(episode_dirs)
    
    # 更新报告：开始转换（保证分母不为0时能正确展示）
    update_conversion_progress(0, total_tasks, "", "开始转换")
    
    for task in valid_tasks:
        episode_path, temp_output_dir, episode_index, gpu_encoder, uuid_to_cutoff_map = task
        preset_uuid = preset_map.get(episode_index)
        
        # 调试信息：显示UUID映射状态
        if preset_uuid:
            print(f"🔗 Episode {episode_index} 使用预设UUID: {preset_uuid}")
        else:
            print(f"⚠️ Episode {episode_index} 未找到预设UUID，将生成新UUID")
        
        # 更新进度：开始处理当前episode
        update_conversion_progress(completed, total_tasks, "", f"处理Episode {episode_index}")
        # 断点：记录开始处理该 episode（不标记完成）
        try:
            update_checkpoint_started(episode_index, completed_from_ckpt)
        except Exception:
            pass
        
        try:
            episode_uuid = process_episode(episode_path, temp_output_dir, episode_index, gpu_encoder, preset_uuid, uuid_to_cutoff_map)
            processed_uuids.append(episode_uuid)
            processed_mappings.append((episode_index, episode_path, episode_uuid))
            completed += 1
            
            # 先更新映射：必须用真实 episode_index 写入一行；只更新映射不改进度数
            update_conversion_progress(episode_index, total_tasks, episode_uuid, f"完成Episode {episode_index}", only_update_mapping=True)
            # 再更新进度计数（不附带 uuid，避免覆盖错误索引）
            update_conversion_progress(completed, total_tasks, "", f"完成Episode {episode_index}")
            # Update checkpoint persistently
            try:
                completed_from_ckpt.add(episode_index)
                save_checkpoint(completed_from_ckpt)
            except Exception as _ck_e:
                print(f"⚠️ 保存断点失败: {_ck_e}")
            
        except Exception as e:
            print(f"❌ Episode {episode_index} 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 更新报告：转换完成
    update_conversion_progress(completed, total_tasks, "", f"转换完成 ({completed}/{total_tasks})")
    
    # 计算实际输出大小和处理统计
    actual_size_gb = calculate_output_size(temp_output_dir)
    size_str = format_size_gb(actual_size_gb)
    processed_count = len(processed_uuids)  # 实际处理成功的episode数量
    processed_duration_seconds = processed_count * 40  # 实际处理的总时长
    processed_duration_str = format_duration(processed_duration_seconds)
    
    # 创建Logistics主目录结构
    logistics_main_dir = os.path.join(OUTPUT_ROOT_PATH, "Logistics")
    os.makedirs(logistics_main_dir, exist_ok=True)
    
    # 先清理旧的同类型输出目录（避免累积）
    import glob
    existing_main_dirs = glob.glob(os.path.join(logistics_main_dir, f"{MAIN_SCENE_NAME}-*"))
    if existing_main_dirs:
        print(f"🧹 清理 {len(existing_main_dirs)} 个旧输出目录...")
        for old_dir in existing_main_dirs:
            if os.path.isdir(old_dir):
                print(f"   删除: {os.path.basename(old_dir)}")
                import shutil
                shutil.rmtree(old_dir)
    
    # 创建task_info文件夹和meta.json文件
    task_info_dir = os.path.join(logistics_main_dir, "task_info")
    os.makedirs(task_info_dir, exist_ok=True)
    
    # 创建完全空的meta.json文件
    meta_filename = f"{MAIN_SCENE_NAME}-{SUB_SCENE_NAME}-{ACTION_NAME}.json"
    meta_file_path = os.path.join(task_info_dir, meta_filename)
    with open(meta_file_path, 'w') as f:
        pass  # 创建完全空的文件
    
    print(f"✅ 创建task_info文件夹和meta.json文件: {meta_filename}")
    
    # 创建多级子目录结构（使用实际处理的数据）
    main_scene_dir_name = f"{MAIN_SCENE_NAME}-{size_str}_{processed_count}counts_{processed_duration_str}"
    main_scene_dir = os.path.join(logistics_main_dir, main_scene_dir_name)
    sub_scene_dir_name = f"{SUB_SCENE_NAME}-{size_str}_{processed_count}counts_{processed_duration_str}"
    sub_scene_dir = os.path.join(main_scene_dir, sub_scene_dir_name)
    action_dir_name = f"{ACTION_NAME}-{size_str}_{processed_count}counts_{processed_duration_str}"
    final_action_dir = os.path.join(sub_scene_dir, action_dir_name)
    
    # 移动临时目录到最终位置
    if os.path.exists(temp_output_dir):
        import shutil
        # 确保目标目录不存在
        if os.path.exists(final_action_dir):
            print(f"⚠️  目标目录已存在，正在删除: {final_action_dir}")
            shutil.rmtree(final_action_dir)
        # 移动临时目录到最终位置
        shutil.move(temp_output_dir, final_action_dir)
        print(f"✅ 成功移动临时目录到: {final_action_dir}")
    else:
        print(f"⚠️  临时目录不存在: {temp_output_dir}")

    # 根据导出模式生成相应的转换报告
    if EXPORT_MODE == 2:  # low-only模式
        # 低清转换报告仅保存在低清输出根目录
        output_root = OUTPUT_LOWRES_ROOT_PATH
        report_path = os.path.join(output_root, "conversion_report_lowres.txt")
        logistics_dir = os.path.join(output_root, "Logistics")
        action_dir = os.path.join(logistics_dir, SUB_SCENE_NAME, ACTION_NAME)
    else:  # high-only或both模式
        # 高清转换报告保存在高清输出根目录
        output_root = OUTPUT_ROOT_PATH
        report_path = os.path.join(output_root, "conversion_report.txt")
        action_dir = final_action_dir
        
    try:
        with open(report_path, 'w', encoding='utf-8') as rf:
            # 报告头
            mode_name = "低清" if EXPORT_MODE == 2 else "高清"
            header = (
                f"Conversion Time: {datetime.datetime.now().isoformat()}\n"
                f"INPUT_ROOT_PATH: {INPUT_ROOT_PATH}\n"
                f"PARQUET_PATH: {PARQUET_PATH}\n"
                f"OUTPUT_ROOT_PATH: {output_root}\n"
                f"FINAL_OUTPUT_DIR: {action_dir}\n"
                f"EXPORT_MODE: {mode_name}\n"
                f"DEPTH_PRECHECK_RAN: {LAST_PRECHECK_RAN}\n"
                f"DEPTH_CHECK_BEHAVIOR: {DEPTH_CHECK_BEHAVIOR}\n"
                "----------------------------------------\n"
            )
            rf.write(header)
            # 写入深度预检异常（若有）
            if LAST_PRECHECK_RAN:
                rf.write("Depth Precheck - Failed Episodes:\n")
                if LAST_PRECHECK_FAILED_EPISODES:
                    for ep in sorted(LAST_PRECHECK_FAILED_EPISODES):
                        rf.write(f"  - {ep}\n")
                else:
                    rf.write("  (none)\n")
                rf.write("----------------------------------------\n")
            # 映射明细（按 episode_index 排序）
            for ep_idx, ep_path, ep_uuid in sorted(processed_mappings, key=lambda x: x[0]):
                parquet_file = os.path.join(PARQUET_PATH, f"episode_{ep_idx:06d}.parquet")
                uuid_out_dir = os.path.join(action_dir, ep_uuid)
                line = (
                    f"episode_{ep_idx:06d} | uuid={ep_uuid} | src={ep_path} | parquet={parquet_file} | out={uuid_out_dir}\n"
                )
                rf.write(line)
        print(f"📝 已生成{mode_name}转换报告: {report_path}")
        print(f"📁 报告保存在原始数据目录: {CURRENT_DATASET_ROOT}")
    except Exception as e:
        print(f"⚠️  生成转换报告失败: {e}")
    
    # =============== 基础结构检查（仅 out；只检查结构完整与重复；不阻塞） ===============
    try:
        print("🔎 基础结构检查（out）：开始...")
        import importlib.util as _il
        _pth = os.path.join(CURRENT_WORKSPACE, '01_basic_structure_check.py')
        spec = _il.spec_from_file_location('basic_check', _pth)
        mod = _il.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        checker = mod.BasicStructureChecker()
        checker.check_batch(final_action_dir)
        integ = checker.results['checks']['data_integrity']
        dup = checker.results['checks']['duplicate_data']
        has_issue = (integ['failed'] > 0) or (dup['failed'] > 0)
        if has_issue:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            # 按你的要求：将异常报告放到 Logistics/task_info 目录（在复制源 task_info 之后）
            task_info_dir = os.path.join(logistics_main_dir, "task_info")
            os.makedirs(task_info_dir, exist_ok=True)
            issue_report = os.path.join(task_info_dir, f"basic_issues_{ts}.txt")
            with open(issue_report, 'w', encoding='utf-8') as rf:
                rf.write("基础结构检查（仅 out）\n")
                rf.write("仅统计：数据完整性、重复\n\n")
                rf.write(f"总episodes: {checker.results['total_episodes']}\n")
                rf.write(f"完整性失败: {integ['failed']}\n")
                rf.write(f"重复失败: {dup['failed']}\n\n")
                rf.write("[完整性缺失]\n")
                for d in integ['details']:
                    if not d.get('passed', True):
                        rf.write(f"- {d['episode']}\n")
                        miss = d.get('missing_critical', []) + d.get('missing_important', [])
                        if miss:
                            rf.write("  缺失: " + ", ".join(miss) + "\n")
                rf.write("\n[重复]\n")
                for d in dup['details']:
                    if not d.get('passed', True):
                        rf.write(f"- {d['episode']} 与 {d.get('duplicate_with')} 重复\n")
            print(f"⚠️ 基础结构检查发现问题，已生成: {issue_report}")
        else:
            print("✅ 基础结构检查通过（无报告文件）")
    except Exception as e:
        print(f"⚠️ 基础结构检查异常（忽略，不阻塞）: {e}")
    
    # =============== 生成 out1 目录（根据导出模式决定） ===============
    from config import EXPORT_MODE
    if EXPORT_MODE in [1, 2]:  # both 或 low-only 模式需要生成out1
        try:
            # 使用配置文件中的低清晰度输出路径
            alt_output_root = OUTPUT_LOWRES_ROOT_PATH
            alt_logistics_dir = os.path.join(alt_output_root, "Logistics")
            os.makedirs(alt_logistics_dir, exist_ok=True)
            # 明确打印 out 与 out1 根路径
            try:
                print(f"📁 out 根路径: {OUTPUT_ROOT_PATH}")
                print(f"📁 out1 根路径: {alt_output_root}")
            except Exception:
                pass

            # 复制 task_info
            alt_task_info_dir = os.path.join(alt_logistics_dir, "task_info")
            os.makedirs(alt_task_info_dir, exist_ok=True)
            try:
                import shutil as _shutil
                _shutil.copy2(meta_file_path, os.path.join(alt_task_info_dir, os.path.basename(meta_file_path)))
            except Exception as e:
                print(f"⚠️  复制 task_info 失败: {e}")

            # 创建与 out 相同的三层结构（空目录）
            alt_main_scene_dir = os.path.join(alt_logistics_dir, main_scene_dir_name)
            alt_sub_scene_dir = os.path.join(alt_main_scene_dir, sub_scene_dir_name)
            alt_action_dir = os.path.join(alt_sub_scene_dir, action_dir_name)
            for d in [alt_main_scene_dir, alt_sub_scene_dir, alt_action_dir]:
                os.makedirs(d, exist_ok=True)

            # 逐 UUID 复制结构，并在每个 UUID 目录下生成低清晰度 color
            if os.path.isdir(final_action_dir):
                for ep_uuid in sorted(os.listdir(final_action_dir)):
                    src_uuid_dir = os.path.join(final_action_dir, ep_uuid)
                    if not os.path.isdir(src_uuid_dir):
                        continue
                    # out1 下对应的 UUID 目录
                    dst_uuid_dir = os.path.join(alt_action_dir, ep_uuid)
                    # 创建空的基本结构
                    camera_dir = os.path.join(dst_uuid_dir, 'camera')
                    video_dir = os.path.join(camera_dir, 'video')
                    os.makedirs(video_dir, exist_ok=True)

                    if EXPORT_MODE == 1:  # both 模式：从高清视频生成低清
                        # 在 out 的对应 UUID 目录下生成低清晰度到 out1 的 video 目录（不在 out 产生 vido_color）
                        src_video_dir = os.path.join(src_uuid_dir, 'camera', 'video')
                        # 直接输出到 out1 的 video 目录
                        generate_lowres_color_videos(src_video_dir, video_dir, width=LOWRES_WIDTH, height=LOWRES_HEIGHT, create_subdir=False)
                    elif EXPORT_MODE == 2:  # low-only 模式：低清视频已经在process_episode中生成
                        # 低清视频已经在process_episode中直接生成到out1，这里不需要额外处理
                        pass
                        
            print(f"✅ out1 目录已就绪: {alt_logistics_dir}")
        except Exception as e:
            print(f"⚠️  生成 out1 目录失败: {e}")
    else:
        print(f"ℹ️  EXPORT_MODE={EXPORT_MODE}，跳过out1目录生成")
    
    # 用源 task_info 覆盖 out 的 task_info（携带标注与异常报告）
    try:
        copy_task_info_from_source(CURRENT_DATASET_ROOT, logistics_main_dir)
    except Exception as _e:
        print(f"⚠️  覆盖 task_info 异常: {_e}")

    print(f"📁 输出目录结构:")
    print(f"  Logistics/")
    print(f"  ├── task_info/")
    print(f"  │   └── {meta_filename}")
    print(f"  └── {main_scene_dir_name}/")
    print(f"      └── {sub_scene_dir_name}/")
    print(f"          └── {action_dir_name}/")
    print(f"              └── [UUID directories]")
    print(f"  实际处理时长: {processed_duration_str}")
    print(f"  实际输出大小: {size_str}")
    
    print(f"\n{'='*60}")
    print("🎉 批量转换完成！")
    print(f"📊 统计信息:")
    print(f"  - 总episode数: {len(episode_dirs)}")
    print(f"  - 成功处理: {processed_count}")
    print(f"  - 处理成功率: {processed_count/len(episode_dirs)*100:.1f}%")
    print(f"  - 总耗时: {total_time:.2f} 秒")
    print(f"  - 平均耗时: {total_time/max(1, processed_count):.2f} 秒/episode")
    print(f"  - 实际输出大小: {size_str}")
    print(f"📁 最终输出路径: {final_action_dir}")
    print(f"📋 生成的UUID ({len(processed_uuids)} 个):")
    for uuid_str in processed_uuids:
        print(f"  - {uuid_str}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 
