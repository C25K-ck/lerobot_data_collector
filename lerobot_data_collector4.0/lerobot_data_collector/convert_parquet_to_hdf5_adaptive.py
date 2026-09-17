#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应Parquet转HDF5转换脚本
支持多种采集任务，自动识别字段和维度，按照相同字段名和维度生成HDF5
"""

import os
import sys
import argparse
import uuid
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import importlib
import json


def ensure_hdf5_compatible(data, str_len=50):
    """
    Ensure data is compatible with HDF5 format
    Convert to numpy array and handle string types
    """
    arr = np.asarray(data)
    if arr.dtype.kind in {'U', 'O'}:
        # String data: convert to fixed-length string array
        if arr.dtype.kind == 'O':
            # Object array (mixed types), convert to string
            arr = np.array([str(x) if x is not None else '' for x in arr])
        # Convert unicode to fixed-length bytes
        max_len = max(len(str(x)) for x in arr) if len(arr) > 0 else str_len
        max_len = max(max_len, str_len)
        return np.array([str(x).encode('utf-8')[:max_len].ljust(max_len, b'\0') 
                        for x in arr], dtype=f'S{max_len}')
    return arr


def analyze_parquet_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze parquet DataFrame structure to understand field names, types, and dimensions
    Returns a dictionary with field information
    """
    structure = {
        'columns': list(df.columns),
        'shape': df.shape,
        'field_info': {}
    }
    
    print(f"📊 分析Parquet结构:")
    print(f"  - 总列数: {len(df.columns)}")
    print(f"  - 数据行数: {df.shape[0]}")
    
    # Analyze each column
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'has_nulls': df[col].isna().any(),
            'sample_value': None,
            'dimension': None,
            'is_nested': False
        }
        
        # Get sample value to understand structure
        sample = df[col].iloc[0] if len(df) > 0 else None
        
        if sample is not None:
            if isinstance(sample, (list, np.ndarray)):
                col_info['is_nested'] = True
                col_info['dimension'] = len(sample) if hasattr(sample, '__len__') else None
                col_info['sample_value'] = sample
                print(f"  - {col}: 嵌套数组, 维度={col_info['dimension']}, dtype={col_info['dtype']}")
            elif isinstance(sample, (dict, pd.Series)):
                col_info['is_nested'] = True
                col_info['sample_value'] = sample
                print(f"  - {col}: 嵌套对象/字典")
            else:
                col_info['sample_value'] = sample
                print(f"  - {col}: 标量, dtype={col_info['dtype']}")
        
        structure['field_info'][col] = col_info
    
    return structure


def detect_hdf5_structure(parquet_structure: Dict[str, Any]) -> Dict[str, str]:
    """
    Detect HDF5 group structure from parquet column names
    Column names with dots (e.g., 'state.joint.position') are interpreted as group paths
    """
    hdf5_structure = {}
    
    for col_name in parquet_structure['columns']:
        if '.' in col_name:
            # Split by dots to create group path
            parts = col_name.split('.')
            hdf5_structure[col_name] = {
                'group_path': '/'.join(parts[:-1]),
                'dataset_name': parts[-1],
                'full_path': col_name.replace('.', '/')
            }
        else:
            # Root level dataset
            hdf5_structure[col_name] = {
                'group_path': '/',
                'dataset_name': col_name,
                'full_path': col_name
            }
    
    return hdf5_structure


def create_hdf5_groups(f: h5py.File, group_path: str):
    """
    Create nested groups in HDF5 file if they don't exist
    """
    if group_path == '/' or group_path == '':
        return f
    
    parts = [p for p in group_path.split('/') if p]
    current = f
    
    for part in parts:
        if part not in current:
            current = current.create_group(part)
        else:
            current = current[part]
    
    return current


def convert_parquet_to_hdf5_adaptive(parquet_file: str, output_hdf5: str, 
                                    max_frames: Optional[int] = None) -> bool:
    """
    Convert parquet file to HDF5 with adaptive field detection
    """
    try:
        print(f"\n{'='*60}")
        print(f"🔄 开始转换: {os.path.basename(parquet_file)}")
        print(f"{'='*60}")
        
        # Check if parquet file exists
        if not os.path.exists(parquet_file):
            print(f"❌ Parquet文件不存在: {parquet_file}")
            return False
        
        # Try to read parquet file with different engines
        print("📖 读取Parquet文件...")
        engines = ['pyarrow', 'fastparquet']
        df = None
        
        for engine in engines:
            try:
                print(f"  尝试使用 {engine} 引擎...")
                df = pd.read_parquet(parquet_file, engine=engine)
                print(f"  ✅ 使用 {engine} 引擎读取成功")
                break
            except Exception as e:
                print(f"  ⚠️  {engine} 引擎失败: {e}")
                continue
        
        if df is None:
            print("❌ 所有引擎都失败，无法读取Parquet文件")
            return False
        
        print(f"✅ 成功读取Parquet文件，形状: {df.shape}")
        
        # Truncate if max_frames is specified
        if isinstance(max_frames, int) and max_frames > 0:
            df = df.iloc[:max_frames]
            print(f"✂️  截断到前 {len(df)} 帧")
        
        # Analyze parquet structure
        parquet_structure = analyze_parquet_structure(df)
        
        # Detect HDF5 structure
        hdf5_structure = detect_hdf5_structure(parquet_structure)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_hdf5), exist_ok=True)
        
        # Create HDF5 file
        print(f"\n📝 创建HDF5文件: {output_hdf5}")
        with h5py.File(output_hdf5, 'w') as f:
            # Add basic metadata
            f.attrs['description'] = 'Adaptive parquet to HDF5 conversion'
            f.attrs['source_parquet'] = os.path.basename(parquet_file)
            f.attrs['num_frames'] = len(df)
            f.attrs['num_fields'] = len(df.columns)
            
            # Process each column
            print(f"\n📊 转换字段数据...")
            processed_count = 0
            
            for col_name in df.columns:
                try:
                    col_data = df[col_name].values
                    col_info = parquet_structure['field_info'][col_name]
                    h5_info = hdf5_structure[col_name]
                    
                    # Get group and dataset name
                    group_path = h5_info['group_path']
                    dataset_name = h5_info['dataset_name']
                    
                    # Create groups if needed
                    if group_path != '/':
                        group = create_hdf5_groups(f, group_path)
                    else:
                        group = f
                    
                    # Handle nested data (arrays/lists)
                    if col_info['is_nested']:
                        # Stack arrays if they are the same length
                        try:
                            # Check if all elements are arrays of the same shape
                            first_elem = col_data[0] if len(col_data) > 0 else None
                            
                            if first_elem is None:
                                # Empty array, skip
                                print(f"  ⚠️  {col_name}: 空数组，跳过")
                                continue
                            
                            # Convert to numpy arrays
                            arrays = []
                            all_same_shape = True
                            first_shape = None
                            
                            for x in col_data:
                                if x is None or (isinstance(x, float) and np.isnan(x)):
                                    # Handle None/NaN by using zeros or skipping
                                    if first_shape is not None:
                                        arrays.append(np.zeros(first_shape, dtype=np.float32))
                                    else:
                                        arrays.append(np.array([0.0]))
                                    all_same_shape = False
                                elif isinstance(x, (list, np.ndarray)):
                                    arr_x = np.array(x)
                                    if first_shape is None:
                                        first_shape = arr_x.shape
                                    elif arr_x.shape != first_shape:
                                        all_same_shape = False
                                    arrays.append(arr_x)
                                else:
                                    # Scalar value
                                    arrays.append(np.array([x]))
                                    all_same_shape = False
                            
                            if all_same_shape and first_shape is not None:
                                # All arrays have the same shape, stack them
                                stacked = np.stack(arrays)
                                group.create_dataset(dataset_name, data=ensure_hdf5_compatible(stacked))
                                print(f"  ✅ {col_name} -> {h5_info['full_path']} (形状: {stacked.shape})")
                            else:
                                # Different shapes or mixed types, try to pad or use variable length
                                try:
                                    # Try to find max shape and pad
                                    max_len = max(len(arr) if arr.ndim == 1 else arr.shape[0] for arr in arrays)
                                    if all(arr.ndim == 1 for arr in arrays):
                                        # 1D arrays, pad to max length
                                        padded = np.array([np.pad(arr, (0, max_len - len(arr)), 
                                                               mode='constant', constant_values=0) 
                                                         if len(arr) < max_len else arr 
                                                         for arr in arrays])
                                        group.create_dataset(dataset_name, data=ensure_hdf5_compatible(padded))
                                        print(f"  ✅ {col_name} -> {h5_info['full_path']} (填充后形状: {padded.shape})")
                                    else:
                                        # Higher dimensions, store as object or convert
                                        arr = ensure_hdf5_compatible(col_data)
                                        group.create_dataset(dataset_name, data=arr)
                                        print(f"  ✅ {col_name} -> {h5_info['full_path']} (变长/混合)")
                                except Exception as e2:
                                    # Final fallback: store as string representation
                                    arr = ensure_hdf5_compatible(col_data)
                                    group.create_dataset(dataset_name, data=arr)
                                    print(f"  ✅ {col_name} -> {h5_info['full_path']} (字符串表示)")
                        except Exception as e:
                            # If stacking fails, try to store as variable length
                            print(f"  ⚠️  {col_name}: 处理嵌套数据失败，使用备用方法: {e}")
                            # Store as object reference or convert to string
                            arr = ensure_hdf5_compatible(col_data)
                            group.create_dataset(dataset_name, data=arr)
                            print(f"  ✅ {col_name} -> {h5_info['full_path']} (备用方法)")
                    else:
                        # Scalar data
                        arr = ensure_hdf5_compatible(col_data)
                        group.create_dataset(dataset_name, data=arr)
                        print(f"  ✅ {col_name} -> {h5_info['full_path']} (形状: {arr.shape})")
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"  ❌ 处理字段 {col_name} 失败: {e}")
                    continue
            
            print(f"\n✅ 成功处理 {processed_count}/{len(df.columns)} 个字段")
        
        print(f"\n✅ Parquet转HDF5完成: {output_hdf5}")
        return True
        
    except Exception as e:
        print(f"\n❌ Parquet转HDF5失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_lerobot_to_copasi(parquet_file: str, output_hdf5: str, 
                              max_frames: Optional[int] = None) -> bool:
    """
    Convert a lerobot-style parquet to COPASI-style HDF5 schema.
    Minimal schema covered:
      - timestamps, timestamps_utc (if available)
      - state/joint/{position, velocity} with shape (N, 14) by stacking left/right arms
      - index dataset and basic file attributes
    Unknown fields are ignored; extend as needed.
    """
    try:
        print(f"\n{'='*60}")
        print(f"🔄 开始 lerobot -> copasi 转换: {os.path.basename(parquet_file)}")
        print(f"{'='*60}")

        if not os.path.exists(parquet_file):
            print(f"❌ Parquet文件不存在: {parquet_file}")
            return False

        print("📖 读取Parquet文件...")
        engines = ['pyarrow', 'fastparquet']
        df = None
        for engine in engines:
            try:
                print(f"  尝试使用 {engine} 引擎...")
                df = pd.read_parquet(parquet_file, engine=engine)
                print(f"  ✅ 使用 {engine} 引擎读取成功, 形状: {df.shape}")
                break
            except Exception as e:
                print(f"  ⚠️  {engine} 引擎失败: {e}")
        if df is None:
            print("❌ 所有引擎都失败，无法读取Parquet文件")
            return False

        if isinstance(max_frames, int) and max_frames > 0:
            df = df.iloc[:max_frames]
            print(f"✂️  截断到前 {len(df)} 帧")

        # Required/optional columns
        has_left_pos = 'observation.left_arm.position' in df.columns
        has_left_vel = 'observation.left_arm.velocity' in df.columns
        has_right_pos = 'observation.right_arm.position' in df.columns
        has_right_vel = 'observation.right_arm.velocity' in df.columns

        if not (has_left_pos and has_right_pos):
            print("❌ 缺少必要的手臂位置列（observation.left_arm.position / observation.right_arm.position）")
            return False

        # Build joint position/velocity (N, 14)
        left_pos = np.stack(df['observation.left_arm.position'].values)
        right_pos = np.stack(df['observation.right_arm.position'].values)
        joint_pos = np.concatenate([left_pos, right_pos], axis=1)

        if has_left_vel and has_right_vel:
            left_vel = np.stack(df['observation.left_arm.velocity'].values)
            right_vel = np.stack(df['observation.right_arm.velocity'].values)
            joint_vel = np.concatenate([left_vel, right_vel], axis=1)
        else:
            joint_vel = np.zeros_like(joint_pos, dtype=np.float32)

        # Timestamps
        unix_ts = df['timestamps'].values if 'timestamps' in df.columns else None
        utc_ts = df['timestamps_utc'].values if 'timestamps_utc' in df.columns else None

        N = joint_pos.shape[0]
        index = np.arange(0, N, dtype=np.int32)

        # Write HDF5 in COPASI-like minimal schema
        os.makedirs(os.path.dirname(output_hdf5) or '.', exist_ok=True)
        with h5py.File(output_hdf5, 'w') as f:
            f.attrs['description'] = 'COPASI-style proprioception data (from lerobot)'
            f.attrs['source_parquet'] = os.path.basename(parquet_file)
            f.attrs['num_frames'] = int(N)

            # timestamps
            if utc_ts is not None:
                utc_bytes = [str(ts).encode('utf-8') for ts in utc_ts]
                f.create_dataset('timestamps_utc', data=utc_bytes)
            if unix_ts is not None:
                f.create_dataset('timestamps', data=unix_ts)

            # state group
            state = f.create_group('state')
            joint = state.create_group('joint')
            joint.create_dataset('position', data=ensure_hdf5_compatible(joint_pos))
            joint.create_dataset('velocity', data=ensure_hdf5_compatible(joint_vel))

            # effector/head/waist/legs/end 可按需扩展，这里先不创建

            # index at root for convenience
            f.create_dataset('index', data=ensure_hdf5_compatible(index))

        # Try to add *_names fields using existing updater if available
        try:
            try:
                from data_processing_tools.update_hdf5_names_fields import update_hdf5_file as _update_h5_names
            except Exception:
                from update_hdf5_names_fields import update_hdf5_file as _update_h5_names  # type: ignore
            print("🔧 更新HDF5 *_names 字段...")
            _update_h5_names(output_hdf5)
            print("✅ HDF5 *_names 字段更新完成")
        except Exception as e:
            print(f"⚠️  跳过*_names更新: {e}")

        print(f"\n✅ lerobot -> copasi 转换完成: {output_hdf5}")
        return True

    except Exception as e:
        print(f"\n❌ lerobot -> copasi 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_episode_structure(output_base_dir: str, episode_uuid: str) -> Dict[str, str]:
    """
    Create episode directory structure following the naming convention
    Returns dictionary with directory paths
    """
    episode_dir = os.path.join(output_base_dir, episode_uuid)
    
    # Create directory structure
    camera_dir = os.path.join(episode_dir, "camera")
    video_dir = os.path.join(camera_dir, "video")
    depth_dir = os.path.join(camera_dir, "depth")
    audio_dir = os.path.join(episode_dir, "audio")
    parameters_dir = os.path.join(episode_dir, "parameters")
    proprio_stats_dir = os.path.join(episode_dir, "proprio_stats")
    
    # Create directories (video, audio, parameters are left empty as requested)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(parameters_dir, exist_ok=True)
    os.makedirs(proprio_stats_dir, exist_ok=True)
    
    return {
        'episode_dir': episode_dir,
        'camera_dir': camera_dir,
        'video_dir': video_dir,
        'depth_dir': depth_dir,
        'audio_dir': audio_dir,
        'parameters_dir': parameters_dir,
        'proprio_stats_dir': proprio_stats_dir
    }


def ensure_copasi_base_dirs(output_root: str) -> Tuple[str, str, str, str]:
    """
    Create COPASI-style base directory structure before UUID level:
      Logistics/<MAIN_SCENE_NAME>/<SUB_SCENE_NAME>/<ACTION_NAME>
    Returns the action directory path.
    """
    # Import names from config if available
    # Prefer local config for this tool, fallback to global config
    MAIN_SCENE_NAME = 'Logistics'
    SUB_SCENE_NAME = 'Materialtransfer'
    ACTION_NAME = 'Distribute_Parcels_To_Corresponding_Regions'
    try:
        cfg = importlib.import_module('parquet2hdf5_config')
        MAIN_SCENE_NAME = getattr(cfg, 'MAIN_SCENE_NAME', MAIN_SCENE_NAME)
        SUB_SCENE_NAME = getattr(cfg, 'SUB_SCENE_NAME', SUB_SCENE_NAME)
        ACTION_NAME = getattr(cfg, 'ACTION_NAME', ACTION_NAME)
    except Exception:
        try:
            from config import MAIN_SCENE_NAME as _M, SUB_SCENE_NAME as _S, ACTION_NAME as _A
            MAIN_SCENE_NAME, SUB_SCENE_NAME, ACTION_NAME = _M, _S, _A
        except Exception:
            pass

    logistics_dir = os.path.join(output_root, 'Logistics')
    os.makedirs(logistics_dir, exist_ok=True)

    # create task_info and empty meta json
    task_info_dir = os.path.join(logistics_dir, 'task_info')
    os.makedirs(task_info_dir, exist_ok=True)
    meta_filename = f"{MAIN_SCENE_NAME}-{SUB_SCENE_NAME}-{ACTION_NAME}.json"
    meta_file_path = os.path.join(task_info_dir, meta_filename)
    if not os.path.exists(meta_file_path):
        with open(meta_file_path, 'w') as _f:
            pass

    main_scene_dir = os.path.join(logistics_dir, MAIN_SCENE_NAME)
    sub_scene_dir = os.path.join(main_scene_dir, SUB_SCENE_NAME)
    action_dir = os.path.join(sub_scene_dir, ACTION_NAME)

    os.makedirs(action_dir, exist_ok=True)
    return logistics_dir, main_scene_dir, sub_scene_dir, action_dir


def format_size_gb(total_bytes: int) -> str:
    """Format bytes as human-readable size string like 123Mb or 1.2Gb"""
    gb = total_bytes / (1024**3)
    if gb >= 1.0:
        return f"{gb:.2f}Gb"
    mb = total_bytes / (1024**2)
    return f"{mb:.0f}Mb"


def format_duration(seconds: int) -> str:
    """Format seconds like 40s, 10m, 1h20m"""
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    if hours:
        return f"{hours}h{minutes}m"
    return f"{minutes}m"


def calculate_output_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except Exception:
                pass
    return total


def finalize_copasi_names(logistics_dir: str, main_scene_dir: str, sub_scene_dir: str, action_dir: str,
                          processed_count: int) -> Tuple[str, str, str]:
    """
    Rename main/sub/action directories to include size/counts/duration postfixes.
    Returns the new (main, sub, action) paths.
    """
    total_bytes = calculate_output_size_bytes(action_dir)
    size_str = format_size_gb(total_bytes)
    duration_seconds = processed_count * 40
    duration_str = format_duration(duration_seconds)

    # Names with postfixes
    new_main_name = f"{os.path.basename(main_scene_dir)}-{size_str}_{processed_count}counts_{duration_str}"
    new_sub_name = f"{os.path.basename(sub_scene_dir)}-{size_str}_{processed_count}counts_{duration_str}"
    new_action_name = f"{os.path.basename(action_dir)}-{size_str}_{processed_count}counts_{duration_str}"

    def _safe_merge_or_rename(src: str, dst: str) -> str:
        if src == dst:
            return dst
        parent = os.path.dirname(dst)
        os.makedirs(parent, exist_ok=True)
        if os.path.exists(dst):
            # Merge src contents into dst, then remove src
            for name in os.listdir(src):
                s = os.path.join(src, name)
                d = os.path.join(dst, name)
                if os.path.isdir(s):
                    if os.path.exists(d):
                        # recurse merge
                        for child in os.listdir(s):
                            cs = os.path.join(s, child)
                            cd = os.path.join(d, child)
                            if os.path.isdir(cs):
                                os.makedirs(cd, exist_ok=True)
                                # move subtree
                                import shutil
                                shutil.move(cs, cd)
                            else:
                                import shutil
                                shutil.move(cs, d)
                        os.rmdir(s)
                    else:
                        import shutil
                        shutil.move(s, dst)
                else:
                    import shutil
                    shutil.move(s, dst)
            os.rmdir(src)
            return dst
        else:
            os.rename(src, dst)
            return dst

    new_main_path = os.path.join(logistics_dir, new_main_name)
    main_scene_dir = _safe_merge_or_rename(main_scene_dir, new_main_path)

    new_sub_path = os.path.join(main_scene_dir, new_sub_name)
    sub_scene_dir = _safe_merge_or_rename(sub_scene_dir, new_sub_path)

    new_action_path = os.path.join(sub_scene_dir, new_action_name)
    action_dir = _safe_merge_or_rename(action_dir, new_action_path)

    return main_scene_dir, sub_scene_dir, action_dir


def process_parquet_files(input_dir: str, output_base_dir: str, 
                         pattern: str = "*.parquet", max_frames: Optional[int] = None) -> Dict[str, Any]:
    """
    Process multiple parquet files in a directory
    Creates standard directory structure: UUID/episode/proprio_stats/proprio_stats.hdf5
    """
    input_path = Path(input_dir)
    # Recursive search when pattern includes '**', otherwise try both levels
    if '**' in pattern:
        parquet_files = list(input_path.rglob(pattern))
    else:
        parquet_files = list(input_path.glob(pattern)) or list(input_path.rglob(pattern))
    
    if not parquet_files:
        print(f"❌ 未找到Parquet文件: {input_dir}/{pattern}")
        return {'success': False, 'processed': 0, 'failed': 0}
    
    print(f"📁 找到 {len(parquet_files)} 个Parquet文件")
    print(f"📁 输出目录: {output_base_dir}")
    
    results = {
        'success': True,
        'processed': 0,
        'failed': 0,
        'episodes': []
    }
    
    # Prepare COPASI base dirs (Logistics/task_info and bare MAIN/SUB/ACTION) directly under configured output root
    logistics_dir, main_scene_dir, sub_scene_dir, action_dir = ensure_copasi_base_dirs(output_base_dir)

    processed_counter = 0

    for parquet_file in sorted(parquet_files):
        try:
            # Generate UUID for episode
            episode_uuid = str(uuid.uuid4())

            # Create episode structure under action_dir
            dirs = create_episode_structure(action_dir, episode_uuid)
            
            # Convert parquet to HDF5 using adaptive mode (read what exists, generate what exists)
            hdf5_path = os.path.join(dirs['proprio_stats_dir'], "proprio_stats.hdf5")
            
            print(f"\n{'='*60}")
            print(f"📄 处理: {parquet_file.name}")
            print(f"📁 UUID: {episode_uuid}")
            print(f"📁 输出路径: {hdf5_path}")
            print(f"{'='*60}")
            
            success = convert_parquet_to_hdf5_adaptive(str(parquet_file), hdf5_path, max_frames)
            
            if success:
                results['processed'] += 1
                processed_counter += 1
                results['episodes'].append({
                    'parquet': str(parquet_file),
                    'uuid': episode_uuid,
                    'hdf5': hdf5_path,
                    'episode_dir': dirs['episode_dir']
                })
                print(f"✅ 完成: {parquet_file.name} -> {episode_uuid}")
            else:
                results['failed'] += 1
                print(f"❌ 失败: {parquet_file.name}")
                
        except Exception as e:
            results['failed'] += 1
            print(f"❌ 处理 {parquet_file.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    # Rename main/sub/action with size/counts/duration postfixes
    try:
        new_main, new_sub, new_action = finalize_copasi_names(logistics_dir, main_scene_dir, sub_scene_dir, action_dir, processed_counter)
        print(f"📁 命名已更新:\n  {os.path.basename(new_main)}\n  {os.path.basename(new_sub)}\n  {os.path.basename(new_action)}")
    except Exception as _e:
        print(f"⚠️  命名更新失败（忽略）: {_e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='自适应Parquet转HDF5转换工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个parquet文件
  python convert_parquet_to_hdf5_adaptive.py input.parquet output.hdf5

  # 批量处理目录中的所有parquet文件
  python convert_parquet_to_hdf5_adaptive.py -d /path/to/parquets -o /path/to/output

  # 指定parquet文件模式
  python convert_parquet_to_hdf5_adaptive.py -d /path/to/parquets -o /path/to/output -p "episode_*.parquet"
        """
    )
    
    parser.add_argument('input', nargs='?', help='输入Parquet文件路径')
    parser.add_argument('output', nargs='?', help='输出HDF5文件路径（单文件模式）')
    parser.add_argument('-d', '--directory', help='输入目录（批量模式）')
    parser.add_argument('-o', '--output-dir', help='输出目录（批量模式）')
    parser.add_argument('-p', '--pattern', default='**/*.parquet', 
                       help='Parquet文件匹配模式（默认: **/*.parquet，递归搜索）')
    parser.add_argument('-m', '--max-frames', type=int, 
                       help='最大帧数（可选，用于截断数据）')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.directory and args.output_dir:
        # Batch mode
        print(f"📁 批量处理模式")
        print(f"  输入目录: {args.directory}")
        print(f"  输出目录: {args.output_dir}")
        print(f"  文件模式: {args.pattern}")
        
        # Ensure output dir exists before writing summary
        os.makedirs(args.output_dir, exist_ok=True)
        # If not provided, read roots from config
        if not args.directory or not args.output_dir:
            # Prefer standalone local config
            DATASET_ROOT = args.directory
            OUTPUT_ROOT_PATH = args.output_dir
            try:
                cfg = importlib.import_module('parquet2hdf5_config')
                if not DATASET_ROOT:
                    DATASET_ROOT = getattr(cfg, 'DATASET_ROOT', None)
                if not OUTPUT_ROOT_PATH:
                    OUTPUT_ROOT_PATH = getattr(cfg, 'OUTPUT_ROOT_PATH', None)
            except Exception:
                pass
            if (not DATASET_ROOT) or (not OUTPUT_ROOT_PATH):
                # Fallback to global config if local config not present or incomplete
                try:
                    from config import DATASET_ROOT as _IN, OUTPUT_ROOT_PATH as _OUT
                    DATASET_ROOT = DATASET_ROOT or _IN
                    OUTPUT_ROOT_PATH = OUTPUT_ROOT_PATH or _OUT
                except Exception:
                    DATASET_ROOT = DATASET_ROOT or '.'
                    OUTPUT_ROOT_PATH = OUTPUT_ROOT_PATH or './out'
            args.directory = DATASET_ROOT
            args.output_dir = OUTPUT_ROOT_PATH
        os.makedirs(args.output_dir, exist_ok=True)
        results = process_parquet_files(args.directory, args.output_dir, args.pattern, args.max_frames)
        
        print(f"\n{'='*60}")
        print(f"📊 处理结果:")
        print(f"  - 成功: {results['processed']}")
        print(f"  - 失败: {results['failed']}")
        print(f"  - 总计: {results['processed'] + results['failed']}")
        print(f"{'='*60}")
        
        # Save results summary
        summary_file = os.path.join(args.output_dir, 'conversion_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"📄 结果摘要已保存: {summary_file}")
        
    elif args.input and args.output:
        # Single file mode
        print(f"📄 单文件处理模式")
        print(f"  输入: {args.input}")
        print(f"  输出: {args.output}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        
        # Always use adaptive mode (read what exists, generate what exists)
        success = convert_parquet_to_hdf5_adaptive(args.input, args.output, args.max_frames)
        
        if success:
            print(f"\n✅ 转换完成!")
            sys.exit(0)
        else:
            print(f"\n❌ 转换失败!")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

