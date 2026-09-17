"""
数据集管理工具：扫描、列出和管理数据集
"""
import os
import json
import importlib
import re
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Callable
from datetime import datetime
import numpy as np
import pandas as pd
import h5py


class DatasetInfo:
    """数据集信息"""
    def __init__(self, repo_id: str, root_path: Path):
        self.repo_id = repo_id
        self.root_path = root_path
        self.info_path = root_path / "meta" / "info.json"
        self.num_episodes = 0
        self.total_frames = 0
        self.fps = 30
        self.robot_type = "unknown"
        self.created_time = None
        self.modified_time = None
        self.parquet_files = []
        self.has_hdf5 = False
        self.has_uploaded = False  # 是否已上传到对象存储
        
        self._load_info()
    
    def _load_info(self):
        """加载数据集信息"""
        if not self.info_path.exists():
            return
        
        try:
            # 读取info.json
            with open(self.info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            self.num_episodes = info.get('total_episodes', 0)
            self.total_frames = info.get('total_frames', 0)
            self.fps = info.get('fps', 30)
            self.robot_type = info.get('robot_type', 'unknown')
            
            # 获取文件时间
            if self.info_path.exists():
                stat = self.info_path.stat()
                self.created_time = datetime.fromtimestamp(stat.st_ctime)
                self.modified_time = datetime.fromtimestamp(stat.st_mtime)
            
            # 扫描parquet文件
            # LeRobot v2.0 结构：data/chunk-*/episode_*.parquet
            # 也支持旧结构：train/*.parquet
            data_dir = self.root_path / "data"
            if data_dir.exists():
                self.parquet_files = list(data_dir.rglob("*.parquet"))
            else:
                # 尝试旧结构
                parquet_dir = self.root_path / "train"
                if parquet_dir.exists():
                    self.parquet_files = list(parquet_dir.glob("*.parquet"))
            
            # 检查是否有本地 hdf5 文件
            hdf5_dir = self.root_path / "hdf5"
            if hdf5_dir.exists():
                hdf5_files = list(hdf5_dir.glob("*.hdf5"))
                self.has_hdf5 = len(hdf5_files) > 0

            # 兼容 COPASI 外部输出：检查转换标记文件
            if not self.has_hdf5:
                converted_flag = self.root_path / "meta" / "hdf5_converted.json"
                if converted_flag.exists():
                    try:
                        with converted_flag.open("r", encoding="utf-8") as f:
                            flag_data = json.load(f)
                        
                        # 验证 HDF5 文件是否真实存在
                        if flag_data.get("use_copasi_structure") and flag_data.get("output_root"):
                            # COPASI 结构：检查 output_root 下是否有对应的 UUID 目录
                            output_root = Path(flag_data["output_root"]).expanduser()
                            logistics_dir = output_root / "Logistics"
                            generated_uuids = flag_data.get("generated_uuids", [])
                            
                            if generated_uuids and logistics_dir.exists():
                                # 至少验证一个 UUID 目录存在且有 HDF5 文件
                                for uuid_str in generated_uuids[:1]:  # 只检查第一个
                                    for main_dir in logistics_dir.iterdir():
                                        if not main_dir.is_dir() or main_dir.name == "task_info":
                                            continue
                                        for sub_dir in main_dir.iterdir():
                                            if not sub_dir.is_dir():
                                                continue
                                            for action_dir in sub_dir.iterdir():
                                                if not action_dir.is_dir():
                                                    continue
                                                uuid_dir = action_dir / uuid_str
                                                hdf5_dir = uuid_dir / "proprio_stats"
                                                if hdf5_dir.exists() and list(hdf5_dir.glob("proprio_stats*.hdf5")):
                                                    self.has_hdf5 = True
                                                    break
                                            if self.has_hdf5:
                                                break
                                        if self.has_hdf5:
                                            break
                                    if self.has_hdf5:
                                        break
                        else:
                            # 默认结构：直接标记为已转换（旧逻辑）
                            self.has_hdf5 = True
                    except Exception:
                        pass
            
            # 检查是否已上传到对象存储
            uploaded_flag = self.root_path / "meta" / "uploaded.json"
            if uploaded_flag.exists():
                try:
                    with uploaded_flag.open("r", encoding="utf-8") as f:
                        upload_data = json.load(f)
                    self.has_uploaded = upload_data.get("uploaded", False)
                except Exception:
                    pass
                
        except Exception as e:
            print(f"读取数据集信息失败 {self.repo_id}: {e}")
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'repo_id': self.repo_id,
            'num_episodes': self.num_episodes,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'robot_type': self.robot_type,
            'created_time': self.created_time.strftime("%Y-%m-%d %H:%M:%S") if self.created_time else "未知",
            'modified_time': self.modified_time.strftime("%Y-%m-%d %H:%M:%S") if self.modified_time else "未知",
            'parquet_count': len(self.parquet_files),
            'has_hdf5': self.has_hdf5,
            'has_uploaded': self.has_uploaded,
            'path': str(self.root_path),
        }


# --------------------------------------------------------------------------------------
# 视频导出辅助常量与工具（借鉴 convert_to_target_format.py 的结构命名）
# --------------------------------------------------------------------------------------

RGB_CAMERA_MAPPINGS: Dict[str, Tuple[str, bool]] = {
    "hand_l_rgb": ("hand_left_color.mp4", False),
    "hand_r_rgb": ("hand_right_color.mp4", False),
    # 历史数据中曾使用 headf_rgbd_color 命名，这里同时兼容
    "headf_rgb": ("head_front_color.mp4", True),
    "headf_rgbd_color": ("head_front_color.mp4", True),
}

DEPTH_CAMERA_MAPPINGS: Dict[str, str] = {
    "hand_l_depth": "hand_left_depth.mkv",
    "hand_r_depth": "hand_right_depth.mkv",
    "headf_depth": "head_front_depth.mkv",
    "headf_rgbd_depth": "head_front_depth.mkv",
}


def _natural_sort_key(path: Path) -> List[Any]:
    """生成自然排序键，确保 10 不会排在 2 前面。"""
    parts: List[Any] = []
    for chunk in re.split(r"(\d+)", path.name):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk)
    return parts


def _infer_sequence_pattern(image_dir: Path) -> Optional[Tuple[str, int]]:
    """
    根据图像文件推断 ffmpeg 可识别的序列模式。
    返回 (pattern, start_number)，pattern 形如 frame_%06d.jpg 或 %d.png。
    """
    for ext in ("jpg", "jpeg", "png"):
        files = sorted(image_dir.glob(f"*.{ext}"), key=_natural_sort_key)
        if not files:
            continue

        match = re.search(r"(\d+)$", files[0].stem)
        if not match:
            continue

        prefix = files[0].stem[: match.start()]
        first_digits = match.group(1)
        zero_padded = first_digits.startswith("0") and len(first_digits) > 1
        width = len(first_digits)
        digits: List[int] = []

        valid = True
        for path in files:
            if path.suffix.lower() != f".{ext}":
                valid = False
                break
            stem_match = re.search(r"(\d+)$", path.stem)
            if not stem_match:
                valid = False
                break
            if stem_match and path.stem[: stem_match.start()] != prefix:
                valid = False
                break
            digits.append(int(stem_match.group(1)))

        if not valid or not digits:
            continue

        start_number = min(digits)
        if zero_padded or width > 1:
            pattern = f"{prefix}%0{width}d.{ext}"
        else:
            pattern = f"{prefix}%d.{ext}"
        return pattern, start_number
    return None


def _run_ffmpeg(cmd: List[str]) -> bool:
    """调用 ffmpeg，并在失败时打印 stderr。"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            print(f"❌ ffmpeg 执行失败: {' '.join(cmd)}")
            if stderr:
                print(stderr)
            return False
        return True
    except FileNotFoundError:
        print("❌ 未检测到 ffmpeg，可执行文件不存在")
        return False
    except Exception as exc:
        print(f"❌ 调用 ffmpeg 异常: {exc}")
        return False


def _convert_images_to_video(
    image_dir: Path, output_path: Path, fps: int = 30, is_head: bool = False
) -> bool:
    """将指定目录下的彩色图序列转换为 MP4。"""
    pattern_info = _infer_sequence_pattern(image_dir)
    if pattern_info is None:
        print(f"⚠️  无法推断图像序列命名，跳过: {image_dir}")
        return False

    pattern, start_number = pattern_info
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        str(image_dir / pattern),
        "-c:v",
        "libx264",
        "-preset",
        "faster",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
    ]

    # Head 画面无需额外裁剪，这里预留钩子便于将来扩展
    cmd.append(str(output_path))
    print(f"🎬 转换彩色视频: {image_dir} -> {output_path}")
    return _run_ffmpeg(cmd)


def _convert_depth_to_video(
    depth_dir: Path, output_path: Path, fps: int = 30
) -> bool:
    """将深度 PNG 序列转换为无损 MKV（FFV1/gray16le）。"""
    pattern_info = _infer_sequence_pattern(depth_dir)
    if pattern_info is None:
        print(f"⚠️  无法推断深度图序列命名，跳过: {depth_dir}")
        return False

    pattern, start_number = pattern_info
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        str(depth_dir / pattern),
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "gray16le",
        "-g",
        "1",
        "-slices",
        "16",
        "-slicecrc",
        "1",
        str(output_path),
    ]
    print(f"🎥 转换深度视频: {depth_dir} -> {output_path}")
    return _run_ffmpeg(cmd)


def _locate_episode_images_dir(dataset_path: Path, parquet_file: Path) -> Optional[Path]:
    """
    根据 parquet 路径推断同 episode 的图像目录。
    data/chunk-000/episode_xxxxx.parquet -> chunk_000/episode_xxxxx/images
    """
    try:
        relative = parquet_file.relative_to(dataset_path)
    except ValueError:
        relative = None

    if relative and len(relative.parts) >= 3 and relative.parts[0] == "data":
        chunk_dash = relative.parts[1]
        episode_name = parquet_file.stem
        chunk_dir = dataset_path / chunk_dash.replace("-", "_")
        candidate = chunk_dir / episode_name / "images"
        if candidate.exists():
            return candidate

    # 兼容旧结构：遍历 chunk_* 目录
    for chunk_dir in dataset_path.glob("chunk_*"):
        candidate = chunk_dir / parquet_file.stem / "images"
        if candidate.exists():
            return candidate
    return None


def _generate_episode_videos(
    images_dir: Path,
    video_dir: Path,
    depth_dir: Path,
    fps: int = 30,
    export_rgb: bool = True,
    export_depth: bool = True,
) -> None:
    """从 images 目录生成彩色/深度视频文件。"""
    if not images_dir.exists():
        print(f"⚠️  未找到图像目录，跳过视频生成: {images_dir}")
        return

    video_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    produced_any = False

    # 彩色视频
    if export_rgb:
        for src_dir, (dst_name, is_head) in RGB_CAMERA_MAPPINGS.items():
            src_path = images_dir / src_dir
            if not src_path.exists():
                continue
            output_path = video_dir / dst_name
            if output_path.exists():
                continue
            if _convert_images_to_video(src_path, output_path, fps=fps, is_head=is_head):
                produced_any = True

    # 深度视频
    if export_depth:
        for src_dir, dst_name in DEPTH_CAMERA_MAPPINGS.items():
            src_path = images_dir / src_dir
            if not src_path.exists():
                continue
            output_path = depth_dir / dst_name
            if output_path.exists():
                continue
            if _convert_depth_to_video(src_path, output_path, fps=fps):
                produced_any = True

    if not produced_any:
        print(f"ℹ️  图像存在但没有生成任何视频（可能输出已存在）: {images_dir}")


def _read_timestamp_txt(path: Path) -> Optional[np.ndarray]:
    """读取简单的一列纳秒时间戳 txt 文件。"""
    if not path.exists():
        return None
    try:
        values: List[int] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    values.append(int(line))
                except Exception:
                    continue
        if not values:
            return None
        return np.asarray(values, dtype=np.int64)
    except Exception as exc:
        print(f"⚠️  读取时间戳失败: {path} ({exc})")
        return None


def _align_stream_indices(
    ref_ts: np.ndarray, cam_ts: np.ndarray, max_delta_ns: int
) -> np.ndarray:
    """
    简单最近邻对齐：对每个 ref_ts 在 cam_ts 中找最近的索引，超过 max_delta_ns 记为 -1。
    返回 shape=(N,) 的索引数组。
    """
    if ref_ts.size == 0 or cam_ts.size == 0:
        return np.full(ref_ts.shape, -1, dtype=np.int64)

    indices = np.full(ref_ts.shape, -1, dtype=np.int64)
    j = 0
    for i, t in enumerate(ref_ts):
        # cam_ts 是单调递增的，向前推进 j 以减小距离
        while j + 1 < cam_ts.size and abs(cam_ts[j + 1] - t) <= abs(cam_ts[j] - t):
            j += 1
        if abs(cam_ts[j] - t) <= max_delta_ns:
            indices[i] = j
    return indices


def _attach_alignment_and_unified_timestamps(
    dataset_path: Path,
    parquet_file: Path,
    hdf5_file: Path,
    fps: int = 30,
) -> None:
    """
    为指定 episode 的 HDF5 附加：
      1) /alignment 下的多模态对齐索引（基于 parquet.timestamps 与各相机 txt）
      2) 统一命名的 *_color_timestamps 数据集（不修改 *_mp4_timestamps 原字段）
    """
    try:
        if not hdf5_file.exists():
            return

        # 尝试读取 parquet 的时间戳列
        try:
            df_ts = pd.read_parquet(parquet_file, columns=["timestamps"])
            ref_ts = df_ts["timestamps"].to_numpy(dtype=np.int64)
        except Exception as exc:
            print(f"⚠️  读取 parquet 时间戳失败，跳过对齐: {parquet_file} ({exc})")
            ref_ts = None

        # 打开 HDF5 追加字段
        with h5py.File(hdf5_file, "a") as f:
            # 统一命名的 color timestamps（从 parquet 列复制）
            try:
                df_extra = pd.read_parquet(
                    parquet_file,
                    columns=[
                        "headf_mp4_timestamps",
                        "hand_l_rgb_mp4_timestamps",
                        "hand_r_rgb_mp4_timestamps",
                    ],
                )
            except Exception:
                df_extra = None

            if df_extra is not None:
                if "headf_mp4_timestamps" in df_extra.columns:
                    ds_name = "head_front_color_timestamps"
                    if ds_name not in f:
                        f.create_dataset(ds_name, data=df_extra["headf_mp4_timestamps"].to_numpy())
                        print(f"✅ 写入统一时间戳字段: {ds_name}")
                if "hand_l_rgb_mp4_timestamps" in df_extra.columns:
                    ds_name = "hand_left_color_timestamps"
                    if ds_name not in f:
                        f.create_dataset(
                            ds_name, data=df_extra["hand_l_rgb_mp4_timestamps"].to_numpy()
                        )
                        print(f"✅ 写入统一时间戳字段: {ds_name}")
                if "hand_r_rgb_mp4_timestamps" in df_extra.columns:
                    ds_name = "hand_right_color_timestamps"
                    if ds_name not in f:
                        f.create_dataset(
                            ds_name, data=df_extra["hand_r_rgb_mp4_timestamps"].to_numpy()
                        )
                        print(f"✅ 写入统一时间戳字段: {ds_name}")

            # 对齐索引
            if ref_ts is None or ref_ts.size == 0:
                return

            images_dir = _locate_episode_images_dir(dataset_path, parquet_file)
            if images_dir is None:
                print(f"⚠️  未找到图像目录，跳过对齐索引写入: {parquet_file.stem}")
                return

            # 加载各相机 txt 时间戳（兼容新旧命名）
            cam_ts: Dict[str, Optional[np.ndarray]] = {}
            cam_ts["hand_l"] = _read_timestamp_txt(
                images_dir / "hand_l_rgb" / "hand_l_timestamps.txt"
            )
            cam_ts["hand_r"] = _read_timestamp_txt(
                images_dir / "hand_r_rgb" / "hand_r_timestamps.txt"
            )
            # 头部：优先新的 headf_rgb/headf_timestamps，其次旧的 headf_rgbd_color/headf_rgbd_timestamps
            head_ts = _read_timestamp_txt(
                images_dir / "headf_rgb" / "headf_timestamps.txt"
            )
            if head_ts is None:
                head_ts = _read_timestamp_txt(
                    images_dir / "headf_rgbd_color" / "headf_rgbd_timestamps.txt"
                )
            cam_ts["headf"] = head_ts

            align_group = f.require_group("alignment")
            if "parquet_indices" not in align_group:
                align_group.create_dataset(
                    "parquet_indices",
                    data=np.arange(ref_ts.shape[0], dtype=np.int64),
                )

            # 允许的最大时间差：略大于 1 帧间隔
            frame_period_ns = int(1e9 / max(1, fps))
            max_delta_ns = int(frame_period_ns * 1.5)

            for name, ts_array in cam_ts.items():
                if ts_array is None or ts_array.size == 0:
                    continue
                ds_name = f"{name}_indices"
                if ds_name in align_group:
                    continue
                indices = _align_stream_indices(ref_ts, ts_array, max_delta_ns)
                align_group.create_dataset(ds_name, data=indices)
                print(f"✅ 写入对齐索引: alignment/{ds_name}")

    except Exception as exc:
        print(f"⚠️  附加对齐索引/统一时间戳失败: {exc}")


def scan_datasets(dataset_root: Optional[str] = None) -> List[DatasetInfo]:
    """
    扫描数据集目录，返回所有数据集的信息
    
    Args:
        dataset_root: 数据集根目录，如果为None则使用默认路径
        
    Returns:
        数据集信息列表
    """
    if dataset_root is None:
        dataset_root = os.environ.get("LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
    
    dataset_root = Path(dataset_root).expanduser()
    if not dataset_root.exists():
        return []
    
    datasets = []
    
    # 遍历数据集根目录下的所有子目录
    for item in dataset_root.iterdir():
        if not item.is_dir():
            continue
        
        # 检查是否是数据集目录（包含meta/info.json）
        info_file = item / "meta" / "info.json"
        if info_file.exists():
            try:
                dataset_info = DatasetInfo(item.name, item)
                datasets.append(dataset_info)
            except Exception as e:
                print(f"加载数据集 {item.name} 失败: {e}")
                continue
    
    # 按修改时间排序（最新的在前）
    datasets.sort(key=lambda x: x.modified_time or datetime.min, reverse=True)
    
    return datasets


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


def detect_hdf5_structure(parquet_structure: Dict[str, Any]) -> Dict[str, Dict]:
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


def convert_parquet_to_hdf5_adaptive(
    parquet_file: str,
    output_hdf5: str,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    verbose: bool = False,
) -> bool:
    """
    Convert parquet file to HDF5 with adaptive field detection
    """
    def _progress(cur: int, total: int, msg: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(int(cur), int(total), str(msg))
        except Exception:
            pass

    def _cancelled() -> bool:
        if should_cancel is None:
            return False
        try:
            return bool(should_cancel())
        except Exception:
            return False

    # 静默模式：大量 print 会显著拖慢转换（尤其在 GUI 场景），默认关闭 verbose
    # 注意：不要直接引用/覆盖局部变量 print（会触发 UnboundLocalError），用 builtins.print 兜底。
    import builtins as _builtins
    _print = _builtins.print
    if not verbose:
        def print(*args, **kwargs):  # type: ignore
            return None

    try:
        print(f"\n{'='*60}")
        print(f"🔄 开始转换: {os.path.basename(parquet_file)}")
        print(f"{'='*60}")

        if _cancelled():
            print("⚠️  用户取消转换（开始前）")
            return False
        
        # Check if parquet file exists
        if not os.path.exists(parquet_file):
            print(f"❌ Parquet文件不存在: {parquet_file}")
            return False
        
        # Try to read parquet file with different engines
        print("📖 读取Parquet文件...")
        engines = ['pyarrow', 'fastparquet']
        df = None
        
        for engine in engines:
            if _cancelled():
                print("⚠️  用户取消转换（读取Parquet阶段）")
                return False
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
        if _cancelled():
            print("⚠️  用户取消转换（读取完成后）")
            return False
        
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
            total_cols = len(df.columns)
            for idx_col, col_name in enumerate(df.columns, start=1):
                if _cancelled():
                    print("⚠️  用户取消转换（写入字段阶段）")
                    return False
                # 进度（字段级）：让 UI 能及时刷新“仍在工作”
                if total_cols > 0 and (idx_col == 1 or idx_col % 5 == 0):
                    _progress(idx_col, total_cols, f"字段 {idx_col}/{total_cols}: {col_name}")
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


def _scan_hdf5_config_files() -> List[str]:
    """
    扫描hdf5_configs目录下的JSON配置文件
    """
    config_files = []
    
    try:
        # 获取项目根目录
        current_file = Path(__file__).resolve()
        # dataset_manager.py 在 lerobot_data_collector/lerobot_data_collector/ 下
        # 向上两级到项目根目录
        proj_root = current_file.parent.parent
        configs_dir = proj_root / "hdf5_configs"
        
        if configs_dir.exists() and configs_dir.is_dir():
            # 扫描所有JSON文件（排除example文件）
            for json_file in configs_dir.glob("*.json"):
                if "example" not in json_file.name.lower():
                    config_files.append(str(json_file))
            
            # 按文件名排序
            config_files.sort()
            if config_files:
                print(f"✓ 找到 {len(config_files)} 个HDF5转换配置文件")
        else:
            print(f"⚠️  HDF5配置文件目录不存在: {configs_dir}")
    except Exception as e:
        print(f"扫描HDF5配置文件失败: {e}")
    
    return config_files


def _format_size_gb(total_bytes: int) -> str:
    """Format bytes as human-readable size string like 123Mb or 1.2Gb"""
    gb = total_bytes / (1024**3)
    if gb >= 1.0:
        return f"{gb:.2f}Gb"
    mb = total_bytes / (1024**2)
    return f"{mb:.0f}Mb"


def _format_duration(seconds: int) -> str:
    """Format seconds like 40s, 10m, 1h20m"""
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    if hours:
        return f"{hours}h{minutes}m"
    return f"{minutes}m"


def _calculate_output_size_bytes(path: Path) -> int:
    """Calculate total size of all files in a directory tree"""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except Exception:
                    pass
    except Exception:
        pass
    return total


def _finalize_copasi_names_with_config(logistics_dir: Path, main_scene_dir: Path, 
                                       sub_scene_dir: Path, action_dir: Path,
                                       processed_count: int, config: Dict[str, Any]) -> None:
    """
    Rename main/sub/action directories to include size/counts/duration postfixes.
    完全按照convert_parquet_to_hdf5_adaptive.py中的finalize_copasi_names逻辑
    """
    total_bytes = _calculate_output_size_bytes(action_dir)
    size_str = _format_size_gb(total_bytes)
    duration_seconds = processed_count * 40  # 假设每个episode 40秒
    duration_str = _format_duration(duration_seconds)
    
    # Names with postfixes
    new_main_name = f"{main_scene_dir.name}-{size_str}_{processed_count}counts_{duration_str}"
    new_sub_name = f"{sub_scene_dir.name}-{size_str}_{processed_count}counts_{duration_str}"
    new_action_name = f"{action_dir.name}-{size_str}_{processed_count}counts_{duration_str}"
    
    def _safe_merge_or_rename(src: Path, dst: Path) -> Path:
        """Safely merge or rename directory"""
        if src == dst:
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            # Merge src contents into dst, then remove src
            import shutil
            for name in os.listdir(src):
                s = src / name
                d = dst / name
                if s.is_dir():
                    if d.exists():
                        # recurse merge
                        for child in os.listdir(s):
                            cs = s / child
                            cd = d / child
                            if cs.is_dir():
                                cd.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(cs), str(cd))
                            else:
                                shutil.move(str(cs), str(d))
                        s.rmdir()
                    else:
                        shutil.move(str(s), str(dst))
                else:
                    shutil.move(str(s), str(dst))
            src.rmdir()
            return dst
        else:
            src.rename(dst)
            return dst
    
    # 关键修复：
    # main_scene_dir 重命名后，sub_scene_dir/action_dir 的旧 Path 会失效（仍指向旧父目录）。
    # 这里在每一步重命名后，重新基于“新父目录 + 旧子目录名”计算源路径，避免 Errno 2。
    orig_sub_name = sub_scene_dir.name
    orig_action_name = action_dir.name

    new_main_path = logistics_dir / new_main_name
    main_scene_dir = _safe_merge_or_rename(main_scene_dir, new_main_path)

    # 父目录可能已移动，刷新子目录源路径
    sub_scene_dir = main_scene_dir / orig_sub_name
    new_sub_path = main_scene_dir / new_sub_name
    if sub_scene_dir.exists():
        sub_scene_dir = _safe_merge_or_rename(sub_scene_dir, new_sub_path)
    else:
        # 防御：若源目录不存在则跳过（避免中断主流程）
        print(f"⚠️  子目录不存在，跳过重命名: {sub_scene_dir}")
        sub_scene_dir = new_sub_path

    # 刷新 action 目录源路径
    action_dir = sub_scene_dir / orig_action_name
    new_action_path = sub_scene_dir / new_action_name
    if action_dir.exists():
        action_dir = _safe_merge_or_rename(action_dir, new_action_path)
    else:
        print(f"⚠️  动作目录不存在，跳过重命名: {action_dir}")
        action_dir = new_action_path
    
    print(f"📁 目录命名已更新:")
    print(f"  {main_scene_dir.name}")
    print(f"  {sub_scene_dir.name}")
    print(f"  {action_dir.name}")


def _load_conversion_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    从JSON配置文件加载转换参数
    支持从 hdf5_configs/ 目录下的JSON文件读取配置
    
    Args:
        config_file: 指定的配置文件路径，如果为None则自动查找第一个配置文件
        
    Returns:
        配置字典
    """
    config = {
        'output_root': None,  # 如果设置，将使用COPASI风格输出结构
        'main_scene_name': None,
        'sub_scene_name': None,
        'action_name': None,
        'use_copasi_structure': False,
        'fps': 30,
        # 是否导出 RGB 视频（很慢）。用户要求默认转换采集后的 RGB+Depth，因此默认开启。
        'export_videos': True,
        # 是否导出深度视频（默认开启）
        'export_depth_videos': True,
    }
    
    # 如果没有指定配置文件，尝试自动查找
    if config_file is None:
        config_files = _scan_hdf5_config_files()
        if config_files:
            config_file = config_files[0]  # 使用第一个找到的配置文件
            print(f"📋 自动使用配置文件: {Path(config_file).name}")
    
    # 尝试从JSON文件读取配置
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                json_config = json.load(f)
            
            # 读取配置项（支持多种命名方式）
            config['output_root'] = json_config.get('output_root_path') or json_config.get('OUTPUT_ROOT_PATH')
            config['main_scene_name'] = json_config.get('main_scene_name') or json_config.get('MAIN_SCENE_NAME')
            config['sub_scene_name'] = json_config.get('sub_scene_name') or json_config.get('SUB_SCENE_NAME')
            config['action_name'] = json_config.get('action_name') or json_config.get('ACTION_NAME')
            config['use_copasi_structure'] = json_config.get('use_copasi_structure', False)
            config['fps'] = json_config.get('fps', config['fps'])
            config['export_videos'] = bool(json_config.get('export_videos', config['export_videos']))
            # 支持深度视频导出独立开关
            if 'export_depth_videos' in json_config:
                config['export_depth_videos'] = bool(json_config.get('export_depth_videos'))
            elif 'EXPORT_DEPTH_VIDEOS' in json_config:
                config['export_depth_videos'] = bool(json_config.get('EXPORT_DEPTH_VIDEOS'))
            
            # 如果设置了output_root，自动启用COPASI结构
            if config['output_root']:
                config['use_copasi_structure'] = True
            
            if config['use_copasi_structure']:
                print(f"📋 使用JSON配置文件: {Path(config_file).name}")
                print(f"   输出目录: {config['output_root']}")
                if config['main_scene_name']:
                    print(f"   场景: {config['main_scene_name']}/{config['sub_scene_name']}/{config['action_name']}")
            
            return config
        except Exception as e:
            print(f"⚠️  读取JSON配置文件失败: {e}")
    
    # 如果JSON配置不存在，尝试向后兼容Python配置
    try:
        try:
            cfg = importlib.import_module('lerobot_data_collector.parquet2hdf5_config')
        except ImportError:
            cfg = importlib.import_module('parquet2hdf5_config')
        
        if hasattr(cfg, 'OUTPUT_ROOT_PATH'):
            config['output_root'] = getattr(cfg, 'OUTPUT_ROOT_PATH')
            config['use_copasi_structure'] = True
        if hasattr(cfg, 'EXPORT_VIDEOS'):
            try:
                config['export_videos'] = bool(getattr(cfg, 'EXPORT_VIDEOS'))
            except Exception:
                pass
        if hasattr(cfg, 'EXPORT_DEPTH_VIDEOS'):
            try:
                config['export_depth_videos'] = bool(getattr(cfg, 'EXPORT_DEPTH_VIDEOS'))
            except Exception:
                pass
        
        if hasattr(cfg, 'MAIN_SCENE_NAME'):
            config['main_scene_name'] = getattr(cfg, 'MAIN_SCENE_NAME', 'Logistics')
        if hasattr(cfg, 'SUB_SCENE_NAME'):
            config['sub_scene_name'] = getattr(cfg, 'SUB_SCENE_NAME')
        if hasattr(cfg, 'ACTION_NAME'):
            config['action_name'] = getattr(cfg, 'ACTION_NAME')
        
        if config['use_copasi_structure']:
            print(f"📋 使用Python配置文件: parquet2hdf5_config.py")
            print(f"   输出目录: {config['output_root']}")
    except ImportError:
        # 配置文件不存在，使用默认行为
        pass
    except Exception as e:
        print(f"⚠️  读取Python配置文件失败: {e}")
    
    return config


def convert_parquet_to_hdf5(
    repo_id: str,
    dataset_root: Optional[str] = None,
    config_file: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    export_videos: Optional[bool] = None,
    export_depth_videos: Optional[bool] = None,
    verbose: bool = False,
    output_root_override: Optional[str] = None,
    timestamped_logistics_dir: bool = True,
) -> bool:
    """
    将数据集的parquet文件转换为hdf5格式（使用自适应转换）
    支持从 hdf5_configs/ 目录下的JSON文件读取自定义配置
    
    Args:
        repo_id: 数据集ID
        dataset_root: 数据集根目录
        config_file: 可选的配置文件路径，如果为None则自动查找
        
    Returns:
        是否转换成功
    """
    def _progress(current: int, total: int, msg: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(int(current), int(total), str(msg))
        except Exception:
            # UI 回调异常不应影响转换主流程
            pass

    def _cancelled() -> bool:
        if should_cancel is None:
            return False
        try:
            return bool(should_cancel())
        except Exception:
            return False

    _progress(0, 0, "正在加载转换配置…")
    # 加载转换配置
    conversion_config = _load_conversion_config(config_file)
    # UI 侧强制指定输出目录：优先级最高（用于确保 COPASI 输出落到用户选择的目录）
    if output_root_override:
        try:
            conversion_config["output_root"] = str(Path(output_root_override).expanduser())
            conversion_config["use_copasi_structure"] = True
        except Exception:
            conversion_config["output_root"] = str(output_root_override)
            conversion_config["use_copasi_structure"] = True
    do_export_videos = bool(conversion_config.get("export_videos", True)) if export_videos is None else bool(export_videos)
    do_export_depth_videos = bool(conversion_config.get("export_depth_videos", True)) if export_depth_videos is None else bool(export_depth_videos)
    
    if dataset_root is None:
        dataset_root = os.environ.get("LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
    
    dataset_root = Path(dataset_root).expanduser()
    dataset_path = dataset_root / repo_id
    
    if not dataset_path.exists():
        print(f"数据集目录不存在: {dataset_path}")
        return False
    
    # LeRobot v2.0 数据集结构：data/chunk-*/episode_*.parquet
    # 也支持旧版本的 train/*.parquet 结构
    parquet_files = []
    
    # 首先尝试新结构：data/chunk-*/
    data_dir = dataset_path / "data"
    if data_dir.exists():
        # 递归查找所有chunk目录下的parquet文件
        parquet_files = list(data_dir.rglob("*.parquet"))
        if parquet_files:
            print(f"找到 {len(parquet_files)} 个parquet文件（在data目录下）")
    
    # 如果新结构没有文件，尝试旧结构：train/
    if not parquet_files:
        train_dir = dataset_path / "train"
        if train_dir.exists():
            parquet_files = list(train_dir.glob("*.parquet"))
            if parquet_files:
                print(f"找到 {len(parquet_files)} 个parquet文件（在train目录下）")
    
    if not parquet_files:
        print(f"未找到parquet文件")
        print(f"  已检查: {data_dir} 和 {dataset_path / 'train'}")
        return False

    # 稳定排序，便于进度显示与可重复性
    try:
        parquet_files = sorted(parquet_files, key=lambda p: str(p))
    except Exception:
        pass
    
    # 本次转换的时间戳（用于输出文件名去重/目录分组）
    try:
        from datetime import datetime as _dt
        run_ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    except Exception:
        run_ts = str(int(time.time()))

    logistics_dir: Optional[Path] = None
    # 注意：不要在输出中额外创建 uuids/ 索引目录（用户认为重复）

    # 确定输出目录
    if conversion_config['use_copasi_structure'] and conversion_config['output_root']:
        # 使用COPASI风格输出结构（完全按照convert_parquet_to_hdf5_adaptive.py的结构）
        output_root = Path(conversion_config['output_root']).expanduser()
        logistics_dir_name = f"Logistics_{run_ts}" if timestamped_logistics_dir else "Logistics"
        logistics_dir = output_root / logistics_dir_name
        # 防止同一秒内多次转换把不同批次写进同一个 Logistics_YYYYMMDD_HHMMSS 目录
        if timestamped_logistics_dir:
            try:
                if logistics_dir.exists() and any(logistics_dir.iterdir()):
                    i = 2
                    while True:
                        cand = output_root / f"{logistics_dir_name}_{i}"
                        if not cand.exists():
                            logistics_dir = cand
                            break
                        i += 1
            except Exception:
                pass
        try:
            logistics_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            # 常见：默认配置指向 /home/dreame/...，当前用户无权限
            print(f"⚠️  无权限创建输出目录 {logistics_dir}，将回退到数据集目录内输出: {e}")
            _progress(0, len(parquet_files), f"输出目录无权限，回退到数据集目录内的 hdf5/（{e}）")
            conversion_config['use_copasi_structure'] = False
            conversion_config['output_root'] = None
        except OSError as e:
            print(f"⚠️  创建输出目录失败 {logistics_dir}，将回退到数据集目录内输出: {e}")
            _progress(0, len(parquet_files), f"创建输出目录失败，回退到数据集目录内的 hdf5/（{e}）")
            conversion_config['use_copasi_structure'] = False
            conversion_config['output_root'] = None
        
        if conversion_config['use_copasi_structure'] and conversion_config['output_root']:
            # 创建场景目录结构
            main_scene = conversion_config['main_scene_name'] or 'Logistics'
            sub_scene = conversion_config['sub_scene_name'] or 'Default'
            action = conversion_config['action_name'] or repo_id
            
            # 创建task_info目录和meta文件（必须在创建场景目录之前）
            task_info_dir = logistics_dir / "task_info"
            task_info_dir.mkdir(exist_ok=True)
            meta_filename = f"{main_scene}-{sub_scene}-{action}.json"
            meta_file = task_info_dir / meta_filename
            if not meta_file.exists():
                meta_file.touch()
            
            # 创建场景目录结构
            # 用户希望顶层就是 Logistics_<ts>，不要再出现 Logistics/Logistics 的重复嵌套：
            # 当 main_scene 默认是 'Logistics' 时，直接把 main_scene_dir 设为 logistics_dir 本身。
            if str(main_scene).strip() == "Logistics":
                main_scene_dir = logistics_dir
            else:
                main_scene_dir = logistics_dir / main_scene
            sub_scene_dir = main_scene_dir / sub_scene
            action_dir = sub_scene_dir / action
            action_dir.mkdir(parents=True, exist_ok=True)
            
            hdf5_base_dir = action_dir
            print(f"📁 使用COPASI输出结构: {hdf5_base_dir}")
        else:
            # 回退：在数据集目录下创建hdf5目录
            hdf5_base_dir = dataset_path / "hdf5"
            hdf5_base_dir.mkdir(exist_ok=True)
            print(f"📁 回退到默认输出结构: {hdf5_base_dir}")
    else:
        # 默认行为：在数据集目录下创建hdf5目录
        hdf5_base_dir = dataset_path / "hdf5"
        hdf5_base_dir.mkdir(exist_ok=True)
        print(f"📁 使用默认输出结构: {hdf5_base_dir}")

    # 明确告诉 UI：最终输出在哪里
    try:
        _progress(0, len(parquet_files), f"输出目录: {hdf5_base_dir}")
    except Exception:
        pass
    
    try:
        print(f"开始转换 {len(parquet_files)} 个parquet文件到hdf5（自适应模式）...")
        total_files = len(parquet_files)
        _progress(0, total_files, f"准备转换（共 {total_files} 个 parquet）…")
        
        success_count = 0
        generated_uuids = []  # 记录本次转换生成的所有 UUID
        import uuid
        
        for idx, parquet_file in enumerate(parquet_files, start=1):
            if _cancelled():
                print("⚠️  用户取消转换")
                _progress(idx - 1, total_files, "已取消")
                return False

            _progress(idx - 1, total_files, f"转换中：{parquet_file.name}（{idx}/{total_files}）")
            if conversion_config['use_copasi_structure']:
                # COPASI结构：完全按照convert_parquet_to_hdf5_adaptive.py的结构
                # 每个episode一个UUID目录，包含完整的子目录结构
                episode_uuid = str(uuid.uuid4())
                episode_dir = hdf5_base_dir / episode_uuid
                
                # 创建完整的episode目录结构（与create_episode_structure一致）
                camera_dir = episode_dir / "camera"
                video_dir = camera_dir / "video"
                depth_dir = camera_dir / "depth"
                audio_dir = episode_dir / "audio"
                parameters_dir = episode_dir / "parameters"
                proprio_stats_dir = episode_dir / "proprio_stats"
                
                # 创建所有目录（video, audio, parameters留空，只创建目录）
                video_dir.mkdir(parents=True, exist_ok=True)
                depth_dir.mkdir(parents=True, exist_ok=True)
                audio_dir.mkdir(parents=True, exist_ok=True)
                parameters_dir.mkdir(parents=True, exist_ok=True)
                proprio_stats_dir.mkdir(parents=True, exist_ok=True)
                
                # 在可能为空的目录下创建占位文件，确保目录结构被保留
                for empty_dir in [video_dir, audio_dir, parameters_dir]:
                    keep_file = empty_dir / ".keep"
                    if not keep_file.exists():
                        keep_file.touch()
                
                # HDF5文件放在proprio_stats目录下（文件名带时间戳，便于区分不同次转换）
                hdf5_file = proprio_stats_dir / f"proprio_stats_{run_ts}.hdf5"
            else:
                # 默认结构：直接在hdf5目录下
                hdf5_file = hdf5_base_dir / f"{parquet_file.stem}_{run_ts}.hdf5"
            
            # 使用自适应转换函数（支持取消/字段级进度）
            def _inner_progress(cur: int, total: int, msg: str) -> None:
                # 把字段级进度折叠成一条文字，让 UI 及时刷新（不改变外层 current/total 语义）
                _progress(idx, total_files, f"{parquet_file.name}: {msg}")

            success = convert_parquet_to_hdf5_adaptive(
                str(parquet_file),
                str(hdf5_file),
                progress_callback=_inner_progress,
                should_cancel=should_cancel,
                verbose=verbose,
            )
            
            if success:
                success_count += 1
                # 附加对齐索引与统一时间戳字段（不修改 *_mp4_timestamps 原始列）
                try:
                    _attach_alignment_and_unified_timestamps(
                        dataset_path=dataset_path,
                        parquet_file=parquet_file,
                        hdf5_file=hdf5_file,
                        fps=conversion_config.get("fps", 30),
                    )
                except Exception as _align_e:
                    print(f"⚠️  写入对齐索引/统一时间戳失败（忽略）: {_align_e}")

                if conversion_config['use_copasi_structure']:
                    print(f"✓ 已转换: {parquet_file.name} -> {episode_uuid}/proprio_stats/{hdf5_file.name}")
                    generated_uuids.append(episode_uuid)  # 记录生成的 UUID

                    images_dir = _locate_episode_images_dir(dataset_path, parquet_file)
                    if images_dir and (do_export_videos or do_export_depth_videos):
                        _progress(idx, total_files, f"{parquet_file.name}: 正在导出视频（很慢）…")
                        _generate_episode_videos(
                            images_dir,
                            video_dir,
                            depth_dir,
                            fps=conversion_config.get('fps', 30),
                            export_rgb=do_export_videos,
                            export_depth=do_export_depth_videos,
                        )
                    elif images_dir and (not do_export_videos) and (not do_export_depth_videos):
                        _progress(idx, total_files, f"{parquet_file.name}: 已跳过视频导出（export_videos=0 且 export_depth_videos=0）")
                    else:
                        print(f"⚠️  未找到对应图像目录，跳过视频导出: {parquet_file.stem}")
                else:
                    print(f"✓ 已转换: {parquet_file.name} -> {hdf5_file.name}")
            else:
                print(f"✗ 转换失败: {parquet_file.name}")

            # 每处理完一个 parquet，推一次进度
            _progress(idx, total_files, f"已完成：{parquet_file.name}（{idx}/{total_files}）")
        
        # 如果使用COPASI结构，在最后重命名目录（添加大小、计数、时长后缀）
        # 但当顶层 Logistics 已带时间戳时，用户希望保持“Logistics_YYYYMMDD_HHMMSS”命名，不再自动改名。
        if conversion_config['use_copasi_structure'] and success_count > 0 and (not timestamped_logistics_dir):
            try:
                _finalize_copasi_names_with_config(
                    logistics_dir, main_scene_dir, sub_scene_dir, action_dir, 
                    success_count, conversion_config
                )
            except Exception as e:
                print(f"⚠️  目录重命名失败（忽略）: {e}")
        
        print(f"\n转换完成！成功: {success_count}/{len(parquet_files)}")
        _progress(total_files, total_files, f"转换完成：成功 {success_count}/{total_files}")

        # 写入数据集级别的"已转换"标记，方便 GUI 状态刷新（不依赖输出目录位置）
        try:
            if success_count > 0:
                meta_dir = dataset_path / "meta"
                meta_dir.mkdir(exist_ok=True)
                flag_path = meta_dir / "hdf5_converted.json"
                payload = {
                    "converted": True,
                    "converted_at": run_ts,
                    "parquet_files": [pf.name for pf in parquet_files],
                    "use_copasi_structure": bool(
                        conversion_config.get("use_copasi_structure")
                        and conversion_config.get("output_root")
                    ),
                    "output_root": str(conversion_config.get("output_root") or ""),
                    "output_dir": str(hdf5_base_dir),
                    "logistics_dir": str(logistics_dir) if logistics_dir is not None else "",
                    "timestamped_logistics_dir": bool(timestamped_logistics_dir),
                    "generated_uuids": generated_uuids,  # 记录本次转换生成的 UUID 列表
                }
                with flag_path.open("w", encoding="utf-8") as f_flag:
                    json.dump(payload, f_flag, ensure_ascii=False, indent=2)
                print(f"📝 已写入转换标记: {flag_path}，包含 {len(generated_uuids)} 个 UUID")
        except Exception as _flag_e:
            print(f"⚠️  写入转换标记失败（忽略）: {_flag_e}")

        return success_count > 0
        
    except ImportError:
        print("错误: 需要安装 pandas, numpy 和 h5py")
        print("请运行: pip install pandas numpy h5py")
        return False
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

