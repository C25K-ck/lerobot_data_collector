#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense 右手相机管理器
======================

该模块将 RealSense D455 的采集流程封装为可复用的类，既能在
数采软件中提供实时预览，也能把 RGB/Depth 帧与时间戳落地到
LeRobot 数据集结构中（images/hand_r_xxx）。

核心特性：
    - 自动寻找指定序列号的摄像头并保持持续预览
    - 支持异步落地 JPEG/PNG 并记录纳秒级时间戳
    - 采集开始/结束可由外部 GUI 控制，确保无相机时也不会阻塞
    - 默认数据根路径改为 /home/dreame/data/hf_dataset，可通过
      LEROBOT_HOME 或 set_dataset_home() 覆盖
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV 在测试环境可能缺失
    cv2 = None

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - RealSense SDK 在测试环境可能缺失
    rs = None

try:
    # 兼容脚本直跑与包内导入两种方式
    from conf import REALSENSE_RIGHT_SN  # type: ignore[import]
except Exception:  # pragma: no cover - 在包模式下尝试相对导入
    try:
        from .conf import REALSENSE_RIGHT_SN  # type: ignore[import]
    except Exception:
        # 如果仍然失败，使用空串占位，后续会自动回退到检测到的第一台设备
        REALSENSE_RIGHT_SN = ""

RS_RIGHT_READ_CORES = [0, 1, 2, 3, 12]
RS_RIGHT_WRITE_CORES = [13, 14, 15]

DEFAULT_DATASET_HOME = Path(
    os.environ.get("LEROBOT_HOME", "/home/dreame/data/hf_dataset")
).expanduser()

logger = logging.getLogger(__name__)


def _bind_current_thread(cores: list[int]) -> None:
    """将当前线程绑定到指定 CPU，失败时静默忽略。"""
    if not cores:
        return
    try:
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, set(cores))
    except Exception:
        pass


@dataclass
class EpisodePaths:
    episode_dir: Path
    rgb_dir: Path
    depth_dir: Path
    timestamp_file: Path


class SaveWorker:
    """使用队列异步落地图像，避免阻塞采集线程。"""

    def __init__(self, maxsize: int = 4096):
        self.queue: queue.Queue[
            Optional[Tuple[np.ndarray, np.ndarray, str, str, int, str]]
        ] = queue.Queue(maxsize=maxsize)
        self._running = True
        self._dropped = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def dropped(self) -> int:
        return self._dropped

    def submit(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        color_path: str,
        depth_path: str,
        timestamp_ns: int,
        timestamp_file: str,
    ) -> None:
        if not self._running:
            return
        try:
            self.queue.put_nowait(
                (color_image, depth_image, color_path, depth_path, timestamp_ns, timestamp_file)
            )
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Exception:
                pass
            try:
                self.queue.put_nowait(
                    (color_image, depth_image, color_path, depth_path, timestamp_ns, timestamp_file)
                )
            except Exception:
                self._dropped += 1

    def flush(self, timeout: Optional[float] = None) -> None:
        """等待队列写空。"""
        start = time.time()
        while True:
            if self.queue.unfinished_tasks == 0:
                return
            if timeout is not None and (time.time() - start) > timeout:
                return
            time.sleep(0.01)

    def stop(self) -> None:
        self._running = False
        try:
            self.queue.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        _bind_current_thread(RS_RIGHT_WRITE_CORES)
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            color_image, depth_image, color_path, depth_path, timestamp_ns, timestamp_file = item
            try:
                if cv2 is None:
                    raise RuntimeError("OpenCV 未安装，无法落地图像")
                if color_image is None or getattr(color_image, "size", 0) == 0:
                    raise RuntimeError("写入彩色图失败: 空图像")
                if depth_image is None or getattr(depth_image, "size", 0) == 0:
                    raise RuntimeError("写入深度图失败: 空图像")
                os.makedirs(os.path.dirname(color_path), exist_ok=True)
                os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                color_ok = cv2.imwrite(color_path, color_image)
                if not color_ok:
                    raise RuntimeError(f"写入彩色图失败: {color_path}")
                depth_ok = cv2.imwrite(depth_path, depth_image)
                if not depth_ok:
                    raise RuntimeError(f"写入深度图失败: {depth_path}")
                with open(timestamp_file, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp_ns}\n")
            except Exception as exc:
                logger.warning("RealSense 右手保存失败: %s", exc)
            finally:
                del color_image
                del depth_image
                self.queue.task_done()


class RealSenseRightCameraManager:
    """负责 RealSense 右手相机的预览与录制。"""

    def __init__(
        self,
        serial_number: Optional[str] = None,
        dataset_home: Path | str | None = None,
        preview_fps: int = 15,
        save_queue_size: int = 4096,
        enable_depth: bool = True,
        frame_timestamp_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.serial_number = serial_number or REALSENSE_RIGHT_SN
        self.dataset_home = Path(dataset_home or DEFAULT_DATASET_HOME).expanduser()
        self.preview_interval = 1.0 / max(1, preview_fps)
        self.save_worker = SaveWorker(maxsize=save_queue_size)
        self.enable_depth = enable_depth
        self._has_depth = enable_depth
        self.log_prefix = "RealSense 右手"

        self.pipeline: Optional["rs.pipeline"] = None
        self.align: Optional["rs.align"] = None
        self._color_format = None
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._recording = False
        self._episode_paths: Optional[EpisodePaths] = None
        self._frame_count = 0
        self._record_target_frames: Optional[int] = None
        self._last_preview_ts = 0.0
        self._preview_lock = threading.Lock()
        self._latest_preview: Optional[np.ndarray] = None
        self.last_error: str = ""
        self.active_serial: Optional[str] = None
        self._frame_timestamp_callback = frame_timestamp_callback

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        """初始化摄像头并启动持续预览线程。"""
        if not self._deps_ready():
            return False
        if self._running:
            return True
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                self._init_camera()
                break
            except Exception as exc:  # pragma: no cover - 实机错误
                self.last_error = str(exc)
                logger.warning(
                    "RealSense 右手初始化失败(第 %d/%d 次): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt >= max_attempts:
                    return False
                time.sleep(0.8)

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("RealSense 右手预览线程已启动")
        return True

    def stop(self) -> None:
        """停止预览线程，但保留流水线，便于后续再次启动。"""
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._capture_thread = None

    def start_recording(
        self, repo_id: str, target_frames: Optional[int] = None
    ) -> Optional[EpisodePaths]:
        """开始向指定 repo 的下一 episode 落地图像。

        Args:
            repo_id: 数据集 ID
            target_frames: 期望采集的图像帧数；如果为 None 则不自动停止
        """
        if not repo_id:
            raise ValueError("repo_id 不能为空")
        if not self._deps_ready():
            logger.warning("RealSense SDK/OpenCV 缺失，跳过相机采集")
            return None
        if not self._running:
            started = self.start()
            if not started:
                return None

        dataset_root = self.dataset_home / repo_id
        episode_paths = self._prepare_episode_dirs(dataset_root)
        self._episode_paths = episode_paths
        self._frame_count = 0
        self._record_target_frames = int(target_frames) if target_frames else None
        self._recording = True
        logger.info(
            "RealSense 右手开始录制，目录：%s，目标帧数：%s",
            episode_paths.episode_dir,
            self._record_target_frames,
        )
        return episode_paths

    def stop_recording(self, wait: bool = True) -> None:
        """结束录制并等待队列写空。"""
        self._recording = False
        self._record_target_frames = None
        if wait:
            self.save_worker.flush(timeout=5.0)

    def shutdown(self) -> None:
        """完全关闭摄像头与保存线程。"""
        self.stop_recording(wait=True)
        self.stop()
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None
        self.align = None
        self.save_worker.stop()

    def set_dataset_home(self, path: Path | str) -> None:
        """更新数据根目录。"""
        self.dataset_home = Path(path).expanduser()

    def get_latest_preview(self) -> Optional[np.ndarray]:
        """返回最新的 RGB 预览图（H, W, 3），若暂无则返回 None。"""
        with self._preview_lock:
            if self._latest_preview is None:
                return None
            return self._latest_preview.copy()

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _deps_ready(self) -> bool:
        if rs is None:
            self.last_error = "pyrealsense2 未安装"
            return False
        if cv2 is None:
            self.last_error = "OpenCV 未安装"
            return False
        return True

    def _init_camera(self) -> None:
        ctx = rs.context()
        devices = ctx.query_devices()
        first_device = None
        target_device = None
        for dev in devices:
            if first_device is None:
                first_device = dev
            try:
                dev_sn = dev.get_info(rs.camera_info.serial_number)
            except Exception:
                dev_sn = None
            if dev_sn and self.serial_number and dev_sn == self.serial_number:
                target_device = dev
                break

        if target_device is None:
            if first_device is None:
                raise RuntimeError("未检测到任何 RealSense 设备")
            if not self.serial_number:
                target_device = first_device
                fallback_sn = target_device.get_info(rs.camera_info.serial_number)
                logger.warning(
                    "未设置序列号，RealSense 将使用检测到的设备 SN=%s",
                    fallback_sn,
                )
                self.serial_number = fallback_sn
            else:
                raise RuntimeError(
                    f"未找到序列号 {self.serial_number} 的 RealSense，请检查连接",
                )

        color_profile, depth_profile = self._select_supported_profiles(target_device)
        self._has_depth = depth_profile is not None
        self._color_format = color_profile[3]

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(
            rs.stream.color,
            color_profile[0],
            color_profile[1],
            color_profile[3],
            color_profile[2],
        )
        if depth_profile:
            config.enable_stream(
                rs.stream.depth,
                depth_profile[0],
                depth_profile[1],
                depth_profile[3],
                depth_profile[2],
            )
        profile = pipeline.start(config)
        sn = profile.get_device().get_info(rs.camera_info.serial_number)
        logger.info("%s连接成功: SN=%s", self.log_prefix, sn)
        self.active_serial = sn
        self.pipeline = pipeline
        self.align = rs.align(rs.stream.color)

    def _capture_loop(self) -> None:  # pragma: no cover - 需实机验证
        _bind_current_thread(RS_RIGHT_READ_CORES)
        while self._running and self.pipeline:
            has_depth = bool(self.enable_depth and self._has_depth)
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=200)
            except Exception:
                continue

            if self.align:
                try:
                    frames = self.align.process(frames)
                except RuntimeError as exc:
                    logger.debug("RealSense 右手对齐失败，重试: %s", exc)
                    continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if has_depth else None
            if not color_frame or (has_depth and not depth_frame):
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
            # 右手相机画面上下翻转，进行垂直翻转纠正方向
            color_image = cv2.flip(color_image, 0)
            if depth_image is not None:
                depth_image = cv2.flip(depth_image, 0)

            now = time.time()
            if now - self._last_preview_ts >= self.preview_interval:
                if rs is not None and self._color_format == rs.format.rgb8:
                    preview_rgb = color_image
                else:
                    preview_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                with self._preview_lock:
                    self._latest_preview = preview_rgb
                self._last_preview_ts = now

            if self._recording and self._episode_paths:
                timestamp_ns = time.time_ns()
                if self._frame_timestamp_callback is not None:
                    try:
                        self._frame_timestamp_callback(timestamp_ns)
                    except Exception:
                        pass
                rgb_path = self._episode_paths.rgb_dir / f"{self._frame_count}.jpg"
                depth_path = self._episode_paths.depth_dir / f"{self._frame_count}.png"
                if has_depth and depth_image is not None:
                    self.save_worker.submit(
                        color_image.copy(),
                        depth_image.copy(),
                        str(rgb_path),
                        str(depth_path),
                        timestamp_ns,
                        str(self._episode_paths.timestamp_file),
                    )
                else:
                    self.save_worker.submit(
                        color_image.copy(),
                        np.zeros_like(color_image[:, :, 0]),
                        str(rgb_path),
                        str(depth_path),
                        timestamp_ns,
                        str(self._episode_paths.timestamp_file),
                    )
                self._frame_count += 1

                # 如果设置了目标帧数，到达后自动停止录制（预览仍继续）
                if (
                    self._record_target_frames is not None
                    and self._frame_count >= self._record_target_frames
                ):
                    logger.info(
                        "RealSense 右手达到目标帧数 %d（相机侧），停止该路写盘以确保帧数对齐",
                        self._record_target_frames,
                    )
                    self._recording = False
                    self._record_target_frames = None

    def _prepare_episode_dirs(self, dataset_root: Path) -> EpisodePaths:
        """根据 lerobot 数据集结构准备 episode 目录。

        约定结构：
            <dataset_root>/
              data/chunk-000/episode_xxxxxx.parquet   # 结构化数据（由 LeRobot 写）
              chunk_000/episode_xxxxxx/               # 图像等原始数据（本模块写）
        """
        dataset_root.mkdir(parents=True, exist_ok=True)

        # 1) data/chunk-000 用于查找已有 episode_xxxxxx.parquet，确定下一个编号
        data_chunk = dataset_root / "data" / "chunk-000"
        data_chunk.mkdir(parents=True, exist_ok=True)

        existing: list[int] = []
        for parquet in data_chunk.glob("episode_*.parquet"):
            try:
                number = int(parquet.stem.split("_")[1])
                existing.append(number)
            except Exception:
                continue
        next_idx = max(existing) + 1 if existing else 0

        # 2) 图像落地到 chunk_000/episode_xxxxxx/，与历史数据保持一致
        images_root = dataset_root / "chunk_000"
        episode_dir = images_root / f"episode_{next_idx:06d}"
        images_dir = episode_dir / "images"
        rgb_dir = images_dir / "hand_r_rgb"
        depth_dir = images_dir / "hand_r_depth"

        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        timestamp_file = rgb_dir / "hand_r_timestamps.txt"
        if timestamp_file.exists():
            timestamp_file.unlink()
        timestamp_file.touch()

        return EpisodePaths(
            episode_dir=episode_dir,
            rgb_dir=rgb_dir,
            depth_dir=depth_dir,
            timestamp_file=timestamp_file,
        )

    def _select_supported_profiles(
        self, device: "rs.device"
    ) -> Tuple[Tuple[int, int, int, rs.format], Optional[Tuple[int, int, int, rs.format]]]:
        color_profile = self._find_profile(device, rs.stream.color)
        if color_profile is None:
            raise RuntimeError("目标设备缺少 1280x720@30 彩色流，无法启动预览")

        depth_profile = None
        if self.enable_depth:
            depth_profile = self._find_profile(device, rs.stream.depth)
            if depth_profile is None:
                logger.warning("%s未检测到 1280x720@30 深度流，将仅输出 RGB", self.log_prefix)

        return color_profile, depth_profile

    def _find_profile(
        self, device: "rs.device", stream_type: int
    ) -> Optional[Tuple[int, int, int, rs.format]]:
        target_resolution = (1280, 720, 30)
        for sensor in device.query_sensors():
            try:
                stream_profiles = sensor.get_stream_profiles()
            except Exception:
                continue
            for profile in stream_profiles:
                if profile.stream_type() != stream_type:
                    continue
                try:
                    video_profile = profile.as_video_stream_profile()
                except Exception:
                    continue
                if (
                    video_profile.width() == target_resolution[0]
                    and video_profile.height() == target_resolution[1]
                    and profile.fps() == target_resolution[2]
                ):
                    return (
                        video_profile.width(),
                        video_profile.height(),
                        profile.fps(),
                        profile.format(),
                    )
        return None


def main() -> None:  # pragma: no cover - 用于手动调试
    """简单的 CLI：启动预览并持续记录到默认 repo。"""
    logging.basicConfig(level=logging.INFO)
    manager = RealSenseRightCameraManager()
    if not manager.start():
        print(f"RealSense 右手不可用：{manager.last_error}")
        return

    repo_id = f"realsense_demo_{int(time.time())}"
    manager.start_recording(repo_id)
    print(f"开始录制到 repo: {repo_id}，按 Ctrl+C 停止")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_recording()
        manager.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()