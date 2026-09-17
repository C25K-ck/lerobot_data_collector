#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奥比中光头部相机管理器
====================

该模块将奥比中光相机的采集流程封装为可复用的类，既能在
数采软件中提供实时预览，也能把 RGB/Depth 帧与时间戳落地到
LeRobot 数据集结构中（images/headf_rgbd_xxx）。

核心特性：
    - 直接控制奥比中光相机硬件，无需ROS2
    - 自动寻找并连接相机设备
    - 支持异步落地 JPEG/PNG 并记录纳秒级时间戳
    - 采集开始/结束可由外部 GUI 控制
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV 在测试环境可能缺失
    cv2 = None

try:
    from pyorbbecsdk import (
        Pipeline,
        Config,
        OBSensorType,
        OBFormat,
        OBAlignMode,
        OBPropertyID,
    )
    ORBBEC_AVAILABLE = True
except ImportError:  # pragma: no cover - Orbbec SDK 在测试环境可能缺失
    ORBBEC_AVAILABLE = False
    Pipeline = None
    Config = None

DEFAULT_DATASET_HOME = Path(
    os.environ.get("LEROBOT_HOME", "/home/dreame/data/hf_dataset")
).expanduser()

logger = logging.getLogger(__name__)

ORBBEC_READ_CORES = [4, 5, 6, 7]
ORBBEC_WRITE_CORES = [13, 14, 15]

# 自定义分辨率设置
COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
COLOR_FPS = 30
DEPTH_WIDTH = 1280
DEPTH_HEIGHT = 720
DEPTH_FPS = 30


def _bind_current_thread(cores: list[int]) -> None:
    """将当前线程绑定到指定 CPU，失败时静默忽略。"""
    if not cores:
        return
    try:
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, set(cores))
    except Exception:
        pass


def _frame_to_bgr_image(frame) -> Optional[np.ndarray]:
    """将奥比中光帧转换为BGR图像。"""
    if frame is None:
        return None
    try:
        # 获取帧数据
        width = frame.get_width()
        height = frame.get_height()
        data = frame.get_data()
        
        # 根据格式转换
        format_type = frame.get_format()
        
        if format_type == OBFormat.RGB:
            # RGB格式，需要转换为BGR
            image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            # RGB转BGR
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return bgr_image
        elif format_type == OBFormat.BGR:
            # 已经是BGR格式
            image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            return image
        else:
            # 其他格式，尝试直接转换
            logger.warning(f"不支持的帧格式: {format_type}")
            return None
    except Exception as e:
        logger.warning(f"帧转换失败: {e}")
        return None


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
        _bind_current_thread(ORBBEC_WRITE_CORES)
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
                logger.warning("奥比中光头部保存失败: %s", exc)
            finally:
                del color_image
                del depth_image
                self.queue.task_done()


class OrbbecHeadCameraManager:
    """负责奥比中光头部相机的预览与录制。"""

    def __init__(
        self,
        dataset_home: Path | str | None = None,
        preview_fps: int = 15,
        save_queue_size: int = 4096,
        enable_depth: bool = True,
    ) -> None:
        self.dataset_home = Path(dataset_home or DEFAULT_DATASET_HOME).expanduser()
        self.preview_interval = 1.0 / max(1, preview_fps)
        self.save_worker = SaveWorker(maxsize=save_queue_size)
        self.enable_depth = enable_depth
        self.log_prefix = "奥比中光头部"

        self.pipeline: Optional["Pipeline"] = None
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
        self._cache_count = 0  # 用于跳过前30帧（预热）
        self._last_loop_ts = 0.0
        self._loop_fps = 0.0

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
                    "奥比中光头部初始化失败(第 %d/%d 次): %s",
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
        logger.info("奥比中光头部预览线程已启动")
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
            logger.warning("奥比中光 SDK/OpenCV 缺失，跳过相机采集")
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
        self._cache_count = 0  # 重置缓存计数
        logger.info(
            "奥比中光头部开始录制，目录：%s，目标帧数：%s",
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

    def get_loop_fps(self) -> float:
        """返回当前采集循环 FPS（诊断用）。"""
        return float(self._loop_fps)

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _deps_ready(self) -> bool:
        if not ORBBEC_AVAILABLE:
            self.last_error = "pyorbbecsdk 未安装"
            return False
        if cv2 is None:
            self.last_error = "OpenCV 未安装"
            return False
        return True

    def _init_camera(self) -> None:
        """初始化奥比中光相机。"""
        pipeline = Pipeline()
        config = Config()

        # 获取彩色流配置
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = profile_list.get_video_stream_profile(
                COLOR_WIDTH, COLOR_HEIGHT, OBFormat.RGB, COLOR_FPS
            )
            logger.info(
                f"使用自定义彩色流配置: {COLOR_WIDTH}x{COLOR_HEIGHT} RGB {COLOR_FPS}fps"
            )
        except Exception:
            color_profile = profile_list.get_default_video_stream_profile()
            logger.info("使用默认彩色流配置")
        config.enable_stream(color_profile)

        # 获取深度流配置
        if self.enable_depth:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                depth_profile = profile_list.get_video_stream_profile(
                    DEPTH_WIDTH, DEPTH_HEIGHT, OBFormat.Y16, DEPTH_FPS
                )
                logger.info(
                    f"使用自定义深度流配置: {DEPTH_WIDTH}x{DEPTH_HEIGHT} Y16 {DEPTH_FPS}fps"
                )
            except Exception:
                depth_profile = profile_list.get_default_video_stream_profile()
                logger.info("使用默认深度流配置")
            config.enable_stream(depth_profile)

        # 启用硬件D2C对齐
        config.set_align_mode(OBAlignMode.HW_MODE)
        logger.info("启用硬件D2C对齐")

        # 启动pipeline
        pipeline.start(config)
        logger.info("%s连接成功", self.log_prefix)

        # 关闭RGB自动曝光并设置手动曝光（微秒）
        try:
            device = pipeline.get_device()
            sensors = device.get_sensor_list()
            for i in range(sensors.get_sensor_count()):
                s = sensors.get_sensor(i)
                if s.get_sensor_type() == OBSensorType.COLOR_SENSOR:
                    s.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
                    s.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 20000)
                    break
        except Exception as e:
            logger.debug("设置曝光参数失败: %s", e)

        self.pipeline = pipeline

    def _capture_loop(self) -> None:  # pragma: no cover - 需实机验证
        """采集循环。"""
        _bind_current_thread(ORBBEC_READ_CORES)
        while self._running and self.pipeline:
            try:
                # 等待帧数据
                # 使用较小超时避免阻塞导致积压
                frames = self.pipeline.wait_for_frames(100)
                if not frames:
                    continue

                # 跳过前30帧（预热）
                if self._cache_count < 30:
                    self._cache_count += 1
                    continue

                # 获取原始帧
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame() if self.enable_depth else None

                if not color_frame:
                    continue

                # 处理彩色图像
                color_image = _frame_to_bgr_image(color_frame)
                if color_image is None:
                    continue

                # 处理深度数据
                depth_image = None
                if depth_frame and self.enable_depth:
                    try:
                        # 直接保存原始 uint16 深度，避免每帧浮点转换/过滤造成卡顿
                        depth_image = np.frombuffer(
                            depth_frame.get_data(), dtype=np.uint16
                        ).reshape((depth_frame.get_height(), depth_frame.get_width()))
                    except Exception as e:
                        logger.debug("深度数据处理失败: %s", e)

                # 更新预览
                now = time.time()
                if now - self._last_preview_ts >= self.preview_interval:
                    preview_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    with self._preview_lock:
                        self._latest_preview = preview_rgb
                    self._last_preview_ts = now

                # 录制处理
                if self._recording and self._episode_paths:
                    timestamp_ns = time.time_ns()
                    rgb_path = self._episode_paths.rgb_dir / f"{self._frame_count}.jpg"
                    depth_path = self._episode_paths.depth_dir / f"{self._frame_count}.png"
                    legacy_headfl = self._episode_paths.episode_dir / "images" / "headfl" / f"{self._frame_count}.jpg"
                    legacy_headf = self._episode_paths.episode_dir / "images" / "headf" / f"{self._frame_count}.jpg"
                    legacy_headfr = self._episode_paths.episode_dir / "images" / "headfr" / f"{self._frame_count}.jpg"

                    if depth_image is not None:
                        self.save_worker.submit(
                            color_image.copy(),
                            depth_image.copy(),
                            str(rgb_path),
                            str(depth_path),
                            timestamp_ns,
                            str(self._episode_paths.timestamp_file),
                        )
                        try:
                            cv2.imwrite(str(legacy_headfl), color_image)
                            cv2.imwrite(str(legacy_headf), color_image)
                            cv2.imwrite(str(legacy_headfr), color_image)
                        except Exception as e:
                            logger.warning("奥比中光头部兼容目录写入失败：%s", e)
                    else:
                        # 如果没有深度图，使用零填充
                        dummy_depth = np.zeros_like(color_image[:, :, 0])
                        self.save_worker.submit(
                            color_image.copy(),
                            dummy_depth,
                            str(rgb_path),
                            str(depth_path),
                            timestamp_ns,
                            str(self._episode_paths.timestamp_file),
                        )
                        try:
                            cv2.imwrite(str(legacy_headfl), color_image)
                            cv2.imwrite(str(legacy_headf), color_image)
                            cv2.imwrite(str(legacy_headfr), color_image)
                        except Exception as e:
                            logger.warning("奥比中光头部兼容目录写入失败：%s", e)
                    self._frame_count += 1

                    # 如果设置了目标帧数，到达后自动停止录制（预览仍继续）
                    if (
                        self._record_target_frames is not None
                        and self._frame_count >= self._record_target_frames
                    ):
                        logger.info(
                            "奥比中光头部达到目标帧数 %d（相机侧），停止该路写盘以确保帧数对齐",
                            self._record_target_frames,
                        )
                        self._recording = False
                        self._record_target_frames = None
                # 统计采集循环频率（用于诊断头部相机是否掉帧）
                loop_now = time.perf_counter()
                if self._last_loop_ts > 0:
                    dt = loop_now - self._last_loop_ts
                    if dt > 0:
                        self._loop_fps = 1.0 / dt
                self._last_loop_ts = loop_now

            except Exception as e:
                # 设备中断时抛出异常
                if "Frame didn't arrive" in str(e) or "timeout" in str(e).lower():
                    logger.debug("检测到设备连接中断: %s", e)
                    continue
                else:
                    logger.debug("采集错误: %s", e)
                    continue

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
        rgb_dir = images_dir / "headf_rgbd_color"
        depth_dir = images_dir / "headf_rgbd_depth"
        legacy_headfl_dir = images_dir / "headfl"
        legacy_headf_dir = images_dir / "headf"
        legacy_headfr_dir = images_dir / "headfr"

        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        legacy_headfl_dir.mkdir(parents=True, exist_ok=True)
        legacy_headf_dir.mkdir(parents=True, exist_ok=True)
        legacy_headfr_dir.mkdir(parents=True, exist_ok=True)

        timestamp_file = rgb_dir / "headf_rgbd_timestamps.txt"
        if timestamp_file.exists():
            timestamp_file.unlink()
        timestamp_file.touch()

        return EpisodePaths(
            episode_dir=episode_dir,
            rgb_dir=rgb_dir,
            depth_dir=depth_dir,
            timestamp_file=timestamp_file,
        )


def main() -> None:  # pragma: no cover - 用于手动调试
    """简单的 CLI：启动预览并持续记录到默认 repo。"""
    logging.basicConfig(level=logging.INFO)
    manager = OrbbecHeadCameraManager()
    if not manager.start():
        print(f"奥比中光头部不可用：{manager.last_error}")
        return

    repo_id = f"orbbec_demo_{int(time.time())}"
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
