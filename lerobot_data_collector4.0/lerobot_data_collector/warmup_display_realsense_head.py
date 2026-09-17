#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense 头部相机管理器（多进程）
================================

该模块在 warmup_display_realsense_right 的基础上，为头部相机提供专
用的数据落地路径（images/headf_xxx），并通过独立进程完成采集，减
少与主 GUI 线程的资源争用。主进程只需通过队列发送指令、接收 JPEG
 预览帧即可。
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from conf import REALSENSE_HEAD_SN  # type: ignore[import]
except Exception:  # pragma: no cover
    try:
        from .conf import REALSENSE_HEAD_SN  # type: ignore[import]
    except Exception:
        REALSENSE_HEAD_SN = ""

from .warmup_display_realsense_right import (
    EpisodePaths,
    RealSenseRightCameraManager,
    DEFAULT_DATASET_HOME,
)

logger = logging.getLogger(__name__)

HEAD_RGB_DIR = "headf_rgb"
HEAD_DEPTH_DIR = "headf_depth"
HEAD_TIMESTAMP_FILE = "headf_timestamps.txt"
PREVIEW_QUEUE_SIZE = 2


class RealSenseHeadCameraManager(RealSenseRightCameraManager):
    """继承右手管理器，仅调整序列号和落地目录。"""

    def __init__(
        self,
        serial_number: Optional[str] = None,
        dataset_home: Path | str | None = None,
        preview_fps: int = 15,
        save_queue_size: int = 4096,
    ) -> None:
        # 头部深度：默认开启（用户需要深度视频/深度帧）
        # 如遇设备不支持或想禁用，可设置环境变量 REALSENSE_HEAD_ENABLE_DEPTH=0
        import os
        _enable_depth = os.environ.get("REALSENSE_HEAD_ENABLE_DEPTH", "1").strip().lower() not in {"0", "false", "no", "off"}
        super().__init__(
            serial_number=serial_number or REALSENSE_HEAD_SN,
            dataset_home=dataset_home or DEFAULT_DATASET_HOME,
            preview_fps=preview_fps,
            save_queue_size=save_queue_size,
            enable_depth=_enable_depth,
        )
        self.log_prefix = "RealSense 头部"

    def _prepare_episode_dirs(self, dataset_root: Path) -> EpisodePaths:
        dataset_root.mkdir(parents=True, exist_ok=True)

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

        images_root = dataset_root / "chunk_000"
        episode_dir = images_root / f"episode_{next_idx:06d}"
        images_dir = episode_dir / "images"
        rgb_dir = images_dir / HEAD_RGB_DIR
        depth_dir = images_dir / HEAD_DEPTH_DIR

        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        timestamp_file = rgb_dir / HEAD_TIMESTAMP_FILE
        if timestamp_file.exists():
            timestamp_file.unlink()
        timestamp_file.touch()

        return EpisodePaths(
            episode_dir=episode_dir,
            rgb_dir=rgb_dir,
            depth_dir=depth_dir,
            timestamp_file=timestamp_file,
        )


def _encode_frame(frame: np.ndarray) -> Optional[bytes]:
    if cv2 is None:
        return None
    try:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )
        if not ok:
            return None
        return encoded.tobytes()
    except Exception:
        return None


def _head_camera_process(
    cmd_queue: "mp.queues.Queue[Dict[str, Any]]",
    preview_queue: "mp.queues.Queue[bytes]",
    status_queue: "mp.queues.Queue[Dict[str, Any]]",
    dataset_home: str,
    preview_fps: int,
    serial_number: Optional[str],
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    manager = RealSenseHeadCameraManager(
        serial_number=serial_number,
        dataset_home=dataset_home,
        preview_fps=preview_fps,
    )
    if not manager.start():
        status_queue.put(
            {"event": "ready", "success": False, "error": manager.last_error}
        )
        return
    status_queue.put({"event": "ready", "success": True})

    running = True
    preview_interval = 1.0 / max(1, preview_fps)
    last_preview_ts = 0.0

    while running:
        try:
            cmd = cmd_queue.get_nowait()
        except queue.Empty:
            cmd = None

        if cmd:
            action = cmd.get("action")
            try:
                if action == "shutdown":
                    running = False
                elif action == "set_dataset_home":
                    manager.set_dataset_home(Path(cmd["path"]))
                elif action == "start_recording":
                    manager.start_recording(
                        cmd["repo_id"], target_frames=cmd.get("target_frames")
                    )
                elif action == "stop_recording":
                    manager.stop_recording(wait=cmd.get("wait", True))
                elif action == "flush":
                    manager.save_worker.flush(timeout=cmd.get("timeout"))
            except Exception as exc:
                status_queue.put(
                    {"event": "error", "context": action, "error": str(exc)}
                )

        now = time.time()
        if now - last_preview_ts >= preview_interval:
            frame = manager.get_latest_preview()
            if frame is not None:
                data = _encode_frame(frame)
                if data is not None:
                    try:
                        preview_queue.put_nowait(data)
                    except queue.Full:
                        try:
                            preview_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            preview_queue.put_nowait(data)
                        except queue.Full:
                            pass
                last_preview_ts = now
        time.sleep(0.01)

    manager.shutdown()
    status_queue.put({"event": "stopped"})


class RealSenseHeadCameraProcess:
    """主进程侧的轻量控制器，负责与子进程通讯。"""

    def __init__(
        self,
        dataset_home: Path | str | None = None,
        preview_fps: int = 15,
        serial_number: Optional[str] = None,
    ) -> None:
        self.dataset_home = Path(dataset_home or DEFAULT_DATASET_HOME).expanduser()
        self.preview_fps = preview_fps
        self.serial_number = serial_number or REALSENSE_HEAD_SN

        self._ctx = mp.get_context("spawn")
        self._cmd_queue: mp.Queue = self._ctx.Queue()
        self._preview_queue: mp.Queue = self._ctx.Queue(maxsize=PREVIEW_QUEUE_SIZE)
        self._status_queue: mp.Queue = self._ctx.Queue()
        self._process: Optional[mp.Process] = None

        self._latest_preview_bytes: Optional[bytes] = None
        self._latest_preview: Optional[np.ndarray] = None
        self._preview_lock = threading.Lock()
        self.last_error: str = ""

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def start(self, timeout: float = 5.0) -> bool:
        if self._process and self._process.is_alive():
            return True

        self._process = self._ctx.Process(
            target=_head_camera_process,
            args=(
                self._cmd_queue,
                self._preview_queue,
                self._status_queue,
                str(self.dataset_home),
                self.preview_fps,
                self.serial_number,
            ),
            daemon=True,
        )
        self._process.start()
        return self._wait_until_ready(timeout=timeout)

    def _wait_until_ready(self, timeout: float) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            try:
                event = self._status_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if event.get("event") == "ready":
                if event.get("success"):
                    self.last_error = ""
                    return True
                self.last_error = event.get("error", "")
                self.shutdown(force=True)
                return False
            if event.get("event") == "error":
                self.last_error = event.get("error", "")
        self.last_error = "头部摄像头进程启动超时"
        self.shutdown(force=True)
        return False

    def shutdown(self, force: bool = False) -> None:
        if not self._process:
            return
        if self._process.is_alive() and not force:
            try:
                self._cmd_queue.put_nowait({"action": "shutdown"})
            except Exception:
                pass
            self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._process = None
        self._latest_preview_bytes = None
        self._latest_preview = None

    # ------------------------------------------------------------------ #
    # 业务接口
    # ------------------------------------------------------------------ #
    def set_dataset_home(self, path: Path | str) -> None:
        self.dataset_home = Path(path).expanduser()
        try:
            self._cmd_queue.put_nowait(
                {"action": "set_dataset_home", "path": str(self.dataset_home)}
            )
        except Exception:
            pass

    def start_recording(self, repo_id: str, target_frames: Optional[int] = None) -> None:
        if not repo_id:
            raise ValueError("repo_id 不能为空")
        if not self._process or not self._process.is_alive():
            started = self.start()
            if not started:
                raise RuntimeError(self.last_error or "头部摄像头未就绪")
        self._cmd_queue.put_nowait(
            {
                "action": "start_recording",
                "repo_id": repo_id,
                "target_frames": target_frames,
            }
        )

    def stop_recording(self, wait: bool = True) -> None:
        if not self._process or not self._process.is_alive():
            return
        try:
            self._cmd_queue.put_nowait({"action": "stop_recording", "wait": wait})
        except Exception:
            pass

    def get_latest_preview(self) -> Optional[np.ndarray]:
        self._drain_status_queue()
        updated = False
        while True:
            try:
                data = self._preview_queue.get_nowait()
                self._latest_preview_bytes = data
                updated = True
            except queue.Empty:
                break

        if not self._latest_preview_bytes:
            return None
        if updated:
            frame = self._decode_preview(self._latest_preview_bytes)
            if frame is not None:
                with self._preview_lock:
                    self._latest_preview = frame
        with self._preview_lock:
            if self._latest_preview is None:
                return None
            return self._latest_preview.copy()

    def _decode_preview(self, data: bytes) -> Optional[np.ndarray]:
        if cv2 is None:
            return None
        try:
            np_data = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            if image is None:
                return None
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return rgb
        except Exception:
            return None

    def _drain_status_queue(self) -> None:
        while True:
            try:
                event = self._status_queue.get_nowait()
            except queue.Empty:
                break
            if event.get("event") == "error":
                self.last_error = event.get("error", "")


def main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="头部 RealSense 预览调试工具")
    parser.add_argument("--repo", default="head_demo", help="测试数据集 ID")
    parser.add_argument("--frames", type=int, default=600, help="录制帧数")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    client = RealSenseHeadCameraProcess()
    if not client.start():
        print(f"头部相机不可用：{client.last_error}")
        return

    client.start_recording(args.repo, target_frames=args.frames)
    print("正在采集，按 Ctrl+C 结束...")
    try:
        while True:
            frame = client.get_latest_preview()
            if frame is not None:
                print(f"\r预览帧: {frame.shape}", end="")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop_recording()
        client.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()

