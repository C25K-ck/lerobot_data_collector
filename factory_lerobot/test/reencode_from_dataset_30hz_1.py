#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import threading
import queue
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从本地数据集读取(color jpg + depth png)，在内存中解码/编码并写回磁盘，模拟30Hz保存线程"
    )
    p.add_argument(
        "--src",
        type=str,
        default="/home/dreame/data/hf_dataset",
        help="源数据集根目录（脚本会递归查找 images/headf_rgbd_color 与 images/headf_rgbd_depth）",
    )
    p.add_argument(
        "--dest",
        type=str,
        default="/home/dreame/data/hf_dataset/reencode_test",
        help="输出目录（将写入 color/ 与 depth/ 子目录）",
    )
    p.add_argument("--fps", type=float, default=30.0, help="目标帧率，默认30")
    p.add_argument("--frames", type=int, default=0, help="最多处理帧数（0表示按照素材长度循环一轮）")
    p.add_argument("--queue-size", type=int, default=32, help="保存队列大小，满则丢旧，默认32")
    p.add_argument("--jpeg-quality", type=int, default=90, help="重编码JPEG质量，默认90")
    p.add_argument("--png-compression", type=int, default=1, help="重编码PNG压缩等级(0-9)，默认1")
    return p.parse_args()


def find_pairs(src_root: Path):
    color_files = []
    depth_files = []
    # 递归查找包含 images/headf_rgbd_* 的目录
    for p in src_root.rglob("images"):
        cdir = p / "headf_rgbd_color"
        ddir = p / "headf_rgbd_depth"
        if cdir.is_dir() and ddir.is_dir():
            color_files.extend(sorted(cdir.glob("*.jpg")))
            depth_files.extend(sorted(ddir.glob("*.png")))

    # 以文件名数字部分配对（基于常见 0.jpg / 0.png 规则）
    def idx_of(path: Path):
        try:
            stem = path.stem
            return int(stem)
        except Exception:
            return None

    cmap = {idx_of(p): p for p in color_files}
    dmap = {idx_of(p): p for p in depth_files}
    common_keys = sorted([k for k in cmap.keys() if k is not None and k in dmap])
    pairs = [(cmap[k], dmap[k]) for k in common_keys]
    return pairs


class SaveWorker:
    def __init__(self, dest_dir: Path, maxsize: int, jpeg_quality: int, png_compression: int):
        self.dest_color = dest_dir / "color"
        self.dest_depth = dest_dir / "depth"
        self.dest_color.mkdir(parents=True, exist_ok=True)
        self.dest_depth.mkdir(parents=True, exist_ok=True)
        self.ts_file = self.dest_color / "timestamps.txt"
        if self.ts_file.exists():
            self.ts_file.unlink()

        self.queue = queue.Queue(maxsize=maxsize)
        self.jpeg_quality = int(jpeg_quality)
        self.png_compression = int(png_compression)
        self.drop_count = 0
        self.processed_count = 0
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, color_img: np.ndarray, depth_img: np.ndarray, idx: int, ts_ns: int):
        if not self.is_running:
            return
        try:
            self.queue.put_nowait((color_img, depth_img, idx, ts_ns))
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.drop_count += 1
                print(f"保存队列已满，丢弃1帧，累计丢弃: {self.drop_count}")
            except Exception:
                pass
            try:
                self.queue.put_nowait((color_img, depth_img, idx, ts_ns))
            except Exception:
                pass

    def _loop(self):
        while self.is_running:
            try:
                color_img, depth_img, idx, ts_ns = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                cpath = str(self.dest_color / f"{idx}.jpg")
                dpath = str(self.dest_depth / f"{idx}.png")
                cv2.imwrite(cpath, color_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                cv2.imwrite(dpath, depth_img, [int(cv2.IMWRITE_PNG_COMPRESSION), self.png_compression])
                with open(self.ts_file, 'a') as f:
                    f.write(f"{ts_ns}\n")
                self.processed_count += 1
            finally:
                self.queue.task_done()

    def stop(self):
        self.is_running = False
        self.thread.join(timeout=2.0)


def main():
    args = parse_args()
    src_root = Path(args.src)
    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(src_root)
    if not pairs:
        print(f"未找到可用的 color/depth 配对文件于: {src_root}")
        return
    total_pairs = len(pairs)
    print(f"找到配对帧数: {total_pairs}")

    save_worker = SaveWorker(dest_root, maxsize=args.queue_size, jpeg_quality=args.jpeg_quality, png_compression=args.png_compression)

    target_interval = 1.0 / max(1e-6, args.fps)
    t_start = time.time()
    submitted = 0
    last_report = time.time()
    last_submitted = 0
    last_processed = 0

    max_frames = args.frames if args.frames > 0 else total_pairs

    for i in range(max_frames):
        color_path, depth_path = pairs[i % total_pairs]

        # 读取与解码（模拟采集输出）
        color_img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color_img is None or depth_img is None:
            continue

        ts_ns = time.time_ns()
        save_worker.submit(color_img, depth_img, i, ts_ns)
        submitted += 1

        # 控制节拍到 ~fps（近似模拟30Hz采集）
        t_next = t_start + submitted * target_interval
        now = time.time()
        if t_next > now:
            time.sleep(t_next - now)

        # 精确Hz（每1秒窗口）
        now = time.time()
        if now - last_report >= 1.0:
            window = now - last_report
            sub_win = submitted - last_submitted
            proc_win = save_worker.processed_count - last_processed
            submit_hz = sub_win / window if window > 0 else 0.0
            save_hz = proc_win / window if window > 0 else 0.0
            print(f"窗口{window:.2f}s: 提交Hz={submit_hz:.2f}, 保存Hz={save_hz:.2f}, 累计提交={submitted}, 丢弃累计={save_worker.drop_count}")
            last_report = now
            last_submitted = submitted
            last_processed = save_worker.processed_count

    # 等待写入完成
    save_worker.queue.join()
    save_worker.stop()

    elapsed = time.time() - t_start
    submit_hz_total = submitted / elapsed if elapsed > 0 else 0.0
    save_hz_total = save_worker.processed_count / elapsed if elapsed > 0 else 0.0
    print(f"完成。用时={elapsed:.2f}s, 平均提交Hz={submit_hz_total:.2f}, 平均保存Hz={save_hz_total:.2f}, 提交帧={submitted}, 保存帧={save_worker.processed_count}, 丢弃累计={save_worker.drop_count}")


if __name__ == "__main__":
    main()


