#!/usr/bin/env python3

import os
import json
from pathlib import Path
import signal
import sys
import time
from typing import Tuple

import numpy as np
import cv2

try:
    import redis  # type: ignore
    import pika  # type: ignore
except Exception as e:
    print(f"缺少依赖: {e}. 请先安装: pip install redis pika")
    sys.exit(1)


def _ensure_dirs(save_dir: str, session_id: str) -> Tuple[Path, Path, Path]:
    session_dir = Path(save_dir) / f"session_{session_id}"
    rgb_dir = session_dir / "rgb"
    depth_dir = session_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    timestamp_path = session_dir / "timestamps.txt"
    if not timestamp_path.exists():
        try:
            timestamp_path.touch()
        except Exception:
            pass
    return session_dir, rgb_dir, depth_dir, timestamp_path


def _msg_to_arrays(r: "redis.Redis", msg: dict) -> Tuple[np.ndarray, np.ndarray]:
    # 拉取字节
    color_meta = msg["color"]
    depth_meta = msg["depth"]

    color_bytes = r.get(color_meta["redis_key"])  # type: ignore
    depth_bytes = r.get(depth_meta["redis_key"])  # type: ignore
    if color_bytes is None or depth_bytes is None:
        raise RuntimeError("Redis 中找不到对应图像数据 (可能过期或被清理)")

    # 反序列化为 numpy
    color_shape = tuple(color_meta["shape"])  # e.g. (720,1280,3)
    depth_shape = tuple(depth_meta["shape"])  # e.g. (720,1280)
    color_dtype = np.dtype(color_meta["dtype"])  # e.g. uint8
    depth_dtype = np.dtype(depth_meta["dtype"])  # e.g. uint16

    color_np = np.frombuffer(color_bytes, dtype=color_dtype).reshape(color_shape)
    depth_np = np.frombuffer(depth_bytes, dtype=depth_dtype).reshape(depth_shape)
    return color_np, depth_np


def main() -> None:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F")
    queue_name = os.getenv("IMAGE_SAVE_QUEUE", "image_saver_right")
    save_dir = os.getenv("SAVE_DIR", "/tmp/realsense_right_images")
    drop_on_missing = os.getenv("DROP_ON_MISSING", "1").lower() in ("1", "true", "yes")
    # 固化日志：禁用详细日志，仅每30帧打印一次累计帧数
    log_detail = False
    log_every_n = 30
    suppress_dup_warn = True
    try:
        processed_ttl_seconds = int(os.getenv("PROCESSED_TTL_SECONDS", "3600"))
    except Exception:
        processed_ttl_seconds = 3600

    r = redis.from_url(redis_url)

    params = pika.URLParameters(rabbitmq_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)
    try:
        prefetch_count = int(os.getenv("PREFETCH_COUNT", "4"))
    except Exception:
        prefetch_count = 4
    if prefetch_count < 1:
        prefetch_count = 1
    channel.basic_qos(prefetch_count=prefetch_count)

    stop_flag = {"stop": False}
    saved_count = {"n": 0}
    log_heartbeat = False
    last_beat = {"t": time.time()}

    def _sigterm(signum, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigterm)
    signal.signal(signal.SIGTERM, _sigterm)

    def _callback(ch, method, properties, body):
        try:
            t0 = time.perf_counter()
            msg = json.loads(body.decode("utf-8"))
            session_id = str(msg["session_id"])  # 与生产端一致
            frame_index = int(msg["frame_index"])  # 文件名用 index
            frame_ts_ns = int(msg.get("frame_ts_ns", msg.get("timestamp_ns", 0)))  # 兼容旧字段
            publish_ts_ns = int(msg.get("publish_ts_ns", frame_ts_ns))
            if log_detail:
                print(f"[worker-right] recv frame={frame_index}", flush=True)

            _, rgb_dir, depth_dir, timestamp_path = _ensure_dirs(save_dir, session_id)

            # 幂等：若已处理过（标记存在）或文件已存在，则直接 ACK 跳过
            processed_key = f"processed:{session_id}:{frame_index}"
            if r.exists(processed_key):  # type: ignore
                if not suppress_dup_warn:
                    print(f"worker: 已处理过 frame={frame_index}, 直接 ACK")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            existing = (rgb_dir / f"{frame_index}.jpg").exists() and (depth_dir / f"{frame_index}.png").exists()
            if existing:
                try:
                    r.setex(processed_key, processed_ttl_seconds, b"1")  # type: ignore
                except Exception:
                    pass
                if not suppress_dup_warn:
                    print(f"worker: 文件已存在 frame={frame_index}, 直接 ACK")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            # 从 Redis 拉取编码字节
            t_get0 = time.perf_counter()
            color_key = msg["color"]["redis_key"]
            depth_key = msg["depth"]["redis_key"]
            color_bytes = r.get(color_key)  # type: ignore
            depth_bytes = r.get(depth_key)  # type: ignore
            if color_bytes is None or depth_bytes is None:
                raise RuntimeError("Redis 中找不到对应图像数据 (可能过期或被清理)")
            t_get1 = time.perf_counter()
            if log_detail:
                print(f"[worker-right] got redis bytes frame={frame_index}", flush=True)

            # 直接将编码字节写盘
            rgb_path = rgb_dir / f"{frame_index}.jpg"
            depth_path = depth_dir / f"{frame_index}.png"
            t_w1_0 = time.perf_counter()
            with open(rgb_path, "wb") as f:
                f.write(color_bytes)
            t_w1_1 = time.perf_counter()
            if log_detail:
                print(f"[worker-right] wrote rgb frame={frame_index}", flush=True)
            t_w2_0 = time.perf_counter()
            with open(depth_path, "wb") as f:
                f.write(depth_bytes)
            t_w2_1 = time.perf_counter()
            if log_detail:
                print(f"[worker-right] wrote depth frame={frame_index}", flush=True)
            if not (ok1 and ok2):
                raise RuntimeError("cv2.imwrite 失败")

            with open(timestamp_path, "a") as f:
                f.write(f"{frame_ts_ns}\n")

            # 先 ACK 再删除 Redis 键，避免删除后崩溃导致消息重投取不到数据
            t_ack0 = time.perf_counter()
            ch.basic_ack(delivery_tag=method.delivery_tag)
            t_ack1 = time.perf_counter()
            if log_detail:
                print(f"[worker-right] acked frame={frame_index}", flush=True)

            # 删除 Redis 中已消费的帧，释放内存（优先使用 UNLINK 异步删除）
            try:
                color_key = msg["color"]["redis_key"]
                depth_key = msg["depth"]["redis_key"]
                t_unlink0 = time.perf_counter()
                try:
                    # Redis 4.0+ 支持 UNLINK
                    r.unlink(color_key, depth_key)  # type: ignore[attr-defined]
                except AttributeError:
                    r.delete(color_key, depth_key)
                t_unlink1 = time.perf_counter()
            except Exception as del_e:
                print(f"清理 Redis 键失败: {del_e}")
                t_unlink0 = t_unlink1 = time.perf_counter()

            t1 = time.perf_counter()

            # 队列等待时延（ms）
            try:
                q_latency_ms = (time.time_ns() - publish_ts_ns) / 1e6
            except Exception:
                q_latency_ms = -1.0

            # 分段耗时（ms）
            get_decode_ms = (t_get1 - t_get0) * 1000.0
            write_rgb_ms = (t_w1_1 - t_w1_0) * 1000.0
            write_depth_ms = (t_w2_1 - t_w2_0) * 1000.0
            ack_ms = (t_ack1 - t_ack0) * 1000.0
            unlink_ms = (t_unlink1 - t_unlink0) * 1000.0
            total_ms = (t1 - t0) * 1000.0

            if log_detail or (frame_index % log_every_n == 0):
                if log_detail:
                    print(
                        f"[worker-right] frame={frame_index} q_ms={q_latency_ms:.1f} "
                        f"get_dec_ms={get_decode_ms:.1f} rgb_ms={write_rgb_ms:.1f} depth_ms={write_depth_ms:.1f} "
                        f"ack_ms={ack_ms:.1f} unlink_ms={unlink_ms:.1f} total_ms={total_ms:.1f}",
                        flush=True,
                    )

            try:
                r.setex(processed_key, processed_ttl_seconds, b"1")  # type: ignore
            except Exception:
                pass

            saved_count["n"] += 1
            if saved_count["n"] == 1 or saved_count["n"] % 30 == 0:
                print(f"已保存 {saved_count['n']} 帧", flush=True)

        except Exception as e:
            # 如果键缺失，可能是此前已写盘但消息因异常被重投；若目标文件已存在则直接 ACK 丢弃
            try:
                msg = json.loads(body.decode("utf-8"))
                session_id = str(msg.get("session_id", ""))
                frame_index = int(msg.get("frame_index", -1))
                _, rgb_dir, depth_dir, _ = _ensure_dirs(save_dir, session_id) if session_id else (None, None, None, None)
                rgb_path = (rgb_dir / f"{frame_index}.jpg") if (rgb_dir is not None and frame_index >= 0) else None
                depth_path = (depth_dir / f"{frame_index}.png") if (depth_dir is not None and frame_index >= 0) else None
            except Exception:
                rgb_path = None
                depth_path = None

            if (rgb_path and rgb_path.exists()) or (depth_path and depth_path.exists()):
                if not suppress_dup_warn:
                    print("worker 警告: Redis 键缺失但文件已存在，直接 ACK 跳过")
                ch.basic_ack(delivery_tag=method.delivery_tag)
            else:
                print(f"worker 处理失败: {e}")
                if drop_on_missing:
                    # 丢弃该消息，避免死循环重投
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                else:
                    # 可选：重投一次（可能导致短暂重试风暴）
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    channel.basic_consume(queue=queue_name, on_message_callback=_callback, auto_ack=False)
    print(f"image_saver_worker 启动，监听队列: {queue_name}", flush=True)

    while not stop_flag["stop"]:
        try:
            connection.process_data_events(time_limit=1.0)
        except pika.exceptions.AMQPError as e:
            print(f"RabbitMQ 异常: {e}")
            time.sleep(1.0)

        if log_heartbeat and (time.time() - last_beat["t"]) >= 5.0:
            print(f"[worker-right] heartbeat saved={saved_count['n']} queue={queue_name}", flush=True)
            last_beat["t"] = time.time()

    try:
        channel.close()
    finally:
        connection.close()


if __name__ == "__main__":
    main()


