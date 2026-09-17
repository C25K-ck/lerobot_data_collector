#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot 通用数据采集平台 - 主入口
"""

import sys
import os
import logging

# 添加包路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _configure_logging() -> None:
    """根据环境变量配置日志等级，默认 INFO，便于查看相机相关日志。"""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _main() -> None:
    _configure_logging()
    # 为优化启动速度：优先走 Qt 启动路径，避免导入包含 Tk GUI 的 `collector.py`
    # （Tkinter/PIL 在 import 阶段开销很大，会拖慢 Qt 启动）。
    try:
        from lerobot_data_collector.qt_gui import main_qt
        from lerobot_data_collector.collector_core import Collector, CollectorConfig

        main_qt(Collector, CollectorConfig)
    except Exception:
        # 兼容旧逻辑：如果 Qt 不可用，再走原入口（可能回退 Tk）
        from lerobot_data_collector.collector import main as collector_main

        collector_main()


if __name__ == "__main__":
    _main()

