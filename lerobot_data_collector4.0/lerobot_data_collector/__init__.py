"""
LeRobot 通用数据采集平台
独立的数据采集软件，支持从ROS2/LCM等数据源采集数据并转换为LeRobot格式
"""

__version__ = "1.0.0"

# 说明：
# ----
# 以前这里会在 import lerobot_data_collector 时立刻导入 `.collector`，从而连带
# 导入 Tkinter/PIL 等 GUI 依赖，导致 Qt 启动也变慢。
# 现在改为“按需导入”（lazy import），只有真的访问这些符号时才加载对应模块。

__all__ = ["Collector", "CollectorConfig", "DataHub", "ROS2Adapter", "lerobotUnit"]


def __getattr__(name: str):
    if name in {"Collector", "CollectorConfig", "DataHub", "ROS2Adapter"}:
        from .collector_core import Collector, CollectorConfig, DataHub, ROS2Adapter

        return {
            "Collector": Collector,
            "CollectorConfig": CollectorConfig,
            "DataHub": DataHub,
            "ROS2Adapter": ROS2Adapter,
        }[name]

    if name == "lerobotUnit":
        from .lerobot_unit import lerobotUnit

        return lerobotUnit

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

