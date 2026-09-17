import logging
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from .collector_core import Collector, CollectorConfig
from .server_api import (
    api_login,
    api_get_my_tasks,
    api_operate_task,
    api_batch_upload_data_info,
    upload_path_to_minio,
    api_get_session_token,
    ApiError,
    LoginResult
)
from .dataset_manager import scan_datasets, convert_parquet_to_hdf5
from .qt_login import USER_TOKEN, SERVER_HOST

logger = logging.getLogger(__name__)

class LeRobotSDK:
    """
    LeRobot 数据采集平台 SDK
    封装了登录、任务获取、数据采集、格式转换及上传等核心功能。
    """
    def __init__(self, server_host: str = None):
        self.server_host = server_host or "http://localhost:8888"
        self.token: Optional[str] = None
        self.user_info: Optional[dict] = None
        self.collector: Optional[Collector] = None

    def login(self, username: str, password: str) -> bool:
        """
        登录平台
        """
        try:
            result: LoginResult = api_login(self.server_host, username, password)
            if result.code == 200:
                self.token = result.token
                self.user_info = result.user_info
                # 设置全局变量以供其他 API 调用
                import lerobot_data_collector.qt_login as qt_login
                qt_login.USER_TOKEN = self.token
                qt_login.SERVER_HOST = self.server_host
                logger.info(f"登录成功: {username}")
                return True
            else:
                logger.error(f"登录失败: {result.message}")
                return False
        except Exception as e:
            logger.exception(f"登录异常: {e}")
            return False

    def get_tasks(self, status: int = None) -> List[Dict[str, Any]]:
        """
        获取当前用户的任务列表
        :param status: 任务状态过滤
        """
        try:
            data = api_get_my_tasks(collection_status=status)
            return data.get("data", {}).get("list", [])
        except ApiError as e:
            logger.error(f"获取任务失败: {e}")
            return []

    def setup_collector(self, config_path: str) -> Optional[Collector]:
        """
        根据配置文件初始化采集器
        """
        import json
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
            cfg = CollectorConfig.from_dict(cfg_dict)
            self.collector = Collector(cfg)
            return self.collector
        except Exception as e:
            logger.exception(f"初始化采集器失败: {e}")
            return None

    def start_collection(self, target_frames: int = 1800):
        """
        开始一次采集会话
        """
        if not self.collector:
            raise RuntimeError("请先调用 setup_collector 初始化采集器")
        
        self.collector.start()
        # 立即开始会话
        self.collector.start_session_immediate(target_frames)
        logger.info(f"采集开始，目标帧数: {target_frames}")

    def stop_collection(self, save: bool = True):
        """
        停止当前采集会话
        """
        if self.collector:
            if save:
                self.collector.stop_and_save()
                logger.info("采集停止并保存")
            else:
                self.collector.stop()
                logger.info("采集停止（未保存）")
            self.collector = None

    def convert_to_hdf5(self, repo_id: str, dataset_root: str = None) -> bool:
        """
        将 Parquet 数据集转换为 HDF5 格式
        """
        try:
            return convert_parquet_to_hdf5(repo_id, dataset_root)
        except Exception as e:
            logger.error(f"转换 HDF5 失败: {e}")
            return False

    def upload_dataset(self, repo_id: str, task_id: int):
        """
        上传数据集到云端平台（MinIO + API 上报）
        """
        # 注意：这里需要实现复杂的上传逻辑，包括获取 STS、扫描文件、逐个上传以及最后的 API 确认。
        # 建议参考 qt_collect.py 中的 _upload_selected_dataset_info 实现。
        logger.warning("upload_dataset 正在开发中，请参考 GUI 实现逻辑。")
        pass

# 使用示例:
# sdk = LeRobotSDK("http://10.204.5.111:9006")
# if sdk.login("user", "pass"):
#     tasks = sdk.get_tasks()
#     sdk.setup_collector("configs/g1.json")
#     sdk.start_collection(target_frames=500)
#     # ... 等待采集完成 ...
#     sdk.stop_collection()

