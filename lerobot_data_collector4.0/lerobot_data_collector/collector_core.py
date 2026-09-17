import json
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 注入 lerobot_factory 到 sys.path，保证可导入顶层包 "lerobot"
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _proj_root = os.path.dirname(_this_dir)       # lerobot_data_collector
    _proj_parent = os.path.dirname(_proj_root)    # 上一级目录

    possible_paths = [
        os.path.join(_proj_root, "lerobot_factory"),
        os.path.join(_proj_parent, "factory_lerobot", "lerobot_factory"),
        os.path.join(_proj_parent, "lerobot_factory"),
        os.environ.get("LEROBOT_FACTORY_PATH", None),
        os.path.expanduser("~/factory_lerobot/lerobot_factory"),
    ]

    _lerobot_factory = None
    for path in possible_paths:
        if path and os.path.exists(path):
            _lerobot_factory = path
            break

    if _lerobot_factory:
        if _lerobot_factory not in sys.path:
            sys.path.insert(0, _lerobot_factory)
        logger.info("lerobot_factory 路径：%s", _lerobot_factory)
    else:
        logger.warning("未找到 lerobot_factory，请检查目录或设置环境变量 LEROBOT_FACTORY_PATH")
except Exception as e:  # pragma: no cover - 防御式日志
    logger.warning("无法添加 lerobot_factory 到 sys.path：%s", e)

# 导入本地模块（支持包运行与脚本直跑）
try:
    from .lerobot_unit import lerobotUnit
    from .lcm_unit import lcmUnit
    from .gen1_lcm_adapter import Gen1LCMAdapter
    from .pico_controller import PicoControllerAdapter, create_pico_control_handler
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from lerobot_unit import lerobotUnit  # type: ignore
    try:
        from lcm_unit import lcmUnit  # type: ignore
    except ImportError:
        lcmUnit = None
        logger.warning("LCM 适配器未实现，将仅使用 ROS2 数据源")

    try:
        from gen1_lcm_adapter import Gen1LCMAdapter  # type: ignore
    except ImportError:
        Gen1LCMAdapter = None
        logger.warning("Gen1 LCM 适配器未实现")

    try:
        from pico_controller import PicoControllerAdapter, create_pico_control_handler  # type: ignore
    except ImportError:
        PicoControllerAdapter = None
        create_pico_control_handler = None
        logger.warning("Pico控制器适配器未实现")


@dataclass
class TopicSpec:
    topic: str
    type: str
    target: str  # 例如: "hands.action"、"hands.state"、"legs.state_q"、"task_mode"
    field: str = "data"  # 对于JointState类型，指定提取哪个字段："position", "velocity", "effort"


class DataHub:
    """兼容旧 API 导出名；当前实现使用 saveInfo / dataFlag 汇聚数据。"""

    pass


@dataclass
class DatasetFieldMapping:
    """数据集字段映射配置：定义数据集字段路径和对应的数据源"""

    dataset_path: str  # 数据集中的字段路径，如 "observation.hands.state"
    source_key: str  # DataHub中的数据源key，如 "hands.state"
    dimension: int  # 数据维度


@dataclass
class CollectorConfig:
    repo_id: str
    fps: int = 30
    hand_dim: int = 30
    leg_dim: int = 12
    ros2_enabled: bool = True
    ros2_topics: List[TopicSpec] | None = None
    lcm_enabled: bool = False
    robot_type: str = "generic"  # 支持: "generic", "gen1"
    pico_enabled: bool = False
    pico_ip: str = "192.168.12.110"
    pico_port: int = 12345
    pico_kinematics_enabled: bool = False
    pico_urdf_left: str | None = None
    pico_urdf_right: str | None = None
    pico_teleop_bridge: bool = False
    pico_only_collection: bool = False
    robot_prep_on_start: bool = False
    dataset_fields: Dict[str, DatasetFieldMapping] | None = None
    dataset_root: str | None = None
    # 与 factory_lerobot 一致：订阅 /multi_camera/sync_img，camera_flag ∧ action_flag 门控
    sync_camera_enabled: bool = False
    task_description: str = "generic"

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "CollectorConfig":
        ros2 = cfg.get("ros2", {}) or {}
        topics: List[TopicSpec] = []
        for t in ros2.get("topics", []) or []:
            field = t.get("field", "data")
            topics.append(
                TopicSpec(
                    topic=t["topic"],
                    type=t["type"],
                    target=t["target"],
                    field=field,
                )
            )

        dataset_fields: Optional[Dict[str, DatasetFieldMapping]] = None
        if cfg.get("dataset_fields"):
            dataset_fields = {}
            default_hand_dim = int(cfg.get("hand_dim", 30))
            default_leg_dim = int(cfg.get("leg_dim", 12))
            for dataset_path, field_cfg in cfg["dataset_fields"].items():
                if isinstance(field_cfg, str):
                    source_key = field_cfg
                    if "hands" in source_key or "hands" in dataset_path:
                        dimension = default_hand_dim
                    elif "legs" in source_key or "legs" in dataset_path:
                        dimension = default_leg_dim
                    else:
                        dimension = default_hand_dim
                    dataset_fields[dataset_path] = DatasetFieldMapping(
                        dataset_path=dataset_path,
                        source_key=source_key,
                        dimension=dimension,
                    )
                elif isinstance(field_cfg, dict):
                    source_key = field_cfg.get("source", dataset_path.split(".")[-1])
                    if "dim" in field_cfg:
                        dimension = int(field_cfg["dim"])
                    elif "hands" in source_key or "hands" in dataset_path:
                        dimension = default_hand_dim
                    elif "legs" in source_key or "legs" in dataset_path:
                        dimension = default_leg_dim
                    else:
                        dimension = default_hand_dim
                    dataset_fields[dataset_path] = DatasetFieldMapping(
                        dataset_path=dataset_path,
                        source_key=source_key,
                        dimension=dimension,
                    )

        dataset_root = cfg.get("dataset_root")
        if dataset_root:
            dataset_root = str(dataset_root)

        ros2_block = cfg.get("ros2") or {}
        ros2_enabled = cfg.get("ros2_enabled")
        if ros2_enabled is None:
            ros2_enabled = ros2_block.get("enabled", True)

        sync_cam = cfg.get("sync_camera_enabled")
        if sync_cam is None:
            sync_cam = ros2_block.get("sync_camera_enabled", False)

        return CollectorConfig(
            repo_id=cfg["repo_id"],
            fps=int(cfg.get("fps", 30)),
            hand_dim=int(cfg.get("hand_dim", 30)),
            leg_dim=int(cfg.get("leg_dim", 12)),
            ros2_enabled=bool(ros2_enabled),
            ros2_topics=topics,
            lcm_enabled=bool(cfg.get("lcm_enabled", False)),
            robot_type=str(cfg.get("robot_type", "generic")),
            pico_enabled=bool(cfg.get("pico_enabled", False)),
            pico_ip=str(cfg.get("pico_ip", "192.168.12.110")),
            pico_port=int(cfg.get("pico_port", 12345)),
            pico_kinematics_enabled=bool(cfg.get("pico_kinematics_enabled", False)),
            pico_urdf_left=cfg.get("pico_urdf_left"),
            pico_urdf_right=cfg.get("pico_urdf_right"),
            pico_teleop_bridge=bool(cfg.get("pico_teleop_bridge", False)),
            pico_only_collection=bool(cfg.get("pico_only_collection", False)),
            robot_prep_on_start=bool(cfg.get("robot_prep_on_start", False)),
            dataset_fields=dataset_fields,
            dataset_root=dataset_root,
            sync_camera_enabled=bool(sync_cam),
            task_description=str(cfg.get("task_description", "generic")),
        )


# ============ factory_lerobot 方案：核心数据结构 ============

class saveInfo:
    """按 factory_lerobot 方案：存储所有数据和时间戳"""
    def __init__(self) -> None:
        # 关节数据
        self.qpos = []  # action 数据
        self.hands_state = []
        self.hands_vel = []
        self.hands_torque = []
        self.hands_current = []
        self.legs_state_q = []
        self.legs_state_qd = []
        self.legs_state_tau = []
        self.legs_cmd_q = []
        self.legs_cmd_qd = []
        self.legs_cmd_tau = []
        self.task_mode = 0
        # 与 factory ros_node_save_data_orbbec.saveInfo 对齐
        self.cmd_vel = np.zeros(6, dtype=np.float32)
        self.squat_des = np.array([0.0, 0.92], dtype=np.float32)

        # 相机数据（按 factory_lerobot 方案）
        self.image_head_list = [None, None, None]  # 左、中、右头部相机
        self.image_list = [None, None, None, None, None, None, None]  # 手部相机等

        # 时间戳存储（与 factory image_timestamps_list：三路头部分别可写同一纳秒）
        timestampes = int(time.time() * 1e9)
        self.timestamp_list = [timestampes]
        self.image_timestamps_list = [timestampes, timestampes, timestampes]


class dataFlag:
    """按 factory_lerobot 方案：数据就绪标志"""
    def __init__(self) -> None:
        self.action_flag = False  # hands.action 就绪标志
        self.hands_state_flag = False
        self.legs_state_q_flag = False
        self.camera_flag = False  # 相机触发标志（按 factory_lerobot 方案）
        self.camera_available = False  # 相机是否可用
        self.all_data_ready = False


class saveSate:
    """按 factory_lerobot 方案：采集状态管理"""
    def __init__(self) -> None:
        self.ready_to_save = False
        self.start_get_data = False
        self.saving = False


class ROS2Adapter:
    """基于 rclpy 的适配器，支持 Float32MultiArray、Int32、String 和 JointState。"""

    def __init__(
        self,
        topics: List[TopicSpec],
        on_update: Callable[[str, Any], None],
        on_control: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.topics = topics
        self.on_update = on_update
        self.on_control = on_control
        self._node = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._first_received: Dict[str, bool] = {}

    def start(self) -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray, Int32, String

        class _Node(Node):
            pass

        def _run() -> None:
            try:
                rclpy.init(args=None)
                logger.debug("ROS2 初始化成功")
            except RuntimeError as e:
                msg = str(e).lower()
                if "already initialized" in msg or "must only be called once" in msg:
                    logger.debug("ROS2 已初始化，跳过重复初始化")
                    try:
                        context = rclpy.get_default_context()
                        if not context.ok():
                            logger.warning("ROS2 上下文已关闭，但无法重新初始化。可能需要重启程序。")
                    except Exception:
                        pass
                else:
                    raise

            import uuid

            node_name = f"generic_collector_{uuid.uuid4().hex[:8]}"
            try:
                self._node = _Node(node_name)
            except Exception as e:
                logger.error("创建 ROS2 节点失败：%s", e)
                raise

            for spec in self.topics:
                if spec.type == "Float32MultiArray":
                    target = spec.target
                    self._node.create_subscription(
                        Float32MultiArray,
                        spec.topic,
                        lambda msg, t=target: self._cb_float_array(msg, t),
                        10,
                    )
                    logger.info("ROS2 订阅 %s -> %s", spec.topic, spec.target)
                elif spec.type == "JointState":
                    try:
                        from sensor_msgs.msg import JointState

                        target = spec.target
                        field = spec.field
                        self._node.create_subscription(
                            JointState,
                            spec.topic,
                            lambda msg, t=target, f=field: self._cb_joint_state(msg, t, f),
                            10,
                        )
                        logger.info("ROS2 订阅 %s -> %s (字段 %s)", spec.topic, spec.target, field)
                    except ImportError:
                        logger.warning("无法导入 sensor_msgs，请安装依赖 pip install sensor-msgs")
                elif spec.type == "Int32":
                    target = spec.target
                    self._node.create_subscription(
                        Int32,
                        spec.topic,
                        lambda msg, t=target: self._cb_int32(msg, t),
                        10,
                    )
                    logger.info("ROS2 订阅 %s -> %s", spec.topic, spec.target)
                elif spec.type == "String":
                    self._node.create_subscription(String, spec.topic, self._cb_string, 10)
                    logger.info("ROS2 订阅 %s -> control", spec.topic)
                elif spec.type == "Twist":
                    from geometry_msgs.msg import Twist

                    target = spec.target
                    self._node.create_subscription(
                        Twist,
                        spec.topic,
                        lambda msg, t=target: self._cb_twist(msg, t),
                        10,
                    )
                    logger.info("ROS2 订阅 %s -> %s (Twist)", spec.topic, spec.target)
                else:
                    logger.warning("未支持的 ROS2 类型 %s (topic %s)", spec.type, spec.topic)

            logger.info("ROS2 适配器已启动，共订阅 %d 个话题", len(self.topics))
            from rclpy.executors import SingleThreadedExecutor

            executor = SingleThreadedExecutor()
            executor.add_node(self._node)
            self._running = True
            try:
                while self._running and rclpy.ok():
                    executor.spin_once(timeout_sec=0.1)
            finally:
                self._running = False
                try:
                    executor.remove_node(self._node)
                except Exception as e:
                    logger.debug("从执行器移除节点时出错：%s", e)
                try:
                    if self._node is not None:
                        self._node.destroy_node()
                        self._node = None
                except Exception as e:
                    logger.debug("销毁 ROS2 节点时出错：%s", e)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("ROS2 适配器线程未能及时停止")
        try:
            if self._node is not None:
                self._node.destroy_node()
                self._node = None
        except Exception as e:
            logger.debug("销毁 ROS2 节点时出错：%s", e)

    def _cb_float_array(self, msg: Any, target: str) -> None:
        try:
            arr = np.array(msg.data, dtype=np.float32)
            if len(arr) > 0:
                self.on_update(target, arr)
                if "action" in target and len(arr) > 0:
                    logger.debug("ROS2 收到 %s: %s... (len=%d)", target, arr[:3], len(arr))
        except Exception as e:
            logger.exception("ROS2 Float32MultiArray 回调错误：%s", e)

    def _cb_int32(self, msg: Any, target: str) -> None:
        try:
            val = int(msg.data)
            self.on_update(target, np.array([val], dtype=np.float32))
            if "task_mode" in target:
                logger.debug("ROS2 收到 %s: %s", target, val)
        except Exception as e:
            logger.exception("ROS2 Int32 回调错误：%s", e)

    def _cb_string(self, msg: Any) -> None:
        try:
            if self.on_control is not None:
                self.on_control(str(msg.data))
        except Exception:
            pass

    def _cb_twist(self, msg: Any, target: str) -> None:
        try:
            linear = msg.linear
            angular = msg.angular
            arr = np.array(
                [linear.x, linear.y, linear.z, angular.x, angular.y, angular.z],
                dtype=np.float32,
            )
            self.on_update(target, arr)
        except Exception as e:
            logger.exception("ROS2 Twist 回调错误：%s", e)

    def _cb_joint_state(self, msg: Any, target: str, field: str) -> None:
        try:
            if field == "position":
                arr = np.array(msg.position, dtype=np.float32) if msg.position else np.array([], dtype=np.float32)
            elif field == "velocity":
                arr = np.array(msg.velocity, dtype=np.float32) if msg.velocity else np.array([], dtype=np.float32)
            elif field == "effort":
                arr = np.array(msg.effort, dtype=np.float32) if msg.effort else np.array([], dtype=np.float32)
            else:
                logger.warning("JointState 字段 %s 不支持，仅支持 position/velocity/effort", field)
                return

            if len(arr) > 0:
                self.on_update(target, arr)
                if target not in self._first_received:
                    self._first_received[target] = True
                    logger.info("JointState 话题已收到数据：%s (字段 %s, len=%d)", target, field, len(arr))
            else:
                logger.debug("JointState 消息字段 %s 为空", field)
        except Exception as e:
            logger.exception("ROS2 JointState 回调错误：%s", e)


class LCMAdapter:
    """从 LCM 获取机器人本体状态/指令并填充 DataHub。"""

    def __init__(self, on_update: Callable[[str, Any], None]) -> None:
        self.on_update = on_update
        self._lcm = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._init_done = False

    def start(self) -> None:
        if lcmUnit is None:
            logger.warning("LCM 适配器未实现，无法启动 LCM 数据源")
            logger.info("如果仅使用 ROS2，请在配置文件中设置 'lcm_enabled: false'")
            self._running = False
            self._init_done = False
            return

        try:
            self._lcm = lcmUnit()
            timeout = 3.0
            start_time = time.time()
            while not self._lcm.update_estimator_once:
                if time.time() - start_time > timeout:
                    logger.warning("LCM 适配器初始化超时，跳过 LCM 数据源")
                    self._running = False
                    return
                time.sleep(0.1)
            self._init_done = True
            logger.info("LCM 适配器初始化完成")
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        except Exception as e:
            logger.warning("LCM 适配器启动失败：%s", e)
            logger.info("如果仅使用 ROS2，请在配置文件中设置 'lcm_enabled: false'")
            self._running = False
            self._init_done = False

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        period = 1.0 / 100.0
        update_count = 0
        while self._running:
            try:
                rs = self._lcm.current_robot_state
                rl = self._lcm.current_robot_leg_state
                if rs is not None:
                    if getattr(rs, "q", None) is not None and len(rs.q) > 0:
                        self.on_update("hands.state", np.array(rs.q, dtype=np.float32))
                    if update_count % 1000 == 0:
                        logger.debug("LCM hands.state: %s... (len=%d)", rs.q[:3], len(rs.q))
                    if getattr(rs, "dot_q", None) is not None and len(rs.dot_q) > 0:
                        self.on_update("hands.vel", np.array(rs.dot_q, dtype=np.float32))
                    if getattr(rs, "torque", None) is not None and len(rs.torque) > 0:
                        self.on_update("hands.torque", np.array(rs.torque, dtype=np.float32))
                    if getattr(rs, "current", None) is not None and len(rs.current) > 0:
                        self.on_update("hands.current", np.array(rs.current, dtype=np.float32))

                if rl is not None:
                    if getattr(rl, "state_q", None) is not None and len(rl.state_q) > 0:
                        self.on_update("legs.state_q", np.array(rl.state_q, dtype=np.float32))
                    if getattr(rl, "state_qd", None) is not None and len(rl.state_qd) > 0:
                        self.on_update("legs.state_qd", np.array(rl.state_qd, dtype=np.float32))
                    if getattr(rl, "state_tau", None) is not None and len(rl.state_tau) > 0:
                        self.on_update("legs.state_tau", np.array(rl.state_tau, dtype=np.float32))
                    if getattr(rl, "cmd_q", None) is not None and len(rl.cmd_q) > 0:
                        self.on_update("legs.cmd_q", np.array(rl.cmd_q, dtype=np.float32))
                    if getattr(rl, "cmd_qd", None) is not None and len(rl.cmd_qd) > 0:
                        self.on_update("legs.cmd_qd", np.array(rl.cmd_qd, dtype=np.float32))
                    if getattr(rl, "cmd_tau", None) is not None and len(rl.cmd_tau) > 0:
                        self.on_update("legs.cmd_tau", np.array(rl.cmd_tau, dtype=np.float32))

                update_count += 1
            except Exception as e:
                if update_count % 1000 == 0:
                    logger.debug("LCM 适配器轮询异常：%s", e)
            time.sleep(period)


class Collector:
    """按 factory_lerobot 方案：将多源数据汇聚为 lerobot 格式"""

    def __init__(self, cfg: CollectorConfig) -> None:
        self.cfg = cfg

        # 按 factory_lerobot 方案：使用 saveInfo、dataFlag、saveSate
        self.data_info = saveInfo()
        self.data_flag = dataFlag()
        self.save_state = saveSate()

        if cfg.dataset_root:
            local_root = os.path.expanduser(cfg.dataset_root)
            logger.info("使用自定义数据集存放目录：%s", local_root)
        else:
            local_root = os.environ.get("LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
            logger.info("使用默认数据集存放目录：%s", local_root)

        custom_fields_mapping = None
        if cfg.dataset_fields:
            custom_fields_mapping = {
                field_mapping.dataset_path: field_mapping.dimension
                for field_mapping in cfg.dataset_fields.values()
            }
        self.dataset = lerobotUnit(repo_id=cfg.repo_id, root=local_root, custom_fields=custom_fields_mapping)
        self._adapters: List[Any] = []
        self._pico_adapter: Optional[PicoControllerAdapter] = None
        self._camera_node = None  # 相机订阅节点（按 factory_lerobot 方案）
        self._camera_executor = None  # 相机执行器
        self._camera_thread: Optional[threading.Thread] = None  # 相机线程
        self._camera_running = False  # 相机线程运行标志
        self._camera_callback_count = 0  # 相机回调计数器（用于诊断）
        self._last_log_time = 0  # 上次输出日志的时间（用于控制日志频率）
        self._sampler_thread: Optional[threading.Thread] = None
        self._sampling = False
        self._session_active = False
        self._allow_no_action = False
        self._current_session_id: Optional[str] = None
        self._pending_session_id: Optional[str] = None
        self._scheduled_start_time: Optional[float] = None
        self._target_frames: Optional[int] = None
        self._frames_written = 0
        self._actual_start_time: Optional[float] = None
        self._target_duration: Optional[float] = None

    def _on_update(self, target: str, value: Any) -> None:
        """按 factory_lerobot 方案：回调中更新数据、设置标志、记录时间戳"""
        # 记录时间戳（按 factory_lerobot 方案）
        current_timestamp_ns = int(time.time() * 1e9)

        # 根据目标类型更新对应的数据和标志
        if target == "hands.action":
            self.data_info.qpos = value
            self.data_info.timestamp_list[0] = current_timestamp_ns  # 记录时间戳
            self.data_flag.action_flag = True
        elif target == "hands.state":
            self.data_info.hands_state = value
            self.data_flag.hands_state_flag = True
            # 如果没有独立的 action 数据源，使用 state 作为 action
            if not self.data_flag.action_flag:
                self.data_info.qpos = value
                self.data_info.timestamp_list[0] = current_timestamp_ns
                self.data_flag.action_flag = True
        elif target == "hands.vel":
            self.data_info.hands_vel = value
        elif target == "hands.torque":
            self.data_info.hands_torque = value
        elif target == "hands.current":
            self.data_info.hands_current = value
        elif target == "legs.state_q":
            self.data_info.legs_state_q = value
            self.data_flag.legs_state_q_flag = True
        elif target == "legs.state_qd":
            self.data_info.legs_state_qd = value
        elif target == "legs.state_tau":
            self.data_info.legs_state_tau = value
        elif target == "legs.cmd_q":
            self.data_info.legs_cmd_q = value
        elif target == "legs.cmd_qd":
            self.data_info.legs_cmd_qd = value
        elif target == "legs.cmd_tau":
            self.data_info.legs_cmd_tau = value
        elif target == "cmd_vel":
            v = np.asarray(value, dtype=np.float32).reshape(-1)
            out = np.zeros(6, dtype=np.float32)
            n = min(6, len(v))
            if n > 0:
                out[:n] = v[:n]
            self.data_info.cmd_vel = out
        elif target == "squat_des":
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                self.data_info.squat_des = arr[:2].copy()
        elif target == "task_mode":
            if len(value) > 0:
                self.data_info.task_mode = int(value[0])

    def setup_adapters(self) -> None:
        if self._adapters:
            logger.info("停止旧的适配器")
            for adapter in self._adapters:
                try:
                    adapter.stop()
                except Exception as e:
                    logger.warning("停止适配器时出错：%s", e)
            self._adapters.clear()

        if self.cfg.ros2_enabled and self.cfg.ros2_topics:
            ros2_adapter = ROS2Adapter(self.cfg.ros2_topics, self._on_update, self._on_control)
            ros2_adapter.start()
            self._adapters.append(ros2_adapter)

        if self.cfg.lcm_enabled:
            if self.cfg.robot_type == "gen1":
                # 使用Gen1专用LCM适配器
                if Gen1LCMAdapter is None:
                    logger.warning("Gen1 LCM 适配器未实现，无法启用 LCM 数据源")
                else:
                    gen1_adapter = Gen1LCMAdapter(self._on_update)
                    if gen1_adapter.start():
                        self._adapters.append(gen1_adapter)
                        logger.info("Gen1 LCM 适配器已启动")
                    else:
                        logger.warning("Gen1 LCM 适配器启动失败")
            else:
                # 使用通用LCM适配器
                if lcmUnit is None:
                    logger.warning("通用LCM 适配器未实现，无法启用 LCM 数据源")
                    logger.info("如仅使用 ROS2，请在配置文件中设置 'lcm_enabled: false'")
                else:
                    lcm_adapter = LCMAdapter(self._on_update)
                    lcm_adapter.start()
                    if lcm_adapter._running or lcm_adapter._init_done:
                        self._adapters.append(lcm_adapter)

        if self.cfg.dataset_fields:
            missing_sources: List[str] = []
            defined_sources = {spec.target for spec in (self.cfg.ros2_topics or [])}
            for dataset_path, field_mapping in self.cfg.dataset_fields.items():
                source_key = field_mapping.source_key
                if source_key not in defined_sources:
                    missing_sources.append(f"{source_key} (用于 {dataset_path})")
            if missing_sources:
                logger.warning("以下数据源未在 ROS2 话题中定义：%s", ", ".join(missing_sources))
                logger.info("请检查配置文件中的 topics 配置")

        if self.cfg.pico_enabled:
            if PicoControllerAdapter is None or create_pico_control_handler is None:
                logger.warning("Pico控制器适配器未实现，无法启用Pico控制")
            else:
                # 创建Pico控制处理器
                pico_control_handler = create_pico_control_handler(self)

                # 创建Pico适配器
                self._pico_adapter = PicoControllerAdapter(
                    server_ip=self.cfg.pico_ip,
                    port=self.cfg.pico_port,
                    on_control=pico_control_handler,
                    on_motion=None,  # 暂时不处理运动数据
                    enable_kinematics=self.cfg.pico_kinematics_enabled,
                    urdf_left_path=self.cfg.pico_urdf_left,
                    urdf_right_path=self.cfg.pico_urdf_right,
                )

                # 启动Pico适配器
                if self._pico_adapter.start():
                    kinematics_status = "启用" if self._pico_adapter.kinematics_enabled else "禁用"
                    logger.info(f"Pico控制器已启用，连接到 {self.cfg.pico_ip}:{self.cfg.pico_port}，运动学计算：{kinematics_status}")
                else:
                    logger.warning("Pico控制器启动失败")

        if self.cfg.sync_camera_enabled:
            logger.info("启用 /multi_camera/sync_img 相机触发（与 factory_lerobot 一致）")
            self._setup_camera_subscription()
        else:
            logger.info("未启用 sync_camera：仅用动作/LCM 触发（与仅客户端 SDK 相机方案兼容）")
            self.data_flag.camera_available = False

    def start(self) -> None:
        self.setup_adapters()
        logger.info("预热数据适配器，等待 ROS2 话题发现")
        time.sleep(2.0)
        logger.info("数据适配器预热完成，开始采集线程")
        self._sampling = True
        self._sampler_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._sampler_thread.start()

    def ensure_sampling_running(self) -> None:
        """确保采样线程处于运行状态（用于多轮 start/stop 会话场景）。"""
        try:
            if self._sampling and self._sampler_thread is not None and self._sampler_thread.is_alive():
                return
        except Exception:
            pass
        self._sampling = True
        self._sampler_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._sampler_thread.start()
        logger.info("采样线程未运行，已自动重启")

    def start_session_immediate(self, target_frames: int, target_duration: Optional[float] = None) -> None:
        self._scheduled_start_time = None
        self._target_frames = max(1, int(target_frames))
        self._target_duration = target_duration
        self._frames_written = 0
        self._actual_start_time = None
        self._session_active = True
        self.save_state.start_get_data = True

    def stop_and_save(self) -> None:
        self._sampling = False
        self._session_active = False
        self.save_state.start_get_data = False
        for ad in self._adapters:
            try:
                ad.stop()
            except Exception:
                pass
        self._adapters.clear()

        # 停止Pico适配器
        if self._pico_adapter:
            try:
                self._pico_adapter.stop()
            except Exception as e:
                logger.warning("停止Pico适配器时出错：%s", e)
            self._pico_adapter = None

        # 停止相机节点（按 factory_lerobot 方案）
        if hasattr(self, '_camera_running'):
            self._camera_running = False
        if hasattr(self, '_camera_node') and self._camera_node is not None:
            try:
                if self._camera_executor:
                    self._camera_executor.shutdown()
                if self._camera_thread and self._camera_thread.is_alive():
                    self._camera_thread.join(timeout=1.0)
                self._camera_node.destroy_node()
                self._camera_node = None
                logger.info("相机节点已停止")
            except Exception as e:
                logger.warning("停止相机节点时出错：%s", e)

        if self._frames_written == 0:
            raise ValueError("没有采集到任何数据，无法保存。请先开始采集并等待至少一帧数据。")
        self.dataset.save_data()

    def _get_vector_safe(self, data, size: int) -> np.ndarray:
        """安全获取向量，处理 None 和尺寸不匹配的情况"""
        if data is None or len(data) == 0:
            return np.zeros((size,), dtype=np.float32)
        arr = np.asarray(data, dtype=np.float32)
        if arr.shape[0] != size:
            if arr.shape[0] > size:
                return arr[:size]
            pad = np.zeros((size,), dtype=np.float32)
            pad[: arr.shape[0]] = arr
            return pad
        return arr

    def _update_lerobot_frame(self):
        """按 factory_lerobot 方案：更新 lerobot 帧数据"""
        # 统一时间轴：有相机时使用相机触发时间戳作为主时钟；否则退化为动作时间戳
        if self.data_flag.camera_available:
            unified_ts_ns = int(self.data_info.image_timestamps_list[1])
        else:
            unified_ts_ns = int(self.data_info.timestamp_list[0])
        unified_ts_s = unified_ts_ns / 1e9

        # 从 data_info 获取数据
        hand_dict = {
            "state": self._get_vector_safe(self.data_info.hands_state, self.cfg.hand_dim),
            "action": self._get_vector_safe(self.data_info.qpos, self.cfg.hand_dim),
            "vel": self._get_vector_safe(self.data_info.hands_vel, self.cfg.hand_dim),
            "current": self._get_vector_safe(self.data_info.hands_current, self.cfg.hand_dim),
            "torque": self._get_vector_safe(self.data_info.hands_torque, self.cfg.hand_dim),
        }

        leg_dict = {
            "state_q": self._get_vector_safe(self.data_info.legs_state_q, self.cfg.leg_dim),
            "state_qd": self._get_vector_safe(self.data_info.legs_state_qd, self.cfg.leg_dim),
            "state_tau": self._get_vector_safe(self.data_info.legs_state_tau, self.cfg.leg_dim),
            "cmd_q": self._get_vector_safe(self.data_info.legs_cmd_q, self.cfg.leg_dim),
            "cmd_qd": self._get_vector_safe(self.data_info.legs_cmd_qd, self.cfg.leg_dim),
            "cmd_tau": self._get_vector_safe(self.data_info.legs_cmd_tau, self.cfg.leg_dim),
            "cmd_vel": self._get_vector_safe(self.data_info.cmd_vel, 6),
        }

        task_mode = np.array([self.data_info.task_mode], dtype=np.float32)
        descrip = self.cfg.task_description
        squat_des = np.asarray(self.data_info.squat_des, dtype=np.float32).reshape(-1)[:2]
        if squat_des.shape[0] < 2:
            squat_des = np.pad(squat_des, (0, 2 - squat_des.shape[0]))

        if self.cfg.dataset_fields:
            leg_custom = {k: v for k, v in leg_dict.items() if k != "cmd_vel"}
            self.dataset.update_frame(
                hand_dict,
                leg_custom,
                task_mode=task_mode,
                descrip=descrip,
            )
        else:
            self.dataset.update_frame(
                hand_dict,
                leg_dict,
                task_mode=task_mode,
                descrip=descrip,
                squat_des_array=squat_des,
                timestamp_list=[unified_ts_ns, unified_ts_ns, unified_ts_ns],
                timestamp=unified_ts_s,
            )

    def _sampling_loop(self) -> None:
        """按 factory_lerobot 方案：period_update_data_m 采样循环"""
        period = 1.0 / max(1, self.cfg.fps)

        while self._sampling:
            t0 = time.time()

            # 检查定时启动
            if (not self._session_active) and (self._scheduled_start_time is not None):
                if time.time() >= self._scheduled_start_time:
                    self._session_active = True
                    self._frames_written = 0
                    self._actual_start_time = None
                    self.save_state.start_get_data = True
                    # 激活后立即清空定时触发，防止同一会话被重复激活
                    self._scheduled_start_time = None
                    self._current_session_id = self._pending_session_id
                    self._pending_session_id = None
                    logger.info("定时启动触发，会话已激活")

            if not self._session_active:
                time.sleep(0.005)
                dt = time.time() - t0
                remain = period - dt
                if remain > 0:
                    time.sleep(remain)
                continue

            # 按 factory_lerobot 方案：检查数据就绪标志
            # 与 factory_lerobot 一致，只检查 camera_flag 和 action_flag（qpos_flag）
            # 不强制要求所有数据源都可用，允许部分数据缺失

            if self.data_flag.camera_available:
                # 有相机：使用相机触发（与 factory_lerobot 一致）
                action_ready = self.data_flag.action_flag or self._allow_no_action
                if not (self.data_flag.camera_flag and action_ready):
                    # 数据未就绪，每5秒记录一次日志（使用时间戳控制，确保每5秒只输出一次）
                    current_time = time.time()
                    if current_time - self._last_log_time >= 5.0:
                        missing = []
                        if not self.data_flag.camera_flag:
                            missing.append("camera")
                        if not action_ready:
                            missing.append("action")
                        logger.info(f"等待数据就绪，缺失：{missing}")
                        self._last_log_time = current_time
                    # 数据未就绪，继续等待
                    time.sleep(0.001)
                    dt = time.time() - t0
                    remain = period - dt
                    if remain > 0:
                        time.sleep(remain)
                    continue
            else:
                # 无相机：只检查 action 数据（降级方案）
                action_ready = self.data_flag.action_flag or self._allow_no_action
                if not action_ready:
                    # 数据未就绪，每5秒记录一次日志
                    current_time = time.time()
                    if current_time - self._last_log_time >= 5.0:
                        logger.info("等待数据就绪（无相机），缺失：action")
                        self._last_log_time = current_time
                    # 数据未就绪，继续等待
                    time.sleep(0.001)
                    dt = time.time() - t0
                    remain = period - dt
                    if remain > 0:
                        time.sleep(remain)
                    continue

            # 第一帧就绪时记录时间
            if self._actual_start_time is None:
                self._actual_start_time = time.time()
                logger.info("数据已就绪，开始写入帧（目标帧数：%d）", self._target_frames)

            # 直接更新帧，避免每帧创建线程导致计数滞后
            self._update_lerobot_frame()

            # 重置标志（为下一帧做准备）
            if self.data_flag.camera_available:
                self.data_flag.camera_flag = False  # 重置相机触发标志（与 factory_lerobot 一致）
            self.data_flag.action_flag = False
            self.data_flag.hands_state_flag = False
            self.data_flag.legs_state_q_flag = False

            self._frames_written += 1

            # 采集进度（与 factory_lerobot 体感对齐：实时可见剩余帧）
            if self._target_frames is not None and self._target_frames > 0 and self._frames_written % 30 == 0:
                remaining_frames = max(0, self._target_frames - self._frames_written)
                logger.info(
                    "采集进度：%d/%d 帧，剩余 %d 帧",
                    self._frames_written,
                    self._target_frames,
                    remaining_frames,
                )

            if self._target_frames is not None and self._frames_written >= self._target_frames:
                logger.info(
                    "主采集达到目标帧数 %d，采集停止并开始统一收尾；请等待三路相机到目标帧后再按右B保存",
                    self._target_frames,
                )
                self._session_active = False
                self._scheduled_start_time = None
                self._pending_session_id = None
                self._current_session_id = None
                self.save_state.start_get_data = False
                self.save_state.ready_to_save = True
                break

            dt = time.time() - t0
            remain = period - dt
            if remain > 0:
                time.sleep(remain)

    def _camera_callback(self, msg) -> None:
        """按 factory_lerobot 方案：相机回调函数，设置触发标志和时间戳"""
        try:
            self.data_flag.camera_flag = True

            current_timestamp_ns = int(time.time() * 1e9)
            self.data_info.image_timestamps_list[0] = current_timestamp_ns
            self.data_info.image_timestamps_list[1] = current_timestamp_ns
            self.data_info.image_timestamps_list[2] = current_timestamp_ns

            if hasattr(msg, "imgfl_array"):
                self.data_info.image_head_list[0] = np.frombuffer(msg.imgfl_array, dtype=np.uint8)
            if hasattr(msg, "imgf_array"):
                self.data_info.image_head_list[1] = np.frombuffer(msg.imgf_array, dtype=np.uint8)
            if hasattr(msg, "imgfr_array"):
                self.data_info.image_head_list[2] = np.frombuffer(msg.imgfr_array, dtype=np.uint8)

        except Exception as e:
            logger.debug("相机回调错误: %s", e)

    def _setup_camera_subscription(self) -> None:
        """按 factory_lerobot 方案：设置相机订阅（作为主时钟触发器）"""
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

            try:
                from multi_camera_msgs.msg import SyncImg

                SYNC_IMG_AVAILABLE = True
            except ImportError as e:
                SyncImg = None
                SYNC_IMG_AVAILABLE = False
                logger.warning("SyncImg 消息类型未找到：%s", e)

            if not SYNC_IMG_AVAILABLE:
                logger.info("相机触发不可用：未找到 multi_camera_msgs.msg.SyncImg")
                return

            try:
                rclpy.init(args=None)
                logger.debug("相机订阅：rclpy 初始化成功")
            except RuntimeError as e:
                msg = str(e).lower()
                if "already initialized" in msg or "must only be called once" in msg:
                    logger.debug("相机订阅：rclpy 已初始化，使用现有上下文")
                else:
                    logger.warning("相机订阅：rclpy 初始化失败：%s", e)
                    raise

            class _CameraNode(Node):
                pass

            import uuid

            node_name = f"camera_subscriber_{uuid.uuid4().hex[:8]}"
            self._camera_node = _CameraNode(node_name)
            self._camera_executor = None

            qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
            )

            # 订阅同步图像话题（与 factory_lerobot 一致）
            self._camera_node.create_subscription(
                SyncImg,
                "/multi_camera/sync_img",
                self._camera_callback,
                qos_profile
            )
            logger.info("已订阅相机话题 /multi_camera/sync_img（作为主时钟触发器）")

            # 在单独的线程中运行相机节点
            # 使用独立标志控制线程生命周期，不依赖 self._sampling
            self._camera_running = True

            def _run_camera_node():
                from rclpy.executors import SingleThreadedExecutor
                self._camera_executor = SingleThreadedExecutor()
                self._camera_executor.add_node(self._camera_node)
                try:
                    while self._camera_running:
                        self._camera_executor.spin_once(timeout_sec=0.1)
                finally:
                    pass

            self._camera_thread = threading.Thread(target=_run_camera_node, daemon=True)
            self._camera_thread.start()

            # 标记相机可用（与 factory_lerobot 一致）
            self.data_flag.camera_available = True
            logger.info("相机触发已启用，使用 /multi_camera/sync_img 作为主时钟")

        except Exception as e:
            self.data_flag.camera_available = False
            logger.warning("设置相机订阅失败：%s", e)
            logger.info("相机触发不可用，将使用 LCM 数据触发")

    def _on_control(self, cmd: str) -> None:
        try:
            if cmd == "abort":
                self._session_active = False
                self._scheduled_start_time = None
                self._pending_session_id = None
                self._current_session_id = None
                self._target_frames = None
                self._frames_written = 0
                self._actual_start_time = None
                self.save_state.start_get_data = False
                # 丢弃当前未保存会话，避免右A重开时先落地旧视频
                try:
                    if hasattr(self.dataset, "dataset") and hasattr(self.dataset.dataset, "clear_episode_buffer"):
                        self.dataset.dataset.clear_episode_buffer()
                        logger.info("中止当前会话：已丢弃未保存数据")
                except Exception as e:
                    logger.warning("中止会话时清空缓冲失败：%s", e)
                try:
                    cfg = getattr(self, "cfg", None)
                    dataset_root = Path(getattr(cfg, "dataset_root", "") or "").expanduser()
                    repo_id = str(getattr(cfg, "repo_id", "") or "").strip()
                    if dataset_root and repo_id:
                        repo_dir = dataset_root / repo_id
                        if repo_dir.exists() and repo_dir.is_dir():
                            shutil.rmtree(repo_dir, ignore_errors=True)
                            logger.info("中止当前会话：已删除未保存目录 %s", repo_dir)
                except Exception as e:
                    logger.warning("中止会话时清理未保存目录失败：%s", e)
                return
            if cmd == "stop":
                self._session_active = False
                self._scheduled_start_time = None
                self._pending_session_id = None
                self._current_session_id = None
                self._target_frames = None
                if self._frames_written == 0:
                    logger.warning("停止命令触发，但未采集到任何数据，跳过保存")
                    return
                self.dataset.save_data()
                return
            if cmd.startswith("start:"):
                parts = cmd.split(":")
                if len(parts) >= 4:
                    session_id = parts[1]
                    start_ts = float(parts[2])
                    target_frames = int(parts[3])
                    # 避免同一 session 被重复 start 导致自动二次采集
                    if session_id and (
                        session_id == self._current_session_id
                        or session_id == self._pending_session_id
                    ):
                        logger.info("忽略重复 start 命令（session_id=%s）", session_id)
                        return
                    self._pending_session_id = session_id
                    self._scheduled_start_time = start_ts
                    self._target_frames = target_frames
        except Exception:
            pass
