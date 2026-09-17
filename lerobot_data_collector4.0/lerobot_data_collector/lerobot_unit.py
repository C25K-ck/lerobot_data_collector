"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>


python scripts/gr00t_finetune.py --dataset-path /data3/T004_data/lerobot/t004_0429 --num-gpus 1 --max-steps 100000 --output-dir /data3/T004_data/model/t004_0429 --data-config gr1_full_upper_body


"""
# 动态注入 lerobot_factory 到 sys.path
import os
import sys

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    # lerobot_data_collector/lerobot_data_collector/lerobot_unit.py
    # 向上1级到 lerobot_data_collector 项目根目录
    _proj_root = os.path.dirname(_this_dir)  # lerobot_data_collector
    _proj_parent = os.path.dirname(_proj_root)  # 项目父目录
    
    # 优先查找路径（按优先级排序）
    possible_paths = [
        # 1. 项目内的 lerobot_factory（最高优先级）
        os.path.join(_proj_root, "lerobot_factory"),
        # 2. 项目父目录下的 factory_lerobot/lerobot_factory
        os.path.join(_proj_parent, "factory_lerobot", "lerobot_factory"),
        # 3. 项目父目录下的 lerobot_factory
        os.path.join(_proj_parent, "lerobot_factory"),
        # 4. 环境变量指定的路径
        os.environ.get("LEROBOT_FACTORY_PATH", None),
        # 5. 用户目录下的路径
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
            print(f"✓ 找到 lerobot_factory: {_lerobot_factory}")
    else:
        print("⚠ 警告: 未找到 lerobot_factory")
        print("   请确保 lerobot_factory 目录存在，或设置环境变量 LEROBOT_FACTORY_PATH")
except Exception as e:
    print(f"警告: 无法添加 lerobot_factory 到 sys.path: {e}")
    pass

# 尝试导入必要的模块
try:
    from __init__ import *
except Exception:
    pass

import dataclasses
from pathlib import Path
from typing import Literal, Dict

try:
    from conf import *
except Exception:
    pass

try:
    import h5py
except Exception:
    pass

# 优先尝试从 lerobot_factory 导入，如果失败则尝试直接导入
try:
    from lerobot_factory.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("警告: 无法导入 LeRobotDataset，请确保 lerobot_factory 路径正确")
        raise

import numpy as np
import datetime

try:
    import torch
except ImportError:
    pass

try:
    import tqdm
except ImportError:
    pass

try:
    import tyro
except ImportError:
    pass

from pathlib import Path

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

class lerobotUnit():
    def __init__(self, repo_id, robot_type="magic_p6", root=None, custom_fields=None):
        # 如果没有指定root，使用默认的本地路径（避免访问HuggingFace Hub）
        if root is None:
            # 使用本地目录，避免尝试从Hub下载
            import os
            # 优先使用环境变量，否则使用默认的本地路径
            default_root = os.environ.get("LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
            root = default_root
        
        from pathlib import Path
        root_path = Path(root)
        # LeRobotDataset 的 root 参数应该是完整的数据集路径（包含repo_id），而不是根目录
        # 所以我们需要传递 root/repo_id 作为数据集的实际路径
        dataset_root_path = root_path / repo_id
        info_file = dataset_root_path / "meta" / "info.json"
        # NOTE:
        # - 继续采集时，dataset_root_path 很可能已经存在。
        # - 我们绝不删除已有目录（避免覆盖/丢失历史数据）。
        # - 优先尝试“加载已有数据集”，即使 meta/info.json 不存在也先尝试加载；
        #   只有在加载失败时，才创建一个不冲突的新目录来写入，避免 Errno 17。

        def _pick_nonconflict_path(p: Path) -> Path:
            """为创建新数据集选择一个不冲突的目录（不会删除旧目录）。"""
            if not p.exists():
                return p
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 先用时间戳后缀，再做递增兜底
            cand = p.with_name(f"{p.name}_new_{ts}")
            if not cand.exists():
                return cand
            i = 1
            while True:
                cand2 = p.with_name(f"{p.name}_new_{ts}_{i}")
                if not cand2.exists():
                    return cand2
                i += 1
        
        # 如果目录已存在，优先尝试加载（即使 info.json 不存在也先试一次）
        if dataset_root_path.exists():
            # 离线/本地优先：确保必要的 meta 文件存在，避免 LeRobotDatasetMetadata 误判并去 Hub 拉取
            try:
                meta_dir = dataset_root_path / "meta"
                meta_dir.mkdir(parents=True, exist_ok=True)
                for fname in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
                    fpath = meta_dir / fname
                    if not fpath.exists():
                        fpath.write_text("", encoding="utf-8")
            except Exception:
                pass
            print(f"检测到数据集目录已存在，尝试加载本地数据集: {dataset_root_path}")
            try:
                self.dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root_path), force_cache_sync=False)
                return
            except Exception as e:
                print(f"加载数据集失败（将创建新目录避免覆盖）: {e}")

        # 数据集不存在（或加载失败），创建新的（确保路径不冲突）
        target_path = _pick_nonconflict_path(dataset_root_path)
        if target_path != dataset_root_path:
            print(f"为避免覆盖，创建新数据集目录: {target_path}（原目录保留: {dataset_root_path}）")
        else:
            print(f"创建新数据集: {target_path}")
        try:
            # create_empty_dataset 期望 root 是完整的数据集路径（包含 repo_id），且该路径通常要求不存在
            # 因此这里只保证 parent 存在，不提前创建 target_path 本身
            target_path.parent.mkdir(parents=True, exist_ok=True)
            self.dataset = create_empty_dataset(
                repo_id,
                robot_type,
                mode="video",
                dataset_config=DEFAULT_DATASET_CONFIG,
                root=str(target_path),
                custom_fields=custom_fields,
            )
        except FileExistsError as e:
            # 处理目录已存在的错误（Errno 17）
            error_msg = str(e)
            print(f"检测到目录已存在错误: {error_msg}")
            # 优先尝试加载已有目录（不依赖 info.json）
            if dataset_root_path.exists():
                try:
                    print("目录已存在，改为加载已有数据集以继续追加采集")
                    self.dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root_path), force_cache_sync=False)
                    return
                except Exception as e2:
                    print(f"加载失败（不会删除旧目录），将创建新目录写入: {e2}")

            # 创建一个新的不冲突目录，避免再次触发 Errno 17
            fallback_path = _pick_nonconflict_path(dataset_root_path)
            print(f"创建新目录以避免冲突: {fallback_path}（原目录保留）")
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            self.dataset = create_empty_dataset(
                repo_id,
                robot_type,
                mode="video",
                dataset_config=DEFAULT_DATASET_CONFIG,
                root=str(fallback_path),
                custom_fields=custom_fields,
            )
        except Exception as e:
            error_msg = str(e)
            # 检查是否是 Hub 访问错误
            if "401" in error_msg or "Repository Not Found" in error_msg or "HuggingFace" in error_msg:
                print("检测到 Hub 访问错误（这是正常的，因为我们只想在本地创建数据集）")
                print(f"错误详情: {error_msg}")
                # 不删除旧目录：改为选择不冲突目录重新创建
                retry_path = _pick_nonconflict_path(dataset_root_path)
                print(f"重新创建数据集（不覆盖旧目录）: {retry_path}")
                retry_path.parent.mkdir(parents=True, exist_ok=True)
                self.dataset = create_empty_dataset(
                    repo_id,
                    robot_type,
                    mode="video",
                    dataset_config=DEFAULT_DATASET_CONFIG,
                    root=str(retry_path),
                    custom_fields=custom_fields,
                )
            else:
                # 其他错误：不清理旧目录，改为创建不冲突的新目录
                print(f"创建数据集失败: {e}")
                print("不会清理旧目录，改为创建不冲突的新目录...")
                retry_path = _pick_nonconflict_path(dataset_root_path)
                retry_path.parent.mkdir(parents=True, exist_ok=True)
                self.dataset = create_empty_dataset(
                    repo_id,
                    robot_type,
                    mode="video",
                    dataset_config=DEFAULT_DATASET_CONFIG,
                    root=str(retry_path),
                    custom_fields=custom_fields,
                )

    def update_frame(
        self,
        hand_list,
        leg_list,
        img_list=None,
        dep_list=None,
        task_mode=None,
        descrip="generic",
        custom_fields=None,
        squat_des_array=None,
        timestamp_list=None,
        timestamp=None,
    ):
        """与 factory_lerobot act_get_data/lerobot_unit.update_frame 字段与语义对齐。"""
        frame = {}

        if custom_fields is not None:
            for dataset_path, value_array in custom_fields.items():
                frame[dataset_path] = value_array
        else:
            for hand_name, value_array in hand_list.items():
                frame[f"observation.hands.{hand_name}"] = value_array
            for leg_name, value_array in leg_list.items():
                frame[f"observation.legs.{leg_name}"] = value_array

        if squat_des_array is not None:
            frame["observation.squat_des"] = squat_des_array

        if task_mode is not None:
            frame["task_mode"] = task_mode
        if descrip is not None:
            frame["task"] = descrip

        if timestamp is not None:
            timestamp_ns = int(timestamp * 1e9)
            frame["timestamps"] = np.array([timestamp_ns], dtype=np.int64)
            utc_time = datetime.datetime.utcfromtimestamp(timestamp)
            frame["timestamps_utc"] = utc_time.strftime("%Y-%m-%dT%H:%M:%S.%f") + "+00:00"
            if timestamp_list is not None and len(timestamp_list) > 0:
                ts0 = int(timestamp_list[0])
                frame["headfr_mp4_timestamps"] = np.array([ts0], dtype=np.int64)
                frame["headf_mp4_timestamps"] = np.array([ts0], dtype=np.int64)
                frame["headfl_mp4_timestamps"] = np.array([ts0], dtype=np.int64)
                frame["hand_r_depth_mp4_timestamps"] = np.array([ts0], dtype=np.int64)
                frame["hand_r_rgb_mp4_timestamps"] = np.array([ts0], dtype=np.int64)
                frame["hand_l_depth_mp4_timestamps"] = np.array([ts0], dtype=np.int64)
                frame["hand_l_rgb_mp4_timestamps"] = np.array([ts0], dtype=np.int64)

        self.dataset.add_frame(frame)
        
    def save_data(self):
        # 检查是否有数据帧需要保存
        if self.dataset.episode_buffer is None:
            raise ValueError("数据集缓冲区未初始化，无法保存")
        buffer_size = self.dataset.episode_buffer.get("size", 0)
        if buffer_size == 0:
            raise ValueError("没有采集到任何数据帧，无法保存。请先调用 update_frame 添加数据帧。")
        self.dataset.save_episode()


def load_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str | Path | None = None,
):
    hand_motors = [
        "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
        "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
        "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4",
        "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
        "left_hand_joint1", "left_hand_joint2", "left_hand_joint3", "left_hand_joint4",
        "left_hand_joint5", "left_hand_joint6",
        "right_hand_joint1", "right_hand_joint2", "right_hand_joint3", "right_hand_joint4",
        "right_hand_joint5", "right_hand_joint6",
        "waist_joint1", "waist_joint2",
        "head_joint1", "head_joint2"
    ]

    features = {
        # 暂时注释掉头部和手部时间戳
        # "headfr_mp4_timestamps": {
        #     "dtype": "int64",
        #     "shape": (1,),
        #     "names": [
        #         "timestamps",
        #     ],
        # },
        # "headf_mp4_timestamps": {
        #     "dtype": "int64",
        #     "shape": (1,),
        #     "names": [
        #         "timestamps",
        #     ],
        # },
        # "headfl_mp4_timestamps": {
        #     "dtype": "int64",
        #     "shape": (1,),
        #     "names": [
        #         "timestamps",
        #     ],
        # },
        # "hand_r_depth_mp4_timestamps": {
        #     "dtype": "int64",
        #     "shape": (1,),
        #     "names": [
        #         "timestamps",
        #     ],
        # },
        # "hand_r_rgb_mp4_timestamps": {
        #     "dtype": "int64",
        #     "shape": (1,),
        #     "names": [
        #         "timestamps",
        #     ],
        # },
        # "hand_l_depth_mp4_timestamps": {
        #     "dtype": "int64",
        #     "shape": (1,),
        #     "names": [
        #         "timestamps",
        #     ],
        # },
        # "hand_l_rgb_mp4_timestamps": {
        #     "dtype": "int64",
        #     "shape": (1,),
        #     "names": [
        #         "timestamps",
        #     ],
        # },
        "task_mode": {
            "dtype": "float32",
            "shape": (1,),
            "names": [
                "task_mode",
            ],
        },
    }
    
    hand_observations = [
        "torque",
        "current",
        "vel",
        "state",
        "action"
    ]

    for obs in hand_observations:
        features[f"observation.hands.{obs}"] = {
            "dtype": "float32",
            "shape": (len(hand_motors),),
            "names": [
                hand_motors,
            ],
        }
    

    leg_motor = [
        "HIP_ROLL_L","HIP_YAW_L","HIP_PITCH_L","KNEE_PITCH_L","ANKLE_PITCH_L","ANKLE_ROLL_L",
        "HIP_ROLL_R","HIP_YAW_R","HIP_PITCH_R","KNEE_PITCH_R","ANKLE_PITCH_R","ANKLE_ROLL_R",
    ]

    leg_observations = [
        "state_q",
        "state_qd",
        "state_tau",
        "cmd_q",
        "cmd_qd",
        "cmd_tau",
    ]

    for obs in leg_observations:
        features[f"observation.legs.{obs}"] = {
            "dtype": "float32",
            "shape": (len(leg_motor),),
            "names": [
                leg_motor,
            ],
        }

    # cameras = [
        
    #     "headfl",
    #     "headf",
    #     "headfr"
    # ]

    # cameras_d = [
    #     "left_hand_rgb",
    #     "right_hand_rgb",
    #     "left_hand_depth",
    #     "right_hand_depth",
    # ]

    # depths = [
        
    # ]

    # for cam in cameras:
    #     features[f"observation.images.{cam}"] = {
    #         "dtype": mode,
    #         "shape": (3, 480, 640),
    #         "names": [
    #             "channels",
    #             "height",
    #             "width",
    #         ],
    #     }

    # for cam in cameras_d:
    #     features[f"observation.images.{cam}"] = {
    #         "dtype": mode,
    #         "shape": (3, 480, 640),
    #         "names": [
    #             "channels",
    #             "height",
    #             "width",
    #         ],
    #     }

    # for cam in depths:
    #     features[f"observation.depths.{cam}"] = {
    #         "dtype": "uint16",
    #         "shape": (480, 640),
    #         "names": [
    #             "height",
    #             "width",
    #         ],
    #     }

    # 直接加载本地数据集（如果存在），而不是从Hub下载
    return LeRobotDataset(repo_id=repo_id, root=root)

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str | Path | None = None,
    custom_fields: Dict[str, int] | None = None,  # 自定义字段映射 {dataset_path: dimension}
) -> LeRobotDataset:
    hand_motors = [
        "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
        "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
        "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4",
        "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
        "left_hand_joint1", "left_hand_joint2", "left_hand_joint3", "left_hand_joint4",
        "left_hand_joint5", "left_hand_joint6",
        "right_hand_joint1", "right_hand_joint2", "right_hand_joint3", "right_hand_joint4",
        "right_hand_joint5", "right_hand_joint6",
        "waist_joint1", "waist_joint2",
        "head_joint1", "head_joint2"
    ]

    # 根据 custom_fields 或 factory_lerobot 默认 schema 构建 features
    if custom_fields:
        features = {
            "task_mode": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["task_mode"],
            },
        }
        print("使用自定义字段映射创建数据集features")
        for dataset_path, dimension in custom_fields.items():
            features[dataset_path] = {
                "dtype": "float32",
                "shape": (dimension,),
                "names": None,
            }
        print(f"生成的自定义features: {list(custom_fields.keys())}")
    else:
        # 与 factory_lerobot act_get_data/lerobot_unit.create_empty_dataset 对齐
        print("使用默认 factory 风格 hands/legs + 时间戳 + squat_des + cmd_vel")
        features = {
            "headfr_mp4_timestamps": {"dtype": "int64", "shape": (1,), "names": ["timestamps"]},
            "headf_mp4_timestamps": {"dtype": "int64", "shape": (1,), "names": ["timestamps"]},
            "headfl_mp4_timestamps": {"dtype": "int64", "shape": (1,), "names": ["timestamps"]},
            "hand_r_depth_mp4_timestamps": {"dtype": "int64", "shape": (1,), "names": ["timestamps"]},
            "hand_r_rgb_mp4_timestamps": {"dtype": "int64", "shape": (1,), "names": ["timestamps"]},
            "hand_l_depth_mp4_timestamps": {"dtype": "int64", "shape": (1,), "names": ["timestamps"]},
            "hand_l_rgb_mp4_timestamps": {"dtype": "int64", "shape": (1,), "names": ["timestamps"]},
            "timestamps": {"dtype": "int64", "shape": (1,), "names": ["timestamps"]},
            "timestamps_utc": {"dtype": "string", "shape": (1,), "names": ["timestamps"]},
            "task_mode": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["task_mode"],
            },
        }
        hand_observations = ["torque", "current", "vel", "state", "action"]
        for obs in hand_observations:
            features[f"observation.hands.{obs}"] = {
                "dtype": "float32",
                "shape": (len(hand_motors),),
                "names": [hand_motors],
            }

        leg_motor = [
            "HIP_ROLL_L", "HIP_YAW_L", "HIP_PITCH_L", "KNEE_PITCH_L", "ANKLE_PITCH_L", "ANKLE_ROLL_L",
            "HIP_ROLL_R", "HIP_YAW_R", "HIP_PITCH_R", "KNEE_PITCH_R", "ANKLE_PITCH_R", "ANKLE_ROLL_R",
        ]
        leg_observations = [
            "state_q", "state_qd", "state_tau", "cmd_q", "cmd_qd", "cmd_tau",
        ]
        for obs in leg_observations:
            features[f"observation.legs.{obs}"] = {
                "dtype": "float32",
                "shape": (len(leg_motor),),
                "names": [leg_motor],
            }
        features["observation.legs.cmd_vel"] = {"dtype": "float32", "shape": (6,)}
        features["observation.squat_des"] = {"dtype": "float32", "shape": (2,)}

    # cameras = [
    #     "headf",
    #     "headfl",
    #     "headfr"
    # ]

    # cameras_d = [
    #     "left_hand_rgb",
    #     "right_hand_rgb",
    #     "left_hand_depth",
    #     "right_hand_depth",
    # ]

    # depths = [
        
    # ]

    # for cam in cameras:
    #     features[f"observation.images.{cam}"] = {
    #         "dtype": mode,
    #         "shape": (3, 480, 640),
    #         "names": [
    #             "channels",
    #             "height",
    #             "width",
    #         ],
    #     }

    # for cam in cameras_d:
    #     features[f"observation.images.{cam}"] = {
    #         "dtype": mode,
    #         "shape": (3, 480, 640),
    #         "names": [
    #             "channels",
    #             "height",
    #             "width",
    #         ],
    #     }

    # for cam in depths:
    #     features[f"observation.depths.{cam}"] = {
    #         "dtype": "uint16",
    #         "shape": (480, 640),
    #         "names": [
    #             "height",
    #             "width",
    #         ],
    #     }
        
    # 创建本地数据集，指定root路径避免访问HuggingFace Hub
    # 注意：LeRobotDataset.create 期望 root 是完整的数据集路径
    # 如果传入的 root 不包含 repo_id，create 方法会自动构建 root/repo_id
    # 但根据文档，如果提供了 root，它应该是完整的数据集路径
    # 所以我们在调用时需要确保传递正确的路径
    try:
        return LeRobotDataset.create(
            repo_id=repo_id,
            fps=30,
            robot_type=robot_type,
            features=features,
            root=root,  # root 应该是完整的数据集路径（root/repo_id）
            use_videos=dataset_config.use_videos,
        )
    except FileExistsError as e:
        # 如果目录已存在错误，检查是否是 parent 目录的问题
        error_msg = str(e)
        if "File exists" in error_msg and root is not None:
            from pathlib import Path
            root_path = Path(root)
            # 如果 root 路径已存在且不是目录，或者 parent 目录已存在，需要处理
            if root_path.exists() and root_path.is_file():
                raise ValueError(f"路径 {root_path} 已存在且是一个文件，无法创建数据集目录")
            elif root_path.parent.exists():
                # parent 目录已存在是正常的，可能是数据集目录已存在
                if root_path.exists():
                    # 数据集目录已存在，尝试使用 exist_ok=True
                    root_path.mkdir(parents=True, exist_ok=True)
                    # 重新尝试创建
                    return LeRobotDataset.create(
                        repo_id=repo_id,
                        fps=30,
                        robot_type=robot_type,
                        features=features,
                        root=root,
                        use_videos=dataset_config.use_videos,
                    )
        raise

def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118

def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep

def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep

if __name__ == "__main__":
    s = lerobotUnit(repo_id="task_test")
