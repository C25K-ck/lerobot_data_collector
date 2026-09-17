"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>


python scripts/gr00t_finetune.py --dataset-path /data3/T004_data/lerobot/t004_0429 --num-gpus 1 --max-steps 100000 --output-dir /data3/T004_data/model/t004_0429 --data-config gr1_full_upper_body


"""
from __init__ import *

import dataclasses
from pathlib import Path
import sys
import os
sys.path.append(f"{os.path.expanduser('~')}/test/factory_lerobot/lerobot_factory/")

from typing import Literal

from conf import *
import h5py

from lerobot_factory.lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import numpy as np
import torch
import tqdm
import tyro
from pathlib import Path
import datetime

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

class lerobotUnit():
    def __init__(self, repo_id,robot_type="magic_p6"):
        try:
            self.dataset = create_empty_dataset(repo_id, robot_type, mode="video", dataset_config=DEFAULT_DATASET_CONFIG)
        except:
            print(f"repo exist : {repo_id} try loading repo \n")
            self.dataset = load_dataset(repo_id, robot_type, mode="video", dataset_config=DEFAULT_DATASET_CONFIG)

    def update_frame(self, hand_list, leg_list, squat_des_array, img_list, dep_list, timestamp_list, task_mode, descrip, timestamp=None):
        frame = {}
        for hand_name, value_array in hand_list.items():
            frame[f"observation.hands.{hand_name}"] = value_array

        for leg_name, value_array in leg_list.items():
            frame[f"observation.legs.{leg_name}"] = value_array

        # print('squat_des:', squat_des_array)
        frame[f'observation.squat_des'] = squat_des_array

        # for camera, img_array in img_list.items():  
        #     frame[f"observation.images.{camera}"] = img_array

        # for camera, img_array in dep_list.items():
        #     frame[f"observation.depths.{camera}"] = img_array
            
        frame["task_mode"] = task_mode
        frame["task"] = descrip
        
        # 处理时间戳
        if timestamp is not None:
            # 生成纳秒级时间戳 (int64)
            timestamp_ns =  int(timestamp*1e9)
            frame["timestamps"] = np.array([timestamp_ns], dtype=np.int64)
            # frame["timestamps"] = timestamp_ns
            # 生成UTC时间戳字符串
            utc_time = datetime.datetime.utcfromtimestamp(timestamp)
            utc_str = utc_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:] + "+00:00"  # 格式: "2025-06-16T02:23:32.954384+00:00"
            frame["timestamps_utc"] = utc_str

            frame["headfr_mp4_timestamps"] =  np.array([timestamp_list[0]], dtype=np.int64)
            frame["headf_mp4_timestamps"] =  np.array([timestamp_list[0]], dtype=np.int64)
            frame["headfl_mp4_timestamps"] =  np.array([timestamp_list[0]], dtype=np.int64)
            frame["hand_r_depth_mp4_timestamps"] =  np.array([timestamp_list[0]], dtype=np.int64)
            frame["hand_r_rgb_mp4_timestamps"] =  np.array([timestamp_list[0]], dtype=np.int64)
            frame["hand_l_depth_mp4_timestamps"] =  np.array([timestamp_list[0]], dtype=np.int64)
            frame["hand_l_rgb_mp4_timestamps"] =  np.array([timestamp_list[0]], dtype=np.int64)

        self.dataset.add_frame(frame)
        
    def save_data(self):
        self.dataset.save_episode()


def load_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,

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
        "headfr_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "headf_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "headfl_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "hand_r_depth_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "hand_r_rgb_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "hand_l_depth_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "hand_l_rgb_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "timestamps_utc": {
            "dtype": "string",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
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

    features["observation.legs.cmd_vel"] = {
            "dtype": "float32",
            "shape": (6,),
    }

    features["observation.squat_des"] =  {
        "dtype": "float32",
        "shape": (2,)
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

    return LeRobotDataset.load_data(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
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

    features = {
        "headfr_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "headf_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "headfl_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "hand_r_depth_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "hand_r_rgb_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "hand_l_depth_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "hand_l_rgb_mp4_timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "timestamps": {
            "dtype": "int64",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
        "timestamps_utc": {
            "dtype": "string",
            "shape": (1,),
            "names": [
                "timestamps",
            ],
        },
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
    
    features["observation.legs.cmd_vel"] = {
            "dtype": "float32",
            "shape": (6,),
    }

    features["observation.squat_des"] =  {
        "dtype": "float32",
        "shape": (2,)
        }

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
        
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

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

