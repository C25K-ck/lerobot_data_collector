


"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>


python scripts/gr00t_finetune.py --dataset-path /data3/T004_data/lerobot/t004_0429 --num-gpus 1 --max-steps 100000 --output-dir /data3/T004_data/model/t004_0429 --data-config gr1_full_upper_body


"""

import cv2

import dataclasses
from pathlib import Path
import sys
sys.path.append("/home/dreame/test/factory_lerobot/")
sys.path.append("/home/dreame/test/factory_lerobot/lerobot_factory/")
sys.path.append("/home/dreame/lifeng/code/factory_lerobot/")
from init_cv import *

from typing import Literal

from conf import *
import h5py

from lerobot_factory.lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import numpy as np
import torch
import tqdm
import tyro
from pathlib import Path

from act_get_data.act_get_data.add_screen import *

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

class lerobotAddscreenData():
    def __init__(self, repo_id,robot_type="magic_p6"):

        self.episode_len = 400
        self.fps = 30
        self.load_data(repo_id,robot_type)
        self.update_frame()



    def load_data(self,repo_id, robot_type):
        
        delta_timestamps = {
            # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
            "observation.images.headf":[t/self.fps  for t in range(self.episode_len)],
            "observation.images.right_hand":[t/self.fps  for t in range(self.episode_len)],
            # loads 6 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
            "observation.state":  [i for i in range(self.episode_len)],
            # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
            "action":  [i for i in range(self.episode_len)],
            "torque":  [i for i in range(self.episode_len)],
            "vel":  [i for i in range(self.episode_len)],
            "task_mode": [i for i in range(self.episode_len)],
        }
        delta_timestamps = None



        # self.source_dataset =  load_dataset(repo_id, robot_type, mode="video", dataset_config=DEFAULT_DATASET_CONFIG,  delta_timestamps=delta_timestamps)
        self.source_dataset =  LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, norm_path=None)
        print(f"self.source_dataset: {self.source_dataset}")
        print(f"self.source_dataset: {self.source_dataset.episode_data_index}")
        
        episode_index = 0
        from_idx = self.source_dataset.episode_data_index["from"][episode_index].item()
        to_idx = self.source_dataset.episode_data_index["to"][episode_index].item()
        print(from_idx)
        print(to_idx)

        

# np.transpose(head_img_np, (2, 0, 1))


    def update_frame(self):

        image_list =["headf", "right_hand"]

        for i in range(len(self.source_dataset)):

            source_data = self.source_dataset[i]

            head_img_np =  np.uint8(source_data["observation.images.headf"].numpy()*255)

            head_img = np.transpose(head_img_np, (1, 2, 0))
            cv2.imshow("head", head_img)


            right_hand_img_np =  np.uint8(source_data["observation.images.right_hand"].numpy()*255)
            right_hand_img = np.transpose(right_hand_img_np, (1, 2, 0))
            cv2.imshow("head", head_img)
            cv2.imshow("right_hand", right_hand_img)
            cv2.waitKey(30)

            # print( np.array([source_data["task_mode"]]))
            # print( np.array([source_data["torque"]]))
            print( np.array([source_data["action"]]))



            
    def save_data(self):
        self.target_dataset.save_episode()


def load_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,

):
    motors = [
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
    cameras = [
        "headf",
        "right_hand",

    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "torque": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "vel": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "task_mode": {
            "dtype": "float32",
            "shape": (1,),
            "names": [
                motors,
            ],
        },
    }

    # if has_velocity:
    #     features["observation.velocity"] = {
    #         "dtype": "float32",
    #         "shape": (len(motors),),
    #         "names": [
    #             motors,
    #         ],
    #     }

    # if has_effort:
    #     features["observation.effort"] = {
    #         "dtype": "float32",
    #         "shape": (len(motors),),
    #         "names": [
    #             motors,
    #         ],
    #     }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }
        
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
    motors = [
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
    cameras = [
        "screen_headf",
        "screen_right_hand",

    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "torque": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "vel": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "task_mode": {
            "dtype": "float32",
            "shape": (1,),
            "names": [
                motors,
            ],
        },
    }

    # if has_velocity:
    #     features["observation.velocity"] = {
    #         "dtype": "float32",
    #         "shape": (len(motors),),
    #         "names": [
    #             motors,
    #         ],
    #     }

    # if has_effort:
    #     features["observation.effort"] = {
    #         "dtype": "float32",
    #         "shape": (len(motors),),
    #         "names": [
    #             motors,
    #         ],
    #     }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }
        
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
    s = lerobotAddscreenData(repo_id="chery/test_chery")
