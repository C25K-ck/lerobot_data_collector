"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>


python scripts/gr00t_finetune.py --dataset-path /data3/T004_data/lerobot/t004_0429 --num-gpus 1 --max-steps 100000 --output-dir /data3/T004_data/model/t004_0429 --data-config gr1_full_upper_body


"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal


import h5py

from lerobot_act.lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import numpy as np
import torch
import tqdm
import tyro
from pathlib import Path

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


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
        "headf",
        "right_hand",
        "merge_headf",
        "merge_right_hand",
        "screen_headf",
        "screen_right_hand"

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

    # if Path(LEROBOT_HOME / repo_id).exists():
    # shutil.rmtree(LEROBOT_HOME + repo_id)

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


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:

        if(f"/observations/images/{camera}"  in ep):
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            if("screen" in camera):
                imgs_array = ep[f"/observations/images/{camera.split('screen_')[1]}"][:]
            elif("merge" in camera):
                imgs_array = ep[f"/observations/images/{camera.split('merge_')[1]}"][:]
        
       
        if imgs_array.shape[-1] == 3:  
            imgs_array = np.transpose(imgs_array, (0, 3, 1, 2))
            
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        # state = torch.from_numpy(ep["/observations/qpos"][:])
        state = torch.from_numpy(ep["/action"][:])
        action = torch.from_numpy(ep["/action"][:])
        task_mode = torch.from_numpy(ep['task_mode'][:])

        velocity = None
        # if "/observations/qvel" in ep:
        #     velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        # if "/observations/effort" in ep:
        #     effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "headf",
                "right_hand",
                "merge_headf",
                "merge_right_hand",
                "screen_headf",
                "screen_right_hand"
            ],
        )

    return imgs_per_cam, state, action, task_mode, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))


    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        print(ep_path)

    
        episodes_idx = int(str(ep_path).split("episode_qpos_")[1].split(".")[0])

        descrip =  "get pear, and it put in busket"


        imgs_per_cam, state, action, task_mode, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        # dataset.episode_buffer = dataset.create_episode_buffer()

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action":action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
                
            frame["task_mode"] = task_mode[i]
            frame["task"] = descrip

            # frame["size"] = 400

            dataset.add_frame(frame)

        # dataset.save_episode(task=task)
        dataset.save_episode()

    return dataset


def port_aloha(
    raw_dir: Path | None = None,
    repo_id: str | None = None,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):

    # shutil.rmtree(LEROBOT_HOME + repo_id)

    #if not raw_dir.exists():
    #    if raw_repo_id is None:
    #        raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
    #    download_raw(raw_dir, repo_id=raw_repo_id)


    #task 1
    # data_dirs =  ['/data1/task_86', '/data1/task_92', '/data1/task_96',"/data3/task_101", "/data3/task_102", "/data3/task_103", "/data3/task_108", "/data3/task_109", "/data3/task_110", "/data3/task_115","/data3/task_116", "/data3/task_121", "/data3/task_123", "/data3/task_124","/data3/task_125"]
    
    #task 2
    data_dirs =  ['/data1/task_87', '/data1/task_89', '/data1/task_90', '/data3/task_98', '/data3/task_100_F', '/data3/task_100_L','/data3/task_113','/data3/task_114', '/data3/task_127', "/data3/task_128"]


    path_dir = []
    for dir in data_dirs:
        print(dir)
        path_dir.append(Path(dir))
    output_path = Path("/data3/factory/lerobot/task_2_0602")


    hdf5_files = []
    for data_dir in path_dir:
        hdf5_files.extend(sorted(data_dir.glob("episode_*.hdf5")))

    dataset = create_empty_dataset(
        output_path,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        #has_effort=has_effort(hdf5_files),
        #has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )

    # dataset = LeRobotDataset(
    #     "/data3/factory/lerobot/task_1/"
    # )

    
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    #dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
