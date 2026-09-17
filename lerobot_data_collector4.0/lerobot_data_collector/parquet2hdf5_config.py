#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone config for the parquet -> HDF5 (COPASI folder structure) converter.
Only serves this tool; does not affect other converters.

Notes:
- Set DATASET_ROOT to the input root directory to scan for parquet files (recursive).
- Set OUTPUT_ROOT_PATH to the output root directory where Logistics/... will be created.
- MAIN_SCENE_NAME, SUB_SCENE_NAME, ACTION_NAME define the folder names before UUID level.
"""

import os

# Input root (where to search parquet files recursively)
DATASET_ROOT = "/home/dreame/data/hf_dataset"

# Output root (will create Logistics/... under this path)
OUTPUT_ROOT_PATH = "/home/dreame/data/suzhikeshucai/task0823"

# Scene naming
MAIN_SCENE_NAME = "Logistics"
SUB_SCENE_NAME = "Materialtransfer"
ACTION_NAME = "Distribute_Parcels_To_Corresponding_Regions"


