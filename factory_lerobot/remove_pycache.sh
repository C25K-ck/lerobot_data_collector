
#!/bin/bash

# 指定要删除 __pycache__ 的目录
target_directory="./"

# 查找并删除 __pycache__ 目录
find "$target_directory" -type d -name '__pycache__' -exec rm -rf {} +

echo "Deleted all __pycache__ directories in $target_directory"
python3 -c "from factory_lerobot.robot_kinemic.robot_kinemic.act_mocap import mocapActionInfo; m=mocapActionInfo(); m.is_start=True; m.start_save_data(0)"

