# 快速开始指南

## 1. 环境准备

### 设置 lerobot_factory 路径

```bash
# 方法1: 设置环境变量（推荐）
export LEROBOT_FACTORY_PATH=/path/to/factory_lerobot/lerobot_factory

# 方法2: 如果lerobot_factory在项目同级目录，会自动查找
# 例如：/home/user/projects/factory_lerobot/lerobot_factory
```

### 安装依赖

```bash
pip install numpy
# ROS2需要单独安装，请参考ROS2官方文档
```

## 2. 创建配置文件

复制示例配置文件并修改：

```bash
cp configs/collector_config_example.json configs/my_config.json
```

编辑 `configs/my_config.json`，配置您的ROS2话题：

```json
{
  "repo_id": "my_robot_data",
  "fps": 30,
  "dataset_root": "/home/user/data/hf_dataset",
  "ros2": {
    "enabled": true,
    "topics": [
      {
        "topic": "/your_robot/joint_states",
        "type": "JointState",
        "target": "robot.state",
        "field": "position"
      }
    ]
  },
  "dataset_fields": {
    "observation.robot.position": {
      "source": "robot.state",
      "dim": 7
    }
  }
}
```

## 3. 启动软件

```bash
# 方法1: 使用main.py
python main.py

# 方法2: 作为模块运行
python -m lerobot_data_collector.collector

# 方法3: 安装后使用
pip install -e .
lerobot-collector
```

## 4. 使用GUI

1. 点击"浏览"选择配置文件
2. 设置采集时长（建议30-120秒）
3. 点击"开始采集"
4. 等待采集完成
5. 点击"结束并保存"

## 项目结构

```
lerobot_data_collector/
├── lerobot_data_collector/     # 核心模块
│   ├── __init__.py
│   ├── collector.py            # 主采集逻辑和GUI
│   ├── lerobot_unit.py          # LeRobot数据集封装
│   └── lcm_unit.py              # LCM适配器（可选）
├── configs/                     # 配置文件示例
│   ├── collector_config_example.json
│   ├── collector_config_myarm_example.json
│   └── collector_config_custom_fields_example.json
├── docs/                        # 文档
│   ├── DATA_INTERFACE_SPECIFICATION.md
│   └── DATASET_FORMAT_FLOW.md
├── main.py                      # 入口文件
├── README.md                    # 项目说明
├── requirements.txt             # 依赖列表
└── setup.py                     # 安装脚本
```

## 常见问题

**Q: 如何设置自定义数据集存放目录？**

A: 在配置文件中设置 `dataset_root` 字段：
```json
{
  "dataset_root": "/home/user/my_datasets"
}
```

**Q: 如何添加新的ROS2消息类型？**

A: 修改 `collector.py` 中的 `ROS2Adapter` 类，添加新的消息类型处理。

**Q: 支持DDS吗？**

A: ROS2底层使用DDS，所以支持。如果需要其他DDS实现，可以扩展适配器。

