# Gen1机器人数据采集集成指南

## 概述

本项目已成功将Gen1机器人的数据采集功能集成到通用机器人数据采集框架中。通过LCM通信协议，Gen1机器人可以与通用数据采集软件无缝协作。

## 支持的功能

### 数据类型
- **上体数据 (30维)**: 手臂、手部、腰部、头部关节状态
  - 关节位置 (hands.state)
  - 关节速度 (hands.vel)
  - 关节力矩 (hands.torque)
  - 关节电流 (hands.current)
- **腿部数据 (12维)**: 腿部关节状态和控制命令
  - 状态位置 (legs.state_q)
  - 状态速度 (legs.state_qd)
  - 状态力矩 (legs.state_tau)
  - 命令位置 (legs.cmd_q)
  - 命令速度 (legs.cmd_qd)
  - 命令力矩 (legs.cmd_tau)

### LCM话题
- `upper_body_data`: 上体传感器数据
- `upper_body_cmd`: 上体控制命令 (可选)
- `state_estimator`: 状态估计数据
- `leg_control_data`: 腿部传感器数据
- `leg_control_command`: 腿部控制命令

## 配置文件

使用 `configs/g1_lcm.json` 配置文件：

```json
{
  "repo_id": "g1_lcm_demo",
  "fps": 30,
  "hand_dim": 30,
  "leg_dim": 12,
  "lcm_enabled": true,
  "ros2_enabled": false,
  "robot_type": "gen1",
  "dataset_root": "/path/to/dataset",
  "dataset_fields": {
    "observation.hands.state": {
      "source": "hands.state",
      "dim": 30
    },
    // ... 其他字段映射
  }
}
```

## 使用方法

### 1. 环境准备

确保安装了必要的LCM库和biped_lcm_types：

```bash
# 安装LCM
pip install lcm

# 编译biped_lcm_types
cd factory_lerobot/biped_lcm_types
make
```

### 2. 启动数据采集

```python
from lerobot_data_collector.collector_core import CollectorConfig, Collector
import json

# 加载配置
with open('configs/g1_lcm.json', 'r') as f:
    config_data = json.load(f)

config = CollectorConfig.from_dict(config_data)
collector = Collector(config)

# 启动采集
collector.start()

# 开始录制会话
collector.start_session_immediate(target_frames=900)  # 30秒数据

# 等待采集完成
import time
time.sleep(30)

# 停止并保存数据
collector.stop_and_save()
```

### 3. 命令行使用

```bash
cd lerobot_data_collector4.0
python -m lerobot_data_collector.collector --config configs/g1_lcm.json
```

## 集成细节

### 核心组件

1. **Gen1LCMAdapter**: 专门的LCM通信适配器
   - 自动连接Gen1 LCM网络
   - 解析多话题数据
   - 线程安全的数据处理

2. **CollectorConfig**: 扩展配置支持
   - 新增 `robot_type` 字段
   - 支持LCM和ROS2的灵活配置

3. **数据集格式**: 完全兼容LeRobot标准格式
   - 结构化数据存储
   - 纳秒级时间戳
   - 自定义字段映射

### 架构优势

- **模块化设计**: Gen1适配器独立于通用框架
- **配置驱动**: 通过JSON配置灵活定制
- **多协议支持**: 同时支持LCM和ROS2
- **数据兼容性**: 无缝集成到现有工作流

## 故障排除

### LCM连接问题
- 检查网络配置: LCM默认使用 `udpm://239.255.76.67:7667?ttl=1`
- 确认biped_lcm_types已正确编译
- 检查防火墙设置

### 数据不同步问题
- 确保所有LCM话题都正常发布
- 检查时间戳同步
- 验证数据维度匹配

### 性能优化
- LCM通信使用专用CPU核心
- 数据异步写入避免阻塞
- 支持多线程处理

## 扩展开发

如需添加新的机器人类型：

1. 创建专用适配器类
2. 更新CollectorConfig支持新类型
3. 添加相应的配置文件
4. 编写集成测试

## 兼容性

- **Python**: 3.8+
- **LCM**: 1.4.0+
- **LeRobot**: 兼容现有数据集格式
- **操作系统**: Linux (推荐Ubuntu 20.04+)
