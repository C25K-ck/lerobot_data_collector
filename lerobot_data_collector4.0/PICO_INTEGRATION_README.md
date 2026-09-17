# Pico遥控器集成指南

## 概述

Pico遥控器现已集成到通用数据采集系统中，支持通过VR控制器按钮控制数据采集过程，并可选地提供运动学计算功能。

## 功能特性

### 控制功能（与factory_lerobot原始逻辑完全一致）
- **开始/停止控制模式**: 按下左控制器 A 按钮
- **切换数据发送**: 按下左控制器 B 按钮（需要先启动控制模式）
- **改变任务模式**: 按下左控制器扳机键
- **开始保存数据任务0**: 按下右控制器 A 按钮（30秒录制）
- **开始保存数据任务1**: 按下右控制器 B 按钮（60秒录制）（不保存）

### 运动学功能（可选）
- **URDF集成**: 支持加载机器人URDF文件进行运动学计算
- **逆运动学**: 计算控制器姿态对应的关节角度
- **实时计算**: 实时将VR控制器姿态转换为机器人关节命令

## 配置文件

### 基本Pico控制配置

```json
{
  "repo_id": "g1_lcm_pico_demo",
  "fps": 30,
  "hand_dim": 30,
  "leg_dim": 12,
  "lcm_enabled": true,
  "ros2_enabled": false,
  "robot_type": "gen1",
  "pico_enabled": true,
  "pico_ip": "192.168.12.110",
  "pico_port": 12345,
  "dataset_root": "/path/to/dataset"
}
```

### 启用运动学计算的配置

```json
{
  "repo_id": "g1_lcm_pico_kinematics_demo",
  "fps": 30,
  "hand_dim": 30,
  "leg_dim": 12,
  "lcm_enabled": true,
  "ros2_enabled": false,
  "robot_type": "gen1",
  "pico_enabled": true,
  "pico_ip": "192.168.12.110",
  "pico_port": 12345,
  "pico_kinematics_enabled": true,
  "pico_urdf_left": "./robot_kinemic/robot_kinemic/model/urdfs/l_arm_v2.urdf",
  "pico_urdf_right": "./robot_kinemic/robot_kinemic/model/urdfs/r_arm_v2.urdf",
  "dataset_root": "/path/to/dataset"
}
```

## 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pico_enabled` | bool | false | 是否启用Pico控制器 |
| `pico_ip` | string | "192.168.12.110" | Pico服务器IP地址 |
| `pico_port` | int | 12345 | Pico服务器端口 |
| `pico_kinematics_enabled` | bool | false | 是否启用运动学计算 |
| `pico_urdf_left` | string | null | 左臂URDF文件路径 |
| `pico_urdf_right` | string | null | 右臂URDF文件路径 |

## 使用方法

### 1. 环境准备

确保已安装必要的依赖：

```bash
# 安装Pico相关依赖
pip install grpcio grpcio-tools

# 确保factory_lerobot路径正确设置
export PYTHONPATH=$PYTHONPATH:/path/to/factory_lerobot/robot_kinemic/robot_kinemic
```

### 2. Pico设备连接

1. 确保Pico设备连接到网络
2. 启动Pico上的数据流应用
3. 确认IP地址和端口配置正确

### 3. 启动数据采集

```bash
cd lerobot_data_collector4.0
python -m lerobot_data_collector.collector --config configs/g1_lcm_pico.json
```

### 4. Pico控制操作（与factory_lerobot完全一致）

#### 独立连接模式（推荐）

**Pico连接完全独立于数据采集，可以随时连接！**

1. **启动数据采集平台**
   - 运行Qt界面应用程序
   - 界面会自动显示Pico控制面板

2. **连接Pico遥控器**
   - **随时点击**右侧面板中的"🔗 连接Pico"按钮
   - 系统会尝试连接到Pico设备（类似于`robot_control.py`的行为）
   - 连接成功后按钮会显示"✅ 已连接"
   - **无需先开始采集！**

3. **选择配置并开始采集（可选）**
   - 在下拉菜单中选择包含Pico的配置文件（如`g1_lcm_pico.json`）
   - 点击"开始采集"按钮启动数据采集系统
   - 如果Pico已连接，系统会自动使用Pico控制器

4. **佩戴Pico头显并开始控制**
   - 佩戴Pico VR头显确保控制器连接正常
   - **按下左控制器A按钮**启动/停止控制模式
   - **按下左控制器B按钮**切换数据发送模式（需要先启动控制模式）
   - **按下右控制器A按钮**开始数据录制任务0（30秒）
   - **按下右控制器B按钮**开始数据录制任务1（60秒）
   - **再次按下左控制器A按钮**停止控制模式

5. **监控状态**
   - 实时查看Pico连接状态指示器（独立于采集状态）
   - 观察数据采集进度和状态
   - Pico连接状态持续有效，直到手动断开或程序退出

#### 传统模式（兼容）

如果您习惯于先开始采集再连接Pico，传统模式仍然可用。系统会自动检测并使用采集器中的Pico配置。

## 运动学计算

### 启用条件

要启用运动学计算，需要：

1. 设置 `pico_kinematics_enabled: true`
2. 提供有效的URDF文件路径
3. 确保Pinocchio库已安装

### URDF文件要求

- 标准URDF格式
- 包含正确的关节和连杆定义
- 关节限制参数正确设置

### 计算结果

启用运动学后，Pico适配器会提供：

- **关节角度**: 从控制器姿态计算出的关节角度
- **运动学验证**: 计算结果的有效性标志
- **实时更新**: 每帧都进行运动学计算

## 故障排除

### Pico连接问题

**问题**: Pico控制器无法连接
**解决**:
- 检查IP地址和端口配置
- 确认Pico设备在同一网络
- 检查防火墙设置
- 重启Pico应用

### 运动学计算失败

**问题**: 运动学计算返回无效结果
**解决**:
- 验证URDF文件路径正确
- 检查URDF文件格式
- 确保Pinocchio库正确安装
- 检查关节限制设置

### 按钮响应延迟

**问题**: 按钮按下后响应延迟
**解决**:
- 检查网络连接质量
- 减少其他程序的CPU占用
- 调整Pico设备位置改善信号

## 高级用法

### 自定义控制映射

可以通过修改 `pico_controller.py` 中的按钮映射来自定义控制逻辑：

```python
# 修改按钮映射
def _check_button_events(self, data: dict, side: str, button: str, command: str):
    # 自定义按钮逻辑
    pass
```

### 运动数据处理

实现自定义的运动数据处理器：

```python
def custom_motion_handler(motion_data: dict):
    # 处理控制器姿态数据
    left_pos = motion_data.get("left_controller", {}).get("position")
    right_pos = motion_data.get("right_controller", {}).get("position")

    # 实现自定义控制逻辑
    pass

# 在初始化时传入
adapter = PicoControllerAdapter(
    on_motion=custom_motion_handler
)
```

## 兼容性

- **Pico设备**: Quest 2/Pro等支持控制器设备
- **操作系统**: Linux (推荐Ubuntu 20.04+)
- **Python**: 3.8+
- **依赖库**: gRPC, Pinocchio (可选)

## 性能建议

- 运动学计算会增加CPU负载
- 建议在高性能机器上启用运动学功能
- Pico数据传输延迟约50ms，适合大多数应用场景
- 网络连接建议使用有线或5GHz WiFi以获得最佳性能
