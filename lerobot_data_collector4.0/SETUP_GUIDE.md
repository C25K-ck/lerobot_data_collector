# 项目设置指南

## 项目结构

```
lerobot_data_collector4.0/
├── lerobot_data_collector/          # 核心数据采集模块
│   ├── collector_core.py           # 采集器核心逻辑
│   ├── gen1_lcm_adapter.py         # Gen1 LCM适配器
│   ├── pico_controller.py          # Pico遥控器适配器
│   ├── orbbec_head_camera_manager.py # 奥比中光相机管理器
│   └── ...
├── robot_kinemic/                  # 运动学模块 (从factory_lerobot复制)
│   └── robot_kinemic/
│       ├── kinemic.py              # URDF加载和运动学计算
│       ├── pico_stream/            # Pico数据流处理
│       ├── teleop_pico.py          # Pico遥控器接口
│       └── model/urdfs/            # URDF模型文件
├── configs/                        # 配置文件
│   ├── g1_lcm.json                # Gen1 LCM配置
│   ├── g1_lcm_pico.json           # Gen1 + Pico配置
│   └── ...
├── GEN1_INTEGRATION_README.md      # Gen1集成说明
├── PICO_INTEGRATION_README.md      # Pico集成说明
└── SETUP_GUIDE.md                 # 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 克隆或复制robot_kinemic模块
# 将factory_lerobot/robot_kinemic/ 复制到本项目根目录

# 安装依赖
pip install -r requirements.txt
pip install grpcio grpcio-tools
pip install pinocchio  # 可选，用于运动学计算
```

### 2. 验证安装

```bash
# 运行Gen1集成测试
python test_gen1_integration.py

# 运行Pico集成测试
python test_pico_integration.py
```

### 3. 配置选择

#### 仅Gen1 LCM数据采集
```bash
python -m lerobot_data_collector.collector --config configs/g1_lcm.json
```

#### Gen1 LCM + Pico遥控器控制（推荐）
```bash
python -m lerobot_data_collector.collector --config configs/g1_lcm_pico.json
```

**Pico控制说明（与factory_lerobot完全一致）：**
- 左控制器A按钮: 开始/停止控制模式
- 左控制器B按钮: 切换数据发送模式
- 左控制器扳机: 改变任务模式
- 右控制器A按钮: 开始数据录制任务0（30秒）
- 右控制器B按钮: 开始数据录制任务1（60秒）

#### 启用运动学计算
修改 `g1_lcm_pico.json` 中的 `pico_kinematics_enabled: true`

## 依赖说明

### 必需依赖
- Python 3.8+
- LCM (Lightweight Communications and Marshalling)
- gRPC (用于Pico通信)

### 可选依赖
- Pinocchio (用于运动学计算)
- OpenCV (用于相机处理)
- ROS2 (用于ROS2通信)

## 网络配置

### Pico遥控器
- 默认IP: `192.168.12.110`
- 默认端口: `12345`
- 确保Pico设备和主机在同一网络

### LCM通信
- 默认组播地址: `udpm://239.255.76.67:7667?ttl=1`
- 确保防火墙允许组播通信

## 故障排除

### 常见问题

1. **AttributeError: 'QtApp' object has no attribute 'small_font'**
   ```bash
   # 这是已修复的问题。如果遇到，请确保使用最新版本
   # 问题出现在UI集成过程中，small_font字体未定义
   # 修复：在qt_collect.py中添加了small_font定义
   ```

2. **ImportError: 无法导入模块**
   ```bash
   # 确保robot_kinemic路径正确
   ls -la robot_kinemic/robot_kinemic/
   ```

2. **Pico连接失败**
   - 检查IP地址和端口
   - 确认Pico设备已启动数据流应用
   - 检查网络连接

3. **运动学计算失败**
   ```bash
   # 安装Pinocchio
   pip install pinocchio
   # 验证URDF文件存在
   ls -la robot_kinemic/robot_kinemic/model/urdfs/
   ```

4. **LCM通信问题**
   - 检查LCM环境变量
   - 确认机器人端LCM服务已启动
   - 检查网络组播设置

### 日志调试

启用详细日志：
```bash
export PYTHONPATH=$PYTHONPATH:.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python test_pico_integration.py
```

## 高级配置

### 自定义URDF路径
```json
{
  "pico_urdf_left": "/custom/path/to/left_arm.urdf",
  "pico_urdf_right": "/custom/path/to/right_arm.urdf"
}
```

### 自定义网络设置
```json
{
  "pico_ip": "192.168.1.100",
  "pico_port": 12346
}
```

### 性能优化
- 对于运动学计算，建议使用高性能CPU
- Pico通信延迟约50ms，适合大多数应用
- LCM通信延迟约1-5ms，适合实时控制

## 支持的机器人类型

- **Gen1**: 支持LCM通信的手臂+腿部机器人
- **通用机器人**: 通过ROS2通信的任意机器人
- **相机系统**: 奥比中光RGB-D相机支持

## 扩展开发

### 添加新机器人类型
1. 创建机器人专用适配器类
2. 更新CollectorConfig支持新配置
3. 添加相应的配置文件
4. 编写集成测试

### 添加新控制设备
1. 实现设备通信接口
2. 创建控制命令映射
3. 集成到Collector系统中
4. 更新配置和文档

## 许可证和贡献

请参考项目根目录的LICENSE文件。
贡献请提交Pull Request并遵循代码规范。
