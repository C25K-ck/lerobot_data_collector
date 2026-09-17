# Magiclab 数据采集平台

一个面向多机器人、多场景的通用采集客户端。软件提供登录鉴权、任务管理、数据采集、数据集管理与 Parquet → HDF5 转换等能力，支持 ROS2 数据流并输出 LeRobot v2.0 数据集。

---

## 目录

- [核心能力](#核心能力)
- [系统架构](#系统架构)
- [准备环境](#准备环境)
- [运行步骤](#运行步骤)
- [配置说明](#配置说明)
- [任务/数据集管理](#任务数据集管理)
- [常见问题](#常见问题)
- [后续计划](#后续计划)

---

## 核心能力

- **Qt UI**：包含登录、任务管理、数据采集、数据集管理四个主要界面，支持加载状态、返回导航、实时帧数显示。
- **任务全生命周期**：支持查询、创建、领取、开始/继续采集等操作，可离线创建本地任务。
- **多数据源采集**：内置 ROS2 适配器（Float32MultiArray / Int32 / String / JointState），可选 LCM。
- **自定义数据集映射**：通过 JSON 配置定义 observation/action 结构，自动转换为 LeRobot Dataset v2 。  
- **断点续采 & 统计**：自动检测已有帧数、继续采集补齐目标帧；实时显示已采集数量。
- **Parquet → HDF5 转换**：可视化配置转换任务，支持自定义输出目录与嵌套结构。
- **Loading / Timeline / Multi-Camera UI**：提供用户等待反馈、动作时间轴、摄像头位占位等视觉组件。

---

## 系统架构

```
lerobot_data_collector/
├─ main.py                       # 入口（自动选择 Qt / Tk GUI）
├─ lerobot_data_collector/
│  ├─ qt_gui.py                  # Qt 主界面（登录 / 任务 / 采集 / 数据集管理）
│  ├─ collector.py               # 采集核心（数据适配器 + LeRobot 写入）
│  ├─ dataset_manager.py         # 数据集扫描、Parquet→HDF5 转换
│  ├─ lerobot_unit.py            # 对 LeRobot dataset 的包装
│  ├─ convert_parquet_to_hdf5_adaptive.py
│  ├─ hdf5_configs/              # HDF5 转换配置示例
│  ├─ configs/                   # 采集配置示例
│  └─ action_steps/              # 动作步骤示例
└─ lerobot_factory/              # 上游 LeRobot 项目（供 dataset 工具使用）
```

> **注意**：`lerobot_factory` 来自原有项目，当前版本直接引用其中的 dataset 构建工具。后续可按需拆分为独立依赖。

---

## 准备环境

1. **操作系统**：Linux（推荐 Ubuntu 22.04），需预装 ROS2 Humble/Foxy（取决于你的机器人环境）。
2. **Python**：3.10 及以上，建议使用 `conda` 或 `venv`。
3. **依赖安装**：

   ```bash
   pip install -r requirements.txt
   ```

4. **LeRobot 工具链**  
   - 项目目录下已包含 `lerobot_factory/`，默认会自动通过 `sys.path` 引用。  
   - 如需使用外部路径，可设置环境变量：

     ```bash
     export LEROBOT_FACTORY_PATH=/path/to/factory_lerobot/lerobot_factory
     ```

5. **ROS2 环境变量**  
   在运行采集客户端前，确保已 `source` 对应的 ROS2 工作空间。

---

## 运行步骤

### 1. 启动客户端

```bash
python main.py
```

程序会尝试加载 PySide6；若不可用则自动回退到 Tkinter（功能受限）。

### 2. 登录

- 输入服务器地址（例如 `http://10.0.0.12:8888`）与账号密码。
- 支持超级用户 `root/root` 离线进入（跳过服务端，但无法访问云端任务）。
- 登录过程使用 `LoadingDialog` 提示网络状态。

### 3. 任务管理

- 查询 / 过滤 / 分页查看任务。
- 本地创建任务：点击 “创建任务” → 填写表单 → 即时加入列表，ID 显示为 `本地-1`、`本地-2`。
- 领取任务：将状态置为 “待采集”。
- 开始采集 / 继续采集：如果任务已有采集数据，会自动补齐剩余帧数。

### 4. 数据采集

- 采集界面提供配置文件、动作步骤下拉选择器、倒计时、三摄像头位占位、动作时间轴等 UI 元素。
- `开始采集`：读取配置 → 初始化适配器 → 自动预热 2 秒 → 正式记帧。  
- `结束采集`：停止并保存为 LeRobot 数据集。  
- `结束采集`（红色按钮）：立即终止当前会话，不保存数据。
- 返回任务管理：若仍在采集，会提示是否停止。

### 5. 数据集管理

- 扫描本地数据集目录（默认 `dataset_root` 或 `~/.cache/huggingface/lerobot`）。
- 显示数据集 ID、帧数、创建时间、是否已转换 HDF5。
- 支持选择数据集执行 Parquet → HDF5 转换，配置文件位于 `hdf5_configs/`。

---

## 配置说明

### 采集配置（`configs/*.json`）

关键字段：

| 字段 | 描述 |
|------|------|
| `repo_id` | 数据集目录名；应用运行期间会自动加上 `任务代码 + 时间戳` 生成会话 ID |
| `fps` | 采样频率（默认 30） |
| `dataset_root` | 数据集输出根目录 |
| `ros2.enabled` | 是否启用 ROS2 适配器 |
| `ros2.topics[]` | 每个话题的订阅配置（topic/type/target/field） |
| `dataset_fields` | 数据集字段映射：`dataset_path` → `source` → `dim` |
| `lcm_enabled` | 可选，启用 LCM 适配器（需实现 `lcm_unit.py`） |

示例请参考 `configs/collector_config_example.json`。

### 动作步骤（`action_steps/*.json`）

供时间轴展示使用，结构示例：

```json
{
  "episode_id": "...",
  "scene_name": "factory",
  "label_info": {
    "action_config": [
      {
        "start_frame": 0,
        "end_frame": 218,
        "action_text": "右臂抓取电机",
        "skill": "Pick"
      }
    ]
  }
}
```

### HDF5 转换配置（`hdf5_configs/*.json`）

定义输出目录、场景树结构、是否裁剪帧等，可参见 `hdf5_configs/README.md`。

---

## 任务/数据集管理

### 任务状态流转

`待发布 (0)` → `待领取 (1)` → `待采集 (2)` → `采集中 (3)` → `任务超时 / 采集完成 / 采集完成-超时 (4/5/6)`。

本地任务默认从 `待发布` 开始，可手动领取后进入采集。

### 数据集目录结构

```
<dataset_root>/
└─ <repo_id>/
   ├─ meta/info.json
   ├─ data/train-00000-of-00001.parquet or .hdf5
   └─ logs/...
```

同一应用会话内生成的多个采集将共享 `session_repo_id`，每次启动应用会生成新的带时间戳的 ID。

## 后续计划

- 拆分 `qt_gui.py` 为模块化结构（login/task/collector/components）。
- 抽象 API Service、配置解析、数据模型以提升可维护性。
- 引入单元测试覆盖关键逻辑（采集、转换、任务流）。
- 根据需求支持更多协议/数据源、云端上传等功能。


