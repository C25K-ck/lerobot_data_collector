# LeRobot 数据采集平台全流程 SDK 手册

本手册详细介绍了如何通过 Python 代码调用数据采集平台的各项功能，实现从机器人硬件采集到数据云端上报的自动化闭环。

## 1. 核心依赖
```bash
pip install requests minio numpy pandas pyarrow h5py PySide6
```

---

## 2. 身份认证 (Auth)
在使用任何业务接口前，必须先获取 `x-auth-tkn`。

### 2.1 用户登录
*   **方法**：`api_login(server_host, user_identity, password)`
*   **作用**：获取全局通用的认证 Token。

```python
from lerobot_data_collector.server_api import api_login

# 执行登录
result = api_login("http://10.204.5.111:9006", "138xxxxxxxx", "your_password")
if result.code == 200:
    token = result.token  # 需存入请求头 x-auth-tkn
    print("登录成功")
```

---

## 3. 任务管理 (Task)
用于获取分配的任务列表，并同步采集进度到服务端。

### 3.1 获取我的任务
*   **方法**：`api_get_my_tasks(status=None, page=1, page_size=20)`
*   **常用状态码**：`1`:待领取, `2`:待采集, `3`:采集中, `5`:已完成。

```python
from lerobot_data_collector.server_api import api_get_my_tasks

tasks = api_get_my_tasks(collection_status=2) # 获取待采集任务
for task in tasks['data']['items']:
    print(f"ID: {task['id']}, 目标数量: {task['collection_count']}")
```

### 3.2 同步任务状态
*   **方法**：`api_operate_task(task_id, op_type=3, collection_status=X)`
*   **作用**：开始采集前将状态置为 `3`，完成后置为 `5`。

```python
from lerobot_data_collector.server_api import api_operate_task

# 将任务ID 123 标记为已完成
api_operate_task(123, op_type=3, collection_status=5)
```

---

## 4. 硬件采集 (Collection)
通过 `Collector` 核心类统一控制，或通过各相机管理器直接操作。

### 4.1 核心采集器 (Collector)
```python
from lerobot_data_collector.collector_core import Collector, CollectorConfig

# 加载机器人配置文件
cfg = CollectorConfig.from_json("configs/g1.json")
collector = Collector(cfg)

collector.start() # 启动所有硬件线程（包括相机预热）
collector.start_session_immediate(target_frames=1800) # 开始录制
# ...
collector.stop_and_save() # 停止并持久化数据到本地 Parquet
```

### 4.2 相机独立接口 (Camera Managers)
SDK 提供了对 RealSense (D455/L515) 和 Orbbec (Femto Bolt) 的底层封装。

#### 支持的管理器：
- `RealSenseRightCameraManager` (右手相机)
- `RealSenseLeftCameraManager` (左手相机)
- `RealSenseHeadCameraManager` (头部 RealSense)
- `OrbbecHeadCameraManager` (头部奥比中光)

#### 核心方法：
- **`start()`**: 初始化底层驱动并启动预览流水线。
- **`get_latest_preview()`**: 返回最新的 RGB 图像（numpy 格式 `H x W x 3`），常用于实时 GUI 显示。
- **`start_recording(repo_id, target_frames=None)`**: 开始将图像（RGB + Depth）异步落地到指定的磁盘路径。
- **`stop_recording(wait=True)`**: 停止落地，并等待保存队列清空。
- **`shutdown()`**: 关闭所有流水线并释放硬件资源。

#### 代码示例：
```python
from lerobot_data_collector.warmup_display_realsense_right import RealSenseRightCameraManager

# 1. 初始化（可指定 SN 序列号）
cam = RealSenseRightCameraManager(serial_number="12345678")

# 2. 启动预览
if cam.start():
    # 获取一帧用于预览
    frame = cam.get_latest_preview()
    if frame is not None:
        print(f"图像尺寸: {frame.shape}")

# 3. 执行录制
cam.start_recording(repo_id="my_dataset_001", target_frames=300)
# ... 等待采集 ...
cam.stop_recording()

# 4. 彻底释放
cam.shutdown()
```

---

## 5. 格式转换 (Conversion)
将原始采集的 Parquet 格式转换为训练常用的 HDF5 格式。

### 5.1 Parquet 转 HDF5
*   **方法**：`convert_parquet_to_hdf5(repo_id, dataset_root=None)`
*   **作用**：自动化处理图像编码、状态对齐并生成标准 HDF5 文件。

```python
from lerobot_data_collector.dataset_manager import convert_parquet_to_hdf5

# 转换指定 Repo ID 的数据集
success = convert_parquet_to_hdf5("my_task_20260106_120000")
if success:
    print("HDF5 转换完成")
```

---

## 6. 云端上传 (Upload)
数据上传遵循：**获取 STS 凭证 -> 物理上传 -> 元信息报备** 三步走策略。

### 6.1 获取上传凭证与物理上传
```python
from lerobot_data_collector.server_api import api_get_session_token, upload_path_to_minio

# 1. 拿临时 Token
session = api_get_session_token()

# 2. 递归上传整个本地数据集目录
upload_info = upload_path_to_minio(
    local_path="/home/user/data/my_repo",
    session=session
)
```

### 6.2 批量报备元信息
*   **方法**：`api_batch_upload_data_info(items)`
*   **作用**：在 MinIO 上传成功后，通知平台该数据集已经可用。

```python
from lerobot_data_collector.server_api import api_batch_upload_data_info, DataInfoItem

item = DataInfoItem(
    task_id=123,
    uuid="my_repo_uuid",
    resource=upload_info['object_root'],
    size=upload_info['size_mb']
)
api_batch_upload_data_info([item])
```

---

## 7. 状态监控 (Monitor)
### 7.1 机器人运行状态上报
用于在后台实时显示机器人是否在线、是否正在忙碌。

```python
from lerobot_data_collector.server_api import api_upload_robot_status, RobotStatus

# 上报当前 SN 编号的机器人为“运行中”
status = RobotStatus(sn="SRIC-G1-001", status=1)
api_upload_robot_status([status])
```

---

## 8. 最佳实践 (Example)
```python
# 自动化采集脚本完整逻辑示例
def auto_collect_and_upload(task_id, config_file):
    # 1. 标记任务开始
    from lerobot_data_collector.server_api import api_operate_task
    api_operate_task(task_id, op_type=3, collection_status=3)
    
    # 2. 执行采集
    from lerobot_data_collector.collector_core import Collector, CollectorConfig
    cfg = CollectorConfig.from_json(config_file)
    collector = Collector(cfg)
    collector.start()
    
    print("开始采集...")
    collector.start_session_immediate(500)
    # 实际应用中此处应有等待逻辑
    collector.stop_and_save()
    
    # 3. 转换并上传
    from lerobot_data_collector.dataset_manager import convert_parquet_to_hdf5
    repo_id = collector.current_repo_id
    convert_parquet_to_hdf5(repo_id)
    
    from lerobot_data_collector.server_api import api_get_session_token, upload_path_to_minio
    sts = api_get_session_token()
    up = upload_path_to_minio(f"outputs/{repo_id}", sts)
    
    # 4. 上报信息并标记完成
    from lerobot_data_collector.server_api import api_batch_upload_data_info, DataInfoItem
    api_batch_upload_data_info([DataInfoItem(task_id, repo_id, up['object_root'], up['size_mb'])])
    api_operate_task(task_id, op_type=3, collection_status=5)
    print("全流程完成！")
```

