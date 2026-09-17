# LeRobot 数据集格式定义流程详解

## 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ 步骤1: 配置文件 (collector_config_myarm_example.json)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    {
      "dataset_fields": {
        "observation.left_arm.position": {
          "source": "left_arm.state",    ← 从ROS2话题获取的数据键
          "dim": 7                        ← 数据维度
        }
      }
    }
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤2: CollectorConfig.from_dict() 解析配置                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    创建 DatasetFieldMapping 对象：
    dataset_fields = {
      "observation.left_arm.position": DatasetFieldMapping(
        dataset_path="observation.left_arm.position",
        source_key="left_arm.state",
        dimension=7
      )
    }
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤3: Collector.__init__() 创建数据集对象                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    custom_fields_mapping = {
      "observation.left_arm.position": 7,  ← 提取 dataset_path 和 dimension
      "observation.left_arm.velocity": 7,
      ...
    }
                              │
                              ▼
    lerobotUnit(
      repo_id="myarm_demo",
      custom_fields=custom_fields_mapping  ← 传递给 lerobotUnit
    )
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤4: lerobotUnit.__init__() 创建/加载数据集                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    调用 create_empty_dataset(
      repo_id="myarm_demo",
      custom_fields=custom_fields_mapping  ← 传递 custom_fields
    )
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤5: create_empty_dataset() 构建 features 字典                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    features = {
      "timestamps": {...},
      "timestamps_utc": {...},
      "task_mode": {...}
    }
                              │
    if custom_fields:  ← 如果提供了自定义字段
                              │
                              ▼
    for dataset_path, dimension in custom_fields.items():
      features[dataset_path] = {
        "dtype": "float32",
        "shape": (dimension,),
        "names": None
      }
                              │
                              ▼
    features = {
      "timestamps": {...},
      "timestamps_utc": {...},
      "task_mode": {...},
      "observation.left_arm.position": {    ← 动态生成
        "dtype": "float32",
        "shape": (7,),
        "names": None
      },
      "observation.left_arm.velocity": {...},
      ...
    }
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤6: LeRobotDataset.create() 创建数据集结构                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    LeRobotDataset.create(
      repo_id="myarm_demo",
      fps=30,
      features=features,  ← 使用 features 定义数据集格式
      root="/home/dreame/data/hf_dataset/myarm_demo"
    )
                              │
                              ▼
    ↓ 创建目录结构：
    /home/dreame/data/hf_dataset/myarm_demo/
      ├── meta/
      │   └── info.json  ← 保存 features 定义
      └── data/
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤7: 数据采集时写入数据 (update_frame)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    在 _sampling_loop() 中：
    1. 从 ROS2 话题获取数据 → DataHub
    2. 从 DataHub 读取数据，构建 frame
                              │
                              ▼
    custom_data = {
      "observation.left_arm.position": [数据数组],  ← 键名必须匹配 features 中的键
      "observation.left_arm.velocity": [数据数组],
      ...
    }
                              │
                              ▼
    dataset.update_frame(
      custom_fields=custom_data  ← 传递数据
    )
                              │
                              ▼
    ↓ 检查：custom_data 的键必须在 features 中定义
    ↓ 写入：frame 数据添加到数据集缓冲区
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤8: 保存数据集 (save_episode)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    dataset.save_episode()
                              │
                              ▼
    ↓ 将缓冲区数据写入 parquet 文件
    ↓ 数据格式必须与 features 定义完全匹配
```

## 关键点说明

### 1. **features 字典的作用**
- `features` 字典定义了数据集的**"模板"**或**"schema"**
- 它告诉 LeRobotDataset：数据集应该包含哪些字段，每个字段的数据类型、形状、名称是什么
- 这个定义会被保存到 `meta/info.json` 文件中

### 2. **custom_fields 的作用**
- `custom_fields` 是一个简化的映射：`{dataset_path: dimension}`
- 它用于**动态生成** `features` 字典
- 如果没有 `custom_fields`，则使用默认的 `hands/legs` 结构

### 3. **数据写入时的匹配**
- `update_frame()` 写入数据时，`frame` 字典的键必须与 `features` 中的键**完全匹配**
- 如果键不匹配，会报错：`Feature mismatch`

### 4. **配置文件到 features 的转换链**
```
JSON配置 → DatasetFieldMapping → custom_fields_mapping → features字典
```

## 实际例子

### 配置示例（collector_config_myarm_example.json）
```json
{
  "dataset_fields": {
    "observation.left_arm.position": {
      "source": "left_arm.state",
      "dim": 7
    }
  }
}
```

### 转换过程
1. **解析配置** → `DatasetFieldMapping(dataset_path="observation.left_arm.position", source_key="left_arm.state", dimension=7)`
2. **提取映射** → `custom_fields_mapping = {"observation.left_arm.position": 7}`
3. **生成features** → 
   ```python
   features["observation.left_arm.position"] = {
       "dtype": "float32",
       "shape": (7,),
       "names": None
   }
   ```
4. **创建数据集** → `LeRobotDataset.create(features=features)`
5. **写入数据** → `frame["observation.left_arm.position"] = [1.0, 2.0, ...]`

## 为什么需要这个流程？

1. **灵活性**：可以支持不同的机器人配置，不需要修改代码
2. **类型安全**：在创建数据集时就确定了数据格式，避免运行时错误
3. **可扩展性**：可以轻松添加新的字段（如传感器数据、图像等）
4. **标准化**：LeRobot 格式要求明确的 schema 定义，确保数据集一致性

