# HDF5转换配置文件

此目录用于存放HDF5转换的JSON配置文件。

## 配置文件格式

创建一个JSON文件（例如 `my_config.json`），包含以下字段：

```json
{
  "output_root_path": "/path/to/output",
  "main_scene_name": "Logistics",
  "sub_scene_name": "Materialtransfer",
  "action_name": "Distribute_Parcels_To_Corresponding_Regions",
  "use_copasi_structure": true
}
```

## 配置字段说明

- `output_root_path` (可选): 输出根目录路径。如果设置，将使用COPASI风格的目录结构
- `main_scene_name` (可选): 主场景名称，默认 "Logistics"
- `sub_scene_name` (可选): 子场景名称
- `action_name` (可选): 动作名称
- `use_copasi_structure` (可选): 是否使用COPASI结构，默认false。如果设置了`output_root_path`，会自动启用

## 使用方式

1. **自动模式**：将配置文件放在此目录下，程序会自动查找并使用第一个非example的JSON文件
2. **手动指定**：在代码中调用 `convert_parquet_to_hdf5()` 时传入 `config_file` 参数

## 输出结构

### 使用COPASI结构（设置了output_root_path）：
```
{output_root_path}/
  Logistics/
    task_info/
      {main_scene_name}-{sub_scene_name}-{action_name}.json
    {main_scene_name}/
      {sub_scene_name}/
        {action_name}/
          {UUID}/
            proprio_stats/
              proprio_stats.hdf5
```

### 默认结构（未设置output_root_path）：
```
{dataset_path}/hdf5/
  {episode_name}.hdf5
```

