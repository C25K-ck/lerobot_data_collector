API对接规范

异常处理

本文状态码非200 都视为异常，客户端按照返回的异常code进行业务处理或提示，判断协议status_code或者直接resp.raise_for_status()

如果status_code 非200, 响应体格式如下：

Code 错误码如下：

接口清单

登录

账号密码登录接口

请求地址：/api/v1/user/login

请求方法：POST

请求参数：

请求样例：

响应参数：

响应样例：

接口调用demo

解析token示例（需要先安装依赖：pip install pyjwt）

任务管理

获取我的任务列表

请求地址：/api/v1/task/my-tasks

请求方法：GET

请求参数：

响应参数：

Task

ActionStep

响应样例：

接口调用demo

操作任务

请求地址：/api/v1/task/tasks/:id/operate

请求方法：PUT

请求参数：

请求样例：

响应参数：

响应样例：

数据信息上传（采集数据上传到存储服务器后需要调用此接口上传数据信息）

请求地址：/api/v1/data-info/batch-upload

请求方法：POST

请求参数：

DataInfo

请求样例：

响应参数：

响应样例：

数据上传获取STS

请求地址： /api/v1/tenant/session-token

请求方法：POST

请求响应：

响应的数据直接传入SDK


<!-- 表格 1 -->

| 注意：登录之后请求接口需要携带token信息，传递在请求头x-auth-tkn中，token 为登录后获取的认证信息 |

| --- |


<!-- 表格 2 -->

| JSON
{
    "code": 10001,
    "message": "xxx error"
} |

| --- |


<!-- 表格 3 -->

| code | 描述 |

| --- | --- |

| 10000 | 系统错误 |

| 10002 | 用户不存在 |


<!-- 表格 4 -->

| 字段名 | 类型 | 位置 | 必填 | 字段说明 |

| --- | --- | --- | --- | --- |

| user_identity | string | body | 是 | 手机号或邮箱地址 |

| password | string | body | 是 | 用户密码 |


<!-- 表格 5 -->

| JSON
{
        "user_identity":"131********",
        "password":"123456"
} |

| --- |


<!-- 表格 6 -->

| 字段名 | 类型 | 字段说明 |

| --- | --- | --- |

| code | int | 状态码（200=成功） |

| data | object | 响应数据集 |

| data.token | string | 用户JWT令牌，后续进行其他操作需使用 |

| data.user_info | object | 用户信息 |

| data.user_info.user_id | int | 用户id |


<!-- 表格 7 -->

| JSON
{
  "code": 200,
  "data": {
    "user_info": {
      "user_id": 1
    },
    "token": "eyJhbGciOiJIUzI1Ni..."
  }
} |

| --- |


<!-- 表格 8 -->

| Python
import requests
import json

# 1. 设置接口地址和登录数据
host="http://localhost:8888" # 此处改为实际的服务地址
url = host+"/api/v1/user/login"
login_data = {
    "user_identity": "13888888888",  # 你的手机号或邮箱
    "password": "your_password"      # 你的密码
}

# 2. 设置请求头
headers = {
    "Content-Type": "application/json"
}

try:
    # 3. 发送登录请求
    response = requests.post(url, json=login_data, headers=headers)
    
    # 4. 解析响应结果
    result = response.json()
    
    # 5. 判断登录是否成功
    if result.get("code") == 200:
        print("✅ 登录成功！")
        token = result["data"]["token"]
        user_id = result["data"]["user_info"]["user_id"]
        print(f"用户ID: {user_id}")
        print(f"Token: {token}")
    else:
        print(f"❌ 登录失败: {result.get('message')}")
        
except Exception as e:
    print(f"❌ 请求出错: {e}") |

| --- |


<!-- 表格 9 -->

| Python
"""
JWT Token 解析脚本
可以直接调用parse_token方法并传入token进行解析
"""

import json
import jwt
import datetime

# 默认密钥（从项目代码中获取）
DEFAULT_SECRET_KEY = "xxxxxxxxxxxxxxxxxxxxxx"


def parse_token(token: str, secret_key: str = None, verify_signature: bool = True) -> dict:
    """
    解析JWT token

    Args:
        token: JWT token字符串
        secret_key: 可选，用于验证签名的密钥，默认使用DEFAULT_SECRET_KEY
        verify_signature: 是否验证签名，默认为True

    Returns:
        解析后的token信息字典
    """
    try:
        # 确定使用的密钥
        key = secret_key if secret_key is not None else DEFAULT_SECRET_KEY

        # 根据参数决定是否验证签名
        if verify_signature:
            # 验证签名
            decoded = jwt.decode(token, key, algorithms=["HS256"])
        else:
            # 不验证签名，仅解析内容
            decoded = jwt.decode(token, options={"verify_signature": False})

        # 添加可读的时间信息
        if 'exp' in decoded:
            exp_time = datetime.datetime.fromtimestamp(decoded['exp'])
            decoded['exp_readable'] = exp_time.strftime("%Y-%m-%d %H:%M:%S")
            # 计算剩余有效期
            now = datetime.datetime.now()
            remaining_seconds = (exp_time - now).total_seconds()

            if remaining_seconds > 0:
                decoded['expires_in_seconds'] = int(remaining_seconds)
                decoded['expires_in_minutes'] = int(remaining_seconds / 60)
                decoded['expires_in_hours'] = int(remaining_seconds / 3600)
            else:
                decoded['expires_in_seconds'] = 0
                decoded['expires_in_minutes'] = 0
                decoded['expires_in_hours'] = 0
                decoded['expired'] = True

        if 'iat' in decoded:
            decoded['iat_readable'] = datetime.datetime.fromtimestamp(decoded['iat']).strftime("%Y-%m-%d %H:%M:%S")

        return decoded
    except jwt.ExpiredSignatureError:
        return {"error": "Token已过期"}
    except jwt.InvalidTokenError as e:
        return {"error": f"Token无效: {str(e)}"}


# 示例使用
if __name__ == "__main__":
    # 示例token - 请替换为实际的token
    example_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

    # 不验证签名，仅解析内容
    print("\n=== 不验证签名，仅解析内容 ===")
    result2 = parse_token(example_token, verify_signature=False)
    print(json.dumps(result2, indent=2, ensure_ascii=False)) |

| --- |


<!-- 表格 10 -->

| 字段名 | 类型 | 位置 | 必填 | 字段说明 |

| --- | --- | --- | --- | --- |

| task_name | string | query | 否 | 任务名称 |

| collection_status | int | query | 否 | 采集状态：
0:待发布 1:待领取 2:待采集 3:采集中 4:任务超时 5:采集完成 6:采集完成-超时 |

| page | int | query | 是 | 页码，例如：1 |

| page_size | int | query | 是 | 每页数量，例如：10 |


<!-- 表格 11 -->

| 字段名 | 类型 | 字段说明 |

| --- | --- | --- |

| code | int | 状态码（200=成功） |

| data | object | 响应数据集 |

| data.page | int | 页码 |

| data.page_size | int | 每页数量 |

| data.total | int | 总数 |

| data.items | array[object(Task)] | 任务集合 |


<!-- 表格 12 -->

| 字段名 | 类型 | 字段说明 |

| --- | --- | --- |

| id | int | 任务ID |

| tenant_id | int | 租户ID |

| task_name | string | 任务名称 |

| task_purpose | string | 任务用途 |

| collection_count | int | 采集数量 |

| collection_body | string | 采集本体 |

| collection_plan | string | 采集方案 |

| project | string | 所属项目 |

| task_description | string | 任务描述 |

| task_start_time | int64 | 任务开始时间,例如：1762428438 |

| task_end_time | int64 | 任务结束时间，例如：1762594428 |

| scene_name | string | 场景名称 |

| sub_scene_name | string | 子场景名称 |

| continuous_action | string | 连续动作 |

| action_difficulty | string | 动作难度 |

| action_duration | int | 动作时长 |

| countdown | int | 倒计时 |

| unconventional_ratio | float64 | 非常规比例 |

| action_steps | array[object(ActionStep)] | 动作步骤json数组 |

| collector_id | int | 采集员ID |

| collection_status | int | 采集状态 |

| approved_count | int | 审核通过数量 |

| created_at | int64 | 创建时间，例如：1762428482 |

| updated_at | int64 | 更新时间，例如：1762428676 |

| collector_name | string | 采集员名称 |


<!-- 表格 13 -->

| 字段名 | 类型 | 字段说明 |

| --- | --- | --- |

| step_number | int | 动作步骤 |

| duration | int | 持续时长 |

| deviation | int | 偏差 |

| action_text | string | 动作文本 |

| atomic_skill | string | 原子技能 |


<!-- 表格 14 -->

| JSON
{
  "code": 200,
  "data": {
    "page": 1,
    "page_size": 10,
    "total": 1,
    "items": [
      {
        "id": 1,
        "tenant_id": 1,
        "task_name": "测试任务1",
        "task_purpose": "测试",
        "collection_count": 500,
        "collection_body": "GEN1",
        "collection_plan": "PICO遥操作",
        "project": "大人形",
        "task_description": "测试使用",
        "task_start_time": 1762428438,
        "task_end_time": 1762594428,
        "scene_name": "家庭",
        "sub_scene_name": "厨房",
        "continuous_action": "炒菜",
        "action_difficulty": "高",
        "action_duration": 80,
        "countdown": 40,
        "unconventional_ratio": 0.012,
        "action_steps": [
          {
            "step_number": 1,
            "duration": 50,
            "deviation": 30,
            "action_text": "起锅",
            "atomic_skill": "抓"
          },
          {
            "step_number": 2,
            "duration": 50,
            "deviation": 30,
            "action_text": "烧油",
            "atomic_skill": "倒"
          },
          {
            "step_number": 3,
            "duration": 50,
            "deviation": 30,
            "action_text": "炒菜",
            "atomic_skill": "炒"
          }
        ],
        "collector_id": 1,
        "collection_status": 1,
        "approved_count": 0,
        "created_at": 1763464936,
        "updated_at": 1763465213,
        "collector_name": "robot"
      }
    ]
  }
} |

| --- |


<!-- 表格 15 -->

| Python
import requests
from typing import Optional, Dict, Any

def get_my_tasks(base_url: str, token: str, task_name: Optional[str] = None,
                 collection_status: Optional[int] = None,
                 page: int = 1, page_size: int = 20) -> Dict[str, Any]:

    url = f"{base_url}/api/v1/task/my-tasks"
    headers = {
        "x-auth-tkn": token,
        "Content-Type": "application/json"
    }

    params = {
        "page": page,
        "page_size": page_size
    }

    if task_name:
        params["task_name"] = task_name
    if collection_status is not None:
        params["collection_status"] = collection_status

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"请求异常: {e}")
        return {}


def print_tasks(tasks_data: Dict[str, Any]) -> None:

    if not tasks_data or tasks_data.get("code") != 200:
        print(f"获取任务失败: {tasks_data}")
        return

    data = tasks_data.get("data", {})
    tasks = data.get("items", [])
    total = data.get("total", 0)

    print(f"共找到 {total} 个任务:")
    print("-" * 80)

    for task in tasks:
        task_id = task.get("id")
        task_name = task.get("task_name")
        project = task.get("project")
        status = task.get("collection_status")
        count = task.get("collection_count")
        approved = task.get("approved_count")

        print(f"ID: {task_id}, 名称: {task_name}, 项目: {project}, 状态: {status}, 采集数: {count}, 通过数: {approved}")


# 示例使用
if __name__ == "__main__":
    # 配置参数
    BASE_URL = "http://localhost:8888"  # 替换为实际API地址
    TOKEN = "eyJhbGciOo..."  # 替换为实际token

    print("=== 获取所有我的任务 ===")
    all_tasks = get_my_tasks(BASE_URL, TOKEN, page=1, page_size=10, task_name="测试", collection_status=0)
    print_tasks(all_tasks) |

| --- |


<!-- 表格 16 -->

| 字段名 | 类型 | 位置 | 必填 | 字段说明 |

| --- | --- | --- | --- | --- |

| id | int | path | 是 | 任务ID |

| type | int | body | 是 | 操作类型：
3：更新采集任务状态
4：更新审核通过数量 |

| collection_status | int | body | 否 | 采集任务状态，type为3时必填
当客户端点领取时，值为2；当客户端点采集时，值为3 |

| approved_count | int | body | 否 | 审核通过数量，type为4时必填 |


<!-- 表格 17 -->

| JSON
{
        "type":3,
        "collection_status":2
} |

| --- |


<!-- 表格 18 -->

| 字段名 | 类型 | 字段说明 |

| --- | --- | --- |

| code | int | 状态码（200=成功） |

| message | string | 返回信息 |


<!-- 表格 19 -->

| JSON
{
  "code": 200,
  "message": "success"
} |

| --- |


<!-- 表格 20 -->

| 字段名 | 类型 | 位置 | 必填 | 字段说明 |

| --- | --- | --- | --- | --- |

| items | array[object(DataInfo)] | body | 是 | 数据集 |


<!-- 表格 21 -->

| 字段名 | 类型 | 字段说明 |

| --- | --- | --- |

| task_id | int | 任务ID |

| uuid | string | 采集数据的uuid |

| resource | string | 采集数据存储的路径 |


<!-- 表格 22 -->

| JSON
{
        "items":[
                {
                        "task_id":1,
                        "uuid":"7c9e33d3-32d1-4091-9f04-678988deaea5",
                        "resource":"/Logistics/Logistics-2p0GB_4counts_0p04h/Materialtransfer-2p0GB_4counts_0p04h/Distribute_Parcels_To_Corresponding_Regions-2p0GB_4counts_0p04h/4ef81915-efad-4978-9d83-c136caa87671"
                },
                {
                        "task_id":1,
                        "uuid":"4ef81915-efad-4978-9d83-c136caa87671",
                        "resource":"/Logistics/Logistics-2p0GB_4counts_0p04h/Materialtransfer-2p0GB_4counts_0p04h/Distribute_Parcels_To_Corresponding_Regions-2p0GB_4counts_0p04h/4ef81915-efad-4978-9d83-c136caa87671"
                }
        ]
} |

| --- |


<!-- 表格 23 -->

| 字段名 | 类型 | 字段说明 |

| --- | --- | --- |

| code | int | 状态码（200=成功） |

| message | string | 返回信息 |

| failed_uuids | string[] | 失败的uuid列表 |


<!-- 表格 24 -->

| JSON
{
  "code": 200,
  "message": "批量上传成功",
  "failed_uuids": [
    "7c9e33d3-32d1-4091-9f04-678988deaea5"
  ]
} |

| --- |


<!-- 表格 25 -->

| JSON
{
    "code": 0,
    "data": {
        "access_key": "AB79AFR5R76MSDT9AV8H",
        "access_secret": "CnhN4d6+cVG1W6XLlwJZB8fjhpN3fKh7aU2w4GRU",
        "security_token": "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJBQjc5QUZSNVI3Nk1TRFQ5QVY4SCIsImV4cCI6MzYwMDAwMDAwMDAwMDAsInBvbGljeSI6ImNvbnNvbGVBZG1pbiIsInNlc3Npb25Qb2xpY3kiOiJld29nSUNKV1pYSnphVzl1SWpvZ0lqSXdNVEl0TVRBdE1UY2lMQW9nSUNKVGRHRjBaVzFsYm5RaU9pQmJDaUFnSUNCN0NpQWdJQ0FnSUNKVGFXUWlPaUFpVTNSdGRERWlMQW9nSUNBZ0lDQWlSV1ptWldOMElqb2dJa0ZzYkc5M0lpd0tJQ0FnSUNBZ0lrRmpkR2x2YmlJNklDSnpNem9xSWl3S0lDQWdJQ0FnSWxKbGMyOTFjbU5sSWpvZ1d3b2dJQ0FnSUNBZ0lDSmhjbTQ2WVhkek9uTXpPam82ZEdWc1pXOXdabk1pTEFvZ0lDQWdJQ0FnSUNKaGNtNDZZWGR6T25Nek9qbzZkR1ZzWlc5d1puTXZkR1Z1WVc1MExURXZLaUlLSUNBZ0lDQWdYUW9nSUNBZ2ZRb2dJRjBLZlE9PSJ9.XAVKrLus8iY1tO_TFda-KDINLCAtECA38c-nmz1VdjEQcxcQ80hRefpl3ORhtDPiS57olzaVrnc_KM9qZB9oDQ",
        "endpoint": "http://10.10.38.134:9000",
        "expiration": 1762195118 
    }
} |

| --- |
