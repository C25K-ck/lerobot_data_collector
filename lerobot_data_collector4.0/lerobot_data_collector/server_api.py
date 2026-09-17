"""
统一封装与服务端的 HTTP API 调用逻辑，严格按照最新 api.md 实现。

说明：
- 这里仅负责 HTTP 协议层和 JSON 解析，不做 GUI 相关处理；
- GUI（如 Qt 界面）应调用这些函数并根据返回结果更新界面；
- 如果 HTTP status_code != 200，将直接 raise_for_status 抛出异常；
- 对于业务错误（JSON 中 code != 200），由调用方根据需要处理。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import math
import pathlib
import tempfile
import requests
from minio import Minio

from .qt_login import get_auth_headers, get_server_host

logger = logging.getLogger(__name__)


class ApiError(RuntimeError):
    """封装业务错误（HTTP 正常但 JSON code 非 200）的异常类型。"""

    def __init__(self, message: str, payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.payload = payload or {}


@dataclass
class LoginResult:
    code: int
    token: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None
    message: str = ""


def api_login(
    server_host: Optional[str],
    user_identity: str,
    password: str,
    timeout: int = 10,
) -> LoginResult:
    """
    账号密码登录接口封装：POST /api/v1/user/login

    注意：此函数不修改全局 USER_TOKEN / SERVER_HOST，只返回结果。
    """
    if not server_host:
        server_host = get_server_host()
    server_host = server_host.rstrip("/")

    url = f"{server_host}/api/v1/user/login"
    logger.info("API 登录请求: %s", url)

    payload = {
        "user_identity": user_identity,
        "password": password,
    }
    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    # 按照规范，非 200 status_code 先作为异常抛出
    resp.raise_for_status()

    data = resp.json()
    code = int(data.get("code", 0))
    if code != 200:
        # 这里不抛异常，交给上层决定如何展示错误信息
        return LoginResult(
            code=code,
            token=None,
            user_info=None,
            message=str(data.get("message", "")),
        )

    token = data.get("data", {}).get("token")
    user_info = data.get("data", {}).get("user_info")
    return LoginResult(code=code, token=token, user_info=user_info, message="")


def api_get_my_tasks(
    page: int = 1,
    page_size: int = 20,
    task_name: Optional[str] = None,
    collection_status: Optional[int] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    获取我的任务列表：GET /api/v1/task/my-tasks

    返回服务端完整 JSON（包含 code / data 等字段）。
    """
    base_url = get_server_host().rstrip("/")
    url = f"{base_url}/api/v1/task/my-tasks"
    headers = get_auth_headers()

    params: Dict[str, Any] = {
        "page": page,
        "page_size": page_size,
    }
    if task_name:
        params["task_name"] = task_name
    if collection_status is not None:
        params["collection_status"] = collection_status

    logger.info("API 获取任务列表: %s params=%s", url, params)
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    code = int(data.get("code", 0))
    if code != 200:
        raise ApiError(str(data.get("message", "获取任务列表失败")), payload=data)
    return data


def api_operate_task(
    task_id: int,
    op_type: int,
    collection_status: Optional[int] = None,
    approved_count: Optional[int] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    操作任务：PUT /api/v1/task/tasks/:id/operate

    参数：
    - type: 操作类型（3：更新采集任务状态；4：更新审核通过数量）
    - collection_status: 当 type 为 3 时必填
    - approved_count: 当 type 为 4 时必填
    """
    base_url = get_server_host().rstrip("/")
    url = f"{base_url}/api/v1/task/tasks/{task_id}/operate"
    headers = get_auth_headers()

    payload: Dict[str, Any] = {"type": op_type}
    if op_type == 3 and collection_status is not None:
        payload["collection_status"] = collection_status
    if op_type == 4 and approved_count is not None:
        payload["approved_count"] = approved_count

    logger.info("API 操作任务: %s payload=%s", url, payload)
    resp = requests.put(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    code = int(data.get("code", 0))
    if code != 200:
        raise ApiError(str(data.get("message", "操作任务失败")), payload=data)
    return data


@dataclass
class SessionToken:
    """数据上传临时凭证（STS）"""

    access_key: str
    access_secret: str
    security_token: str
    endpoint: str
    expiration: int
    tenant_id: int = 0  # 从 token 中解析出的 tenant_id


def _extract_tenant_id_from_token(security_token: str) -> int:
    """
    从 STS 的 security_token (JWT) 中解析 tenant_id。
    
    JWT 的 sessionPolicy 字段是 base64 编码的 JSON，里面包含类似：
    "teleopfs/tenant-1/*" 的资源路径，从中提取 tenant_id。
    """
    import base64
    import re
    
    try:
        # JWT 格式: header.payload.signature
        parts = security_token.split(".")
        if len(parts) < 2:
            return 0
        
        # 解码 payload (第二部分)
        payload_b64 = parts[1]
        # 补齐 base64 padding
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        
        payload_json = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
        payload_data = json.loads(payload_json)
        
        # sessionPolicy 是 base64 编码的策略 JSON
        session_policy_b64 = payload_data.get("sessionPolicy", "")
        if session_policy_b64:
            # 补齐 padding
            padding = 4 - len(session_policy_b64) % 4
            if padding != 4:
                session_policy_b64 += "=" * padding
            
            policy_json = base64.urlsafe_b64decode(session_policy_b64).decode("utf-8")
            
            # 在策略中查找 tenant-N 模式
            # 例如: "teleopfs/tenant-1/*" 或 "arn:aws:s3:::teleopfs/tenant-1/*"
            match = re.search(r"tenant-(\d+)", policy_json)
            if match:
                tid = int(match.group(1))
                logger.info("从 STS token 中解析到 tenant_id=%d", tid)
                return tid
    except Exception as e:
        logger.warning("解析 STS token 中的 tenant_id 失败: %s", e)
    
    return 0


def api_get_session_token(timeout: int = 10) -> SessionToken:
    """
    数据上传获取 STS：POST /api/v1/tenant/session-token

    按照 api.md 说明：
    - 无额外 body 参数（如后续有扩展，可在此处补充）；
    - 返回的数据直接传入对象存储 SDK 使用。
    """
    import time as _time
    base_url = get_server_host().rstrip("/")
    url = f"{base_url}/api/v1/tenant/session-token"
    headers = get_auth_headers()

    logger.info("API 获取会话 STS: %s", url)
    logger.info("请求头: %s", {k: (v[:20] + '...' if len(str(v)) > 20 else v) for k, v in headers.items()})
    
    _start = _time.time()
    try:
        resp = requests.post(url, json={}, headers=headers, timeout=(5, timeout))
        logger.info("STS 请求返回，耗时 %.2f 秒，状态码 %s", _time.time() - _start, resp.status_code)
    except requests.exceptions.Timeout as e:
        logger.error("STS 请求超时（%.2f 秒）: %s", _time.time() - _start, e)
        raise ApiError(f"STS 请求超时: {e}") from e
    except requests.exceptions.RequestException as e:
        logger.error("STS 请求异常（%.2f 秒）: %s", _time.time() - _start, e)
        raise ApiError(f"STS 请求失败: {e}") from e
    
    resp.raise_for_status()
    data = resp.json()
    logger.info("STS 响应 JSON: code=%s, data=%s", data.get("code"), data.get("data"))

    code = int(data.get("code", 0))
    # 兼容 code=0 和 code=200 都表示成功
    if code not in (0, 200):
        raise ApiError(str(data.get("message", "获取会话凭证失败")), payload=data)

    payload = data.get("data") or {}
    try:
        security_token = str(payload["security_token"])
        
        # 尝试从 security_token (JWT) 中解析 tenant_id
        tenant_id = _extract_tenant_id_from_token(security_token)
        
        session = SessionToken(
            access_key=str(payload["access_key"]),
            access_secret=str(payload["access_secret"]),
            security_token=security_token,
            endpoint=str(payload["endpoint"]),
            expiration=int(payload["expiration"]),
            tenant_id=tenant_id,
        )
        logger.info("STS 获取成功，endpoint=%s，过期时间=%s，tenant_id=%s", session.endpoint, session.expiration, session.tenant_id)
        return session
    except KeyError as exc:
        logger.error("STS 响应缺少字段: %s, 完整响应: %s", exc, data)
        raise ApiError(f"STS 响应缺少字段: {exc}", payload=data) from exc
    except Exception as exc:  # pragma: no cover - 防御式
        logger.error("STS 响应格式异常: %s, 完整响应: %s", exc, data)
        raise ApiError(f"STS 响应格式异常: {exc}", payload=data) from exc


def _default_tenant_id(user_info: Optional[Dict[str, Any]] = None) -> int:
    """
    从用户信息或环境变量中推断 tenant_id。
    优先顺序：
      1) 传入的 user_info['tenant_id']
      2) 全局登录信息 qt_login.USER_INFO['tenant_id']
      3) 环境变量 LEROBOT_TENANT_ID
      4) 回退为 0
    """
    # 尝试从传入的 user_info 获取
    if isinstance(user_info, dict):
        try:
            tid = user_info.get("tenant_id")
            if isinstance(tid, int):
                return tid
            if isinstance(tid, str) and tid.isdigit():
                return int(tid)
        except Exception:
            pass
    
    # 尝试从全局登录信息获取
    try:
        from lerobot_data_collector.qt_login import get_user_info
        global_user_info = get_user_info()
        if isinstance(global_user_info, dict):
            tid = global_user_info.get("tenant_id")
            if isinstance(tid, int):
                return tid
            if isinstance(tid, str) and tid.isdigit():
                return int(tid)
    except Exception:
        pass
    
    env_v = os.environ.get("LEROBOT_TENANT_ID")
    if env_v and env_v.isdigit():
        return int(env_v)
    return 0


def upload_path_to_minio(
    local_path: str,
    session: SessionToken,
    tenant_id: Optional[int] = None,
    bucket: str = "teleopfs",
    progress_callback: Optional[callable] = None,
    custom_object_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    使用 STS 将本地文件/目录上传到对象存储。

    约定：
      - bucket 固定为 'teleopfs'（可通过参数覆盖）；
      - object_root: 如果提供 custom_object_prefix 则使用，否则为 f'tenant-{tenant_id}/dataset/{basename}'
      - 若 local_path 是目录，则递归上传该目录下所有文件到
        object_root/ 相对路径，不再进行 zip/tar.gz 打包；
      - 若 local_path 是文件，则直接上传为 object_root；
      - progress_callback: 可选的进度回调函数，签名为：
          callback(filename: str, current_file: int, total_files: int, 
                   uploaded_bytes: int, total_bytes: int)
      - 返回字典包含：
          {
            "bucket": ...,
            "object_root": ...,
            "size_mb": ...,
          }
    """
    src = pathlib.Path(local_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"本地路径不存在: {src}")

    # 1) 规范 tenant_id：优先使用 session 中解析出的 tenant_id
    if tenant_id is None:
        if session.tenant_id > 0:
            tenant_id = session.tenant_id
        else:
            tenant_id = _default_tenant_id(None)

    logger = logging.getLogger(__name__)

    # 2) 计算对象前缀与总大小（目录：递归所有文件；文件：自身）
    if custom_object_prefix:
        object_root = custom_object_prefix
    else:
        object_root = f"tenant-{tenant_id}/dataset/{src.name}"
    total_bytes = 0

    if src.is_file():
        total_bytes = src.stat().st_size
    else:
        for file in src.rglob("*"):
            if file.is_file():
                try:
                    total_bytes += file.stat().st_size
                except Exception:
                    continue

    # 向上取整，避免小文件被算成 0 MB 导致参数缺失/校验失败
    size_mb = int(math.ceil(total_bytes / (1024 * 1024))) if total_bytes > 0 else 0

    # 4) 构造 Minio 客户端并上传
    # 支持通过环境变量覆盖 MinIO endpoint（用于服务端配置错误的情况）
    minio_override = os.environ.get("MINIO_ENDPOINT_OVERRIDE")
    if minio_override:
        logger.info("使用环境变量覆盖 MinIO endpoint: %s -> %s", session.endpoint, minio_override)
        actual_endpoint = minio_override
    else:
        actual_endpoint = session.endpoint
    
    endpoint = actual_endpoint.replace("http://", "").replace("https://", "")
    secure = actual_endpoint.startswith("https://")

    logger.info("正在连接 MinIO: endpoint=%s, secure=%s", endpoint, secure)
    # 使用 MinIO 默认的 HTTP 客户端，不设置自定义超时
    # 因为分片上传时每个分片都有独立的超时，自定义超时可能导致问题
    client = Minio(
        endpoint=endpoint,
        access_key=session.access_key,
        secret_key=session.access_secret,
        session_token=session.security_token,
        secure=secure,
    )

    # 跳过 bucket 存在性检查/创建，直接上传（由服务端预先创建好 bucket）
    uploaded_bytes = 0
    
    if src.is_file():
        # 单文件：直接上传为 object_root
        logger.info(
            "开始上传单文件到对象存储: bucket=%s, object=%s, size=%d MB",
            bucket,
            object_root,
            size_mb,
        )
        if progress_callback:
            progress_callback(src.name, 1, 1, 0, total_bytes)
        # 禁用并行上传，使用单线程顺序上传，避免连接池问题
        client.fput_object(
            bucket_name=bucket,
            object_name=object_root,
            file_path=str(src),
            num_parallel_uploads=1,
        )
        if progress_callback:
            progress_callback(src.name, 1, 1, total_bytes, total_bytes)
        logger.info("上传完成: %s/%s", bucket, object_root)
    else:
        # 目录：递归上传所有文件，保留相对路径结构
        logger.info(
            "开始上传目录到对象存储: bucket=%s, prefix=%s, total_size=%d MB",
            bucket,
            object_root,
            size_mb,
        )
        # 先收集所有文件
        all_files = [f for f in src.rglob("*") if f.is_file()]
        total_files = len(all_files)
        
        # 并行上传（默认开 3 个 worker）
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        lock = threading.Lock()

        def upload_one(file: pathlib.Path, idx: int) -> None:
            nonlocal uploaded_bytes
            rel = file.relative_to(src).as_posix()
            obj_name = f"{object_root}/{rel}"
            file_size = file.stat().st_size

            if progress_callback:
                with lock:
                    progress_callback(rel, idx, total_files, uploaded_bytes, total_bytes)

            logger.info("上传文件: %s -> %s/%s", file, bucket, obj_name)
            client.fput_object(
                bucket_name=bucket,
                object_name=obj_name,
                file_path=str(file),
                num_parallel_uploads=3,
            )

            with lock:
                uploaded_bytes += file_size
                if progress_callback:
                    progress_callback(rel, idx, total_files, uploaded_bytes, total_bytes)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(upload_one, file, idx) for idx, file in enumerate(all_files, 1)]
            for fut in as_completed(futures):
                # 让异常向上抛出
                fut.result()
        logger.info("目录上传完成: %s (共 %.2f MB)", object_root, total_bytes / (1024 * 1024))

    return {
        "bucket": bucket,
        "object_root": object_root,
        "size_mb": size_mb,
    }


@dataclass
class DataInfoItem:
    """上传到平台的数据集条目描述。"""

    task_id: int
    uuid: str
    resource: str
    # 单位：MB（由 upload_path_to_minio 计算并向上取整）
    size: int = 0
    # 视频时长（秒）。你当前需求固定为 60。
    duration: int = 60

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "task_id": self.task_id,
            "uuid": self.uuid,
            "resource": self.resource,
            "size": int(self.size) if isinstance(self.size, int) else 0,
            "duration": int(self.duration) if isinstance(self.duration, int) else 60,
        }
        return payload


def api_batch_upload_data_info(
    items: List[DataInfoItem],
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    数据信息上传：POST /api/v1/data-info/batch-upload

    语义：采集数据已经通过对象存储 SDK 上传完成后，调用此接口把
    task_id / uuid / resource 等“元信息”一次性告知平台。
    """
    if not items:
        raise ValueError("items 不能为空")

    base_url = get_server_host().rstrip("/")
    url = f"{base_url}/api/v1/data-info/batch-upload"
    headers = get_auth_headers()
    # 关键：该接口需要登录 token（x-auth-tkn）。如果缺失，后端常返回 400 invalid request parameter
    if not headers.get("x-auth-tkn"):
        raise ApiError("未检测到登录 token（x-auth-tkn），请先在界面完成登录后再上传/上报。")

    payload = {"items": [it.to_dict() for it in items]}
    logger.info("API 批量上传数据信息: %s (items=%d)", url, len(items))
    logger.info(
        "batch-upload 请求头: %s",
        {k: (str(v)[:20] + "..." if len(str(v)) > 20 else v) for k, v in headers.items()},
    )
    # 打印第一个 item 的详细信息用于调试
    if items:
        first_item = items[0].to_dict()
        uuid_v = first_item.get("uuid")
        res_v = first_item.get("resource") or ""
        res_head = (res_v[:80] + "...") if len(res_v) > 80 else res_v
        res_tail = ("..." + res_v[-80:]) if len(res_v) > 80 else res_v
        logger.info(
            "第一个 item 示例: task_id=%s, uuid=%s, resource(head)=%s, resource(tail)=%s, size=%s, resource_endswith_uuid=%s",
            first_item.get("task_id"),
            (str(uuid_v)[:8] + "...") if uuid_v else None,
            res_head,
            res_tail,
            first_item.get("size"),
            bool(uuid_v) and res_v.endswith("/" + str(uuid_v)),
        )
    
    import time as _time
    _start = _time.time()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=(5, timeout))
        logger.info("batch-upload 请求返回，耗时 %.2f 秒，状态码 %s", _time.time() - _start, resp.status_code)
    except requests.exceptions.Timeout as e:
        logger.error("batch-upload 请求超时（%.2f 秒）: %s", _time.time() - _start, e)
        raise ApiError(f"数据信息上报超时: {e}") from e
    except requests.exceptions.RequestException as e:
        logger.error("batch-upload 请求异常（%.2f 秒）: %s", _time.time() - _start, e)
        raise ApiError(f"数据信息上报失败: {e}") from e
    
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # HTTP 错误（4xx, 5xx），尝试解析响应体
        try:
            error_data = resp.json()
            logger.error("batch-upload HTTP 错误 %s: %s", resp.status_code, error_data)
            raise ApiError(f"HTTP {resp.status_code}: {error_data.get('message', str(e))}", payload=error_data) from e
        except Exception:
            logger.error("batch-upload HTTP 错误 %s: %s", resp.status_code, resp.text[:500])
            raise ApiError(f"HTTP {resp.status_code}: {resp.text[:200]}") from e
    
    data = resp.json()

    code = int(data.get("code", 0))
    if code not in (0, 200):
        raise ApiError(str(data.get("message", "数据信息批量上传失败")), payload=data)
    
    logger.info("batch-upload 响应: code=%s, failed_uuids=%s", code, data.get("failed_uuids", []))
    return data


@dataclass
class RobotStatus:
    """机器人状态"""
    sn: str  # 机器人 SN
    status: int  # 0: 未运行, 1: 运行中


def api_upload_robot_status(
    robots: List[RobotStatus],
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    上传机器人运行状态: POST /api/v1/robot/upload-status
    
    Args:
        robots: 机器人状态列表
        timeout: 超时时间（秒）
    
    Returns:
        响应 JSON（code, message）
    
    Raises:
        ApiError: 业务错误
        requests.HTTPError: HTTP 错误
    """
    server_host = get_server_host().rstrip("/")
    url = f"{server_host}/api/v1/robot/upload-status"
    headers = get_auth_headers()
    
    payload = {
        "robots": [{"sn": r.sn, "status": r.status} for r in robots]
    }
    
    logger.info("API 上传机器人状态: %s", url)
    logger.info("请求数据: %s", payload)
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.HTTPError as e:
        try:
            error_data = resp.json()
            logger.error("upload-robot-status HTTP 错误 %s: %s", resp.status_code, error_data)
            raise ApiError(f"HTTP {resp.status_code}: {error_data.get('message', str(e))}", payload=error_data) from e
        except Exception:
            logger.error("upload-robot-status HTTP 错误 %s: %s", resp.status_code, resp.text[:500])
            raise ApiError(f"HTTP {resp.status_code}: {resp.text[:200]}") from e
    
    data = resp.json()
    code = int(data.get("code", 0))
    
    if code not in (0, 200):
        raise ApiError(str(data.get("message", "上传机器人状态失败")), payload=data)
    
    logger.info("upload-robot-status 响应: code=%s, message=%s", code, data.get("message", ""))
    return data


__all__ = [
    "ApiError",
    "LoginResult",
    "SessionToken",
    "DataInfoItem",
    "RobotStatus",
    "api_login",
    "api_get_my_tasks",
    "api_operate_task",
    "api_get_session_token",
    "api_batch_upload_data_info",
    "api_upload_robot_status",
]


