from datetime import datetime
import os

TASK_CODE = os.getenv("TASK_CODE", "zwj_test")
TIME_FORMAT = "%Y%m%d_%H%M%S"

# 优先使用环境变量中的固定REPO_ID，确保所有进程使用相同的ID
FIXED_REPO_ID = os.getenv("FIXED_REPO_ID")
if FIXED_REPO_ID:
    REPO_ID = FIXED_REPO_ID
    print(f"🔗 使用固定REPO_ID: {REPO_ID}")
else:
    CURRENT_TIME = datetime.now().strftime(TIME_FORMAT)
    REPO_ID = f"t008_task_{TASK_CODE}_{CURRENT_TIME}"
    print(f"🆕 生成新REPO_ID: {REPO_ID}")

MAX_STEP = 2400

# QPOS = "TEST"
QPOS = "ACT"

# 音频配置
AUDIO_SAMPLE_RATE = 16000  # 音频采样率 (Hz)
AUDIO_CHANNELS = 1         # 音频通道数 (单声道)
AUDIO_FORMAT = "WAV"    # 音频格式
AUDIO_FRAME_SIZE = 1024    # 音频帧大小

# 音频采集配置
AUDIO_COLLECT_BF = True         # 是否采集波束形成音频
AUDIO_ENABLE_CONTINUOUS_COLLECTION = True  # 是否启用连续音频采集
AUDIO_CONTINUOUS_DURATION = 80.0  # 连续音频采集时长（秒）

# 音频话题
ORIGIN_AUDIO_TOPIC = "/voice/origin_audio"
BF_AUDIO_TOPIC = "/voice/bf_audio"

# 音频数据缓存
AUDIO_QUEUE_SIZE = 1000    # 音频队列大小

# 数据同步参数
SYNC_TIMEOUT = 0.1         # 数据同步超时时间 (秒)
FRAME_RATE = 30            # 视频帧率 (fps) 

#REALSENSE_LEFT_SN = "230322276432"
#REALSENSE_RIGHT_SN = "230422271658"341222300752

#
# RealSense 序列号配置
# -------------------
# - 推荐通过环境变量覆盖，避免改代码：
#   - REALSENSE_RIGHT_SN=xxxxxxxxxxxx
#   - REALSENSE_LEFT_SN=xxxxxxxxxxxx
#   - REALSENSE_HEAD_SN=xxxxxxxxxxxx
# - 若某一路不使用，可设为 "DISABLED" 或空串（Qt 会跳过初始化，避免抢占同一设备）。
#
# 当前机器实测连接的设备：Intel RealSense D455 SN=341522300238
# 为避免只有一台相机时左右手/头部同时抢占导致无画面，默认仅启用头部。
REALSENSE_LEFT_SN = os.getenv("REALSENSE_LEFT_SN", "230322276597")
REALSENSE_RIGHT_SN = os.getenv("REALSENSE_RIGHT_SN", "230322276916")
REALSENSE_HEAD_SN = os.getenv("REALSENSE_HEAD_SN", "DISABLED")
