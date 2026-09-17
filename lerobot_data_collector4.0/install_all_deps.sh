#!/usr/bin/env bash
# =============================================================================
# LeRobot 数据采集平台 — 一键安装全部依赖
#
# 用法:
#   cd lerobot_data_collector4.0
#   chmod +x install_all_deps.sh
#   ./install_all_deps.sh
#
# 可选环境变量:
#   USE_VENV=0          不使用虚拟环境，直接 pip install --user（默认 1 使用 .venv）
#   SKIP_APT=1          跳过 apt 系统包（无 sudo 时）
#   SKIP_ROS2=1         跳过 ROS2 Humble 安装
#   SKIP_REALSENSE=1    跳过 Intel RealSense 系统库
#   SKIP_ORBBEC=1       跳过奥比中光 SDK 尝试
#   SKIP_LCM_BUILD=1    跳过 biped_lcm_types 编译
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
PARENT_DIR="$(dirname "${PROJECT_ROOT}")"
FACTORY_LEROBOT="${FACTORY_LEROBOT:-${PARENT_DIR}/factory_lerobot}"
BIPED_LCM_TYPES="${FACTORY_LEROBOT}/biped_lcm_types"
VENV_DIR="${PROJECT_ROOT}/.venv"
ENV_FILE="${PROJECT_ROOT}/.env_collector"

USE_VENV="${USE_VENV:-1}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_ROS2="${SKIP_ROS2:-0}"
SKIP_REALSENSE="${SKIP_REALSENSE:-0}"
SKIP_ORBBEC="${SKIP_ORBBEC:-0}"
SKIP_LCM_BUILD="${SKIP_LCM_BUILD:-0}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

run_apt() {
    if [[ "${SKIP_APT}" == "1" ]]; then
        log_warn "SKIP_APT=1，跳过: $*"
        return 0
    fi
    if ! command -v apt-get &>/dev/null; then
        log_warn "未检测到 apt-get，跳过系统包: $*"
        return 0
    fi
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}

pip_install() {
    "${PIP_CMD[@]}" install "$@"
}

pip_install_optional() {
    if ! pip_install "$@"; then
        log_warn "可选包安装失败（可忽略）: $*"
    fi
}

detect_cuda_for_cupy() {
    if command -v nvidia-smi &>/dev/null; then
        local ver
        ver="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)"
        case "${ver}" in
            12.*) echo "cupy-cuda12x" ;;
            11.*) echo "cupy-cuda11x" ;;
            *) echo "" ;;
        esac
    else
        echo ""
    fi
}

write_env_file() {
    cat > "${ENV_FILE}" <<EOF
# LeRobot 数据采集平台环境变量 — 由 install_all_deps.sh 生成
# 使用前执行: source ${ENV_FILE}

export LEROBOT_COLLECTOR_ROOT="${PROJECT_ROOT}"

# lerobot_factory（项目内已捆绑）
export LEROBOT_FACTORY_PATH="${PROJECT_ROOT}/lerobot_factory"

# factory_lerobot（Gen1 LCM / biped_lcm_types / 真实 lcm_unit）
export FACTORY_LEROBOT="${FACTORY_LEROBOT}"

# robot_kinemic（Pico 遥操作 / 运动学）
export ROBOT_KINEMIC_ROOT="${PROJECT_ROOT}/robot_kinemic/robot_kinemic"

# PYTHONPATH：factory_lerobot + robot_kinemic
export PYTHONPATH="${FACTORY_LEROBOT}:${PROJECT_ROOT}/robot_kinemic/robot_kinemic:\${PYTHONPATH:-}"

# 数据集默认目录（可按需修改）
export LEROBOT_HOME="\${LEROBOT_HOME:-\$HOME/.cache/huggingface/lerobot}"

# Qt 高 DPI（界面显示）
export QT_AUTO_SCREEN_SCALE_FACTOR=1

# 日志
export LOG_LEVEL="\${LOG_LEVEL:-INFO}"
EOF
    log_ok "已写入环境文件: ${ENV_FILE}"
}

write_run_helper() {
    cat > "${PROJECT_ROOT}/run_collector.sh" <<'RUNEOF'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${DIR}/.venv/bin/activate"
fi
if [[ -f "${DIR}/.env_collector" ]]; then
    # shellcheck disable=SC1091
    source "${DIR}/.env_collector"
fi
# ROS2（若已安装）
if [[ -f /opt/ros/humble/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi
cd "${DIR}"
exec python3 main.py "$@"
RUNEOF
    chmod +x "${PROJECT_ROOT}/run_collector.sh"
    log_ok "已创建启动脚本: ${PROJECT_ROOT}/run_collector.sh"
}

verify_import() {
    local name="$1"
    local code="$2"
    if python3 -c "${code}" 2>/dev/null; then
        log_ok "  ✓ ${name}"
        return 0
    else
        log_warn "  ✗ ${name}（缺失或不可用）"
        return 1
    fi
}

main() {
    echo ""
    echo "============================================================"
    echo " LeRobot 数据采集平台 — 依赖一键安装"
    echo " 项目目录: ${PROJECT_ROOT}"
    echo "============================================================"
    echo ""

    # -------------------------------------------------------------------------
    # 1. 系统包（apt）
    # -------------------------------------------------------------------------
    log_info ">>> [1/8] 安装系统依赖 (apt)..."

    run_apt \
        python3 python3-pip python3-venv python3-dev \
        build-essential cmake pkg-config git curl wget \
        python3-tk \
        fontconfig fonts-noto-cjk fonts-wqy-microhei fonts-wqy-zenhei \
        libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
        libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg \
        libusb-1.0-0 libudev-dev \
        libjpeg-dev libpng-dev libtiff-dev \
        libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
        default-jdk \
        lcm-lib lcm-tools \
        libeigen3-dev

    # pinocchio 系统包（Ubuntu 22.04 可选，与 pip pin 二选一）
    run_apt python3-pinocchio 2>/dev/null || log_warn "apt python3-pinocchio 不可用，将使用 pip pin"

    log_ok "系统基础包安装完成"

    # -------------------------------------------------------------------------
    # 2. ROS2 Humble
    # -------------------------------------------------------------------------
    log_info ">>> [2/8] 安装 ROS2 Humble（机器人 ROS 通信）..."

    if [[ "${SKIP_ROS2}" == "1" ]]; then
        log_warn "SKIP_ROS2=1，跳过 ROS2"
    elif [[ -f /opt/ros/humble/setup.bash ]]; then
        log_ok "ROS2 Humble 已存在"
    elif command -v apt-get &>/dev/null && [[ "${SKIP_APT}" != "1" ]]; then
        if ! dpkg -l | grep -q ros-humble-desktop; then
            log_info "添加 ROS2 apt 源并安装 ros-humble-desktop（耗时较长）..."
            sudo apt-get install -y software-properties-common curl
            if [[ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]]; then
                sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
                    -o /usr/share/keyrings/ros-archive-keyring.gpg
                echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME}") main" \
                    | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
            fi
            sudo apt-get update -qq
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
                ros-humble-desktop \
                ros-humble-std-msgs \
                ros-humble-geometry-msgs \
                ros-humble-sensor-msgs \
                python3-colcon-common-extensions \
                || log_warn "ROS2 部分包安装失败，请手动检查"
        fi
        log_ok "ROS2 Humble 安装完成（使用前需 source /opt/ros/humble/setup.bash）"
        log_warn "自定义包 multi_camera_msgs 不在本仓库，同步相机功能需自行编译安装对应 ROS2 工作空间"
    else
        log_warn "无法自动安装 ROS2，请手动安装 ROS2 Humble"
    fi

    # -------------------------------------------------------------------------
    # 3. Intel RealSense
    # -------------------------------------------------------------------------
    log_info ">>> [3/8] 安装 Intel RealSense 系统库..."

    if [[ "${SKIP_REALSENSE}" == "1" ]]; then
        log_warn "SKIP_REALSENSE=1，跳过 RealSense 系统库"
    elif command -v apt-get &>/dev/null && [[ "${SKIP_APT}" != "1" ]]; then
        if ! dpkg -l | grep -q librealsense2; then
            log_info "添加 Intel RealSense apt 源..."
            if [[ ! -f /etc/apt/keyrings/librealsense.gpg ]]; then
                sudo mkdir -p /etc/apt/keyrings
                curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
                    | sudo tee /etc/apt/keyrings/librealsense.gpg >/dev/null
                echo "deb [signed-by=/etc/apt/keyrings/librealsense.gpg] https://librealsense.intel.com/Debian/apt-repo $(. /etc/os-release && echo "${VERSION_CODENAME}") main" \
                    | sudo tee /etc/apt/sources.list.d/librealsense.list >/dev/null
                sudo apt-get update -qq
            fi
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
                librealsense2-dkms librealsense2-utils librealsense2-dev \
                || log_warn "RealSense 系统库安装失败，pyrealsense2 可能无法使用 USB 相机"
        else
            log_ok "librealsense2 已安装"
        fi
    fi

    # -------------------------------------------------------------------------
    # 4. 奥比中光 Orbbec SDK
    # -------------------------------------------------------------------------
    log_info ">>> [4/8] 尝试安装奥比中光 Orbbec SDK..."

    if [[ "${SKIP_ORBBEC}" == "1" ]]; then
        log_warn "SKIP_ORBBEC=1，跳过 Orbbec SDK"
    else
        ORBBEC_INSTALLED=0
        if ldconfig -p 2>/dev/null | grep -qi orbbec; then
            ORBBEC_INSTALLED=1
            log_ok "系统已存在 Orbbec 库"
        fi
        if [[ "${ORBBEC_INSTALLED}" == "0" ]] && command -v apt-get &>/dev/null && [[ "${SKIP_APT}" != "1" ]]; then
            # 尝试常见安装路径 / 用户自行放置的 deb
            ORBBEC_DEB="$(find "${PARENT_DIR}" "${HOME}/Downloads" -maxdepth 3 -name '*Orbbec*SDK*.deb' 2>/dev/null | head -1 || true)"
            if [[ -n "${ORBBEC_DEB}" ]]; then
                log_info "发现 Orbbec deb: ${ORBBEC_DEB}"
                sudo dpkg -i "${ORBBEC_DEB}" || sudo apt-get install -f -y
                ORBBEC_INSTALLED=1
            fi
        fi
        if [[ "${ORBBEC_INSTALLED}" == "0" ]]; then
            log_warn "未检测到 Orbbec SDK（libOrbbecSDK.so）"
            log_warn "请从奥比中光官网下载 Linux SDK 安装后，再执行: pip install pyorbbecsdk"
            log_warn "下载页: https://orbbec.github.io/pyorbbecsdk/source/2_installation.html"
        fi
    fi

    # -------------------------------------------------------------------------
    # 5. Python 虚拟环境 + pip 依赖
    # -------------------------------------------------------------------------
    log_info ">>> [5/8] 安装 Python 依赖 (pip)..."

    if [[ "${USE_VENV}" == "1" ]]; then
        if [[ ! -d "${VENV_DIR}" ]]; then
            python3 -m venv "${VENV_DIR}"
        fi
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
        PIP_CMD=(python -m pip)
        log_ok "使用虚拟环境: ${VENV_DIR}"
    else
        PIP_CMD=(python3 -m pip)
        pip_install --upgrade pip setuptools wheel
    fi

    pip_install --upgrade pip setuptools wheel

    # 主依赖清单
    pip_install -r "${PROJECT_ROOT}/requirements-full.txt"

    # pinocchio：若 pin 失败，尝试 pinocchio / 系统包
    if ! python3 -c "import pinocchio" 2>/dev/null; then
        pip_install_optional pinocchio
    fi

    # CuPy（GPU 深度缩放，可选）
    CUPY_PKG="$(detect_cuda_for_cupy)"
    if [[ -n "${CUPY_PKG}" ]]; then
        log_info "检测到 NVIDIA CUDA，尝试安装 ${CUPY_PKG}..."
        pip_install_optional "${CUPY_PKG}"
    else
        log_warn "未检测到 CUDA，跳过 cupy（convert_to_target_format.py GPU 加速不可用）"
    fi

    # copasi_data_preprocess 为外部私有模块，不在公开 PyPI
    log_warn "copasi_data_preprocess / config 为 convert_to_target_format.py 外部模块，不在本安装脚本范围"

    log_ok "Python pip 依赖安装完成"

    # -------------------------------------------------------------------------
    # 6. biped_lcm_types 编译
    # -------------------------------------------------------------------------
    log_info ">>> [6/8] 编译 biped_lcm_types（Gen1 LCM 消息类型）..."

    if [[ "${SKIP_LCM_BUILD}" == "1" ]]; then
        log_warn "SKIP_LCM_BUILD=1，跳过"
    elif [[ ! -d "${BIPED_LCM_TYPES}" ]]; then
        log_warn "未找到 ${BIPED_LCM_TYPES}，跳过 LCM 类型编译"
        log_warn "Gen1 真机采集需要 factory_lerobot/biped_lcm_types"
    elif [[ -f "${BIPED_LCM_TYPES}/python/upper_body_data_package.py" ]]; then
        log_ok "biped_lcm_types Python 文件已存在，跳过编译"
    elif command -v lcm-gen &>/dev/null; then
        (
            cd "${BIPED_LCM_TYPES}"
            bash make_types.sh
        )
        log_ok "biped_lcm_types 编译完成"
    else
        log_warn "lcm-gen 不可用，无法编译 biped_lcm_types"
    fi

    # -------------------------------------------------------------------------
    # 7. 环境配置与启动脚本
    # -------------------------------------------------------------------------
    log_info ">>> [7/8] 写入环境配置..."

    write_env_file
    write_run_helper

    # -------------------------------------------------------------------------
    # 8. 验证
    # -------------------------------------------------------------------------
    log_info ">>> [8/8] 验证依赖..."

    FAIL=0
    verify_import "numpy"           "import numpy"           || FAIL=$((FAIL+1))
    verify_import "PySide6"         "import PySide6"         || FAIL=$((FAIL+1))
    verify_import "PyQt6"           "import PyQt6"           || true
    verify_import "pandas"          "import pandas"          || FAIL=$((FAIL+1))
    verify_import "h5py"            "import h5py"            || FAIL=$((FAIL+1))
    verify_import "pyarrow"         "import pyarrow"         || FAIL=$((FAIL+1))
    verify_import "requests"        "import requests"        || FAIL=$((FAIL+1))
    verify_import "minio"           "import minio"           || FAIL=$((FAIL+1))
    verify_import "torch"           "import torch"           || FAIL=$((FAIL+1))
    verify_import "PIL"             "from PIL import Image"  || FAIL=$((FAIL+1))
    verify_import "datasets"        "import datasets"        || FAIL=$((FAIL+1))
    verify_import "huggingface_hub" "import huggingface_hub" || FAIL=$((FAIL+1))
    verify_import "jsonlines"       "import jsonlines"       || FAIL=$((FAIL+1))
    verify_import "torchvision"     "import torchvision"     || FAIL=$((FAIL+1))
    verify_import "cv2"             "import cv2"             || FAIL=$((FAIL+1))
    verify_import "lcm"             "import lcm"             || true
    verify_import "grpc"            "import grpc"            || true
    verify_import "google.protobuf" "import google.protobuf" || true
    verify_import "scipy"           "import scipy"           || true
    verify_import "pinocchio"       "import pinocchio"       || true
    verify_import "pyrealsense2"    "import pyrealsense2"    || true
    verify_import "pyorbbecsdk"     "import pyorbbecsdk"     || true
    verify_import "pynput"          "import pynput"          || true
    verify_import "psutil"          "import psutil"          || true
    verify_import "matplotlib"      "import matplotlib"      || true

    # lerobot_factory
    if PYTHONPATH="${PROJECT_ROOT}/lerobot_factory:${FACTORY_LEROBOT}:${PROJECT_ROOT}/robot_kinemic/robot_kinemic" python3 -c \
        "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset" 2>/dev/null; then
        log_ok "  ✓ lerobot_factory / LeRobotDataset"
    else
        log_warn "  ✗ lerobot_factory / LeRobotDataset"
        FAIL=$((FAIL+1))
    fi

    # 字体
    if fc-list :lang=zh 2>/dev/null | head -1 | grep -q .; then
        log_ok "  ✓ 中文字体 (fontconfig)"
    else
        log_warn "  ✗ 中文字体可能缺失"
    fi

    # ffmpeg
    if command -v ffmpeg &>/dev/null; then
        log_ok "  ✓ ffmpeg"
    else
        log_warn "  ✗ ffmpeg 未找到"
    fi

    # ROS2
    if [[ -f /opt/ros/humble/setup.bash ]]; then
        log_ok "  ✓ ROS2 Humble (/opt/ros/humble)"
    else
        log_warn "  ✗ ROS2 Humble 未安装（ROS2 机器人采集需要）"
    fi

    echo ""
    echo "============================================================"
    if [[ "${FAIL}" -eq 0 ]]; then
        log_ok "核心依赖验证通过！"
    else
        log_warn "有 ${FAIL} 项核心依赖未通过，请查看上方日志"
    fi
    echo ""
    echo "后续使用:"
    echo "  1) 启动采集客户端:"
    echo "       ./run_collector.sh"
    echo "  或:"
    echo "       source .env_collector && source .venv/bin/activate && python3 main.py"
    echo ""
    echo "  2) ROS2 机器人采集前:"
    echo "       source /opt/ros/humble/setup.bash"
    echo ""
    echo "  3) 可选跳过项（重装时使用）:"
    echo "       SKIP_ROS2=1 SKIP_REALSENSE=1 SKIP_ORBBEC=1 ./install_all_deps.sh"
    echo "============================================================"
    echo ""
}

main "$@"
