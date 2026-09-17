#!/usr/bin/env bash
# 智身 ZSL-M1 / ZSL-L2：编译并启动 Genisom SDK → ROS2 桥接
#
# 用法:
#   ./scripts/run_zsl_m1_bridge.sh              # 启动 ZSL-M1 桥接
#   ./scripts/run_zsl_l2_bridge.sh              # 启动 ZSL-L2 桥接
#   ./scripts/run_zsl_m1_bridge.sh --build      # 强制重新编译
#   ROBOT_IP=192.168.234.1 ./scripts/run_zsl_m1_bridge.sh
#
# 环境变量:
#   GENISOM_SDK   默认 $HOME/Desktop/genisom_robot_sdk-main
#   ROBOT_IP      默认 192.168.234.1
#   ROBOT_PORT    默认 8082

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/genisom_env.sh"

SERIES="${ZS_SERIES:-m1}"
FORCE_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --build) FORCE_BUILD=1 ;;
    m1|l2) SERIES="$arg" ;;
  esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
TAG="ZSL-${SERIES^^}"
info()  { echo -e "${GREEN}[$TAG]${NC} $*"; }
warn()  { echo -e "${YELLOW}[$TAG]${NC} $*"; }
error() { echo -e "${RED}[$TAG]${NC} $*" >&2; }

source_colcon_setup() {
  local setup_file="$1"
  if [ ! -f "$setup_file" ]; then
    error "找不到 setup.bash: $setup_file"
    return 1
  fi
  export COLCON_TRACE="${COLCON_TRACE:-}"
  export COLCON_PREFIX_PATH="${COLCON_PREFIX_PATH:-}"
  export COLCON_PYTHON_EXECUTABLE="${COLCON_PYTHON_EXECUTABLE:-}"
  set +u
  # shellcheck disable=SC1090
  source "$setup_file"
  set -u
}

detect_ros_distro() {
  if [ -n "${ROS_DISTRO:-}" ]; then
    echo "$ROS_DISTRO"
    return 0
  fi
  local d
  for d in humble jazzy iron rolling; do
    if [ -f "/opt/ros/$d/setup.bash" ]; then
      echo "$d"
      return 0
    fi
  done
  echo "unknown"
}

ROS_DISTRO_NAME="$(detect_ros_distro)"
if [ "$ROS_DISTRO_NAME" = "unknown" ]; then
  error "未检测到 ROS2，请先安装 Humble/Jazzy"
  exit 1
fi
source_colcon_setup "/opt/ros/$ROS_DISTRO_NAME/setup.bash"

if [ ! -f "$GENISOM_SDK/include/robot_sdk/sdk_client.hpp" ]; then
  error "找不到 Genisom SDK: $GENISOM_SDK"
  error "请设置: export GENISOM_SDK=/path/to/genisom_robot_sdk-main"
  exit 1
fi

PKG_DIR="$GENISOM_WS/src/zs_dog_genisom"
if [ ! -d "$PKG_DIR" ]; then
  error "找不到 ROS2 包: $PKG_DIR"
  exit 1
fi

NEED_BUILD=0
if [ "$FORCE_BUILD" = "1" ]; then
  NEED_BUILD=1
elif [ ! -f "$GENISOM_WS/install/zs_dog_genisom/share/zs_dog_genisom/package.bash" ] && \
     [ ! -f "$GENISOM_WS/install/setup.bash" ]; then
  NEED_BUILD=1
elif [ ! -x "$GENISOM_WS/install/zs_dog_genisom/lib/zs_dog_genisom/zs_dog_bridge_node" ]; then
  NEED_BUILD=1
fi

if [ "$NEED_BUILD" = "1" ]; then
  info "编译 zs_dog_genisom ..."
  mkdir -p "$GENISOM_WS"
  cd "$GENISOM_WS"
  colcon build --packages-select zs_dog_genisom --event-handlers console_direct+
fi

source_colcon_setup "$GENISOM_WS/install/setup.bash"

ROBOT_IP="${ROBOT_IP:-192.168.234.1}"
ROBOT_PORT="${ROBOT_PORT:-8082}"
info "启动桥接: series=$SERIES ip=$ROBOT_IP:$ROBOT_PORT"
info "客户端请选择配置 ZSL-${SERIES^^}"
exec ros2 run zs_dog_genisom zs_dog_bridge_node --ros-args \
  -p series:="$SERIES" \
  -p robot_ip:="$ROBOT_IP" \
  -p robot_port:="$ROBOT_PORT"
