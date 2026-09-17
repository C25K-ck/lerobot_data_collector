#!/usr/bin/env bash
# Casbot 轮臂 W1：开启全身关节主动上报，并检查采集话题是否有数据
#
# 用法:
#   ./scripts/enable_casbot_w1_joint_push.sh
#   ./scripts/enable_casbot_w1_joint_push.sh --check   # 只检查话题，不调 Service
#
# 网络: 电脑与机器人同一网段，机器人默认 172.16.0.10
# 采集客户端请选择配置 Casbot-W1

set -euo pipefail

CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then
  CHECK_ONLY=1
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
info()  { echo -e "${GREEN}[Casbot-W1]${NC} $*"; }
warn()  { echo -e "${YELLOW}[Casbot-W1]${NC} $*"; }
error() { echo -e "${RED}[Casbot-W1]${NC} $*" >&2; }

source_ros() {
  if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    # shellcheck disable=SC1090
    set +u
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    set -u
    return 0
  fi
  local d
  for d in humble jazzy iron rolling; do
    if [ -f "/opt/ros/$d/setup.bash" ]; then
      set +u
      # shellcheck disable=SC1090
      source "/opt/ros/$d/setup.bash"
      set -u
      return 0
    fi
  done
  return 1
}

if ! source_ros; then
  error "未检测到 ROS2，请先 source 对应环境"
  exit 1
fi

STATE_TOPIC="/motion_unified/get/joint_state"
CMD_TOPIC="/motion_unified/control/Movej_transparent"

if [ "$CHECK_ONLY" != "1" ]; then
  info "尝试开启全身关节主动上报 (50 Hz)"
  if ros2 interface show crb_ros_msg/srv/MotionUnifiedControl >/dev/null 2>&1; then
    ros2 service call /motion_unified/control crb_ros_msg/srv/MotionUnifiedControl "{
      motion_unified: {
        func_name: Set_realtimePush_enable,
        int_name: ['hz', 'all'],
        int_val: [50, 1]
      }
    }"
    info "已请求 Set_realtimePush_enable"
  else
    warn "本机没有 crb_ros_msg，无法在这里调用 Service。"
    warn "请在机器人或已 source 厂家工作空间的终端执行："
    cat <<'EOF'
ros2 service call /motion_unified/control crb_ros_msg/srv/MotionUnifiedControl "{
  motion_unified: {
    func_name: Set_realtimePush_enable,
    int_name: ['hz', 'all'],
    int_val: [50, 1]
  }
}"
EOF
  fi
fi

info "检查状态话题: $STATE_TOPIC"
if timeout 3 ros2 topic echo --once "$STATE_TOPIC" >/tmp/casbot_w1_joint_once.txt 2>/dev/null; then
  names="$(python3 - <<'PY'
from pathlib import Path
text = Path("/tmp/casbot_w1_joint_once.txt").read_text(errors="ignore")
print("ok", "names" in text, "position" in text)
PY
)"
  info "已收到 $STATE_TOPIC"
  info "可启动采集客户端，配置选择 Casbot-W1"
else
  error "3 秒内没有收到 $STATE_TOPIC"
  error "请确认：1) 与机器人同网段  2) 已开启 realtime push  3) ros2 topic list 能看到该话题"
  exit 1
fi

info "指令话题(遥操时才有数据): $CMD_TOPIC"
ros2 topic info "$CMD_TOPIC" >/dev/null 2>&1 && info "已发现 $CMD_TOPIC" || warn "暂时看不到 $CMD_TOPIC，遥操开始后才会有发布者"
