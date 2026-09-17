#!/usr/bin/env bash
# 智身 Genisom Robot SDK 环境（ZSL-M1 / ZSL-L2）
# 用法: source scripts/genisom_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export GENISOM_SDK="${GENISOM_SDK:-$HOME/Desktop/genisom_robot_sdk-main}"
export GENISOM_ARCH="$(uname -m | sed 's/amd64/x86_64/')"
export GENISOM_LIB="$GENISOM_SDK/lib/$GENISOM_ARCH"
export GENISOM_WS="${GENISOM_WS:-$CLIENT_ROOT/ros2_ws}"

if [ ! -d "$GENISOM_LIB" ]; then
  echo "[Genisom] 找不到 SDK 库目录: $GENISOM_LIB" >&2
  echo "         请设置 GENISOM_SDK，例如: export GENISOM_SDK=/home/ck/Desktop/genisom_robot_sdk-main" >&2
else
  export LD_LIBRARY_PATH="$GENISOM_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
