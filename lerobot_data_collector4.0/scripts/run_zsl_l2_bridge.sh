#!/usr/bin/env bash
# 智身 ZSL-L2：启动 Genisom SDK → ROS2 桥接
export ZS_SERIES=l2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_zsl_m1_bridge.sh" l2 "$@"
