#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据采集系统启动状态检查工具
用于测试和验证启动检测逻辑
"""

import time
import os
import sys
from pathlib import Path

def monitor_startup_log(log_file, timeout=60):
    """监控启动日志文件"""
    print(f"🔍 监控启动日志: {log_file}")
    
    # 需要检测的关键启动信息
    startup_markers = [
        "lcm unit is ok!",           # LCM单元初始化完成
        "audio is starting!",        # 音频系统初始化
        "等待开始录制"               # 完全初始化完成标志
    ]
    
    found_markers = set()
    last_file_size = 0
    start_time = time.time()
    
    print("等待检测的启动标记:")
    for i, marker in enumerate(startup_markers, 1):
        print(f"  {i}. {marker}")
    print()
    
    try:
        while time.time() - start_time < timeout:
            if not os.path.exists(log_file):
                print(f"   等待日志文件创建...")
                time.sleep(1)
                continue
                
            try:
                current_size = os.path.getsize(log_file)
                if current_size > last_file_size:
                    # 读取新增的内容
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(last_file_size)
                        new_content = f.read()
                        last_file_size = current_size
                        
                        # 显示新增内容（用于调试）
                        if new_content.strip():
                            print(f"[日志] {new_content.strip()}")
                        
                        # 检查是否包含关键标记
                        for marker in startup_markers:
                            if marker in new_content and marker not in found_markers:
                                found_markers.add(marker)
                                print(f"✓ 检测到启动标记: {marker}")
                        
                        # 如果找到所有关键标记，认为启动完成
                        if len(found_markers) >= len(startup_markers):
                            elapsed = time.time() - start_time
                            print(f"\n🎉 检测到所有启动标记！用时{elapsed:.1f}秒")
                            return True
                        elif len(found_markers) >= 2:
                            elapsed = time.time() - start_time
                            print(f"\n✅ 检测到主要启动标记！用时{elapsed:.1f}秒")
                            print(f"   检测到 {len(found_markers)}/{len(startup_markers)} 个标记")
                            return True
                            
            except Exception as e:
                print(f"❌ 读取日志文件出错: {e}")
                
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                print(f"   [{elapsed:.0f}s] 已检测到 {len(found_markers)}/{len(startup_markers)} 个启动标记")
                
            time.sleep(1)
        
        print(f"\n⚠️  监控超时 ({timeout}秒)")
        print(f"   最终检测到 {len(found_markers)}/{len(startup_markers)} 个启动标记")
        return False
        
    except KeyboardInterrupt:
        print(f"\n🛑 用户中断监控")
        return False

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python check_startup_status.py <log_file_path>")
        print("或者: python check_startup_status.py auto")
        sys.exit(1)
    
    if sys.argv[1] == "auto":
        # 自动查找最新的启动日志
        script_dir = Path(__file__).parent
        logs_dir = script_dir / "logs"
        
        if not logs_dir.exists():
            print("❌ 日志目录不存在，请先运行启动脚本")
            sys.exit(1)
        
        log_files = list(logs_dir.glob("*数据采集*_startup.log"))
        if not log_files:
            print("❌ 未找到数据采集启动日志文件")
            sys.exit(1)
        
        # 使用最新的日志文件
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        log_file = latest_log
    else:
        log_file = Path(sys.argv[1])
    
    print("🔍 数据采集系统启动状态检查器")
    print("=" * 50)
    print(f"监控日志文件: {log_file}")
    print()
    
    success = monitor_startup_log(log_file)
    
    if success:
        print("\n🎯 数据采集系统启动完成！")
    else:
        print("\n❌ 数据采集系统启动检测失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
