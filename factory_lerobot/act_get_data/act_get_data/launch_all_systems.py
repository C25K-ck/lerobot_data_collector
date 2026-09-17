#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一启动脚本
同时启动：
1. save_act_data.py - 数据采集系统
2. warmup_display_orbbec.py - Orbbec相机预热显示
3. warmup_display_realsense_left.py - RealSense左相机预热显示
4. warmup_display_realsense_right.py - RealSense右相机预热显示
"""

import subprocess
import time
import signal
import sys
import os
from pathlib import Path
from conf import REPO_ID  # 导入REPO_ID

class SystemLauncher:
    def __init__(self):
        self.processes = []
        self.script_dir = Path(__file__).parent
        
        # 生成统一的REPO_ID，确保所有进程使用相同的ID
        self.unified_repo_id = REPO_ID
        print(f"🎯 统一REPO_ID: {self.unified_repo_id}")
        
        # 脚本路径配置
        self.scripts = {
            "数据采集": {
                "path": self.script_dir / "save_act_data.py",
                "cwd": self.script_dir,
                "env": {"FIXED_REPO_ID": self.unified_repo_id}
            },
            "Orbbec相机": {
                "path": self.script_dir / "orbbec_depth_720_show_queue.py", 
                "cwd": self.script_dir,
                "env": {"FIXED_REPO_ID": self.unified_repo_id}
            },
            "RealSense左相机": {
                "path": self.script_dir / "warmup_display_realsense_left.py",
                "cwd": self.script_dir, 
                "env": {"FIXED_REPO_ID": self.unified_repo_id}
            },
            "RealSense右相机": {
                "path": self.script_dir / "warmup_display_realsense_right.py",
                "cwd": self.script_dir,
                "env": {"FIXED_REPO_ID": self.unified_repo_id}
            }
        }
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """处理Ctrl+C等信号"""
        print(f"\n🛑 收到信号{signum}，正在关闭所有进程...")
        self.stop_all_processes()
        sys.exit(0)
        
    def start_process(self, name, script_info):
        """启动单个进程"""
        try:
            script_path = script_info["path"]
            cwd = script_info["cwd"]
            env = script_info.get("env")
            
            if not script_path.exists():
                print(f"❌ 脚本不存在: {script_path}")
                return None
                
            print(f"🚀 启动{name}: {script_path.name}")
            
            # 构建环境变量
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            # 启动进程 - 让子进程输出直接显示到终端
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(cwd),
                env=process_env,
                universal_newlines=True
            )
            
            return {
                "name": name,
                "process": process,
                "script_path": script_path
            }
            
        except Exception as e:
            print(f"❌ 启动{name}失败: {e}")
            return None
    
    def start_all_processes(self):
        """启动所有进程"""
        print("🎬 开始启动所有系统...")
        print("=" * 60)
        
        for name, script_info in self.scripts.items():
            process_info = self.start_process(name, script_info)
            if process_info:
                self.processes.append(process_info)
                time.sleep(1)  # 错开启动时间
            else:
                print(f"⚠️  跳过{name}")
        
        print("=" * 60)
        print(f"✅ 已启动{len(self.processes)}个进程")
        
        # 显示进程状态
        self.show_process_status()
        
    def show_process_status(self):
        """显示所有进程状态"""
        print("\n📊 进程状态:")
        print("-" * 50)
        for i, proc_info in enumerate(self.processes, 1):
            name = proc_info["name"]
            process = proc_info["process"]
            script_name = proc_info["script_path"].name
            
            if process.poll() is None:
                status = "🟢 运行中"
                pid = process.pid
                print(f"{i}. {name} ({script_name}) - {status} PID:{pid}")
            else:
                status = "🔴 已停止"
                return_code = process.returncode
                print(f"{i}. {name} ({script_name}) - {status} 退出码:{return_code}")
        print("-" * 50)
        
    def monitor_processes(self):
        """监控所有进程"""
        print("\n🔍 开始监控进程状态...")
        print("💡 按Ctrl+C停止所有进程")
        print("=" * 60)
        
        try:
            while True:
                # 检查进程状态
                running_count = 0
                stopped_processes = []
                
                for proc_info in self.processes:
                    process = proc_info["process"]
                    name = proc_info["name"]
                    
                    if process.poll() is None:
                        running_count += 1
                    else:
                        stopped_processes.append((name, process.returncode))
                
                # 报告停止的进程
                for name, return_code in stopped_processes:
                    if return_code == 0:
                        print(f"✅ {name} 正常退出")
                    else:
                        print(f"❌ {name} 异常退出 (退出码: {return_code})")
                
                # 如果所有进程都停止了，退出监控
                if running_count == 0:
                    print("🏁 所有进程已停止")
                    break
                    
                # 定期显示状态
                current_time = time.strftime("%H:%M:%S")
                print(f"[{current_time}] 运行中: {running_count}/{len(self.processes)} 个进程")
                
                time.sleep(5)  # 每5秒检查一次
                
        except KeyboardInterrupt:
            print(f"\n🛑 用户中断，停止监控")
            
    def stop_all_processes(self):
        """停止所有进程"""
        if not self.processes:
            return
            
        print("🛑 正在停止所有进程...")
        
        # 首先尝试优雅关闭
        for proc_info in self.processes:
            name = proc_info["name"]
            process = proc_info["process"]
            
            if process.poll() is None:  # 进程还在运行
                print(f"🔄 正在停止{name}...")
                try:
                    process.terminate()
                except:
                    pass
        
        # 等待进程结束
        time.sleep(3)
        
        # 强制杀死仍在运行的进程
        for proc_info in self.processes:
            name = proc_info["name"]
            process = proc_info["process"]
            
            if process.poll() is None:  # 进程仍在运行
                print(f"💀 强制杀死{name}...")
                try:
                    process.kill()
                except:
                    pass
        
        # 等待所有进程完全结束
        for proc_info in self.processes:
            try:
                proc_info["process"].wait(timeout=2)
            except:
                pass
                
        print("✅ 所有进程已停止")
        
    def run(self):
        """运行主流程"""
        print("🎯 统一系统启动器 (修复版)")
        print("🔧 已修复15fps限制问题")
        print("📋 将启动以下系统:")
        for i, name in enumerate(self.scripts.keys(), 1):
            print(f"  {i}. {name}")
        print()
        
        try:
            # 启动所有进程
            self.start_all_processes()
            
            if not self.processes:
                print("❌ 没有成功启动任何进程")
                return
            
            # 监控进程
            self.monitor_processes()
            
        except Exception as e:
            print(f"❌ 运行过程中发生错误: {e}")
            
        finally:
            # 清理所有进程
            self.stop_all_processes()

def main():
    """主函数"""
    launcher = SystemLauncher()
    launcher.run()

if __name__ == "__main__":
    main() 
