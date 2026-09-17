import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
logger = logging.getLogger(__name__)

# 注入 lerobot_factory 到 sys.path，保证可导入顶层包 "lerobot"
# 支持多种路径查找方式（按优先级排序）：
# 1. 项目内的 lerobot_factory（最高优先级）
# 2. 项目父目录下的 factory_lerobot/lerobot_factory
# 3. 环境变量 LEROBOT_FACTORY_PATH
# 4. 其他常见路径
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    # lerobot_data_collector/lerobot_data_collector/collector.py
    # 向上1级到 lerobot_data_collector 项目根目录
    _proj_root = os.path.dirname(_this_dir)  # lerobot_data_collector
    _proj_parent = os.path.dirname(_proj_root)  # 项目父目录
    
    # 优先查找路径（按优先级排序）
    possible_paths = [
        # 1. 项目内的 lerobot_factory（最高优先级）
        os.path.join(_proj_root, "lerobot_factory"),
        # 2. 项目父目录下的 factory_lerobot/lerobot_factory
        os.path.join(_proj_parent, "factory_lerobot", "lerobot_factory"),
        # 3. 项目父目录下的 lerobot_factory
        os.path.join(_proj_parent, "lerobot_factory"),
        # 4. 环境变量指定的路径
        os.environ.get("LEROBOT_FACTORY_PATH", None),
        # 5. 用户目录下的路径
        os.path.expanduser("~/factory_lerobot/lerobot_factory"),
    ]
    
    _lerobot_factory = None
    for path in possible_paths:
        if path and os.path.exists(path):
            _lerobot_factory = path
            break
    
    if _lerobot_factory:
        if _lerobot_factory not in sys.path:
            sys.path.insert(0, _lerobot_factory)
        logger.info("lerobot_factory 路径：%s", _lerobot_factory)
    else:
        logger.warning("未找到 lerobot_factory，请检查目录或设置环境变量 LEROBOT_FACTORY_PATH")
except Exception as e:
    logger.warning("无法添加 lerobot_factory 到 sys.path：%s", e)

# 导入本地模块（支持包运行与脚本直跑）
try:
    from .lerobot_unit import lerobotUnit
    from .lcm_unit import lcmUnit
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    sys.path.insert(0, os.path.dirname(__file__))
    from lerobot_unit import lerobotUnit
    try:
        from lcm_unit import lcmUnit
    except ImportError:
        # LCM适配器是可选的
        lcmUnit = None
        logger.warning("LCM 适配器未实现，将仅使用 ROS2 数据源")

# 采集核心逻辑独立到 collector_core，便于复用与测试
from .collector_core import (  # noqa: E402
    TopicSpec,
    DatasetFieldMapping,
    CollectorConfig,
    DataHub,
    ROS2Adapter,
    LCMAdapter,
    Collector,
)

try:
    from .warmup_display_realsense_right import RealSenseRightCameraManager
except ImportError:
    RealSenseRightCameraManager = None  # 摄像头为可选组件

try:
    from .warmup_display_realsense_left import RealSenseLeftCameraManager
except ImportError:
    RealSenseLeftCameraManager = None  # 左手摄像头为可选组件

try:
    from .warmup_display_realsense_head import RealSenseHeadCameraProcess
except ImportError:
    RealSenseHeadCameraProcess = None  # 头部摄像头可选

try:
    from .orbbec_head_camera_manager import OrbbecHeadCameraManager
except ImportError:
    OrbbecHeadCameraManager = None  # 奥比中光头部摄像头为可选组件


# --- GUI ---
# 优先使用Qt（PySide6）以获得更好的字体渲染
# 如果Qt 导入失败，再回退到Tkinter
USE_QT = True
try:
    from .qt_gui import main_qt
except Exception as e:  # pragma: no cover - 防御式回退
    # 任何原因导致 Qt GUI 不可用时，都回退到 Tkinter，并打印原因
    import logging as _logging
    _logging.getLogger(__name__).warning("Qt GUI 不可用，回退到 Tkinter：%s", e)
    USE_QT = False

# 确保Tkinter总是可用（作为备选）
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk  # moved up
import numpy as np


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LeRobot 数据采集")
        # 更大的窗口，更优雅的比例（增加高度以容纳数据集列表和摄像头视图）
        self.root.geometry("1200x900")
        # macOS风格的背景色 - 柔和的灰白色
        self.root.configure(bg="#FAFAFA")
        
        # 设计系统配色（类似macOS/iOS的设计语言）
        self.colors = {
            'bg_primary': '#FAFAFA',      # 主背景（柔和的灰白）
            'bg_card': '#FFFFFF',          # 卡片背景（纯白）
            'bg_secondary': '#F5F5F7',     # 次要背景
            'text_primary': '#1D1D1F',     # 主文本（深灰黑）
            'text_secondary': '#86868B',   # 次要文本（柔和的灰）
            'text_tertiary': '#AEAEB2',    # 三级文本
            'accent_blue': '#007AFF',      # 主色调（iOS蓝）
            'accent_green': '#34C759',      # 成功色（iOS绿）
            'accent_red': '#FF3B30',       # 警告/错误色
            'accent_orange': '#FF9500',    # 提示色
            'border': '#E5E5EA',           # 边框色（非常淡的灰）
            'shadow': '#E8E8ED',           # 阴影色
        }
        
        # 优化字体渲染（抗锯齿和平滑度）
        try:
            # 设置DPI感知，改善高分辨率显示
            current_scale = self.root.tk.call('tk', 'scaling', '.')
            # 如果缩放因子过小，适当调整以改善字体显示
            if current_scale < 1.0:
                self.root.tk.call('tk', 'scaling', 1.0)
            # 尝试设置字体渲染选项（某些系统支持）
            try:
                # X11系统可能需要设置字体渲染选项
                import os
                if 'DISPLAY' in os.environ:
                    # Linux X11环境
                    self.root.tk.call('option', 'add', '*Font', 'default', 'userDefault')
            except:
                pass
        except:
            pass
        
        # 专业字体选择策略 - 优先使用高质量现代字体
        # 参考：专业软件（如VS Code、Figma、Notion）都使用高质量系统字体
        try:
            import tkinter.font as tkfont
            import subprocess
            
            chinese_font = None
            fonts = tkfont.families()
            fonts_lower = [f.lower() for f in fonts]
            
            # 专业软件常用字体优先级列表（按质量排序）
            professional_fonts = [
                'noto sans cjk sc', 'noto sans cjk', 'noto sans',
                'source han sans sc', 'source han sans', 'source han sans cn',
                '思源黑体', '思源黑体 cn', '思源黑体 sc',
                'microsoft yahei', 'microsoft yahei ui', '微软雅黑', '微软雅黑 ui',
                'inter', 'inter ui',
                    'wenquanyi micro hei', '文泉驿微米黑',
                    'wenquanyi zen hei', '文泉驿正黑',
                'pingfang sc', '苹方',
                'hiragino sans gb', '冬青黑体',
                    'dejavu sans', 'liberation sans',
                'sans', 'sans-serif',
                ]
                
            # 精确匹配优先
            for candidate in professional_fonts:
                candidate_lower = candidate.lower().strip()
                for i, font_lower in enumerate(fonts_lower):
                    if font_lower == candidate_lower:
                        chinese_font = fonts[i]
                        logger.info("Tkinter 字体选择：%s", chinese_font)
                        break
                if chinese_font:
                    break
                
            # 如果精确匹配失败，尝试模糊匹配
            if not chinese_font:
                keywords = ['noto', 'source han', 'microsoft yahei', 'inter',
                            'wenquanyi', 'pingfang', 'hiragino', 'dejavu']
                for kw in keywords:
                    for i, font_lower in enumerate(fonts_lower):
                        if kw in font_lower:
                            chinese_font = fonts[i]
                            logger.debug("Tkinter 字体模糊匹配：%s", chinese_font)
                            break
                    if chinese_font:
                        break
                
            # 使用系统 fc-match 推荐（Linux）
            if not chinese_font:
                try:
                    result = subprocess.run(
                        ['fc-match', '-s', ':lang=zh', 'sans'],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if lines:
                            first_match = lines[0].split(':')[0].strip()
                            if first_match:
                                for i, font_lower in enumerate(fonts_lower):
                                    if first_match.lower() == font_lower:
                                        chinese_font = fonts[i]
                                        logger.info("Tkinter 字体系统推荐：%s", chinese_font)
                                        break
                                if not chinese_font:
                                    chinese_font = first_match
                                    logger.debug("Tkinter 字体采用 fc-match 默认值：%s", chinese_font)
                except Exception:
                    pass
                
            # 最后的备选：使用 Tkinter 默认字体
            if not chinese_font:
                try:
                    default_font = tkfont.nametofont('TkDefaultFont')
                    chinese_font = default_font.cget('family')
                    logger.info("Tkinter 字体使用默认：%s", chinese_font)
                except Exception:
                    chinese_font = 'Sans'
                    logger.info("Tkinter 字体使用通用备选 Sans")
                
        except Exception as e:
            logger.debug("Tkinter 字体选择异常：%s", e)
            try:
                default_font = tkfont.nametofont('TkDefaultFont')
                chinese_font = default_font.cget('family')
            except:
                chinese_font = 'Sans'
        
        # 配置TTK样式 - 更优雅的设计
        style = ttk.Style()
        style.theme_use('clam')
        
        # 自定义样式 - 使用更大的字体和更好的行高（专业软件通常使用更大的字体）
        # 参考：VS Code使用13-14px，Figma使用14-16px，Notion使用14-16px
        style.configure('Title.TLabel', 
                       font=(chinese_font, 22, 'normal'),  # 增大标题字体
                       background=self.colors['bg_primary'],
                       foreground=self.colors['text_primary'])
        style.configure('Header.TLabel', 
                       font=(chinese_font, 13, 'normal'),  # 增大头部字体
                       background=self.colors['bg_card'],
                       foreground=self.colors['text_primary'])
        style.configure('Subtext.TLabel',
                       font=(chinese_font, 11, 'normal'),  # 增大副文本字体
                       background=self.colors['bg_card'],
                       foreground=self.colors['text_secondary'])

        self.cfg_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪")
        self.duration_var = tk.StringVar(value="60")
        self.status_indicator_color = self.colors['text_tertiary']  # 状态指示器颜色
        self.collection_start_time: Optional[float] = None
        self.collection_duration: float = 0.0  # 秒
        self.collection_target_frames: int = 0
        self.collection_fps: int = 30
        self.chinese_font = chinese_font

        self.collector: Optional[Collector] = None
        self.right_camera_manager: Optional[
            "RealSenseRightCameraManager"
        ] = None
        self.head_camera_manager: Optional["RealSenseHeadCameraProcess"] = None
        self.current_dataset_home: Optional[Path] = None
        self._camera_preview_job: Optional[str] = None
        self._head_camera_recording: bool = False

        # 在软件启动时生成唯一的repo_id（基于时间戳）
        # 同一次运行中多次采集都使用同一个repo_id
        # 关闭软件再启动时会生成新的repo_id
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 尝试从conf.py获取任务代码作为前缀，如果没有则使用默认值
        try:
            from .conf import TASK_CODE
            task_code = TASK_CODE
        except ImportError:
            task_code = "data_collection"
        self.session_repo_id = f"{task_code}_{timestamp}"
        logger.info("本次会话数据集 ID：%s", self.session_repo_id)

        # 主容器 - 使用Notebook实现标签页
        main_notebook = ttk.Notebook(root)
        main_notebook.pack(fill=tk.BOTH, expand=True, padx=32, pady=32)
        
        # 标签页1: 数据采集
        collect_frame = tk.Frame(main_notebook, bg=self.colors['bg_primary'])
        main_notebook.add(collect_frame, text="数据采集")
        
        # 标签页2: 数据集管理
        datasets_frame = tk.Frame(main_notebook, bg=self.colors['bg_primary'])
        main_notebook.add(datasets_frame, text="数据集管理")
        
        # 主容器 - 增加更多的留白
        main_frame = tk.Frame(collect_frame, bg=self.colors['bg_primary'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题区域 - 极简设计
        title_frame = tk.Frame(main_frame, bg=self.colors['bg_primary'])
        title_frame.pack(fill=tk.X, pady=(0, 32))
        title_label = ttk.Label(title_frame, text="数据采集", style='Title.TLabel')
        title_label.pack(anchor=tk.W)
        
        # 副标题（可选）
        subtitle_label = tk.Label(title_frame, 
                                 text="LeRobot 通用数据采集平台",
                                 font=(self.chinese_font, 12),  # 增大副标题字体
                                 bg=self.colors['bg_primary'],
                                 fg=self.colors['text_secondary'])
        subtitle_label.pack(anchor=tk.W, pady=(6, 0))  # 增加间距

        # 配置区域（优雅的卡片设计）
        config_frame = tk.Frame(main_frame, 
                               bg=self.colors['bg_card'],
                               relief=tk.FLAT,
                               bd=0)
        config_frame.pack(fill=tk.X, pady=(0, 16))
        
        # 卡片阴影效果（使用边框模拟）
        shadow_frame = tk.Frame(main_frame, bg=self.colors['shadow'], height=2)
        shadow_frame.place(in_=config_frame, relx=0, rely=1, relwidth=1, height=1)
        
        config_inner = tk.Frame(config_frame, bg=self.colors['bg_card'])
        config_inner.pack(padx=24, pady=20)

        # 配置文件选择 - 下拉菜单设计
        cfg_label = ttk.Label(config_inner, text="配置文件", style='Header.TLabel')
        cfg_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        cfg_entry_frame = tk.Frame(config_inner, bg=self.colors['bg_card'])
        cfg_entry_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=(0, 20))
        
        # 扫描配置文件目录
        config_files = self._scan_config_files()
        
        # 下拉菜单容器（带边框）
        combobox_container = tk.Frame(cfg_entry_frame, 
                                     bg=self.colors['border'],
                                     bd=0,
                                     highlightthickness=0)
        combobox_container.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=0)
        
        # 下拉菜单（Combobox）
        self.cfg_combobox = ttk.Combobox(combobox_container,
                                        textvariable=self.cfg_path_var,
                                        values=config_files,
                                        font=(self.chinese_font, 12),
                                        state="readonly",  # 只读模式，只能从下拉列表选择
                                        width=50)
        self.cfg_combobox.pack(fill=tk.BOTH, expand=True, padx=1, pady=1, ipady=12, ipadx=14)
        
        # 如果有配置文件，选择第一个
        if config_files:
            self.cfg_path_var.set(config_files[0])
        
        # 浏览按钮 - 用于选择其他位置的配置文件
        browse_btn = tk.Button(cfg_entry_frame, 
                               text="浏览其他...",
                               command=self.browse,
                               bg=self.colors['accent_blue'],
                               fg="white",
                               font=(self.chinese_font, 12, 'normal'),
                               relief=tk.FLAT,
                               bd=0,
                               cursor="hand2",
                               padx=20,
                               pady=12,
                               activebackground="#0051D5",
                               activeforeground="white")
        browse_btn.pack(side=tk.LEFT, padx=(12, 0))
        
        # 添加悬停效果
        def on_browse_enter(e):
            browse_btn.config(bg="#0051D5")
        def on_browse_leave(e):
            browse_btn.config(bg=self.colors['accent_blue'])
        browse_btn.bind("<Enter>", on_browse_enter)
        browse_btn.bind("<Leave>", on_browse_leave)
        
        # 刷新按钮 - 用于刷新配置文件列表
        refresh_cfg_btn = tk.Button(cfg_entry_frame,
                                    text="刷新",
                                    command=self._refresh_config_files,
                                    bg=self.colors['text_secondary'],
                                    fg="white",
                                    font=(self.chinese_font, 11, 'normal'),
                                    relief=tk.FLAT,
                                    bd=0,
                                    cursor="hand2",
                                    padx=16,
                                    pady=12,
                                    activebackground="#6E6E73",
                                    activeforeground="white")
        refresh_cfg_btn.pack(side=tk.LEFT, padx=(8, 0))
        
        # 刷新按钮悬停效果
        def on_refresh_enter(e):
            refresh_cfg_btn.config(bg="#6E6E73")
        def on_refresh_leave(e):
            refresh_cfg_btn.config(bg=self.colors['text_secondary'])
        refresh_cfg_btn.bind("<Enter>", on_refresh_enter)
        refresh_cfg_btn.bind("<Leave>", on_refresh_leave)

        # 采集时长设置 - 更优雅的布局
        duration_label = ttk.Label(config_inner, text="采集帧数", style='Header.TLabel')
        duration_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 10))
        
        duration_frame = tk.Frame(config_inner, bg=self.colors['bg_card'])
        duration_frame.grid(row=3, column=0, sticky=tk.W)
        
        # 时长输入框容器
        duration_container = tk.Frame(duration_frame, 
                                     bg=self.colors['border'],
                                     bd=0)
        duration_container.pack(side=tk.LEFT, ipady=0)
        
        duration_entry = tk.Entry(duration_container,
                                 textvariable=self.duration_var,
                                 width=8,
                                 font=(self.chinese_font, 14, 'normal'),  # 增大数字输入字体
                                 relief=tk.FLAT,
                                 bd=0,
                                 bg=self.colors['bg_card'],
                                 fg=self.colors['text_primary'],
                                 highlightthickness=0,
                                 justify=tk.CENTER,
                                 insertbackground=self.colors['accent_blue'])
        duration_entry.pack(fill=tk.BOTH, expand=True, padx=1, pady=1, ipady=12, ipadx=14)  # 增加内边距
        
        # 单位标签
        unit_label = tk.Label(duration_frame,
                             text="帧",
                             font=(self.chinese_font, 12),  # 增大单位标签字体
                             bg=self.colors['bg_card'],
                             fg=self.colors['text_secondary'])
        unit_label.pack(side=tk.LEFT, padx=(12, 0), pady=0)
        
        # 提示文本
        hint_label = ttk.Label(config_inner,
                              text="建议帧数：1800（约60秒@30fps）",
                              style='Subtext.TLabel')
        hint_label.grid(row=4, column=0, sticky=tk.W, pady=(8, 0))

        # 操作按钮区域 - 大按钮设计（类似iOS的大按钮）
        button_frame = tk.Frame(main_frame, bg=self.colors['bg_primary'])
        button_frame.pack(fill=tk.X, pady=(24, 0))
        
        # 按钮容器（用于居中）
        button_container = tk.Frame(button_frame, bg=self.colors['bg_primary'])
        button_container.pack(expand=True)
        
        button_font = (self.chinese_font, 14, 'normal')  # 增大按钮字体（专业软件通常14-16px）
        
        # 开始按钮 - iOS风格的大按钮
        start_btn = tk.Button(button_container,
                             text="开始采集",
                             command=self.start,
                             bg=self.colors['accent_blue'],
                             fg="white",
                             font=button_font,
                             relief=tk.FLAT,
                             bd=0,
                             cursor="hand2",
                             padx=40,
                             pady=14,
                             activebackground="#0051D5",
                             activeforeground="white")
        start_btn.pack(side=tk.LEFT, padx=(0, 12))
        
        # 开始按钮悬停效果
        def on_start_enter(e):
            start_btn.config(bg="#0051D5")
        def on_start_leave(e):
            start_btn.config(bg=self.colors['accent_blue'])
        start_btn.bind("<Enter>", on_start_enter)
        start_btn.bind("<Leave>", on_start_leave)
        
        # 结束按钮 - 次要按钮（更柔和的颜色）
        stop_btn = tk.Button(button_container,
                             text="结束并保存",
                             command=self.stop,
                             bg=self.colors['text_secondary'],
                             fg="white",
                             font=button_font,
                             relief=tk.FLAT,
                             bd=0,
                             cursor="hand2",
                             padx=40,
                             pady=14,
                             activebackground="#6E6E73",
                             activeforeground="white")
        stop_btn.pack(side=tk.LEFT)
        
        # 结束按钮悬停效果
        def on_stop_enter(e):
            stop_btn.config(bg="#6E6E73")
        def on_stop_leave(e):
            stop_btn.config(bg=self.colors['text_secondary'])
        stop_btn.bind("<Enter>", on_stop_enter)
        stop_btn.bind("<Leave>", on_stop_leave)

        # 摄像头视频显示区域
        camera_frame = tk.Frame(main_frame,
                               bg=self.colors['bg_card'],
                               relief=tk.FLAT,
                               bd=0)
        camera_frame.pack(fill=tk.X, pady=(24, 0))
        
        camera_inner = tk.Frame(camera_frame, bg=self.colors['bg_card'])
        camera_inner.pack(padx=24, pady=20)
        
        # 摄像头标题
        camera_title = ttk.Label(camera_inner, text="摄像头视图", style='Header.TLabel')
        camera_title.pack(anchor=tk.W, pady=(0, 16))
        
        # 三个摄像头视图容器
        cameras_container = tk.Frame(camera_inner, bg=self.colors['bg_card'])
        cameras_container.pack(fill=tk.X)
        
        # 左手摄像头（左侧）
        left_hand_camera_frame = tk.Frame(cameras_container,
                                          bg=self.colors['bg_secondary'],
                                          relief=tk.FLAT,
                                          bd=2,
                                          highlightbackground=self.colors['border'],
                                          highlightthickness=2)
        left_hand_camera_frame.pack(side=tk.LEFT, padx=(0, 12), fill=tk.BOTH, expand=True)
        
        left_hand_camera_label = tk.Label(left_hand_camera_frame,
                                         text="左手",
                                         font=(self.chinese_font, 11),
                                         bg=self.colors['bg_secondary'],
                                         fg=self.colors['text_secondary'])
        left_hand_camera_label.pack(pady=(8, 8))
        
        self.left_hand_camera_display = tk.Label(left_hand_camera_frame,
                                                 text="等待图像...",
                                                 font=(self.chinese_font, 10),
                                                 bg=self.colors['bg_primary'],
                                                 fg=self.colors['text_secondary'],
                                                 width=40,
                                                 height=15,
                                                 relief=tk.SOLID,
                                                 bd=1)
        self.left_hand_camera_display.pack(padx=8, pady=(0, 8))
        
        # 头部摄像头（中间）
        head_camera_frame = tk.Frame(cameras_container,
                                    bg=self.colors['bg_secondary'],
                                    relief=tk.FLAT,
                                    bd=2,
                                    highlightbackground=self.colors['border'],
                                    highlightthickness=2)
        head_camera_frame.pack(side=tk.LEFT, padx=(0, 12), fill=tk.BOTH, expand=True)
        
        head_camera_label = tk.Label(head_camera_frame,
                                    text="头部",
                                    font=(self.chinese_font, 11),
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text_secondary'])
        head_camera_label.pack(pady=(8, 8))
        
        self.head_camera_display = tk.Label(head_camera_frame,
                                           text="等待图像...",
                                           font=(self.chinese_font, 10),
                                           bg=self.colors['bg_primary'],
                                           fg=self.colors['text_secondary'],
                                           width=40,
                                           height=15,
                                           relief=tk.SOLID,
                                           bd=1)
        self.head_camera_display.pack(padx=8, pady=(0, 8))
        
        # 右手摄像头（右侧）
        right_hand_camera_frame = tk.Frame(cameras_container,
                                           bg=self.colors['bg_secondary'],
                                           relief=tk.FLAT,
                                           bd=2,
                                           highlightbackground=self.colors['border'],
                                           highlightthickness=2)
        right_hand_camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_hand_camera_label = tk.Label(right_hand_camera_frame,
                                           text="右手",
                                           font=(self.chinese_font, 11),
                                           bg=self.colors['bg_secondary'],
                                           fg=self.colors['text_secondary'])
        right_hand_camera_label.pack(pady=(8, 8))
        
        self.right_hand_camera_display = tk.Label(right_hand_camera_frame,
                                                  text="等待图像...",
                                                  font=(self.chinese_font, 10),
                                                  bg=self.colors['bg_primary'],
                                                  fg=self.colors['text_secondary'],
                                                  width=40,
                                                  height=15,
                                                  relief=tk.SOLID,
                                                  bd=1)
        self.right_hand_camera_display.pack(padx=8, pady=(0, 8))

        # 状态显示区域 - 优雅的状态卡片
        status_frame = tk.Frame(main_frame, 
                               bg=self.colors['bg_card'],
                               relief=tk.FLAT,
                               bd=0)
        status_frame.pack(fill=tk.X, pady=(24, 0))
        
        status_inner = tk.Frame(status_frame, bg=self.colors['bg_card'])
        status_inner.pack(padx=24, pady=20)
        
        # 状态标题和指示器行
        status_header = tk.Frame(status_inner, bg=self.colors['bg_card'])
        status_header.pack(fill=tk.X, pady=(0, 12))
        
        status_title = ttk.Label(status_header, text="状态", style='Header.TLabel')
        status_title.pack(side=tk.LEFT)
        
        # 状态指示器（圆形LED灯效果）
        self.status_indicator = tk.Canvas(status_header,
                                         width=10,
                                         height=10,
                                         bg=self.colors['bg_card'],
                                         highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=(12, 0))
        self._draw_status_indicator(self.colors['text_tertiary'])
        
        # 状态文本显示 - 大字体，清晰易读
        self.status_display = tk.Label(status_inner,
                                      textvariable=self.status_var,
                                      font=(self.chinese_font, 14, 'normal'),  # 增大状态字体
                                      bg=self.colors['bg_card'],
                                      fg=self.colors['text_primary'],
                                      anchor=tk.W,
                                      justify=tk.LEFT,
                                      wraplength=600)
        self.status_display.pack(fill=tk.X, anchor=tk.W, pady=(0, 4))  # 增加间距
        
        # 进度信息（如果正在采集）
        self.progress_label = tk.Label(status_inner,
                                      text="",
                                      font=(self.chinese_font, 11),  # 增大进度信息字体
                                      bg=self.colors['bg_card'],
                                      fg=self.colors['text_secondary'],
                                      anchor=tk.W)
        self.progress_label.pack(fill=tk.X, anchor=tk.W, pady=(8, 0))
        
        # === 数据集管理标签页 ===
        self._create_datasets_tab(datasets_frame)

    def _create_datasets_tab(self, parent: tk.Frame) -> None:
        """创建数据集管理标签页"""
        # 数据集管理区域
        datasets_layout = tk.Frame(parent, bg=self.colors['bg_primary'])
        datasets_layout.pack(fill=tk.BOTH, expand=True, padx=32, pady=32)
        
        # 标题
        datasets_title = ttk.Label(datasets_layout, text="已采集数据集", style='Title.TLabel')
        datasets_title.pack(anchor=tk.W, pady=(0, 16))
        
        # 数据集列表卡片
        list_frame = tk.Frame(datasets_layout,
                             bg=self.colors['bg_card'],
                             relief=tk.FLAT,
                             bd=0)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 16))
        
        # 工具栏
        toolbar = tk.Frame(list_frame, bg=self.colors['bg_card'])
        toolbar.pack(fill=tk.X, padx=20, pady=(20, 12))
        
        refresh_btn = tk.Button(toolbar,
                               text="刷新",
                               command=self._refresh_datasets,
                               bg=self.colors['accent_blue'],
                               fg="white",
                               font=(self.chinese_font, 11, 'normal'),
                               relief=tk.FLAT,
                               bd=0,
                               cursor="hand2",
                               padx=20,
                               pady=8,
                               activebackground="#0051D5")
        refresh_btn.pack(side=tk.LEFT)
        
        toolbar.pack_propagate(False)
        toolbar.config(height=40)
        
        # 数据集列表（使用Treeview）
        list_container = tk.Frame(list_frame, bg=self.colors['bg_card'])
        list_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # 创建Treeview
        columns = ("数据集ID", "Episodes", "总帧数", "创建时间", "状态")
        self.dataset_tree = ttk.Treeview(list_container,
                                        columns=columns,
                                        show="headings",
                                        height=12)
        
        # 配置列
        self.dataset_tree.column("数据集ID", width=200, anchor=tk.W)
        self.dataset_tree.column("Episodes", width=80, anchor=tk.CENTER)
        self.dataset_tree.column("总帧数", width=100, anchor=tk.CENTER)
        self.dataset_tree.column("创建时间", width=180, anchor=tk.CENTER)
        self.dataset_tree.column("状态", width=100, anchor=tk.CENTER)
        
        # 设置列标题
        for col in columns:
            self.dataset_tree.heading(col, text=col)
        
        # 滚动条
        scrollbar_y = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.dataset_tree.yview)
        scrollbar_x = ttk.Scrollbar(list_container, orient=tk.HORIZONTAL, command=self.dataset_tree.xview)
        self.dataset_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # 布局
        self.dataset_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        list_container.grid_rowconfigure(0, weight=1)
        list_container.grid_columnconfigure(0, weight=1)
        
        # 操作按钮区域
        action_frame = tk.Frame(list_frame, bg=self.colors['bg_card'])
        action_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        convert_btn = tk.Button(action_frame,
                               text="转换为HDF5",
                               command=self._convert_selected_dataset,
                               bg=self.colors['accent_green'],
                               fg="white",
                               font=(self.chinese_font, 12, 'normal'),
                               relief=tk.FLAT,
                               bd=0,
                               cursor="hand2",
                               padx=24,
                               pady=10,
                               activebackground="#2FB55A")
        convert_btn.pack(side=tk.LEFT, padx=(0, 12))
        
        # 存储数据集信息
        self.datasets_info: List[Dict] = []
        self.current_dataset_root: Optional[str] = None
        
        # 初始化时刷新列表
        self._refresh_datasets()
        self._init_camera_manager()
        self._init_head_camera_manager()
        self._schedule_camera_preview_update()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _refresh_datasets(self) -> None:
        """刷新数据集列表"""
        # 清空现有列表
        for item in self.dataset_tree.get_children():
            self.dataset_tree.delete(item)
        
        # 获取数据集根目录（从配置或环境变量）
        try:
            if self.collector and self.collector.cfg.dataset_root:
                dataset_root = self.collector.cfg.dataset_root
            else:
                import os
                dataset_root = os.environ.get("LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
        except:
            import os
            dataset_root = os.environ.get("LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
        
        self.current_dataset_root = dataset_root
        
        # 扫描数据集
        try:
            from .dataset_manager import scan_datasets
            datasets = scan_datasets(dataset_root)
            self.datasets_info = [ds.to_dict() for ds in datasets]
            
            # 添加到列表
            for ds_info in self.datasets_info:
                status = "✓ 已转换" if ds_info['has_hdf5'] else "Parquet"
                self.dataset_tree.insert("", tk.END, values=(
                    ds_info['repo_id'],
                    ds_info['num_episodes'],
                    ds_info['total_frames'],
                    ds_info['created_time'],
                    status
                ))
            
            # 更新状态显示
            if datasets:
                self.status_var.set(f"找到 {len(datasets)} 个数据集")
            else:
                self.status_var.set("未找到数据集")
                
        except Exception as e:
            logger.error("刷新数据集列表失败：%s", e)
            messagebox.showerror("错误", f"刷新数据集列表失败: {e}")

    def _resolve_dataset_home(self, cfg: Optional[CollectorConfig]) -> Path:
        """根据配置或环境变量推断数据根目录。"""
        if cfg and cfg.dataset_root:
            return Path(os.path.expanduser(cfg.dataset_root))
        env_path = os.environ.get("LEROBOT_HOME")
        if env_path:
            return Path(os.path.expanduser(env_path))
        preferred = Path("/home/dreame/data/hf_dataset").expanduser()
        if preferred.exists():
            return preferred
        return Path(os.path.expanduser("~/.cache/huggingface/lerobot"))

    def _init_camera_manager(self) -> None:
        """初始化右手摄像头（若模块可用）。"""
        if RealSenseRightCameraManager is None:
            logger.info("未启用 RealSense 右手摄像头模块")
            return
        try:
            logger.info("初始化 RealSense 右手摄像头管理器...")
            self.current_dataset_home = self._resolve_dataset_home(None)
            self.right_camera_manager = RealSenseRightCameraManager(
                dataset_home=self.current_dataset_home
            )
            started = self.right_camera_manager.start()
            if not started:
                logger.warning(
                    "右手摄像头不可用：%s",
                    self.right_camera_manager.last_error,
                )
            else:
                logger.info(
                    "右手摄像头预览线程已启动，dataset_home=%s",
                    self.current_dataset_home,
                )
        except Exception as e:
            logger.warning("初始化右手摄像头失败：%s", e)
            self.right_camera_manager = None

    def _init_head_camera_manager(self) -> None:
        """初始化头部摄像头多进程。"""
        if RealSenseHeadCameraProcess is None:
            logger.info("未启用头部摄像头模块")
            return
        try:
            logger.info("初始化头部摄像头多进程管理器...")
            head_manager = RealSenseHeadCameraProcess(
                dataset_home=self._resolve_dataset_home(None)
            )
            if not head_manager.start():
                logger.warning(
                    "头部摄像头不可用：%s",
                    getattr(head_manager, "last_error", ""),
                )
                return
            self.head_camera_manager = head_manager
            logger.info("头部摄像头预览进程已启动")
        except Exception as e:
            logger.warning("初始化头部摄像头失败：%s", e)
            self.head_camera_manager = None

    def _schedule_camera_preview_update(self) -> None:
        """轮询摄像头预览帧并更新 GUI。"""
        if not self.root.winfo_exists():
            return
        head_frame = None
        right_frame = None
        if self.head_camera_manager:
            try:
                head_frame = self.head_camera_manager.get_latest_preview()
            except Exception as e:
                logger.debug("获取头部预览失败：%s", e)
        if self.right_camera_manager:
            try:
                right_frame = self.right_camera_manager.get_latest_preview()
            except Exception as e:
                logger.debug("获取右手预览失败：%s", e)
        if head_frame is not None or right_frame is not None:
            self.update_camera_images(
                head_image=head_frame,
                right_hand_image=right_frame,
            )
        self._camera_preview_job = self.root.after(66, self._schedule_camera_preview_update)

    def _start_right_camera_recording(
        self, cfg: CollectorConfig, target_frames: int
    ) -> None:
        if not self.right_camera_manager:
            return
        dataset_home = self._resolve_dataset_home(cfg)
        self.current_dataset_home = dataset_home
        try:
            self.right_camera_manager.set_dataset_home(dataset_home)
            paths = self.right_camera_manager.start_recording(
                cfg.repo_id, target_frames=target_frames
            )
            logger.info(
                "右手摄像头开始录制: repo_id=%s, episode_dir=%s",
                cfg.repo_id,
                getattr(paths, "episode_dir", None),
            )
        except Exception as e:
            logger.warning("启动右手摄像头录制失败：%s", e)

    def _stop_right_camera_recording(self) -> None:
        if not self.right_camera_manager:
            return
        try:
            self.right_camera_manager.stop_recording(wait=True)
            logger.info("右手摄像头录制已停止")
        except Exception as e:
            logger.debug("停止右手摄像头录制时出错：%s", e)

    def _start_head_camera_recording(
        self, cfg: CollectorConfig, target_frames: int
    ) -> None:
        if not self.head_camera_manager:
            return
        dataset_home = self._resolve_dataset_home(cfg)
        try:
            self.head_camera_manager.set_dataset_home(dataset_home)
            self.head_camera_manager.start_recording(
                cfg.repo_id, target_frames=target_frames
            )
            logger.info("头部摄像头开始录制: repo_id=%s", cfg.repo_id)
            self._head_camera_recording = True
        except Exception as e:
            logger.warning("启动头部摄像头录制失败：%s", e)

    def _stop_head_camera_recording(self) -> None:
        if not self.head_camera_manager:
            return
        try:
            self.head_camera_manager.stop_recording(wait=True)
            logger.info("头部摄像头录制已停止")
            self._head_camera_recording = False
        except Exception as e:
            logger.debug("停止头部摄像头录制时出错：%s", e)

    def _on_close(self) -> None:
        if self._camera_preview_job:
            try:
                self.root.after_cancel(self._camera_preview_job)
            except Exception:
                pass
            self._camera_preview_job = None
        if self.right_camera_manager:
            try:
                self.right_camera_manager.shutdown()
            except Exception:
                pass
        if self.head_camera_manager:
            try:
                self.head_camera_manager.shutdown()
            except Exception:
                pass
            self.head_camera_manager = None
        self.root.destroy()
    
    def _convert_selected_dataset(self) -> None:
        """转换选中的数据集为HDF5"""
        selected = self.dataset_tree.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选择要转换的数据集")
            return
        
        # 获取选中的数据集ID
        item = self.dataset_tree.item(selected[0])
        repo_id = item['values'][0]
        
        # 确认转换
        if not messagebox.askyesno("确认", f"确定要将数据集 '{repo_id}' 转换为HDF5格式吗？\n\n这可能需要一些时间..."):
            return
        
        try:
            from .dataset_manager import convert_parquet_to_hdf5
            self.status_var.set("正在转换...")
            self.progress_label.config(text=f"转换数据集: {repo_id}")
            
            # 在后台线程中执行转换
            import threading
            def convert_thread():
                success = convert_parquet_to_hdf5(repo_id, self.current_dataset_root)
                self.root.after(0, lambda: self._on_convert_done(repo_id, success))
            
            thread = threading.Thread(target=convert_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("错误", f"转换失败: {e}")
            self.status_var.set("转换失败")
            self.progress_label.config(text=str(e))
    
    def _on_convert_done(self, repo_id: str, success: bool) -> None:
        """转换完成回调"""
        if success:
            messagebox.showinfo("完成", f"数据集 '{repo_id}' 已成功转换为HDF5格式")
            self.status_var.set("转换成功")
            self.progress_label.config(text=f"数据集 {repo_id} 已转换为HDF5")
            # 刷新列表
            self._refresh_datasets()
        else:
            messagebox.showerror("失败", f"数据集 '{repo_id}' 转换失败")
            self.status_var.set("转换失败")

    def _draw_status_indicator(self, color: str) -> None:
        """绘制状态指示器（圆形LED灯）"""
        self.status_indicator.delete("all")
        self.status_indicator.create_oval(2, 2, 8, 8, fill=color, outline="", width=0)
    
    def _scan_config_files(self) -> List[str]:
        """扫描configs目录下的JSON配置文件"""
        import os
        from pathlib import Path
        
        config_files = []
        
        # 查找configs目录（项目根目录下的configs）
        try:
            # 获取项目根目录（向上查找）
            current_file = Path(__file__).resolve()
            # collector.py 在 lerobot_data_collector/lerobot_data_collector/ 下
            # 向上两级到项目根目录
            proj_root = current_file.parent.parent
            configs_dir = proj_root / "configs"
            
            if configs_dir.exists() and configs_dir.is_dir():
                # 扫描所有JSON文件
                for json_file in configs_dir.glob("*.json"):
                    config_files.append(str(json_file))
                
                # 按文件名排序
                config_files.sort()
                logger.info("发现 %d 个配置文件", len(config_files))
            else:
                logger.warning("配置文件目录不存在：%s", configs_dir)
        except Exception as e:
            logger.warning("扫描配置文件失败：%s", e)
        
        return config_files
    
    def _refresh_config_files(self) -> None:
        """刷新配置文件列表"""
        config_files = self._scan_config_files()
        self.cfg_combobox['values'] = config_files
        if config_files:
            # 如果当前选择的文件不在列表中，选择第一个
            current = self.cfg_path_var.get()
            if current not in config_files:
                self.cfg_path_var.set(config_files[0])
            messagebox.showinfo("刷新完成", f"找到 {len(config_files)} 个配置文件")
        else:
            messagebox.showwarning("未找到配置文件", "configs目录下没有找到JSON配置文件")

    def browse(self) -> None:
        """浏览选择其他位置的配置文件"""
        path = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if path:
            # 如果选择的文件不在下拉列表中，添加到列表
            current_values = list(self.cfg_combobox['values'])
            if path not in current_values:
                new_values = list(current_values) + [path]
                self.cfg_combobox['values'] = new_values
            self.cfg_path_var.set(path)

    def _update_countdown(self) -> None:
        """更新倒计时显示 - 优雅的状态更新"""
        if self.collection_start_time is None or not self.collector:
            return
        
        frames_written = getattr(self.collector, "_frames_written", 0)
        frames_left = max(0, self.collection_target_frames - frames_written)
        fps = max(1, self.collection_fps)
        # 使用实际开始写入数据的时间来计算已采集时间（用于估算剩余时间）
        if self.collector._actual_start_time is not None:
            elapsed = time.time() - self.collector._actual_start_time
        else:
            elapsed = 0.0
        remaining = frames_left / fps
        
        if remaining > 0 and self.collector._session_active:
            if self.collector._actual_start_time is not None:
                status_text = "正在采集数据"
                progress_text = (
                    f"已采集 {frames_written}/{self.collection_target_frames} 帧 • "
                    f"预计剩余 {remaining:.1f} 秒"
                )
                self.status_display.config(fg=self.colors['accent_blue'])
                self._draw_status_indicator(self.colors['accent_blue'])
            else:
                status_text = "等待数据就绪"
                progress_text = f"目标帧数：{self.collection_target_frames} 帧"
                self.status_display.config(fg=self.colors['text_secondary'])
                self._draw_status_indicator(self.colors['accent_orange'])
            self.status_var.set(status_text)
            self.progress_label.config(text=progress_text)
            self.root.after(1000, self._update_countdown)
        elif remaining <= 0 and self.collector._session_active:
            self.collector._session_active = False
            self.status_var.set("采集完成")
            self.progress_label.config(
                text=f"共采集 {frames_written}/{self.collection_target_frames} 帧，"
                     "请点击'结束并保存'按钮"
            )
            self.status_display.config(fg=self.colors['accent_orange'])
            self._draw_status_indicator(self.colors['accent_orange'])
            if self._head_camera_recording:
                self._stop_head_camera_recording()

    def start(self) -> None:
        path = self.cfg_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("错误", "请先选择有效的配置 JSON 文件")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cc = CollectorConfig.from_dict(cfg)
            # 每次开始采集时生成新的repo_id（基于当前时间戳）
            # 这样每次采集都会使用不同的文件夹
            from datetime import datetime
            import os
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                from .conf import TASK_CODE
                task_code = TASK_CODE
            except ImportError:
                task_code = "data_collection"
            cc.repo_id = f"{task_code}_{timestamp}"
            # 设置环境变量，让其他进程（如orbbec_depth_720_show_queue.py）使用相同的repo_id
            os.environ["FIXED_REPO_ID"] = cc.repo_id
            logger.info("使用数据集 ID：%s", cc.repo_id)
            self.collector = Collector(cc)
            self.collector.start()
            # 读取目标帧数（用户输入的即为帧数）
            try:
                target_frames = int(float(self.duration_var.get().strip()))
            except Exception:
                target_frames = 1800
            target_frames = max(1, target_frames)
            self.collection_fps = max(1, int(cc.fps))
            target_duration = target_frames / self.collection_fps
            self.collector.start_session_immediate(
                target_frames, target_duration=target_duration
            )
            self._start_right_camera_recording(cc, target_frames)
            self._start_head_camera_recording(cc, target_frames)
            self.collection_start_time = time.time()
            self.collection_target_frames = target_frames
            self.collection_duration = target_duration
            self.status_var.set("正在启动")
            self.progress_label.config(
                text=f"目标帧数：{target_frames}（≈{target_duration:.1f}s @ {self.collection_fps}fps）"
            )
            self.status_display.config(fg=self.colors['accent_blue'])
            self._draw_status_indicator(self.colors['accent_blue'])
            self._update_countdown()
        except Exception as e:
            self._stop_right_camera_recording()
            self._stop_head_camera_recording()
            messagebox.showerror("启动失败", str(e))
            self.status_var.set("启动失败")
            self.progress_label.config(text=str(e))
            self.status_display.config(fg=self.colors['accent_red'])
            self._draw_status_indicator(self.colors['accent_red'])
            logger.exception("Tk 采集启动失败：%s", e)
    
    def update_camera_images(
        self, head_image=None, left_hand_image=None, right_hand_image=None
    ) -> None:
        """
        更新摄像头图像显示
        
        Args:
            head_image: 头部摄像头图像 (numpy array, shape: H, W, 3, dtype: uint8)
            left_hand_image: 左手摄像头图像 (numpy array, shape: H, W, 3, dtype: uint8)
            right_hand_image: 右手摄像头图像 (numpy array, shape: H, W, 3, dtype: uint8)
        """

        def numpy_to_photo(img_array):
            if img_array is None:
                return None
            try:
                # 确保是 uint8 并裁剪到 [0, 255]
                if img_array.dtype != np.uint8:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

                # 灰度图转三通道
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                # 去掉 alpha 通道
                elif img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                # 非 3 通道直接忽略
                elif img_array.shape[2] != 3:
                    return None

                pil_image = Image.fromarray(img_array)

                display_width = self.right_hand_camera_display.winfo_width() or 320
                display_height = self.right_hand_camera_display.winfo_height() or 240
                if display_width > 1 and display_height > 1:
                    pil_image.thumbnail(
                        (display_width, display_height),
                        Image.Resampling.LANCZOS,
                    )

                return ImageTk.PhotoImage(image=pil_image)
            except Exception as e:  # pragma: no cover - 容错
                logger.debug("Tk 图像转换失败：%s", e)
                return None

        if head_image is not None:
            photo = numpy_to_photo(head_image)
            if photo:
                self.head_camera_display.config(image=photo)
            self.head_camera_display.image = photo

        if left_hand_image is not None:
            photo = numpy_to_photo(left_hand_image)
            if photo:
                self.left_hand_camera_display.config(image=photo)
            self.left_hand_camera_display.image = photo

        if right_hand_image is not None:
            photo = numpy_to_photo(right_hand_image)
            if photo:
                self.right_hand_camera_display.config(image=photo)
            self.right_hand_camera_display.image = photo

    def stop(self) -> None:
        self._stop_right_camera_recording()
        self._stop_head_camera_recording()
        if not self.collector:
            messagebox.showwarning("提示", "尚未开始采集，无法保存")
            return
        # 检查是否有数据帧
        if self.collector._frames_written == 0:
            messagebox.showwarning("保存失败", "没有采集到任何数据，无法保存。\n\n请先点击'开始采集'并等待数据采集后再保存。")
            self.status_var.set("保存失败")
            self.progress_label.config(text="没有采集到任何数据")
            self.status_display.config(fg=self.colors['accent_orange'])
            self._draw_status_indicator(self.colors['accent_orange'])
            return
        try:
            self.collection_start_time = None
            self.status_var.set("正在保存")
            self.progress_label.config(text="保存数据到数据集...")
            self.status_display.config(fg=self.colors['text_secondary'])
            self._draw_status_indicator(self.colors['accent_orange'])
            self.root.update()  # 立即更新UI
            
            self.collector.stop_and_save()
            self.status_var.set("保存成功")
            self.progress_label.config(text=f"共采集 {self.collector._frames_written} 帧数据")
            self.status_display.config(fg=self.colors['accent_green'])
            self._draw_status_indicator(self.colors['accent_green'])
            messagebox.showinfo("完成", 
                              f"采集已结束并保存为 LeRobot 数据集\n\n共采集 {self.collector._frames_written} 帧数据",
                              icon='info')
        except ValueError as e:
            error_msg = str(e)
            if "没有采集到任何数据" in error_msg or "没有采集到任何数据帧" in error_msg:
                messagebox.showwarning("保存失败", "没有采集到任何数据，无法保存。\n\n请先点击'开始采集'并等待数据采集后再保存。")
                self.status_var.set("保存失败")
                self.progress_label.config(text="没有采集到任何数据")
                self.status_display.config(fg=self.colors['accent_orange'])
                self._draw_status_indicator(self.colors['accent_orange'])
            else:
                messagebox.showerror("保存失败", error_msg)
                self.status_var.set("保存失败")
                self.progress_label.config(text=error_msg)
                self.status_display.config(fg=self.colors['accent_red'])
                self._draw_status_indicator(self.colors['accent_red'])
        except Exception as e:
            messagebox.showerror("保存失败", str(e))
            self.status_var.set("保存失败")
            self.progress_label.config(text=str(e))
            self.status_display.config(fg=self.colors['accent_red'])
            self._draw_status_indicator(self.colors['accent_red'])


def main() -> None:
    """主函数 - 自动选择最佳GUI框架"""
    if USE_QT:
        # 使用Qt（PySide6）- 更好的字体渲染
        from .qt_gui import main_qt
        main_qt(Collector, CollectorConfig)
    else:
        # 回退到Tkinter
        root = tk.Tk()
        App(root)
        root.mainloop()


if __name__ == "__main__":
    main()


