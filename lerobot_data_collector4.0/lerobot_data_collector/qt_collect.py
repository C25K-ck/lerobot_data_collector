import logging
import os
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

def _sn_disabled(sn: object) -> bool:
    """判断某一路相机是否被配置为禁用。

    约定：空串/None/0/"DISABLED"/"DISABLE"/"NONE"/"NULL" 都视为禁用。
    """
    if sn is None:
        return True
    if not isinstance(sn, str):
        sn = str(sn)
    s = sn.strip()
    if not s:
        return True
    return s.lower() in {"0", "disable", "disabled", "none", "null"}


def _env_flag(name: str, default: bool = False) -> bool:
    """读取布尔环境变量：1/true/yes/on 为 True，其它为 False。"""
    v = os.getenv(name)
    if v is None:
        return default
    s = v.strip().lower()
    if not s:
        return default
    return s in {"1", "true", "yes", "y", "on"}


def _list_realsense_serials() -> Tuple[List[str], str]:
    """枚举当前 RealSense 设备序列号列表。

    返回: (serials, error_message)。成功时 error_message 为空串。
    """
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:
        return [], f"pyrealsense2 不可用：{exc}"
    try:
        ctx = rs.context()
        devs = ctx.query_devices()
        serials: List[str] = []
        for d in devs:
            try:
                serials.append(d.get_info(rs.camera_info.serial_number))
            except Exception:
                continue
        return serials, ""
    except Exception as exc:
        return [], f"枚举 RealSense 失败：{exc}"


def _resolve_realsense_sn(preferred_sn: str) -> Tuple[Optional[str], str]:
    """根据配置SN解析出实际要使用的 RealSense SN（单机一台时支持自动回退）。"""
    serials, err = _list_realsense_serials()
    if err:
        return None, err
    if not serials:
        return None, "未检测到任何 RealSense 设备"

    s = (preferred_sn or "").strip()
    if s and (s in serials):
        return s, ""

    # 没配SN / 配置SN不匹配：单机一台时自动回退到唯一设备
    if len(serials) == 1:
        picked = serials[0]
        if s and (s != picked):
            return picked, f"未找到SN={s}，已自动回退到唯一设备 SN={picked}"
        return picked, f"未配置SN，已自动选择唯一设备 SN={picked}"

    if s:
        return None, f"未找到SN={s}，当前检测到的设备SN={serials}"
    return None, f"检测到多台 RealSense：SN={serials}，请在环境变量 REALSENSE_HEAD_SN 指定其中一个"

try:
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QSplitter,
        QScrollArea,
        QLabel,
        QLineEdit,
        QCheckBox,
        QPushButton,
        QFileDialog,
        QMessageBox,
        QFrame,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QComboBox,
        QListView,
        QListWidget,
        QListWidgetItem,
        QDialog,
        QSpinBox,
        QTextEdit,
        QDateTimeEdit,
        QProgressBar,
        QSizePolicy,
    )
    from PySide6.QtCore import Qt, QTimer, QRect, QDateTime, QSettings
    from PySide6.QtGui import (
        QFont,
        QFontMetrics,
        QFontDatabase,
        QColor,
        QPalette,
        QPainter,
        QPen,
        QBrush,
        QPixmap,
        QImage,
    )
    PYSIDE6_AVAILABLE = True
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QApplication,
            QMainWindow,
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QGridLayout,
            QSplitter,
            QScrollArea,
            QLabel,
            QLineEdit,
            QCheckBox,
            QPushButton,
            QFileDialog,
            QMessageBox,
            QFrame,
            QTabWidget,
            QTableWidget,
            QTableWidgetItem,
            QHeaderView,
        QComboBox,
        QListView,
        QListWidget,
            QListWidgetItem,
            QDialog,
            QSpinBox,
            QTextEdit,
            QDateTimeEdit,
            QProgressBar,
            QSizePolicy,
        )
        from PyQt6.QtCore import Qt, QTimer, QRect, QDateTime, QSettings
        from PyQt6.QtGui import (
            QFont,
            QFontMetrics,
            QFontDatabase,
            QColor,
            QPalette,
            QPainter,
            QPen,
            QBrush,
            QPixmap,
            QImage,
        )
        PYSIDE6_AVAILABLE = True
    except ImportError:
        PYSIDE6_AVAILABLE = False
        logger.warning("PySide6/PyQt6 未安装，将回退到 Tkinter")
        logger.info("请安装依赖: pip install PySide6")

if TYPE_CHECKING:
    from .server_api import ApiError, DataInfoItem, SessionToken

# 说明：这些模块（网络/存储/相机）导入较慢，改为按需导入以优化启动速度。

RealSenseRightCameraManager = None  # type: ignore[assignment]
RealSenseLeftCameraManager = None  # type: ignore[assignment]
RealSenseHeadCameraProcess = None  # type: ignore[assignment]
OrbbecHeadCameraManager = None  # type: ignore[assignment]

# 相机序列号配置（可选）。不强依赖，读取失败就当作未配置。
try:
    from .conf import REALSENSE_LEFT_SN, REALSENSE_RIGHT_SN, REALSENSE_HEAD_SN  # type: ignore
except Exception:  # pragma: no cover
    REALSENSE_LEFT_SN = ""
    REALSENSE_RIGHT_SN = ""
    REALSENSE_HEAD_SN = ""

try:
    from PySide6.QtCore import Signal, Slot
except ImportError:
    from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot


class StatusGridWidget(QWidget):
    """用于显示采集状态的绿色点阵指示器"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_count = 0
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        cols, rows = 5, 2
        dot_size = 6
        spacing = 8
        start_x = (self.width() - (cols * (dot_size + spacing) - spacing)) / 2
        start_y = (self.height() - (rows * (dot_size + spacing) - spacing)) / 2
        
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * (dot_size + spacing)
                y = start_y + row * (dot_size + spacing)
                
                index = row * cols + col
                if index < self.active_count:
                    color = QColor(52, 199, 89)  # 绿色
                else:
                    color = QColor(60, 60, 65)  # 深灰色
                
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(int(x), int(y), dot_size, dot_size)
        
        painter.end()
    
    def set_active_count(self, count):
        self.active_count = count
        self.update()


class TimelineWidget(QWidget):
    """时间轴视图，显示动作步骤的时间轴和任务目标（垂直布局）"""
    def __init__(self, parent=None, colors=None, chinese_font_family=None):
        super().__init__(parent)
        self.steps_data = []  # 存储步骤数据：[{start_time, end_time, action_text, skill}]
        self.total_duration = 0.0  # 总时长（秒）
        self.current_time = 0.0  # 当前时间（秒）
        self.colors = colors or {}  # 保存颜色字典
        self.chinese_font_family = chinese_font_family or "Microsoft YaHei"  # 保存字体
        self.setMinimumHeight(200)
    
    def set_steps(self, steps_data, total_duration):
        """设置步骤数据"""
        self.steps_data = steps_data
        if steps_data:
            max_end_time = max([s['end_time'] for s in steps_data])
            self.total_duration = max(total_duration, max_end_time)
        else:
            self.total_duration = total_duration
        # 注意：不要在这里动态增高控件，否则会把右侧面板撑出一屏导致滚动；
        # 时间轴的显示密度应在 paintEvent 里通过缩放/压缩行高来适配固定高度。
        self.update()
    
    def set_current_time(self, current_time):
        """设置当前时间，用于高亮当前步骤"""
        self.current_time = current_time
        self.update()
    
    def paintEvent(self, event):
        """绘制垂直时间轴和动作步骤列表"""
        painter = QPainter(self)
        if not painter.isActive():
            return
        
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            width = self.width()
            height = self.height()
            time_label_width = 85  # 左侧时间标签区域宽度（显示 0-52s 这类范围）
            timeline_width = 50  # 时间轴区域宽度
            margin_top = 15
            margin_bottom = 15
            step_item_height = 72  # 允许文字换行显示完整描述
            timeline_line_width = 2
            
            # 背景
            painter.fillRect(0, 0, width, height, QColor(self.colors.get('bg_card', QColor(30, 30, 30))))
            
            if not self.steps_data or self.total_duration <= 0:
                painter.setPen(QColor(self.colors.get('text_secondary', QColor(150, 150, 150))))
                painter.setFont(QFont(self.chinese_font_family, 12))
                painter.drawText(width // 2 - 100, height // 2, "暂无动作步骤数据")
                return
            
            available_height = height - margin_top - margin_bottom
            timeline_x = time_label_width + timeline_width // 2
            timeline_y_start = margin_top
            timeline_y_end = height - margin_bottom
            
            painter.setPen(QPen(QColor(self.colors.get('border', QColor(100, 100, 100))), timeline_line_width))
            painter.drawLine(timeline_x, timeline_y_start, timeline_x, timeline_y_end)
            
            for i, step in enumerate(self.steps_data):
                if self.total_duration > 0:
                    step_y = timeline_y_start + (step['start_time'] / self.total_duration) * available_height
                else:
                    step_y = timeline_y_start
                
                step_y = max(margin_top + 5, min(step_y, height - margin_bottom - 5))
                is_current = step['start_time'] <= self.current_time <= step['end_time']
                
                point_radius = 6 if is_current else 4
                point_color = QColor(52, 199, 89) if is_current else QColor(self.colors.get('accent_blue', QColor(0, 122, 255)))
                painter.setBrush(QBrush(point_color))
                painter.setPen(QPen(point_color, 2))
                painter.drawEllipse(int(timeline_x - point_radius), int(step_y - point_radius), 
                                   point_radius * 2, point_radius * 2)
                
                # 显示完整时间范围，避免只看见起始时间
                time_text = f"{int(step['start_time'])}-{int(step['end_time'])}s"
                painter.setPen(QColor(self.colors.get('text_secondary', QColor(150, 150, 150))))
                painter.setFont(QFont(self.chinese_font_family, 9))
                text_rect = painter.fontMetrics().boundingRect(time_text)
                label_x = time_label_width - text_rect.width() - 5
                label_y = int(step_y + text_rect.height() // 2 - text_rect.height() // 2)
                painter.drawText(int(label_x), int(label_y + text_rect.height()), time_text)
                
                line_x_start = timeline_x + point_radius + 5
                line_x_end = time_label_width + timeline_width + 20
                painter.setPen(QPen(QColor(self.colors.get('border', QColor(60, 60, 60))), 1))
                painter.drawLine(int(line_x_start), int(step_y), int(line_x_end), int(step_y))
                
                text_x = line_x_end + 10
                text_y = step_y - step_item_height // 2
                text_width = width - text_x - 10
                
                if is_current:
                    highlight_rect = QRect(int(text_x - 5), int(text_y), 4, step_item_height)
                    painter.fillRect(highlight_rect, QColor(52, 199, 89))
                
                painter.setPen(QColor(255, 255, 255) if is_current else 
                             QColor(self.colors.get('text_primary', QColor(255, 255, 255))))
                painter.setFont(QFont(self.chinese_font_family, 10, 
                                   QFont.Weight.Bold if is_current else QFont.Weight.Normal))
                # 时间信息已在左侧展示，这里只显示描述文本，给更多宽度避免裁剪
                text = str(step.get('action_text') or "")
                
                text_rect = QRect(int(text_x), int(text_y), int(text_width), step_item_height)
                painter.drawText(
                    text_rect,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
                    text,
                )
            
            if 0 <= self.current_time <= self.total_duration and self.total_duration > 0:
                current_y = timeline_y_start + (self.current_time / self.total_duration) * available_height
                current_y = max(margin_top, min(current_y, height - margin_bottom))
                painter.setPen(QPen(QColor(255, 59, 48), 2))
                painter.drawLine(0, int(current_y), width, int(current_y))
                
                painter.setPen(QColor(255, 59, 48))
                painter.setFont(QFont(self.chinese_font_family, 9, QFont.Weight.Bold))
                current_text = f"{self.current_time:.1f}s"
                text_rect = painter.fontMetrics().boundingRect(current_text)
                label_x = int(timeline_x + 8)
                label_y = int(current_y - text_rect.height() // 2 - 2)
                if label_y < margin_top:
                    label_y = margin_top
                painter.fillRect(label_x, label_y, 
                               text_rect.width() + 4, text_rect.height() + 4, 
                               QColor(255, 59, 48))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(label_x + 2, label_y + text_rect.height(), current_text)
        finally:
            painter.end()


class QtApp(QMainWindow):
    """使用Qt（PySide6）的GUI，提供更好的字体渲染"""
    
    # 定义信号用于线程间通信
    upload_success_signal = Signal(list, int, int)  # (failed_uuids, task_id, total_items)
    upload_error_signal = Signal(str)  # (error_message)
    hdf5_progress_signal = Signal(int, int, str)  # (current, total, message)
    hdf5_done_signal = Signal(str, bool, bool)  # (repo_id, success, cancelled)
    hdf5_item_progress_signal = Signal(str, int, int, str)  # (repo_id, current, total, message)
    hdf5_batch_item_done_signal = Signal(str, bool, object)  # (repo_id, success, config_file)
    hdf5_upload_scan_done_signal = Signal(list)  # (items)
    hdf5_upload_progress_signal = Signal(int, int, str)  # (current, total, message)
    hdf5_upload_done_signal = Signal(bool, str)  # (success, message)
    pico_control_signal = Signal(str)
    
    def __init__(self, collector_class, collector_config_class, task: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # 连接信号到槽
        self.upload_success_signal.connect(self._on_upload_success)
        self.upload_error_signal.connect(self._on_upload_error)
        self.hdf5_progress_signal.connect(self._on_hdf5_progress)
        self.hdf5_done_signal.connect(self._on_hdf5_done)
        self.hdf5_item_progress_signal.connect(self._on_hdf5_item_progress)
        self.hdf5_batch_item_done_signal.connect(self._on_one_batch_convert_done)
        self.hdf5_upload_scan_done_signal.connect(self._on_hdf5_upload_scan_done)
        self.hdf5_upload_progress_signal.connect(self._on_hdf5_upload_progress)
        self.hdf5_upload_done_signal.connect(self._on_hdf5_upload_done)
        self.pico_control_signal.connect(self._on_pico_control_main)
        self.collector_class = collector_class
        self.collector_config_class = collector_config_class
        self.task_manager = None  # 任务管理窗口的引用，用于返回
        # 当前任务信息（来自任务管理页）
        self.current_task: Optional[Dict[str, Any]] = task or {}
        # 采集落盘根目录：优先使用 /home/magiclab/data/hf_dataset（不可用时再降级），
        # 当天所有采集都放在同一个“当天目录”下；每次采集仍会创建一个独立 repo_id 子目录（不复用）。
        try:
            import os as _os
            def _is_writable_dir(p: str) -> bool:
                try:
                    return _os.path.isdir(p) and _os.access(p, _os.W_OK | _os.X_OK)
                except Exception:
                    return False

            def _ensure_writable_dir(p: str) -> bool:
                try:
                    _os.makedirs(p, exist_ok=True)
                except Exception:
                    pass
                return _is_writable_dir(p)

            # 候选根目录（允许用环境变量覆盖，但若不可写会自动降级）
            candidates: list[str] = []
            env_root = _os.environ.get("HF_DATASET_ROOT")
            if env_root:
                candidates.append(_os.path.expanduser(env_root))
            candidates.extend(
                [
                    "/home/magiclab/data/hf_dataset",
                    "/data/hf_dataset",
                    "/hf_dataset",
                    "/home/dreame/data/hf_dataset",
                    _os.path.expanduser("~/.cache/huggingface/lerobot"),
                ]
            )

            chosen: str | None = None
            for c in candidates:
                if _ensure_writable_dir(c):
                    chosen = c
                    break
            if chosen is None:
                # 兜底：选一个存在的目录（即使不可写也打印出来，便于排查）
                for c in candidates:
                    try:
                        if _os.path.isdir(c):
                            chosen = c
                            break
                    except Exception:
                        continue
            self._base_dataset_root = chosen or "/home/magiclab/data/hf_dataset"
            logger.info("采集根目录(base_dataset_root)：%s", self._base_dataset_root)
        except Exception:
            self._base_dataset_root = "/home/magiclab/data/hf_dataset"
        
        # 苹果风格配色（更专业、更现代）
        self.colors = {
            'bg_primary': QColor(20, 20, 23),         # 极深色背景
            'bg_card': QColor(28, 28, 32),           # 卡片背景
            'bg_secondary': QColor(38, 38, 42),      # 次要背景
            'bg_tertiary': QColor(48, 48, 52),       # 三级背景
            'text_primary': QColor(255, 255, 255),   # 主文本
            'text_secondary': QColor(170, 170, 175), # 次要文本
            'text_tertiary': QColor(110, 110, 115),  # 三级文本
            'accent_blue': QColor(10, 132, 255),    # 苹果蓝
            'accent_green': QColor(48, 209, 88),    # 苹果绿
            'accent_red': QColor(255, 69, 58),      # 苹果红
            'accent_purple': QColor(191, 90, 242),  # 苹果紫
            'accent_orange': QColor(255, 159, 10),  # 苹果橙
            'border': QColor(50, 50, 55),            # 边框色
            'shadow': QColor(15, 15, 18),            # 阴影色
            'status_green': QColor(48, 209, 88),
        }
        
        # 设置窗口属性
        self.setWindowTitle("SRIC 数据采集平台")
        self.resize(1800, 950) # 进一步增加初始宽度以适配双列布局和 480P 图像
        
        # 重要：不要用 setWindowFlags 覆盖默认 flags（某些 WM 下会导致不可缩放）。
        # 在默认 flags 基础上确保常规窗口按钮与可缩放能力。
        flags = self.windowFlags()
        flags |= Qt.WindowType.Window
        flags |= Qt.WindowType.WindowTitleHint
        flags |= Qt.WindowType.WindowSystemMenuHint
        flags |= Qt.WindowType.WindowMinimizeButtonHint
        flags |= Qt.WindowType.WindowMaximizeButtonHint
        flags |= Qt.WindowType.WindowCloseButtonHint
        # 显式关闭“固定大小对话框”提示（若上层环境/主题意外注入）
        try:
            flags &= ~Qt.WindowType.MSWindowsFixedSizeDialogHint
        except Exception:
            pass
        self.setWindowFlags(flags)
        # 给一个合理的最小尺寸，避免布局被挤爆；不限制最大尺寸，允许自由缩放
        self.setMinimumSize(1200, 800)
        
        # 配置字体渲染
        self._setup_fonts()
        
        # 设置样式
        self._setup_styles()
        
        # 创建主窗口（使用标签页）
        self._create_ui_with_tabs()
        
        # 初始化变量
        self.collector = None
        self.collection_start_time = None
        self.collection_duration = 0.0
        self.action_steps = []  # 动作步骤列表
        self.current_step_index = -1  # 当前步骤索引
        self._existing_frames = 0  # 已有数据帧数（用于断点续采）
        self._pico_adapter = None  # 独立的Pico适配器，不依赖采集器
        self._pico_teleop_runtime = None  # pico_teleop_bridge.PicoTeleopRuntime（做法1：单路流遥操）
        # Pico 连接配置（持久化）
        try:
            self._settings = QSettings("SRIC", "DataCollector")
        except Exception:
            self._settings = None
        self._existing_episodes = 0  # 已有episode数（用于断点续采与目标对比）
        self._target_episodes: Optional[int] = None  # 目标episode数（来自任务）
        # 目标帧数（用于UI显示 X / Y 帧）
        self._target_frames_ui: int = 0
        # 摄像头相关
        self.right_camera_manager: Optional["RealSenseRightCameraManager"] = None
        self.left_camera_manager: Optional["RealSenseLeftCameraManager"] = None
        self.head_camera_manager: Optional["RealSenseHeadCameraProcess"] = None
        self.orbbec_head_camera_manager: Optional["OrbbecHeadCameraManager"] = None
        self._camera_timer: Optional[QTimer] = None
        self._camera_recording: bool = False
        self._left_camera_recording: bool = False
        self._head_camera_recording: bool = False
        self._pico_save_ready_announced: bool = False

        # HDF5 转换进度 UI（按需创建）
        self._hdf5_progress_dialog: Optional[QDialog] = None
        self._hdf5_progress_bar: Optional[QProgressBar] = None
        self._hdf5_progress_label: Optional[QLabel] = None
        self._hdf5_cancelled: bool = False
        self._hdf5_cancel_event = None

        # 上传HDF5页状态
        self._upload_hdf5_items: List[Dict[str, Any]] = []
        self._upload_hdf5_scanning: bool = False
        self._upload_hdf5_uploading: bool = False
        self._upload_hdf5_cancel_event = None
        
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
        logger.info("当前会话数据集 ID：%s", self.session_repo_id)
        
        # 定时器用于倒计时
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_countdown)

        # 初始化摄像头预览（延迟启动，提升首屏渲染速度）
        self._schedule_camera_preview_init()
        
    def _setup_fonts(self):
        """设置专业字体 - Qt提供更好的字体渲染"""
        # 极速启动模式：跳过字体库遍历（在部分系统上较慢）
        if _env_flag("FAST_STARTUP", default=False):
            self.chinese_font_family = QFont().family()
            self.title_font = QFont(self.chinese_font_family, 20, QFont.Weight.Bold)
            self.header_font = QFont(self.chinese_font_family, 11, QFont.Weight.Medium)
            self.body_font = QFont(self.chinese_font_family, 12, QFont.Weight.Normal)
            self.large_font = QFont(self.chinese_font_family, 14, QFont.Weight.Normal)
            self.status_font = QFont(self.chinese_font_family, 14, QFont.Weight.Normal)
            self.subtitle_font = QFont(self.chinese_font_family, 10, QFont.Weight.Normal)
            return
        # Qt的字体渲染引擎支持：
        # 1. 抗锯齿（Antialiasing）
        # 2. 亚像素渲染（Subpixel rendering）
        # 3. 更好的DPI缩放
        
        # 获取系统字体数据库
        font_db = QFontDatabase()
        available_fonts = font_db.families()
        available_lower = [f.lower() for f in available_fonts]
        
        # 专业字体优先级列表
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
        ]
        
        # 查找最佳字体
        self.chinese_font_family = None
        for candidate in professional_fonts:
            candidate_lower = candidate.lower().strip()
            if candidate_lower in available_lower:
                idx = available_lower.index(candidate_lower)
                self.chinese_font_family = available_fonts[idx]
                logger.info("Qt 字体选择：%s", self.chinese_font_family)
                break
        
        # 如果找不到，使用系统默认
        if not self.chinese_font_family:
            self.chinese_font_family = QFont().family()
            logger.debug("Qt 字体使用系统默认：%s", self.chinese_font_family)
        
        # 创建字体对象（Qt的字体渲染更平滑）
        self.title_font = QFont(self.chinese_font_family, 20, QFont.Weight.Bold)
        self.header_font = QFont(self.chinese_font_family, 11, QFont.Weight.Medium)
        self.body_font = QFont(self.chinese_font_family, 12, QFont.Weight.Normal)
        self.large_font = QFont(self.chinese_font_family, 14, QFont.Weight.Normal)
        self.status_font = QFont(self.chinese_font_family, 14, QFont.Weight.Normal)
        self.subtitle_font = QFont(self.chinese_font_family, 10, QFont.Weight.Normal)
        self.small_font = QFont(self.chinese_font_family, 9, QFont.Weight.Normal)
        
        # 启用抗锯齿和亚像素渲染（Qt的优势）
        self.title_font.setStyleHint(QFont.StyleHint.SansSerif)
        self.header_font.setStyleHint(QFont.StyleHint.SansSerif)
        self.body_font.setStyleHint(QFont.StyleHint.SansSerif)
        self.large_font.setStyleHint(QFont.StyleHint.SansSerif)
        self.status_font.setStyleHint(QFont.StyleHint.SansSerif)
        self.subtitle_font.setStyleHint(QFont.StyleHint.SansSerif)
        
    def _setup_styles(self):
        """设置全局 QSS 样式（苹果风格）"""
        self.setStyleSheet(f"""
            QMainWindow, QDialog, QWidget {{
                background-color: {self.colors['bg_primary'].name()};
                color: {self.colors['text_primary'].name()};
                font-family: "{self.chinese_font_family}";
            }}
            
            QTabWidget::pane {{
                border: none;
                background-color: {self.colors['bg_primary'].name()};
            }}
            
            QTabBar::tab {{
                background-color: transparent;
                color: {self.colors['text_secondary'].name()};
                padding: 12px 28px;
                margin-right: 4px;
                font-size: 15px;
                font-weight: 500;
                border-bottom: 2px solid transparent;
            }}
            
            QTabBar::tab:selected {{
                color: {self.colors['accent_blue'].name()};
                border-bottom: 2px solid {self.colors['accent_blue'].name()};
            }}
            
            QTabBar::tab:hover:!selected {{
                color: {self.colors['text_primary'].name()};
                background-color: {self.colors['bg_secondary'].name()};
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }}

            QFrame {{
                border: none;
            }}

            QPushButton {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_primary'].name()};
                border-radius: 12px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                border: 1px solid {self.colors['border'].name()};
            }}
            
            QPushButton:hover {{
                background-color: {self.colors['bg_tertiary'].name()};
                border: 1px solid {self.colors['accent_blue'].name()};
            }}

            QPushButton#PrimaryButton {{
                background-color: {self.colors['accent_blue'].name()};
                color: white;
                border: none;
                font-weight: 700;
            }}
            
            QPushButton#PrimaryButton:hover {{
                background-color: {self.colors['accent_blue'].lighter(115).name()};
            }}

            QPushButton#SuccessButton {{
                background-color: {self.colors['accent_green'].name()};
                color: white;
                border: none;
                font-weight: 700;
            }}
            
            QPushButton#SuccessButton:hover {{
                background-color: {self.colors['accent_green'].lighter(115).name()};
            }}

            QPushButton#DangerButton {{
                background-color: {self.colors['accent_red'].name()};
                color: white;
                border: none;
                font-weight: 700;
            }}
            
            QPushButton#DangerButton:hover {{
                background-color: {self.colors['accent_red'].lighter(115).name()};
            }}

            QLineEdit, QSpinBox, QComboBox {{
                background-color: {self.colors['bg_secondary'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 10px;
                padding: 10px 14px;
                color: {self.colors['text_primary'].name()};
                font-size: 14px;
                combobox-popup: 0; /* 强制使用 Qt 弹出框而非原生，解决 Linux 白边问题 */
            }}
            
            QComboBox QAbstractItemView {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_primary'].name()};
                selection-background-color: {self.colors['accent_blue'].name()};
                selection-color: white;
                border: 1px solid {self.colors['border'].name()};
                outline: none;
            }}
            
            /* 针对 Linux 的深度覆盖：锁定弹出层所有可能的容器背景 */
            QComboBox QFrame, QComboBox QListView, QComboBox QAbstractItemView::item {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_primary'].name()};
                padding: 8px 12px;
                border: none;
                margin: 0px;
            }}

            QComboBox QListView::item:selected {{
                background-color: {self.colors['accent_blue'].name()};
                color: white;
            }}
            
            QLineEdit:focus {{
                border: 2px solid {self.colors['accent_blue'].name()};
                background-color: {self.colors['bg_card'].name()};
            }}

            QProgressBar {{
                border: none;
                background-color: {self.colors['bg_secondary'].name()};
                height: 6px;
                border-radius: 3px;
                text-align: center;
            }}
            
            QProgressBar::chunk {{
                background-color: {self.colors['accent_blue'].name()};
                border-radius: 3px;
            }}

            QTableWidget {{
                background-color: {self.colors['bg_card'].name()};
                alternate-background-color: {self.colors['bg_secondary'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 12px;
                gridline-color: transparent;
                color: {self.colors['text_primary'].name()};
                selection-background-color: {self.colors['accent_blue'].name()};
                selection-color: white;
                outline: none;
            }}
            
            QTableWidget::item {{
                padding: 10px;
                border-bottom: 1px solid {self.colors['border'].name()};
                color: {self.colors['text_primary'].name()};
            }}

            QTableWidget::item:selected {{
                background-color: {self.colors['accent_blue'].name()};
                color: white;
                border-radius: 0px;
            }}
            
            QHeaderView::section {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_secondary'].name()};
                padding: 12px;
                border: none;
                border-bottom: 2px solid {self.colors['border'].name()};
                font-weight: 600;
                font-size: 13px;
            }}

            /* 美化滚动条 */
            QScrollBar:vertical {{
                background: transparent;
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {self.colors['bg_tertiary'].name()};
                min-height: 30px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    def _schedule_camera_preview_init(self) -> None:
        """延迟启动相机预览，避免阻塞窗口首次渲染。"""
        delay_ms = 800 if _env_flag("FAST_STARTUP", default=False) else 0
        QTimer.singleShot(delay_ms, self._init_camera_preview_now)

    def _init_camera_preview_now(self) -> None:
        """真正启动相机预览（可被延迟调用）。"""
        try:
            self._init_right_camera()
            self._init_left_camera()
            self._init_head_camera()
            self._start_camera_preview()
        except Exception as e:
            logger.warning("启动相机预览失败：%s", e)
        
    def _create_ui_with_tabs(self):
        """创建带标签页的UI界面"""
        # 创建标签页容器
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # 设置标签页样式 - 深色主题
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {self.colors['border'].name()};
                background-color: {self.colors['bg_primary'].name()};
            }}
            QTabBar::tab {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_secondary'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 10px 24px;
                margin-right: 2px;
                font-size: 13px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.colors['bg_card'].name()};
                color: {self.colors['text_primary'].name()};
                border-bottom: 2px solid {self.colors['accent_blue'].name()};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {self.colors['bg_card'].name()};
                color: {self.colors['text_primary'].name()};
            }}
        """)
        
        # 标签页1: 数据采集
        collect_widget = QWidget()
        self._create_collect_tab(collect_widget)
        self.tab_widget.addTab(collect_widget, "数据采集")
        
        # 标签页2: 转换HDF5数据（按模板：可浏览选择+勾选+进度）
        convert_widget = QWidget()
        self._create_convert_hdf5_tab_ui(convert_widget)
        self.tab_widget.addTab(convert_widget, "转换HDF5数据")

        # 标签页3: 上传HDF5数据（按截图：可浏览选择+勾选+进度）
        upload_hdf5_widget = QWidget()
        self._create_upload_hdf5_tab_ui(upload_hdf5_widget)
        self.tab_widget.addTab(upload_hdf5_widget, "上传HDF5数据")
        
        # 标签页4: 机器人管理
        robot_widget = QWidget()
        self._create_robot_tab_ui(robot_widget)
        self.tab_widget.addTab(robot_widget, "机器人管理")
    
    def _create_collect_tab(self, parent: QWidget):
        """创建数据采集标签页 - 左右分栏布局，苹果风格"""
        # 主布局：使用 QSplitter 实现真正可响应缩放的左右分栏
        # - 左侧：相机预览/信息
        # - 右侧：控制面板（可滚动，避免缩小时裁切/错位）
        main_layout = QVBoxLayout(parent)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        splitter.setHandleWidth(6)
        # 保存引用，便于窗口显示后重新分配尺寸
        self._collect_splitter = splitter
        main_layout.addWidget(splitter)
        
        # ========== 左侧面板：摄像头视图 ==========
        left_panel = QWidget()
        # 不要把左侧锁死为超大最小宽度，否则窗口缩小时右侧会被挤爆导致错位
        left_panel.setMinimumWidth(520)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(24, 24, 24, 24)
        left_layout.setSpacing(20)
        
        # 左侧顶部信息条：极简苹果设计
        left_info_bar = QFrame()
        left_info_bar.setObjectName("InfoBar")
        left_info_bar.setStyleSheet(f"""
            QFrame#InfoBar {{
                background-color: {self.colors['bg_card'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 12px;
            }}
        """)
        left_info_layout = QHBoxLayout(left_info_bar)
        left_info_layout.setContentsMargins(16, 12, 16, 12)
        left_info_layout.setSpacing(15)

        # 当前任务标签
        task_text = "-"
        if getattr(self, "current_task", None):
            _name = self.current_task.get("task_name") or self.current_task.get("name") or "-"
            _tid = self.current_task.get("id") or self.current_task.get("task_id")
            if _tid is not None and str(_tid).strip() != "":
                task_text = f"当前任务: {_name}（ID: {_tid}）"
            else:
                task_text = f"当前任务: {_name}"
        self.task_info_label_left = QLabel(task_text)
        self.task_info_label_left.setFont(self.header_font)
        self.task_info_label_left.setStyleSheet(f"color: {self.colors['text_primary'].name()}; font-weight: 600;")
        left_info_layout.addWidget(self.task_info_label_left, stretch=1)

        # 进度信息
        self.episode_progress_label_left = QLabel("0 条")
        self.episode_progress_label_left.setFont(self.body_font)
        self.episode_progress_label_left.setStyleSheet(f"color: {self.colors['text_secondary'].name()};")
        left_info_layout.addWidget(self.episode_progress_label_left)

        self.frames_progress_label_left = QLabel("0 帧")
        self.frames_progress_label_left.setFont(self.body_font)
        self.frames_progress_label_left.setStyleSheet(f"color: {self.colors['accent_blue'].name()}; font-weight: 600;")
        left_info_layout.addWidget(self.frames_progress_label_left)
        
        left_layout.addWidget(left_info_bar)
        
        # 摄像头布局改为网格，防止重叠
        cameras_grid = QGridLayout()
        cameras_grid.setSpacing(20)
        cameras_grid.setRowStretch(0, 1)
        cameras_grid.setColumnStretch(0, 1)
        cameras_grid.setColumnStretch(1, 1)

        # 头部大画面模式：隐藏左右手视频框，腾出空间让头部框更大
        # - 默认开启；如需恢复三路显示：QT_ONLY_HEAD_CAMERA_VIEW=0
        only_head_view = _env_flag("QT_ONLY_HEAD_CAMERA_VIEW", default=True)
        show_left_view = (not only_head_view) and (not _sn_disabled(REALSENSE_LEFT_SN))
        show_right_view = (not only_head_view) and (not _sn_disabled(REALSENSE_RIGHT_SN))

        # 顶部大摄像头 (头部) - 占据第0行，跨2列
        self.head_camera_display = QLabel("等待图像...")
        self.head_camera_display.setObjectName("CameraDisplay")
        self.head_camera_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 允许窗口自由缩放：这里只给一个较小的最小值，避免布局被最小尺寸“撑爆”
        # 实际显示大小交给布局与 SizePolicy 决定
        if only_head_view:
            self.head_camera_display.setMinimumSize(480, 320)
        else:
            self.head_camera_display.setMinimumSize(420, 280)
        self.head_camera_display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.head_camera_display.setStyleSheet(f"background-color: {self.colors['bg_card'].name()}; border: 1px solid {self.colors['border'].name()}; border-radius: 16px;")
        cameras_grid.addWidget(self.head_camera_display, 0, 0, 1, 2)
        
        # 左手摄像头 - 第1行，第0列
        self.left_hand_camera_display = QLabel("等待图像...")
        self.left_hand_camera_display.setObjectName("CameraDisplay")
        self.left_hand_camera_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_hand_camera_display.setMinimumSize(320, 240)
        self.left_hand_camera_display.setStyleSheet(f"background-color: {self.colors['bg_card'].name()}; border: 1px solid {self.colors['border'].name()}; border-radius: 16px;")
        if show_left_view:
            cameras_grid.addWidget(self.left_hand_camera_display, 1, 0)
        else:
            self.left_hand_camera_display.setVisible(False)
        
        # 右手摄像头 - 第1行，第1列
        self.right_hand_camera_display = QLabel("等待图像...")
        self.right_hand_camera_display.setObjectName("CameraDisplay")
        self.right_hand_camera_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_hand_camera_display.setMinimumSize(320, 240)
        self.right_hand_camera_display.setStyleSheet(f"background-color: {self.colors['bg_card'].name()}; border: 1px solid {self.colors['border'].name()}; border-radius: 16px;")
        if show_right_view:
            cameras_grid.addWidget(self.right_hand_camera_display, 1, 1)
        else:
            self.right_hand_camera_display.setVisible(False)

        if (not show_left_view) and (not show_right_view):
            cameras_grid.setRowStretch(1, 0)
        
        left_layout.addLayout(cameras_grid)
        splitter.addWidget(left_panel)
        
        # ========== 右侧面板：控制面板（可滚动，避免缩放时裁切/错位） ==========
        right_panel = QWidget()
        right_panel.setObjectName("ControlPanel")
        right_panel.setStyleSheet(
            f"background-color: {self.colors['bg_card'].name()}; border-left: 1px solid {self.colors['border'].name()};"
        )
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(15)
        # 右侧也不要锁死过大的最小宽度；缩小时用滚动承接
        right_panel.setMinimumWidth(420)
        right_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setWidget(right_panel)
        splitter.addWidget(right_scroll)

        # 初始分配：必须在 widget show 之后再 setSizes，否则会被 Qt 的 sizeHint 覆盖
        def _apply_initial_splitter_sizes() -> None:
            try:
                splitter.setStretchFactor(0, 3)  # 相机区
                splitter.setStretchFactor(1, 2)  # 控制区
                # 以当前窗口宽度为基准分配，避免不同屏幕下比例失真
                w = max(1400, int(self.width() or 0))
                left_w = int(w * 0.62)
                right_w = max(420, w - left_w)
                splitter.setSizes([left_w, right_w])
            except Exception:
                pass

        try:
            QTimer.singleShot(0, _apply_initial_splitter_sizes)
        except Exception:
            _apply_initial_splitter_sizes()
        
        # 返回按钮 (跨列显示)
        back_btn = QPushButton("← 返回任务管理")
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.setFixedHeight(45)
        back_btn.setStyleSheet(f"background-color: {self.colors['bg_secondary'].name()}; border-radius: 12px; font-weight: bold;")
        back_btn.clicked.connect(self._go_back_to_task_manager)
        right_layout.addWidget(back_btn)

        # 功能网格布局
        functional_grid = QGridLayout()
        functional_grid.setSpacing(15)
        # 两列响应式：让两列都可伸缩，避免缩放时某一列被挤到不可见
        try:
            functional_grid.setColumnStretch(0, 1)
            functional_grid.setColumnStretch(1, 1)
        except Exception:
            pass
        
        # --- 第一列第一行：实时状态（竖着）---
        status_card = QFrame()
        status_card.setStyleSheet(f"background-color: {self.colors['bg_secondary'].name()}; border-radius: 12px; padding: 5px;")
        status_card_layout = QVBoxLayout(status_card)
        status_card_layout.setSpacing(4)
        status_card_layout.setContentsMargins(10, 10, 10, 10)

        status_title = QLabel("实时状态")
        status_title.setFont(self.header_font)
        status_title.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-weight: 600; text-transform: uppercase;")
        status_card_layout.addWidget(status_title)

        self.timer_display = QLabel("00:00")
        timer_font = QFont(self.chinese_font_family, 32, QFont.Weight.Bold)
        self.timer_display.setFont(timer_font)
        self.timer_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            fm = QFontMetrics(timer_font)
            rect = fm.tightBoundingRect("00:00")
            self.timer_display.setFixedHeight(rect.height() + 28)
        except Exception:
            self.timer_display.setFixedHeight(68)
        self.timer_display.setStyleSheet("padding-top: 14px; padding-bottom: 6px;")
        status_card_layout.addWidget(self.timer_display)

        self.status_display = QLabel("就绪")
        self.status_display.setFont(self.status_font)
        self.status_display.setStyleSheet(f"color: {self.colors['accent_blue'].name()}; font-weight: 600;")
        self.status_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_card_layout.addWidget(self.status_display)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        status_card_layout.addWidget(self.progress_bar)

        self.collection_count_display = QLabel("0 / 1800 帧")
        self.collection_count_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.collection_count_display.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-size: 12px;")
        status_card_layout.addWidget(self.collection_count_display)

        self.episode_progress_label = QLabel("0 条")
        self.episode_progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.episode_progress_label.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-size: 12px;")
        status_card_layout.addWidget(self.episode_progress_label)
        
        status_card.setMinimumHeight(210)
        status_card.setMaximumHeight(240)
        functional_grid.addWidget(status_card, 0, 0)

        # --- 第二列第一行：采集控制 ---
        control_card = QFrame()
        control_card.setStyleSheet(f"background-color: {self.colors['bg_secondary'].name()}; border-radius: 12px; padding: 10px;")
        control_card_layout = QVBoxLayout(control_card)
        control_card_layout.setSpacing(8)

        collect_title = QLabel("采集控制")
        collect_title.setFont(self.header_font)
        collect_title.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-weight: 600; text-transform: uppercase;")
        control_card_layout.addWidget(collect_title)

        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("目标帧数"))
        self.duration_input = QSpinBox()
        self.duration_input.setRange(1, 100000)
        self.duration_input.setValue(1800)
        self.duration_input.setMinimumHeight(36)
        settings_layout.addWidget(self.duration_input)
        control_card_layout.addLayout(settings_layout)

        # Pico控制器状态显示
        pico_status_layout = QHBoxLayout()
        pico_status_layout.addWidget(QLabel("Pico状态:"))

        self.pico_status_label = QLabel("未启用")
        self.pico_status_label.setStyleSheet(f"color: {self.colors['text_tertiary'].name()}; font-weight: 500;")
        pico_status_layout.addWidget(self.pico_status_label)

        # Pico连接状态指示器
        self.pico_indicator = QLabel("●")
        self.pico_indicator.setStyleSheet(f"color: {self.colors['accent_red'].name()}; font-size: 16px; font-weight: bold;")
        self.pico_indicator.setFixedWidth(20)
        pico_status_layout.addWidget(self.pico_indicator)

        pico_status_layout.addStretch()
        control_card_layout.addLayout(pico_status_layout)

        # Pico控制选项（独立于采集/配置，始终显示；连接由按钮手动触发）
        self.pico_control_frame = QFrame()
        self.pico_control_frame.setVisible(True)
        self.pico_control_frame.setStyleSheet(f"background-color: {self.colors['bg_tertiary'].name()}; border-radius: 8px; padding: 8px; margin-top: 5px;")
        pico_control_layout = QVBoxLayout(self.pico_control_frame)
        pico_control_layout.setSpacing(6)

        pico_control_title = QLabel("Pico遥控器控制")
        pico_control_title.setFont(self.body_font)
        pico_control_title.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-weight: 600;")
        pico_control_layout.addWidget(pico_control_title)

        # Pico 连接参数（可编辑 + 记忆上次输入）
        pico_param_layout = QGridLayout()
        pico_param_layout.setHorizontalSpacing(10)
        pico_param_layout.setVerticalSpacing(6)

        pico_ip_label = QLabel("Pico IP")
        pico_ip_label.setStyleSheet(f"color: {self.colors['text_secondary'].name()};")
        pico_param_layout.addWidget(pico_ip_label, 0, 0)

        self.pico_ip_input = QLineEdit()
        self.pico_ip_input.setMinimumHeight(30)
        self.pico_ip_input.setPlaceholderText("例如：192.168.12.110")
        self.pico_ip_input.setStyleSheet(
            f"background-color: {self.colors['bg_secondary'].name()}; border-radius: 6px; padding: 4px 8px;"
        )
        pico_param_layout.addWidget(self.pico_ip_input, 0, 1)

        pico_port_label = QLabel("端口")
        pico_port_label.setStyleSheet(f"color: {self.colors['text_secondary'].name()};")
        pico_param_layout.addWidget(pico_port_label, 1, 0)

        self.pico_port_input = QSpinBox()
        self.pico_port_input.setRange(1, 65535)
        self.pico_port_input.setValue(12345)
        self.pico_port_input.setMinimumHeight(30)
        pico_param_layout.addWidget(self.pico_port_input, 1, 1)

        # 恢复上次输入（若存在）
        try:
            if getattr(self, "_settings", None) is not None:
                last_ip = str(self._settings.value("pico/last_ip", "192.168.12.110"))
                last_port = int(self._settings.value("pico/last_port", 12345))
                self.pico_ip_input.setText(last_ip)
                self.pico_port_input.setValue(last_port)
        except Exception:
            # 兜底默认
            self.pico_ip_input.setText("192.168.12.110")
            self.pico_port_input.setValue(12345)

        # 任意修改时立即写入设置（记忆上次输入）
        def _persist_pico_params():
            try:
                if getattr(self, "_settings", None) is None:
                    return
                self._settings.setValue("pico/last_ip", self.pico_ip_input.text().strip())
                self._settings.setValue("pico/last_port", int(self.pico_port_input.value()))
                self._settings.sync()
            except Exception:
                pass

        self.pico_ip_input.textChanged.connect(lambda _t: _persist_pico_params())
        self.pico_port_input.valueChanged.connect(lambda _v: _persist_pico_params())

        pico_control_layout.addLayout(pico_param_layout)

        # Pico连接/断开按钮（水平布局）
        pico_btn_layout = QHBoxLayout()
        pico_btn_layout.setSpacing(8)

        # Pico连接按钮
        self.pico_connect_btn = QPushButton("🔗 连接Pico")
        self.pico_connect_btn.setMinimumHeight(32)
        self.pico_connect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pico_connect_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['accent_blue'].name()};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {self.colors['accent_blue'].name()}dd;
            }}
            QPushButton:pressed {{
                background-color: {self.colors['accent_blue'].name()}bb;
            }}
            QPushButton:disabled {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_tertiary'].name()};
            }}
        """)
        self.pico_connect_btn.clicked.connect(self._connect_pico)
        # 初始状态：未连接
        self.pico_connect_btn.setEnabled(True)
        self.pico_connect_btn.setText("🔗 连接Pico")
        pico_btn_layout.addWidget(self.pico_connect_btn)

        # Pico断开按钮
        self.pico_disconnect_btn = QPushButton("🔌 断开Pico")
        self.pico_disconnect_btn.setMinimumHeight(32)
        self.pico_disconnect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pico_disconnect_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['accent_red'].name()};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {self.colors['accent_red'].name()}dd;
            }}
            QPushButton:pressed {{
                background-color: {self.colors['accent_red'].name()}bb;
            }}
            QPushButton:disabled {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_tertiary'].name()};
            }}
        """)
        self.pico_disconnect_btn.clicked.connect(self._disconnect_pico)
        # 初始状态：未连接时禁用
        self.pico_disconnect_btn.setEnabled(False)
        pico_btn_layout.addWidget(self.pico_disconnect_btn)

        pico_control_layout.addLayout(pico_btn_layout)

        control_card_layout.addWidget(self.pico_control_frame)

        self.robot_prep_cb = QCheckBox("开始前执行机器人准备动作（zhunbei 轨迹，完成后左 A/B 遥操、右 A/B 采图）")
        self.robot_prep_cb.setChecked(True)
        self.robot_prep_cb.setStyleSheet(f"color: {self.colors['text_secondary'].name()};")
        control_card_layout.addWidget(self.robot_prep_cb)

        self.pico_only_collection_cb = QCheckBox("相机由 Pico 右 A 启动（界面「开始采集」不启相机）")
        self.pico_only_collection_cb.setChecked(False)
        self.pico_only_collection_cb.setStyleSheet(f"color: {self.colors['text_secondary'].name()};")
        control_card_layout.addWidget(self.pico_only_collection_cb)

        btns_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始采集")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setStyleSheet(f"background-color: {self.colors['accent_blue'].name()}; color: white; border-radius: 12px; font-weight: bold; font-size: 15px;")
        self.start_btn.clicked.connect(self.start)
        btns_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("结束采集")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setStyleSheet(f"background-color: {self.colors['accent_red'].name()}; color: white; border-radius: 12px; font-weight: bold; font-size: 15px;")
        self.stop_btn.clicked.connect(self.stop)
        btns_layout.addWidget(self.stop_btn)
        control_card_layout.addLayout(btns_layout)
        functional_grid.addWidget(control_card, 0, 1)

        # --- 第二行：机器人配置（横着，跨2列）---
        config_card = QFrame()
        config_card.setStyleSheet(f"background-color: {self.colors['bg_secondary'].name()}; border-radius: 12px; padding: 10px;")
        config_card_layout = QHBoxLayout(config_card)  # 改为水平布局
        config_card_layout.setSpacing(15)
        config_card_layout.setContentsMargins(15, 10, 15, 10)
        
        cfg_label = QLabel("机器人配置")
        cfg_label.setFont(self.header_font)
        cfg_label.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-weight: 600; text-transform: uppercase;")
        cfg_label.setFixedWidth(120)  # 固定标签宽度
        config_card_layout.addWidget(cfg_label)
        
        config_files = self._scan_config_files()
        self.cfg_combobox = QComboBox()
        self.cfg_combobox.setView(QListView())
        self.cfg_combobox.setMinimumHeight(44)
        for cfg in config_files:
            self.cfg_combobox.addItem(cfg['display'], cfg['path'])
        config_card_layout.addWidget(self.cfg_combobox, stretch=1)  # 下拉框占据剩余空间
        functional_grid.addWidget(config_card, 1, 0, 1, 2)

        # --- 第三行：动作步骤 (单独占一行) ---
        steps_card = QFrame()
        steps_card.setStyleSheet(f"background-color: {self.colors['bg_secondary'].name()}; border-radius: 12px; padding: 10px;")
        steps_card_layout = QVBoxLayout(steps_card)
        steps_card_layout.setSpacing(8)

        steps_title = QLabel("动作步骤")
        steps_title.setFont(self.header_font)
        steps_title.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-weight: 600; text-transform: uppercase;")
        steps_card_layout.addWidget(steps_title)

        self.steps_combobox = QComboBox()
        self.steps_combobox.setView(QListView()) # 显式设置视图以解决样式失效问题
        self.steps_combobox.setMinimumHeight(40)
        action_steps_files = self._scan_action_steps_files()
        for steps_file in action_steps_files:
            self.steps_combobox.addItem(steps_file['display'], steps_file['path'])
        self.steps_combobox.currentTextChanged.connect(self._on_action_steps_file_changed)
        steps_card_layout.addWidget(self.steps_combobox)

        self.timeline_widget = TimelineWidget(self, colors=self.colors, chinese_font_family=self.chinese_font_family)
        # 缩短时间轴区域，给“实时状态”更多空间，尽量做到单页无需滚动
        self.timeline_widget.setMinimumSize(400, 190)
        steps_card_layout.addWidget(self.timeline_widget)
        functional_grid.addWidget(steps_card, 2, 0, 1, 2)

        # 右侧上半部分：固定内容
        right_layout.addLayout(functional_grid)

        # 让“动作步骤/时间轴”区域优先增长；在较小窗口下由右侧滚动承接
        try:
            right_layout.setStretchFactor(steps_card, 1)
        except Exception:
            pass

        # 状态指示灯（放在底部按钮上方）
        indicator_layout = QHBoxLayout()
        self.left_status_grid = StatusGridWidget()
        self.left_status_grid.setFixedSize(40, 30)
        self.right_status_grid = StatusGridWidget()
        self.right_status_grid.setFixedSize(40, 30)
        indicator_layout.addWidget(self.left_status_grid)
        indicator_layout.addStretch()
        indicator_layout.addWidget(self.right_status_grid)
        right_layout.addLayout(indicator_layout)

        # 底部取消按钮（固定贴底）
        self.cancel_btn = QPushButton("取消采集")
        self.cancel_btn.setFixedHeight(45)
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {self.colors['accent_red'].name()};
                border: 1px solid {self.colors['accent_red'].name()};
                border-radius: 12px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['accent_red'].name()};
                color: white;
            }}
        """)
        self.cancel_btn.clicked.connect(self.cancel_collection)
        right_layout.addWidget(self.cancel_btn)
        # 注意：right_panel 已经作为 right_scroll 的内容加入 splitter，
        # 不能再次 addWidget，否则会导致布局错乱/右侧空白。

        # 初始化动作步骤列表和状态指示器
        self._init_action_steps()
        self._draw_status_grids()
    
    def _create_datasets_tab_ui(self, parent: QWidget):
        """创建数据集管理标签页UI - 苹果风格"""
        datasets_layout = QVBoxLayout(parent)
        datasets_layout.setContentsMargins(32, 32, 32, 32)
        datasets_layout.setSpacing(24)
        
        # 顶部标题
        datasets_title = QLabel("数据集管理")
        datasets_title.setFont(self.title_font)
        datasets_title.setStyleSheet(f"color: {self.colors['text_primary'].name()}; font-weight: 700;")
        datasets_layout.addWidget(datasets_title)
        
        # 工具栏卡片
        toolbar_card = QFrame()
        toolbar_card.setStyleSheet(f"""
            QFrame {{
                background-color: {self.colors['bg_card'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 14px;
                padding: 10px;
            }}
        """)
        toolbar_layout = QHBoxLayout(toolbar_card)
        toolbar_layout.setContentsMargins(15, 10, 15, 10)
        toolbar_layout.setSpacing(15)
        
        refresh_btn = QPushButton("刷新列表")
        refresh_btn.setObjectName("PrimaryButton")
        refresh_btn.setFixedHeight(40)
        refresh_btn.clicked.connect(self._refresh_datasets)
        toolbar_layout.addWidget(refresh_btn)
        
        toolbar_layout.addSpacing(20)
        
        task_filter_label = QLabel("按任务筛选:")
        task_filter_label.setFont(self.body_font)
        task_filter_label.setStyleSheet(f"color: {self.colors['text_secondary'].name()};")
        toolbar_layout.addWidget(task_filter_label)
        
        self.task_filter_combo = QComboBox()
        self.task_filter_combo.setView(QListView()) # 显式设置视图以解决样式失效问题
        self.task_filter_combo.setMinimumWidth(200)
        self.task_filter_combo.setFixedHeight(40)
        self.task_filter_combo.addItem("全部任务", None)
        self.task_filter_combo.currentIndexChanged.connect(self._refresh_datasets)
        toolbar_layout.addWidget(self.task_filter_combo)
        
        toolbar_layout.addStretch()
        datasets_layout.addWidget(toolbar_card)
        
        # 数据集列表表格
        self.dataset_table = QTableWidget()
        self.dataset_table.setColumnCount(10)
        self.dataset_table.setHorizontalHeaderLabels([
            "数据集ID", "任务ID", "任务名", "Episodes", "目标条数", "任务进度", "总帧数", "创建时间", "状态", "上传"
        ])
        self.dataset_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.dataset_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.dataset_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.dataset_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.dataset_table.setAlternatingRowColors(False) # 禁用交替颜色，避免颜色冲突
        self.dataset_table.verticalHeader().setVisible(False)
        self.dataset_table.setShowGrid(False)
        datasets_layout.addWidget(self.dataset_table)
        
        # 底部操作栏
        action_layout = QHBoxLayout()
        action_layout.setSpacing(15)
        
        self.convert_btn = QPushButton("转换为 HDF5 格式")
        self.convert_btn.setMinimumHeight(44)
        self.convert_btn.clicked.connect(self._convert_selected_dataset)
        action_layout.addWidget(self.convert_btn)

        self.browse_convert_btn = QPushButton("浏览并转换HDF5")
        self.browse_convert_btn.setMinimumHeight(44)
        self.browse_convert_btn.clicked.connect(self._browse_and_convert_hdf5)
        action_layout.addWidget(self.browse_convert_btn)

        self.upload_info_btn = QPushButton("上报数据信息到平台")
        self.upload_info_btn.setObjectName("PrimaryButton")
        self.upload_info_btn.setMinimumHeight(44)
        self.upload_info_btn.clicked.connect(self._upload_selected_dataset_info)
        action_layout.addWidget(self.upload_info_btn)
        
        action_layout.addStretch()
        datasets_layout.addLayout(action_layout)
        
        # 存储数据集信息
        self.datasets_info = []
        self.current_dataset_root = None
        
        # 初始化时刷新列表
        self._refresh_datasets()

    def _create_convert_hdf5_tab_ui(self, parent: QWidget):
        """转换HDF5数据标签页：可浏览选择+勾选+进度（模板化）"""
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(18)

        title = QLabel("转换HDF5数据")
        title.setFont(self.title_font)
        title.setStyleSheet(f"color: {self.colors['text_primary'].name()}; font-weight: 700;")
        layout.addWidget(title)

        desc = QLabel("选择包含 LeRobot 数据集的根目录，扫描后勾选要转换的 repo 目录（包含 data/train 下的 parquet），并开始转换。")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {self.colors['text_secondary'].name()};")
        layout.addWidget(desc)

        top_card = QFrame()
        top_card.setStyleSheet(f"""
            QFrame {{
                background-color: {self.colors['bg_card'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 14px;
                padding: 12px;
            }}
        """)
        top_row = QHBoxLayout(top_card)
        top_row.setContentsMargins(12, 10, 12, 10)
        top_row.setSpacing(10)

        top_row.addWidget(QLabel("数据集根目录："))
        self.hdf5_convert_root_edit = QLineEdit(os.environ.get("LEROBOT_HOME", str(Path.home())))
        top_row.addWidget(self.hdf5_convert_root_edit, stretch=1)
        browse_btn = QPushButton("浏览…")
        scan_btn = QPushButton("扫描数据集")

        def _browse():
            picked = QFileDialog.getExistingDirectory(self, "选择数据集根目录", self.hdf5_convert_root_edit.text())
            if picked:
                self.hdf5_convert_root_edit.setText(picked)

        browse_btn.clicked.connect(_browse)
        scan_btn.clicked.connect(self._scan_convert_candidates)
        top_row.addWidget(browse_btn)
        top_row.addWidget(scan_btn)
        layout.addWidget(top_card)

        # 配置选择卡片（自动/手动 JSON）
        cfg_card = QFrame()
        cfg_card.setStyleSheet(f"""
            QFrame {{
                background-color: {self.colors['bg_card'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 14px;
                padding: 12px;
            }}
        """)
        cfg_layout = QHBoxLayout(cfg_card)
        cfg_layout.setContentsMargins(12, 10, 12, 10)
        cfg_layout.setSpacing(10)
        cfg_layout.addWidget(QLabel("配置文件(JSON)："))
        self.hdf5_convert_cfg_edit = QLineEdit("")
        self.hdf5_convert_cfg_edit.setPlaceholderText("留空=自动使用 hdf5_configs/ 下第一个配置或 Python 默认配置")
        cfg_layout.addWidget(self.hdf5_convert_cfg_edit, stretch=1)
        self.hdf5_convert_auto_cfg = QCheckBox("自动选择配置（推荐）")
        self.hdf5_convert_auto_cfg.setChecked(True)
        cfg_layout.addWidget(self.hdf5_convert_auto_cfg)
        cfg_browse_btn = QPushButton("浏览…")

        def _cfg_initial_dir() -> str:
            try:
                proj_root = Path(__file__).resolve().parent.parent
                d = proj_root / "hdf5_configs"
                if d.exists():
                    return str(d)
            except Exception:
                pass
            return str(Path.home())

        def _browse_cfg():
            picked, _ = QFileDialog.getOpenFileName(
                self,
                "选择 HDF5 转换配置文件",
                _cfg_initial_dir(),
                "JSON Files (*.json);;All Files (*)",
            )
            if picked:
                self.hdf5_convert_cfg_edit.setText(picked)
                self.hdf5_convert_auto_cfg.setChecked(False)

        def _toggle_cfg():
            use_auto = self.hdf5_convert_auto_cfg.isChecked()
            self.hdf5_convert_cfg_edit.setEnabled(not use_auto)
            cfg_browse_btn.setEnabled(not use_auto)

        cfg_browse_btn.clicked.connect(_browse_cfg)
        self.hdf5_convert_auto_cfg.stateChanged.connect(_toggle_cfg)
        _toggle_cfg()

        cfg_layout.addWidget(cfg_browse_btn)
        layout.addWidget(cfg_card)

        # 输出目录（COPASI output_root）：改为全自动（按天目录），不再要求用户手动选择
        out_card = QFrame()
        out_card.setStyleSheet(f"""
            QFrame {{
                background-color: {self.colors['bg_card'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 14px;
                padding: 12px;
            }}
        """)
        out_layout = QHBoxLayout(out_card)
        out_layout.setContentsMargins(12, 10, 12, 10)
        out_layout.setSpacing(10)
        out_layout.addWidget(QLabel("输出目录(output_root)："))
        self.hdf5_convert_output_root_edit = QLineEdit("")
        self.hdf5_convert_output_root_edit.setPlaceholderText("已改为自动：使用配置文件 output_root 下的 YYYYMMDD_host 目录")
        out_layout.addWidget(self.hdf5_convert_output_root_edit, stretch=1)
        out_browse = QPushButton("浏览…")

        def _browse_out():
            picked = QFileDialog.getExistingDirectory(self, "选择输出目录（将创建 Logistics/…）", self.hdf5_convert_output_root_edit.text() or str(Path.home()))
            if picked:
                self.hdf5_convert_output_root_edit.setText(picked)

        out_browse.clicked.connect(_browse_out)
        out_layout.addWidget(out_browse)
        layout.addWidget(out_card)

        # 隐藏手动输出目录与时间戳选项（按用户需求：无需手动选择，按天自动归档）
        try:
            out_card.setVisible(False)
            out_browse.setVisible(False)
            self.hdf5_convert_output_root_edit.setVisible(False)
        except Exception:
            pass

        self.hdf5_convert_output_root_add_ts = QCheckBox("output_root目录自动加时间戳（已弃用）")
        self.hdf5_convert_output_root_add_ts.setChecked(False)
        self.hdf5_convert_output_root_add_ts.setVisible(False)
        layout.addWidget(self.hdf5_convert_output_root_add_ts)

        self.hdf5_convert_timestamped_logistics = QCheckBox("Logistics目录加时间戳（已弃用）")
        self.hdf5_convert_timestamped_logistics.setChecked(False)
        self.hdf5_convert_timestamped_logistics.setVisible(False)
        layout.addWidget(self.hdf5_convert_timestamped_logistics)

        self.hdf5_convert_table = QTableWidget()
        self.hdf5_convert_table.setColumnCount(6)
        self.hdf5_convert_table.setHorizontalHeaderLabels(["选择", "repo_id/目录名", "parquet数量", "路径", "进度", "状态"])
        self.hdf5_convert_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.hdf5_convert_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.hdf5_convert_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.hdf5_convert_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.hdf5_convert_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.hdf5_convert_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.hdf5_convert_table.verticalHeader().setVisible(False)
        self.hdf5_convert_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.hdf5_convert_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.hdf5_convert_table.setShowGrid(False)
        layout.addWidget(self.hdf5_convert_table)

        bottom = QHBoxLayout()
        self.hdf5_convert_status = QLabel("未扫描")
        self.hdf5_convert_status.setStyleSheet(f"color: {self.colors['text_secondary'].name()};")
        bottom.addWidget(self.hdf5_convert_status)
        bottom.addStretch(1)
        select_all = QPushButton("全选")
        unselect_all = QPushButton("取消全选")
        start_btn = QPushButton("开始转换选中项")
        start_btn.setObjectName("PrimaryButton")

        select_all.clicked.connect(lambda: self._set_table_check_all(self.hdf5_convert_table, True))
        unselect_all.clicked.connect(lambda: self._set_table_check_all(self.hdf5_convert_table, False))
        start_btn.clicked.connect(self._convert_checked_candidates)

        bottom.addWidget(select_all)
        bottom.addWidget(unselect_all)
        bottom.addWidget(start_btn)
        layout.addLayout(bottom)

        self._hdf5_convert_candidates: List[Dict[str, Any]] = []

        # 批量转换进度（本页内显示）
        self.hdf5_convert_overall = QProgressBar()
        self.hdf5_convert_overall.setMinimum(0)
        self.hdf5_convert_overall.setMaximum(100)
        self.hdf5_convert_overall.setValue(0)
        self.hdf5_convert_overall.setFormat("总进度 0%")
        self.hdf5_convert_overall.setTextVisible(True)
        layout.addWidget(self.hdf5_convert_overall)

        self.hdf5_convert_current = QProgressBar()
        self.hdf5_convert_current.setMinimum(0)
        self.hdf5_convert_current.setMaximum(0)
        self.hdf5_convert_current.setValue(0)
        self.hdf5_convert_current.setFormat("当前数据集：等待开始…")
        self.hdf5_convert_current.setTextVisible(True)
        layout.addWidget(self.hdf5_convert_current)

        # 视频导出：用户要求默认转换 RGB + Depth 视频
        self.hdf5_convert_export_videos = QCheckBox("导出RGB视频（很慢）")
        self.hdf5_convert_export_videos.setChecked(True)
        layout.addWidget(self.hdf5_convert_export_videos)

        self.hdf5_convert_export_depth_videos = QCheckBox("导出深度视频（默认）")
        self.hdf5_convert_export_depth_videos.setChecked(True)
        layout.addWidget(self.hdf5_convert_export_depth_videos)

        self.hdf5_convert_cancel_btn = QPushButton("取消转换")
        self.hdf5_convert_cancel_btn.setEnabled(False)
        self.hdf5_convert_cancel_btn.clicked.connect(self._cancel_batch_convert)
        layout.addWidget(self.hdf5_convert_cancel_btn)

        self._batch_convert_queue: List[Dict[str, Any]] = []
        self._batch_convert_total: int = 0
        self._batch_convert_done: int = 0
        self._batch_convert_running: bool = False
        self._batch_convert_cancel_event = None

        self._hdf5_convert_row_map: Dict[str, int] = {}
        # 批量转换输出根目录（按天）：每个 repo_id 会落到 <daily_root>/<repo_id>/ 下
        self._batch_convert_output_root_base: Optional[str] = None

    def _create_upload_hdf5_tab_ui(self, parent: QWidget):
        """上传HDF5数据标签页：按截图风格（目录选择+扫描+勾选+上传进度）"""
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(18)

        title = QLabel("上传HDF5数据")
        title.setFont(self.title_font)
        title.setStyleSheet(f"color: {self.colors['text_primary'].name()}; font-weight: 700;")
        layout.addWidget(title)

        tip = QLabel("上传已转换好的 HDF5 数据集到对象存储。请选择输出目录（包含 Logistics 子目录），扫描后勾选要上传的 UUID 数据集目录。")
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {self.colors['text_secondary'].name()};")
        layout.addWidget(tip)

        # 顶部输入区
        input_card = QFrame()
        input_card.setStyleSheet(f"""
            QFrame {{
                background-color: {self.colors['bg_card'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 14px;
                padding: 12px;
            }}
        """)
        input_layout = QGridLayout(input_card)
        input_layout.setContentsMargins(12, 10, 12, 10)
        input_layout.setHorizontalSpacing(10)
        input_layout.setVerticalSpacing(10)

        input_layout.addWidget(QLabel("任务ID："), 0, 0)
        self.hdf5_upload_task_id = QLineEdit("1")
        self.hdf5_upload_task_id.setFixedWidth(120)
        input_layout.addWidget(self.hdf5_upload_task_id, 0, 1)
        hint = QLabel("（上传后会自动调用 batch-upload 接口上报数据信息）")
        hint.setStyleSheet(f"color: {self.colors['text_secondary'].name()};")
        input_layout.addWidget(hint, 0, 2, 1, 3)

        input_layout.addWidget(QLabel("数据集根目录："), 1, 0)
        self.hdf5_upload_root_edit = QLineEdit(str(Path.home()))
        input_layout.addWidget(self.hdf5_upload_root_edit, 1, 1, 1, 3)
        browse_btn = QPushButton("浏览…")
        scan_btn = QPushButton("扫描数据集")
        browse_btn.clicked.connect(self._browse_hdf5_upload_root)
        scan_btn.clicked.connect(self._scan_hdf5_upload_root)
        input_layout.addWidget(browse_btn, 1, 4)
        input_layout.addWidget(scan_btn, 1, 5)

        layout.addWidget(input_card)

        # 表格
        self.hdf5_upload_table = QTableWidget()
        self.hdf5_upload_table.setColumnCount(5)
        self.hdf5_upload_table.setHorizontalHeaderLabels(["选择", "UUID/数据集名", "文件数", "大小", "路径"])
        self.hdf5_upload_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.hdf5_upload_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.hdf5_upload_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.hdf5_upload_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.hdf5_upload_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.hdf5_upload_table.verticalHeader().setVisible(False)
        self.hdf5_upload_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.hdf5_upload_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.hdf5_upload_table.setShowGrid(False)
        layout.addWidget(self.hdf5_upload_table)

        # 底部操作
        bottom = QHBoxLayout()
        self.hdf5_upload_status = QLabel("未扫描")
        self.hdf5_upload_status.setStyleSheet(f"color: {self.colors['text_secondary'].name()};")
        bottom.addWidget(self.hdf5_upload_status)
        bottom.addStretch(1)

        select_all = QPushButton("全选")
        unselect_all = QPushButton("取消全选")
        upload_btn = QPushButton("上传选中文件")
        upload_btn.setObjectName("PrimaryButton")
        select_all.clicked.connect(lambda: self._set_table_check_all(self.hdf5_upload_table, True))
        unselect_all.clicked.connect(lambda: self._set_table_check_all(self.hdf5_upload_table, False))
        upload_btn.clicked.connect(self._upload_checked_hdf5_items)
        cancel_btn = QPushButton("取消上传")
        cancel_btn.clicked.connect(self._cancel_hdf5_upload)
        bottom.addWidget(select_all)
        bottom.addWidget(unselect_all)
        bottom.addWidget(upload_btn)
        bottom.addWidget(cancel_btn)
        layout.addLayout(bottom)

        # 进度条
        self.hdf5_upload_progress = QProgressBar()
        self.hdf5_upload_progress.setMinimum(0)
        self.hdf5_upload_progress.setMaximum(100)
        self.hdf5_upload_progress.setValue(0)
        self.hdf5_upload_progress.setFormat("0%")
        self.hdf5_upload_progress.setTextVisible(True)
        layout.addWidget(self.hdf5_upload_progress)

    def _set_table_check_all(self, table: QTableWidget, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is not None:
                item.setCheckState(state)

    def _browse_hdf5_upload_root(self) -> None:
        picked = QFileDialog.getExistingDirectory(self, "选择输出目录（包含 Logistics 子目录）", self.hdf5_upload_root_edit.text())
        if picked:
            self.hdf5_upload_root_edit.setText(picked)

    def _scan_hdf5_upload_root(self) -> None:
        """扫描 Logistics 下的 UUID 目录并填充表格。"""
        if self._upload_hdf5_scanning:
            return
        root = (self.hdf5_upload_root_edit.text() or "").strip()
        if not root:
            QMessageBox.warning(self, "提示", "请先选择输出目录")
            return
        self._upload_hdf5_scanning = True
        self.hdf5_upload_status.setText("扫描中…")
        self.hdf5_upload_progress.setValue(0)
        self.hdf5_upload_progress.setFormat("扫描中…")

        import threading
        from pathlib import Path as _Path
        import os as _os

        def worker():
            base = _Path(root).expanduser()
            # 兼容：
            # - output_root/Logistics
            # - output_root/Logistics_YYYYMMDD_HHMMSS（或其它 Logistics*）
            # - 用户直接选中了某个 Logistics* 目录
            logistics_roots: List[_Path] = []
            if base.is_dir() and base.name.lower().startswith("logistics"):
                logistics_roots = [base]
            else:
                # 优先找 output_root 下的所有 Logistics* 目录
                try:
                    logistics_roots = sorted([p for p in base.glob("Logistics*") if p.is_dir()], key=lambda p: p.name)
                except Exception:
                    logistics_roots = []
                if not logistics_roots:
                    # 回退到旧逻辑：base/Logistics 或 base 本身
                    logistics_roots = [base / "Logistics"] if (base / "Logistics").exists() else [base]
            items: List[Dict[str, Any]] = []
            # 过滤不存在的 root
            logistics_roots = [p for p in logistics_roots if p.exists()]
            if not logistics_roots:
                self.hdf5_upload_scan_done_signal.emit([])
                return

            for logistics in logistics_roots:
                # 扫描支持两种布局：
                # A) Logistics*/Logistics-xxx/<sub>/<action>/<uuid>/proprio_stats/proprio_stats*.hdf5   (api.md 示例)
                # B) Logistics*/<sub>/<action>/<uuid>/proprio_stats/proprio_stats*.hdf5               (更常见的“无 batch 层”输出)

                def _maybe_add_uuid(uuid_dir: _Path) -> None:
                    if not uuid_dir.is_dir():
                        return
                    ps_dir = uuid_dir / "proprio_stats"
                    if (not ps_dir.exists()) or (not list(ps_dir.glob("proprio_stats*.hdf5"))):
                        return
                    file_count = 0
                    total_bytes = 0
                    for r, _ds, fs in _os.walk(uuid_dir):
                        for fn in fs:
                            fp = _Path(r) / fn
                            try:
                                total_bytes += fp.stat().st_size
                                file_count += 1
                            except Exception:
                                pass
                    try:
                        rel_path = str(uuid_dir.relative_to(logistics.parent))
                    except Exception:
                        rel_path = str(uuid_dir)
                    items.append(
                        {
                            "uuid": uuid_dir.name,
                            "path": str(uuid_dir),
                            "rel_path": rel_path,
                            "file_count": file_count,
                            "size_bytes": total_bytes,
                            "logistics_root": str(logistics),
                        }
                    )

                # 先判断是否存在 Layout A 的 batch 层（Logistics-xxx 或 Logistics）
                main_candidates = [p for p in logistics.iterdir() if p.is_dir() and p.name != "task_info"]
                batch_dirs = [p for p in main_candidates if p.name.startswith("Logistics-") or p.name == "Logistics"]
                if batch_dirs:
                    # Layout A
                    for main_dir in batch_dirs:
                        for sub_dir in main_dir.iterdir():
                            if not sub_dir.is_dir():
                                continue
                            for action_dir in sub_dir.iterdir():
                                if not action_dir.is_dir():
                                    continue
                                for uuid_dir in action_dir.iterdir():
                                    _maybe_add_uuid(uuid_dir)
                else:
                    # Layout B：直接从 Logistics* 下的子目录开始
                    for sub_dir in main_candidates:
                        for action_dir in sub_dir.iterdir():
                            if not action_dir.is_dir():
                                continue
                            for uuid_dir in action_dir.iterdir():
                                _maybe_add_uuid(uuid_dir)
            self.hdf5_upload_scan_done_signal.emit(items)

        threading.Thread(target=worker, daemon=True).start()

    def _on_hdf5_upload_scan_done(self, items: list) -> None:
        self._upload_hdf5_scanning = False
        self._upload_hdf5_items = list(items or [])
        self.hdf5_upload_table.setRowCount(len(self._upload_hdf5_items))

        def _fmt_size(b: int) -> str:
            mb = b / (1024 * 1024)
            if mb < 1024:
                return f"{mb:.1f} MB"
            return f"{mb/1024:.2f} GB"

        for i, it in enumerate(self._upload_hdf5_items):
            chk = QTableWidgetItem("")
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk.setCheckState(Qt.CheckState.Checked)
            self.hdf5_upload_table.setItem(i, 0, chk)
            self.hdf5_upload_table.setItem(i, 1, QTableWidgetItem(str(it.get("uuid", ""))))
            self.hdf5_upload_table.setItem(i, 2, QTableWidgetItem(str(it.get("file_count", 0))))
            self.hdf5_upload_table.setItem(i, 3, QTableWidgetItem(_fmt_size(int(it.get("size_bytes", 0)))))
            self.hdf5_upload_table.setItem(i, 4, QTableWidgetItem(str(it.get("rel_path", ""))))

        self.hdf5_upload_status.setText(f"扫描完成：{len(self._upload_hdf5_items)} 条")
        self.hdf5_upload_progress.setValue(0)
        self.hdf5_upload_progress.setFormat("0%")

    def _upload_checked_hdf5_items(self) -> None:
        """上传勾选的 UUID 目录，并 batch-upload 上报。"""
        if self._upload_hdf5_uploading:
            QMessageBox.warning(self, "提示", "已有上传任务进行中，请等待完成")
            return

        # batch-upload 需要登录 token（x-auth-tkn）。如果未登录，则先弹出登录框。
        try:
            from .qt_login import get_auth_headers, LoginDialog
            if not get_auth_headers().get("x-auth-tkn"):
                dlg = LoginDialog(self)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    QMessageBox.warning(self, "提示", "未登录，已取消上传")
                    return
        except Exception:
            # 防御：如果登录模块不可用，继续走后续逻辑，由 server_api 给出明确错误
            pass
        # task_id
        task_id = (self.hdf5_upload_task_id.text() or "").strip()
        if not task_id.isdigit():
            QMessageBox.warning(self, "提示", "任务ID必须是数字")
            return

        selected: List[Dict[str, Any]] = []
        for row, it in enumerate(self._upload_hdf5_items):
            chk = self.hdf5_upload_table.item(row, 0)
            if chk is not None and chk.checkState() == Qt.CheckState.Checked:
                selected.append(it)
        if not selected:
            QMessageBox.warning(self, "提示", "请先勾选要上传的数据")
            return

        # 防止混选多个 Logistics 根目录导致 resource 计算异常（更稳）
        roots = sorted({str(it.get("logistics_root") or "") for it in selected if it.get("logistics_root")})
        if len(roots) > 1:
            reply = QMessageBox.question(
                self,
                "提示",
                "你选择了多个 Logistics 根目录的数据（可能来自不同批次）。\n\n"
                "这会导致 resource 规则更复杂，后端可能严格校验失败。\n"
                "建议一次只上传同一个 Logistics 根目录。\n\n"
                "仍要继续上传吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # 确认
        reply = QMessageBox.question(
            self,
            "确认上传",
            f"将上传 {len(selected)} 个 UUID 目录，并调用 batch-upload 上报。\n是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        import threading
        self._upload_hdf5_uploading = True
        self.hdf5_upload_status.setText("上传中…")
        self.hdf5_upload_progress.setValue(0)
        self.hdf5_upload_progress.setFormat("0%")
        cancel_event = threading.Event()
        self._upload_hdf5_cancel_event = cancel_event

        def worker():
            try:
                # 延迟导入（启动更快）
                from .server_api import (
                    api_get_session_token,
                    upload_path_to_minio,
                    api_batch_upload_data_info,
                    api_get_my_tasks,
                    ApiError,
                    DataInfoItem,
                )
                from pathlib import Path as _Path
                import re as _re

                sts = api_get_session_token()
                items_to_report: List[DataInfoItem] = []
                total = len(selected)

                # 预检：task_id 是否属于“我的任务”（否则后端很可能 400 invalid request parameter）
                try:
                    task_id_int = int(task_id)
                    tasks_resp = api_get_my_tasks(page=1, page_size=200)
                    task_items = (tasks_resp.get("data") or {}).get("items") or []
                    allowed_ids = {int(t.get("id")) for t in task_items if str(t.get("id", "")).isdigit()}
                    if task_id_int not in allowed_ids:
                        sample = ", ".join(str(x) for x in sorted(list(allowed_ids))[:20])
                        raise RuntimeError(
                            f"任务ID={task_id_int} 不在“我的任务”列表中（可能无权限/不存在）。"
                            f"请从任务列表复制正确的ID。当前可见ID示例: {sample}"
                        )
                except Exception as e:
                    raise RuntimeError(f"任务ID预检失败：{e}") from e

                for idx, it in enumerate(selected, start=1):
                    if cancel_event.is_set():
                        self.hdf5_upload_done_signal.emit(False, "用户取消")
                        return
                    uuid_dir = _Path(it["path"])
                    uuid_str = it["uuid"]
                    logistics_root = _Path(it.get("logistics_root") or "").expanduser()
                    if not logistics_root.exists():
                        # 如果缺少 logistics_root，则无法构造规范 resource，直接失败提示
                        raise RuntimeError(f"缺少有效 logistics_root，无法上传：uuid={uuid_str}")

                    # resource 路径：按参考文件格式（带 tenant 前缀，保持完整 Logistics 目录层级）
                    # 计算 UUID 在 Logistics 父目录下的相对路径（例如: Logistics/Logistics-xxx/Materialtransfer-xxx/Action-xxx/uuid）
                    try:
                        rel_path = uuid_dir.relative_to(logistics_root.parent)
                        rel_path_str = f"/{rel_path.as_posix()}"
                    except Exception:
                        # 如果无法计算相对路径，使用 uuid 作为相对路径
                        rel_path_str = f"/{uuid_str}"
                    
                    # 生成与 MinIO 上传一致的对象前缀（tenant-N 开头，保持 Logistics 目录层级）
                    minio_upload_prefix = f"tenant-{sts.tenant_id}{rel_path_str}".lstrip("/")
                    
                    # 直接使用与本地上传一致的路径上报给平台（前置一个 "/" 作为绝对路径）
                    resource = f"/{minio_upload_prefix}"

                    # 再做一次本地校验：uuid 必须是 resource 的最后一段
                    if not resource.endswith(f"/{uuid_str}"):
                        raise RuntimeError(f"resource 末段不是 uuid：uuid={uuid_str}, resource={resource}")
                    # 正则校验：resource 格式应为 /tenant-{tenant_id}/Logistics/.../<uuid>
                    resource_pattern = rf"^/tenant-{sts.tenant_id}/Logistics/.+/.+/[0-9a-fA-F\\-]{{36}}$"
                    if not _re.match(resource_pattern, resource):
                        logger.warning(f"resource 格式可能不符合预期：{resource}（继续上传）")

                    def progress_callback(filename, current_file, total_files, uploaded_bytes, total_bytes):
                        # 在本页进度条里展示“总体进度”：(已完成uuid + 当前uuid文件百分比)/总uuid
                        try:
                            file_pct = 0
                            if total_bytes and total_bytes > 0:
                                file_pct = int(uploaded_bytes / total_bytes * 100)
                            overall = int(((idx - 1) + (file_pct / 100.0)) / total * 100)
                            self.hdf5_upload_progress_signal.emit(overall, 100, f"上传中 {idx}/{total}：{uuid_str}（{file_pct}%）")
                        except Exception:
                            pass

                    # 对象存储实际对象前缀：使用与 resource 一致的 minio_upload_prefix（去掉前导 /）
                    custom_prefix = minio_upload_prefix
                    up_result = upload_path_to_minio(
                        str(uuid_dir),
                        sts,
                        progress_callback=progress_callback,
                        custom_object_prefix=custom_prefix,
                    )
                    size_mb = int(up_result.get("size_mb", 0))
                    # 按你的需求：上报 task_id, uuid, resource, size, duration（duration 固定 60）
                    items_to_report.append(
                        DataInfoItem(task_id=int(task_id), uuid=uuid_str, resource=resource, size=size_mb, duration=60)
                    )
                    # 记录备用 resource（已与主 resource 一致，保留字段以防重试逻辑需要）
                    it["_resource_with_tenant"] = resource
                    it["_size_mb"] = size_mb

                    self.hdf5_upload_progress_signal.emit(idx, total, f"已上传 {idx}/{total}")

                # batch upload
                try:
                    api_batch_upload_data_info(items_to_report)
                except ApiError as e:
                    # 重试逻辑（当前 resource 已包含 tenant 前缀，保留此逻辑以防后端格式要求变化）
                    code = None
                    try:
                        code = int((e.payload or {}).get("code", 0))
                    except Exception:
                        code = None
                    if code == 20001:
                        alt_items: List[DataInfoItem] = []
                        for it in selected:
                            uuid_str = it.get("uuid", "")
                            alt_res = it.get("_resource_with_tenant") or ""  # 已与主 resource 一致
                            if not alt_res:
                                raise RuntimeError(f"缺少备用 resource 用于重试：uuid={uuid_str}")
                            alt_size = int(it.get("_size_mb") or 0)
                            alt_items.append(DataInfoItem(task_id=int(task_id), uuid=uuid_str, resource=alt_res, size=alt_size, duration=60))
                        api_batch_upload_data_info(alt_items)
                    else:
                        raise
                self.hdf5_upload_done_signal.emit(True, f"上传完成：{len(items_to_report)} 条")
            except Exception as e:
                self.hdf5_upload_done_signal.emit(False, f"上传失败：{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_hdf5_upload_progress(self, current: int, total: int, message: str) -> None:
        # 兼容两种形态：
        # - (idx,total) 形式：转成百分比
        # - (percent,100) 形式：直接作为百分比
        if total > 0:
            percent = int(current / total * 100) if total != 100 else int(current)
            percent = max(0, min(100, percent))
            self.hdf5_upload_progress.setValue(percent)
            self.hdf5_upload_progress.setFormat(f"{percent}%")
        self.hdf5_upload_status.setText(message)

    def _on_hdf5_upload_done(self, success: bool, message: str) -> None:
        self._upload_hdf5_uploading = False
        self._upload_hdf5_cancel_event = None
        self.hdf5_upload_status.setText(message)
        if success:
            self.hdf5_upload_progress.setValue(100)
            self.hdf5_upload_progress.setFormat("100%")
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.critical(self, "失败", message)

    def _cancel_hdf5_upload(self) -> None:
        if self._upload_hdf5_cancel_event is not None:
            try:
                self._upload_hdf5_cancel_event.set()
            except Exception:
                pass
        self.hdf5_upload_status.setText("正在取消…（将在当前文件结束后停止）")

    def _scan_convert_candidates(self) -> None:
        """扫描可转换的数据集（repo 目录）。"""
        root = (getattr(self, "hdf5_convert_root_edit", None).text() or "").strip()  # type: ignore[union-attr]
        if not root:
            QMessageBox.warning(self, "提示", "请先选择数据集根目录")
            return
        base = Path(root).expanduser()
        if not base.exists():
            QMessageBox.warning(self, "提示", "目录不存在")
            return

        candidates: List[Dict[str, Any]] = []
        # 仅扫描一级子目录（repo）
        for p in base.iterdir():
            if not p.is_dir():
                continue
            data_dir = p / "data"
            train_dir = p / "train"
            parquet_count = 0
            try:
                if data_dir.exists():
                    parquet_count = len(list(data_dir.rglob("*.parquet")))
                if parquet_count == 0 and train_dir.exists():
                    parquet_count = len(list(train_dir.glob("*.parquet")))
            except Exception:
                parquet_count = 0
            if parquet_count > 0:
                candidates.append({"repo_id": p.name, "dataset_root": str(base), "path": str(p), "parquet_count": parquet_count})

        self._hdf5_convert_candidates = candidates
        self._hdf5_convert_row_map = {it["repo_id"]: idx for idx, it in enumerate(candidates)}
        self.hdf5_convert_table.setRowCount(len(candidates))
        for i, it in enumerate(candidates):
            chk = QTableWidgetItem("")
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk.setCheckState(Qt.CheckState.Checked)
            self.hdf5_convert_table.setItem(i, 0, chk)
            self.hdf5_convert_table.setItem(i, 1, QTableWidgetItem(it["repo_id"]))
            self.hdf5_convert_table.setItem(i, 2, QTableWidgetItem(str(it["parquet_count"])))
            self.hdf5_convert_table.setItem(i, 3, QTableWidgetItem(it["path"]))

            pb = QProgressBar()
            pb.setMinimum(0)
            pb.setMaximum(100)
            pb.setValue(0)
            pb.setFormat("0%")
            pb.setTextVisible(True)
            pb.setFixedWidth(140)
            self.hdf5_convert_table.setCellWidget(i, 4, pb)

            st = QTableWidgetItem("待转换")
            st.setForeground(QColor(self.colors['text_secondary']))
            self.hdf5_convert_table.setItem(i, 5, st)

        self.hdf5_convert_status.setText(f"扫描完成：{len(candidates)} 个可转换数据集")

    def _convert_checked_candidates(self) -> None:
        """批量转换勾选的数据集（总进度 + 当前数据集进度）。"""
        selected: List[Dict[str, Any]] = []
        for row, it in enumerate(getattr(self, "_hdf5_convert_candidates", [])):
            chk = self.hdf5_convert_table.item(row, 0)
            if chk is not None and chk.checkState() == Qt.CheckState.Checked:
                selected.append(it)
        if not selected:
            QMessageBox.warning(self, "提示", "请先勾选要转换的数据集")
            return
        if self._batch_convert_running:
            QMessageBox.warning(self, "提示", "已有批量转换在进行中")
            return

        # 读取配置
        cfg_file: Optional[str] = None
        try:
            if getattr(self, "hdf5_convert_auto_cfg").isChecked():  # type: ignore[union-attr]
                cfg_file = None
            else:
                cfg_file = (getattr(self, "hdf5_convert_cfg_edit").text() or "").strip() or None  # type: ignore[union-attr]
        except Exception:
            cfg_file = None

        import threading
        self._batch_convert_cancel_event = threading.Event()
        self._batch_convert_queue = selected
        self._batch_convert_total = len(selected)
        self._batch_convert_done = 0
        self._batch_convert_running = True
        self.hdf5_convert_cancel_btn.setEnabled(True)
        self.hdf5_convert_overall.setValue(0)
        self.hdf5_convert_overall.setFormat("总进度 0%")
        self.hdf5_convert_current.setMaximum(0)
        self.hdf5_convert_current.setValue(0)
        self.hdf5_convert_current.setFormat("当前数据集：准备开始…")
        self.hdf5_convert_status.setText(f"转换中：0/{self._batch_convert_total}")

        # 输出目录：按天自动构建，不再需要手动选择
        # 规则：使用配置文件的 output_root 作为 base，在其下创建 YYYYMMDD_<hostname> 目录；
        # 然后每个 repo_id 再各自落到 <daily_dir>/<repo_id>/ 下，避免不同原始数据混在一起。
        self._batch_convert_output_root_base = None
        try:
            import socket, re
            from pathlib import Path
            from .dataset_manager import _load_conversion_config

            cfg = _load_conversion_config(cfg_file)
            base_root = (cfg.get("output_root") or "").strip()
            if not base_root:
                # 若配置未给出 output_root，回退到用户主目录下的 hdf5_output
                base_root = str(Path.home() / "hdf5_output")
            base_root = str(Path(base_root).expanduser())
            Path(base_root).mkdir(parents=True, exist_ok=True)

            day = datetime.now().strftime("%Y%m%d")
            host = socket.gethostname() or "host"
            host_safe = re.sub(r"[^A-Za-z0-9._-]+", "-", host).strip("-") or "host"
            daily_dir = str(Path(base_root) / f"{day}_{host_safe}")
            Path(daily_dir).mkdir(parents=True, exist_ok=True)
            self._batch_convert_output_root_base = daily_dir
            self.hdf5_convert_status.setText(
                f"转换中：0/{self._batch_convert_total}（输出：{daily_dir}）"
            )
        except Exception:
            self._batch_convert_output_root_base = None

        self._run_next_batch_convert(cfg_file)

    def _cancel_batch_convert(self) -> None:
        if self._batch_convert_cancel_event is not None:
            self._batch_convert_cancel_event.set()
        self.hdf5_convert_cancel_btn.setEnabled(False)
        self.hdf5_convert_current.setFormat("当前数据集：正在取消…（将在当前文件结束后停止）")

    def _run_next_batch_convert(self, config_file: Optional[str]) -> None:
        """启动队列里的下一个转换（后台线程）。"""
        if self._batch_convert_cancel_event is not None and self._batch_convert_cancel_event.is_set():
            self._batch_convert_running = False
            self.hdf5_convert_status.setText("已取消")
            self.hdf5_convert_cancel_btn.setEnabled(False)
            return
        if not self._batch_convert_queue:
            self._batch_convert_running = False
            self.hdf5_convert_cancel_btn.setEnabled(False)
            self.hdf5_convert_overall.setValue(100)
            self.hdf5_convert_overall.setFormat("总进度 100%")
            self.hdf5_convert_current.setMaximum(1)
            self.hdf5_convert_current.setValue(1)
            self.hdf5_convert_current.setFormat("当前数据集：完成")
            self.hdf5_convert_status.setText("全部转换完成")
            # 刷新数据集管理页列表
            try:
                self._refresh_datasets()
            except Exception:
                pass
            return

        it = self._batch_convert_queue.pop(0)
        repo_id = it["repo_id"]
        dataset_root = it["dataset_root"]
        self.hdf5_convert_current.setMaximum(0)
        self.hdf5_convert_current.setValue(0)
        self.hdf5_convert_current.setFormat(f"当前数据集：{repo_id}")
        # 行状态
        try:
            row = self._hdf5_convert_row_map.get(repo_id)
            if row is not None:
                st_item = self.hdf5_convert_table.item(row, 5)
                if st_item is not None:
                    st_item.setText("转换中…")
        except Exception:
            pass

        import threading

        def progress_cb(cur: int, total: int, msg: str) -> None:
            # 当前数据集进度
            try:
                # 更新当前数据集总进度条（parquet级），同时推到“每行进度”
                self.hdf5_progress_signal.emit(cur, total, msg if msg else "")
                self.hdf5_item_progress_signal.emit(repo_id, cur, total, msg if msg else "")
            except Exception:
                pass

        def should_cancel() -> bool:
            return bool(self._batch_convert_cancel_event.is_set()) if self._batch_convert_cancel_event is not None else False

        def worker():
            from .dataset_manager import convert_parquet_to_hdf5
            export_videos = False
            try:
                export_videos = bool(self.hdf5_convert_export_videos.isChecked())
            except Exception:
                export_videos = False
            export_depth_videos = True
            try:
                export_depth_videos = bool(self.hdf5_convert_export_depth_videos.isChecked())
            except Exception:
                export_depth_videos = True
            # 用户希望每次转换生成顶层 Logistics_YYYYMMDD_HHMMSS 目录（便于肉眼统计批次）
            ts_logistics = True
            # 输出目录：直接落到当天目录（点开就看到 Logistics_YYYYMMDD_HHMMSS），不再嵌套 repo_id
            output_root_override = None
            try:
                from pathlib import Path
                if self._batch_convert_output_root_base:
                    output_root_override = str(Path(self._batch_convert_output_root_base))
                    Path(output_root_override).mkdir(parents=True, exist_ok=True)
            except Exception:
                output_root_override = None
            ok = convert_parquet_to_hdf5(
                repo_id,
                dataset_root,
                config_file=config_file,
                progress_callback=progress_cb,
                should_cancel=should_cancel,
                export_videos=export_videos,
                export_depth_videos=export_depth_videos,
                verbose=False,
                output_root_override=output_root_override,
                timestamped_logistics_dir=ts_logistics,
            )
            # 不要在子线程里用 QTimer.singleShot（可能不触发）；用信号回主线程更新 UI
            self.hdf5_batch_item_done_signal.emit(repo_id, bool(ok), config_file)

        threading.Thread(target=worker, daemon=True).start()

    def _on_one_batch_convert_done(self, repo_id: str, success: bool, config_file: Optional[str]) -> None:
        self._batch_convert_done += 1
        total = max(1, self._batch_convert_total)
        percent = int(self._batch_convert_done / total * 100)
        self.hdf5_convert_overall.setValue(percent)
        self.hdf5_convert_overall.setFormat(f"总进度 {percent}%")
        self.hdf5_convert_status.setText(f"转换中：{self._batch_convert_done}/{self._batch_convert_total}")

        # 如果用户点了取消：立即停止后续队列，并给出明确提示
        if self._batch_convert_cancel_event is not None and self._batch_convert_cancel_event.is_set():
            self._batch_convert_running = False
            self._batch_convert_queue = []
            self.hdf5_convert_cancel_btn.setEnabled(False)
            self.hdf5_convert_status.setText("已取消")
            QMessageBox.information(self, "已取消", "HDF5 转换已取消（当前文件已停止/或在当前文件结束后停止）")
            return

        if not success:
            # 不中断批处理：继续下一个，但在状态里提示失败
            self.hdf5_convert_current.setMaximum(1)
            self.hdf5_convert_current.setValue(1)
            self.hdf5_convert_current.setFormat(f"当前数据集：{repo_id} 转换失败（继续下一项）")
            try:
                row = self._hdf5_convert_row_map.get(repo_id)
                if row is not None:
                    st_item = self.hdf5_convert_table.item(row, 5)
                    if st_item is not None:
                        st_item.setText("转换失败")
                    pb = self.hdf5_convert_table.cellWidget(row, 4)
                    if isinstance(pb, QProgressBar):
                        pb.setFormat("失败")
            except Exception:
                pass
        else:
            self.hdf5_convert_current.setMaximum(1)
            self.hdf5_convert_current.setValue(1)
            self.hdf5_convert_current.setFormat(f"当前数据集：{repo_id} 转换完成")
            try:
                row = self._hdf5_convert_row_map.get(repo_id)
                if row is not None:
                    st_item = self.hdf5_convert_table.item(row, 5)
                    if st_item is not None:
                        st_item.setText("已完成")
                    pb = self.hdf5_convert_table.cellWidget(row, 4)
                    if isinstance(pb, QProgressBar):
                        pb.setValue(100)
                        pb.setFormat("100%")
            except Exception:
                pass

        self._run_next_batch_convert(config_file)
    
    def _create_robot_tab_ui(self, parent: QWidget):
        """创建机器人管理标签页UI - 苹果风格"""
        import json
        from .server_api import RobotStatus, api_upload_robot_status, ApiError
        
        main_layout = QVBoxLayout(parent)
        main_layout.setContentsMargins(32, 32, 32, 32)
        main_layout.setSpacing(24)
        
        # 标题
        title_label = QLabel("机器人管理")
        title_label.setFont(self.title_font)
        title_label.setStyleSheet(f"color: {self.colors['text_primary'].name()}; font-weight: 700;")
        main_layout.addWidget(title_label)
        
        # 说明卡片
        desc_card = QFrame()
        desc_card.setStyleSheet(f"""
            QFrame {{
                background-color: {self.colors['bg_secondary'].name()};
                border-radius: 12px;
                padding: 15px;
            }}
        """)
        desc_layout = QVBoxLayout(desc_card)
        desc_label = QLabel("管理机器人状态，可以添加机器人并上报运行状态到服务器。")
        desc_label.setFont(self.body_font)
        desc_label.setStyleSheet(f"color: {self.colors['text_secondary'].name()};")
        desc_layout.addWidget(desc_label)
        main_layout.addWidget(desc_card)
        
        # 加载机器人型号配置
        self.robot_models = self._load_robot_models()
        
        # 机器人列表表格
        self.robot_table = QTableWidget()
        self.robot_table.setColumnCount(6)
        self.robot_table.setHorizontalHeaderLabels(["设备名称", "SN", "型号", "是否双足", "状态", "操作"])
        header = self.robot_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.robot_table.setColumnWidth(2, 120)
        self.robot_table.setColumnWidth(3, 80)
        self.robot_table.setColumnWidth(4, 140)
        self.robot_table.setColumnWidth(5, 120)
        self.robot_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.robot_table.setAlternatingRowColors(False)
        self.robot_table.verticalHeader().setVisible(False)
        self.robot_table.setShowGrid(False)
        main_layout.addWidget(self.robot_table)
        
        # 添加机器人区域
        add_layout = QHBoxLayout()
        
        # 下拉菜单样式
        combo_style = f"""
            QComboBox {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_primary'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 150px;
            }}
            QComboBox:hover {{
                border-color: {self.colors['accent_blue'].name()};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {self.colors['text_secondary'].name()};
                margin-right: 10px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_primary'].name()};
                border: 1px solid {self.colors['border'].name()};
                selection-background-color: {self.colors['accent_blue'].name()};
            }}
        """
        
        # 选择机器人（下拉菜单显示名称和SN）
        robot_label = QLabel("选择机器人:")
        robot_label.setFont(self.body_font)
        robot_label.setStyleSheet(f"color: {self.colors['text_primary'].name()};")
        add_layout.addWidget(robot_label)
        
        self.robot_model_combo = QComboBox()
        self.robot_model_combo.setView(QListView()) # 显式设置视图以解决样式失效问题
        self.robot_model_combo.setFont(self.body_font)
        self.robot_model_combo.setMinimumWidth(300)
        self.robot_model_combo.setStyleSheet(combo_style)
        for robot in self.robot_models:
            display_text = f"{robot['name']} - {robot['sn']} ({robot['model']})"
            self.robot_model_combo.addItem(display_text, robot)
        add_layout.addWidget(self.robot_model_combo)
        
        add_btn = QPushButton("添加机器人")
        add_btn.setFont(self.body_font)
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['accent_green'].name()};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['accent_green'].lighter(110).name()};
            }}
        """)
        add_btn.clicked.connect(self._add_robot)
        add_layout.addWidget(add_btn)
        
        add_layout.addStretch()
        main_layout.addLayout(add_layout)
        
        # 操作按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        # 全部设为运行中
        set_running_btn = QPushButton("全部设为运行中")
        set_running_btn.setFont(self.body_font)
        set_running_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['accent_blue'].name()};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['accent_blue'].lighter(110).name()};
            }}
        """)
        set_running_btn.clicked.connect(lambda: self._set_all_robot_status(1))
        btn_layout.addWidget(set_running_btn)
        
        # 全部设为未运行
        set_stopped_btn = QPushButton("全部设为未运行")
        set_stopped_btn.setFont(self.body_font)
        set_stopped_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['bg_tertiary'].name()};
                color: {self.colors['text_primary'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 8px;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['bg_secondary'].name()};
            }}
        """)
        set_stopped_btn.clicked.connect(lambda: self._set_all_robot_status(0))
        btn_layout.addWidget(set_stopped_btn)
        
        # 上传状态按钮
        upload_btn = QPushButton("上传机器人状态")
        upload_btn.setFont(self.body_font)
        upload_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['accent_purple'].name()};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.colors['accent_purple'].lighter(110).name()};
            }}
        """)
        upload_btn.clicked.connect(self._upload_robot_status)
        btn_layout.addWidget(upload_btn)
        
        main_layout.addLayout(btn_layout)
        
        # 存储机器人列表
        self.robot_list: List[Dict[str, Any]] = []
        
        # 从配置加载机器人列表（如果有的话）
        self._load_robot_list()
    
    def _load_robot_models(self):
        """从配置文件加载机器人设备列表"""
        import json as _json
        config_path = Path(__file__).parent.parent / "configs" / "robot_models.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                    return data.get("robots", [])
            except Exception as e:
                logger.warning("加载机器人配置失败: %s", e)
        return []
    
    def _add_robot(self):
        """添加机器人到列表"""
        # 获取选中的机器人信息
        robot_data = self.robot_model_combo.currentData()
        if not robot_data:
            QMessageBox.warning(self, "添加失败", "请选择机器人")
            return
        
        sn = robot_data["sn"]
        
        # 检查是否已存在
        for robot in self.robot_list:
            if robot["sn"] == sn:
                QMessageBox.warning(self, "添加失败", f"机器人 {robot_data['name']} (SN: {sn}) 已在列表中")
                return
        
        # 添加到列表（包含完整信息）
        self.robot_list.append({
            "name": robot_data["name"],
            "sn": sn,
            "model": robot_data["model"],
            "is_biped": robot_data["is_biped"],
            "status": 0
        })
        self._refresh_robot_table()
        self._save_robot_list()
    
    def _remove_robot(self, sn: str):
        """从列表移除机器人"""
        self.robot_list = [r for r in self.robot_list if r["sn"] != sn]
        self._refresh_robot_table()
        self._save_robot_list()
    
    def _toggle_robot_status(self, sn: str):
        """切换机器人状态"""
        for robot in self.robot_list:
            if robot["sn"] == sn:
                robot["status"] = 1 if robot["status"] == 0 else 0
                break
        self._refresh_robot_table()
        self._save_robot_list()
    
    def _set_all_robot_status(self, status: int):
        """设置所有机器人的状态"""
        for robot in self.robot_list:
            robot["status"] = status
        self._refresh_robot_table()
        self._save_robot_list()
    
    def _refresh_robot_table(self):
        """刷新机器人表格"""
        self.robot_table.setRowCount(len(self.robot_list))
        
        for row, robot in enumerate(self.robot_list):
            # 设置行高
            self.robot_table.setRowHeight(row, 50)
            
            # 列0: 设备名称
            name_item = QTableWidgetItem(robot.get("name", "-"))
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.robot_table.setItem(row, 0, name_item)
            
            # 列1: SN
            sn_item = QTableWidgetItem(robot["sn"])
            sn_item.setFlags(sn_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.robot_table.setItem(row, 1, sn_item)
            
            # 列2: 型号
            model_item = QTableWidgetItem(robot.get("model", "-"))
            model_item.setFlags(model_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.robot_table.setItem(row, 2, model_item)
            
            # 列3: 是否双足
            is_biped = robot.get("is_biped", False)
            biped_item = QTableWidgetItem("是" if is_biped else "否")
            biped_item.setFlags(biped_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.robot_table.setItem(row, 3, biped_item)
            
            # 列4: 状态 - 可点击切换
            status_text = "🟢 运行中" if robot["status"] == 1 else "⚫ 未运行"
            status_btn = QPushButton(status_text)
            status_btn.setFont(self.body_font)
            status_btn.setMinimumSize(100, 36)
            if robot["status"] == 1:
                status_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {self.colors['accent_green'].name()};
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 6px 12px;
                        font-size: 13px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: {self.colors['accent_green'].lighter(110).name()};
                    }}
                """)
            else:
                status_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {self.colors['bg_tertiary'].name()};
                        color: {self.colors['text_primary'].name()};
                        border: 1px solid {self.colors['border'].name()};
                        border-radius: 6px;
                        padding: 6px 12px;
                        font-size: 13px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: {self.colors['bg_secondary'].name()};
                    }}
                """)
            sn_for_toggle = robot["sn"]
            status_btn.clicked.connect(lambda checked, s=sn_for_toggle: self._toggle_robot_status(s))
            self.robot_table.setCellWidget(row, 4, status_btn)
            
            # 列5: 删除按钮
            del_btn = QPushButton("删除")
            del_btn.setFont(self.body_font)
            del_btn.setMinimumSize(70, 36)
            del_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.colors['accent_red'].name()};
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-size: 13px;
                    font-weight: bold;
                }}
            QPushButton:hover {{
                background-color: {self.colors['accent_red'].lighter(110).name()};
            }}
            """)
            sn_for_del = robot["sn"]
            del_btn.clicked.connect(lambda checked, s=sn_for_del: self._remove_robot(s))
            self.robot_table.setCellWidget(row, 5, del_btn)
    
    def _load_robot_list(self):
        """从配置文件加载机器人列表"""
        import json as _json
        config_path = Path(__file__).parent / "robot_list.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.robot_list = _json.load(f)
                self._refresh_robot_table()
            except Exception as e:
                logger.warning("加载机器人列表失败: %s", e)
    
    def _save_robot_list(self):
        """保存机器人列表到配置文件"""
        import json as _json
        config_path = Path(__file__).parent / "robot_list.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                _json.dump(self.robot_list, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存机器人列表失败: %s", e)
    
    def _upload_robot_status(self):
        """上传机器人状态到服务器"""
        from .server_api import RobotStatus, api_upload_robot_status, ApiError
        
        if not self.robot_list:
            QMessageBox.warning(self, "上传失败", "请先添加机器人")
            return
        
        try:
            robots = [RobotStatus(sn=r["sn"], status=r["status"]) for r in self.robot_list]
            result = api_upload_robot_status(robots)
            
            if result.get("code") == 200:
                QMessageBox.information(self, "上传成功", f"成功上传 {len(robots)} 个机器人的状态")
            else:
                QMessageBox.warning(self, "上传失败", result.get("message", "未知错误"))
        except ApiError as e:
            QMessageBox.critical(self, "上传失败", f"API 错误: {e}")
        except Exception as e:
            QMessageBox.critical(self, "上传失败", f"错误: {e}")

    def _refresh_datasets(self):
        """刷新数据集列表"""
        # 清空表格
        self.dataset_table.setRowCount(0)
        
        import os
        import json
        from pathlib import Path
        from datetime import datetime
        from .dataset_manager import scan_datasets
        
        # 收集所有可能的数据集根目录
        dataset_roots = set()
        
        # 1. 默认目录
        default_root = os.environ.get("LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
        dataset_roots.add(os.path.expanduser(default_root))
        
        # 2. 从配置文件中提取 dataset_root
        config_dir = Path(__file__).parent.parent / "configs"
        if config_dir.exists():
            for config_file in config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        cfg = json.load(f)
                        if 'dataset_root' in cfg and cfg['dataset_root']:
                            dataset_root = os.path.expanduser(cfg['dataset_root'])
                            if os.path.exists(dataset_root):
                                dataset_roots.add(dataset_root)
                except Exception as e:
                    logger.debug("读取配置文件 %s 失败：%s", config_file, e)
        
        # 3. 扫描所有数据集目录
        all_datasets = []
        for dataset_root in dataset_roots:
            try:
                datasets = scan_datasets(dataset_root)
                all_datasets.extend(datasets)
                logger.debug("从目录 %s 扫描到 %d 个数据集", dataset_root, len(datasets))
            except Exception as e:
                logger.warning("扫描目录 %s 失败：%s", dataset_root, e)
        
        # 去重（按 repo_id）
        seen_ids = set()
        unique_datasets = []
        for ds in all_datasets:
            if ds.repo_id not in seen_ids:
                seen_ids.add(ds.repo_id)
                unique_datasets.append(ds)
        
        # 按修改时间排序（最新的在前）
        unique_datasets.sort(key=lambda x: x.modified_time or datetime.min, reverse=True)
        
        # 设置当前数据集根目录（用于转换，使用第一个找到的目录）
        if dataset_roots:
            self.current_dataset_root = list(dataset_roots)[0]
        else:
            self.current_dataset_root = default_root
        
        try:
            # 读取 task.json，构建任务聚合
            import json as _json
            task_agg: Dict[str, Dict] = {}
            enriched = []
            for ds in unique_datasets:
                info = ds.to_dict()
                task_id = None
                task_name = "-"
                target_eps = None
                meta_task = Path(ds.root_path) / "meta" / "task.json"
                try:
                    if meta_task.exists():
                        with open(meta_task, "r", encoding="utf-8") as f:
                            t = _json.load(f)
                        task_id = t.get("task_id")
                        task_name = t.get("task_name") or "-"
                        target_eps = t.get("target_episodes")
                except Exception as ex:
                    logger.debug("读取 %s 失败: %s", str(meta_task), ex)
                info["task_id"] = task_id
                info["task_name"] = task_name
                info["target_episodes"] = target_eps
                enriched.append(info)
                # 任务聚合
                key = "__no_task__" if task_id is None else str(task_id)
                if key not in task_agg:
                    task_agg[key] = {
                        "task_id": task_id,
                        "task_name": "未关联任务" if task_id is None else task_name,
                        "completed": 0,
                        "target": None,
                    }
                task_agg[key]["completed"] += int(info.get("num_episodes", 0) or 0)
                if target_eps:
                    if task_agg[key]["target"] is None:
                        task_agg[key]["target"] = int(target_eps)
                    else:
                        task_agg[key]["target"] = max(int(target_eps), int(task_agg[key]["target"]))
            
            # 刷新任务筛选下拉
            if hasattr(self, "task_filter_combo"):
                current = self.task_filter_combo.currentData()
                self.task_filter_combo.blockSignals(True)
                self.task_filter_combo.clear()
                self.task_filter_combo.addItem("全部任务", None)
                items = list(task_agg.values())
                items.sort(key=lambda x: (x["task_id"] is None, str(x["task_name"])))
                for it in items:
                    display = f'{it["task_name"]} ({it["task_id"]})' if it["task_id"] is not None else "未关联任务"
                    self.task_filter_combo.addItem(display, it["task_id"])
                # 恢复选择
                set_idx = 0
                if current is not None:
                    for i in range(self.task_filter_combo.count()):
                        if self.task_filter_combo.itemData(i) == current:
                            set_idx = i
                            break
                self.task_filter_combo.setCurrentIndex(set_idx)
                self.task_filter_combo.blockSignals(False)
                selected_task = self.task_filter_combo.currentData()
            else:
                selected_task = None
            
            # 过滤
            filtered = []
            for info in enriched:
                tid = info.get("task_id")
                if selected_task is None or tid == selected_task:
                    filtered.append(info)
            
            self.datasets_info = filtered
            self.dataset_table.setRowCount(len(filtered))
            
            # 任务进度映射
            task_progress = {}
            for key, it in task_agg.items():
                if key == "__no_task__":
                    continue
                task_progress[str(it["task_id"])] = {"completed": it["completed"], "target": it["target"], "name": it["task_name"]}
            
            from PySide6.QtWidgets import QProgressBar
            for i, info in enumerate(filtered):
                status = "✓ 已转换" if info.get('has_hdf5') else "Parquet"
                task_id = info.get("task_id")
                task_name = info.get("task_name") or "-"
                target_eps = info.get("target_episodes")
                # 任务聚合进度
                progress_text = "-"
                progress_widget = None
                if task_id is not None:
                    key = str(task_id)
                    agg = task_progress.get(key)
                    if agg:
                        comp = int(agg["completed"] or 0)
                        targ = agg["target"]
                        if targ:
                            progress_text = f"{comp} / {targ}"
                            progress_widget = QProgressBar()
                            progress_widget.setRange(0, int(targ))
                            progress_widget.setValue(int(comp))
                            progress_widget.setFormat(progress_text)
                            progress_widget.setTextVisible(True)
                            progress_widget.setStyleSheet(f"""
                                QProgressBar {{
                                    background-color: {self.colors['bg_secondary'].name()};
                                    border: 1px solid {self.colors['border'].name()};
                                    border-radius: 6px;
                                    color: {self.colors['text_primary'].name()};
                                    height: 16px;
                                }}
                                QProgressBar::chunk {{
                                    background-color: {self.colors['accent_blue'].name()};
                                    border-radius: 6px;
                                }}
                            """)
                        else:
                            progress_text = f"{comp}"
                
                upload_status = "✓ 已上传" if info.get('has_uploaded') else "未上传"
                
                row_values = [
                    info.get('repo_id', ''),
                    "-" if task_id is None else str(task_id),
                    task_name,
                    str(info.get('num_episodes', 0)),
                    "-" if not target_eps else str(target_eps),
                    progress_text,
                    str(info.get('total_frames', 0)),
                    info.get('created_time', ''),
                    status,
                    upload_status
                ]
                for col_idx, value in enumerate(row_values):
                    if col_idx == 5 and progress_widget is not None:
                        self.dataset_table.setCellWidget(i, col_idx, progress_widget)
                        continue
                    item = QTableWidgetItem(value)
                    item.setForeground(QColor(self.colors['text_primary']))
                    self.dataset_table.setItem(i, col_idx, item)
            
            if not filtered:
                logger.info("未找到任何数据集（已扫描 %d 个目录）", len(dataset_roots))
        
        except Exception as e:
            logger.error("刷新数据集列表失败：%s", e)
            QMessageBox.critical(self, "错误", f"刷新数据集列表失败: {e}")
    
    def _convert_selected_dataset(self):
        """转换选中的数据集为HDF5"""
        selected = self.dataset_table.currentRow()
        if selected < 0 or selected >= len(self.datasets_info):
            QMessageBox.warning(self, "提示", "请先选择要转换的数据集")
            return
        
        repo_id = self.datasets_info[selected]['repo_id']
        dataset_path = self.datasets_info[selected].get('path', '')
        
        # 从路径中提取数据集根目录
        if dataset_path:
            from pathlib import Path
            dataset_root = str(Path(dataset_path).parent)
        else:
            dataset_root = self.current_dataset_root

        # ---------------------------
        # 选择转换参数：数据集根目录 + (可选) HDF5 配置文件
        # ---------------------------
        dlg = QDialog(self)
        dlg.setWindowTitle("HDF5 转换设置")
        dlg.setModal(True)

        dlg_layout = QVBoxLayout(dlg)
        form = QGridLayout()
        form.setSpacing(10)

        root_label = QLabel("数据集根目录")
        root_edit = QLineEdit(dataset_root or "")
        root_browse = QPushButton("浏览...")

        def _browse_root():
            picked = QFileDialog.getExistingDirectory(self, "选择数据集根目录", root_edit.text() or dataset_root or "")
            if picked:
                root_edit.setText(picked)

        root_browse.clicked.connect(_browse_root)

        config_label = QLabel("配置文件(JSON)")
        config_edit = QLineEdit("")
        config_edit.setPlaceholderText("留空=自动使用 hdf5_configs/ 下第一个配置或 Python 默认配置")
        config_browse = QPushButton("浏览...")
        auto_cfg_chk = QCheckBox("自动选择配置（推荐）")
        auto_cfg_chk.setChecked(True)

        def _config_initial_dir() -> str:
            try:
                # qt_collect.py 在 lerobot_data_collector/lerobot_data_collector/
                proj_root = Path(__file__).resolve().parent.parent
                cfg_dir = proj_root / "hdf5_configs"
                if cfg_dir.exists():
                    return str(cfg_dir)
            except Exception:
                pass
            return str(Path.home())

        def _browse_config():
            init_dir = _config_initial_dir()
            picked, _ = QFileDialog.getOpenFileName(
                self,
                "选择 HDF5 转换配置文件",
                init_dir,
                "JSON Files (*.json);;All Files (*)",
            )
            if picked:
                config_edit.setText(picked)
                auto_cfg_chk.setChecked(False)

        def _toggle_auto_cfg():
            use_auto = auto_cfg_chk.isChecked()
            config_edit.setEnabled(not use_auto)
            config_browse.setEnabled(not use_auto)

        config_browse.clicked.connect(_browse_config)
        auto_cfg_chk.stateChanged.connect(_toggle_auto_cfg)
        _toggle_auto_cfg()

        form.addWidget(root_label, 0, 0)
        form.addWidget(root_edit, 0, 1)
        form.addWidget(root_browse, 0, 2)
        form.addWidget(config_label, 1, 0)
        form.addWidget(config_edit, 1, 1)
        form.addWidget(config_browse, 1, 2)
        form.addWidget(auto_cfg_chk, 2, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        ok_btn = QPushButton("开始转换")
        cancel_btn = QPushButton("取消")
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)

        dlg_layout.addLayout(form)
        dlg_layout.addLayout(btn_row)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        dataset_root = (root_edit.text() or "").strip()
        if not dataset_root:
            QMessageBox.warning(self, "提示", "数据集根目录不能为空")
            return

        config_file: Optional[str]
        if auto_cfg_chk.isChecked():
            config_file = None
        else:
            config_file = (config_edit.text() or "").strip() or None
        

        # 确认转换
        reply = QMessageBox.question(self, "确认", 
                                   f"确定要将数据集 '{repo_id}' 转换为HDF5格式吗？\n\n"
                                   f"- 数据集根目录: {dataset_root}\n"
                                   f"- 配置文件: {config_file or '自动'}\n\n"
                                   f"这可能需要一些时间...",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            from .dataset_manager import convert_parquet_to_hdf5
            logger.info("开始转换数据集：%s（目录：%s, config=%s）", repo_id, dataset_root, config_file or "auto")
            self._start_hdf5_conversion(repo_id, dataset_root, config_file)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"转换失败: {e}")
            logger.error("转换失败：%s", e)

    def _browse_and_convert_hdf5(self):
        """在数据集管理页浏览选择本地数据集目录并转换为 HDF5。"""
        # 1) 选择数据集目录（repo 目录）
        base_dir = self.current_dataset_root or os.environ.get("LEROBOT_HOME") or str(Path.home())
        picked_dir = QFileDialog.getExistingDirectory(self, "选择要转换的数据集目录（repo目录）", str(base_dir))
        if not picked_dir:
            return

        ds_path = Path(picked_dir).expanduser()
        repo_id = ds_path.name
        dataset_root = str(ds_path.parent)

        # 2) 选择配置文件（可选）
        use_auto_cfg = QMessageBox.question(
            self,
            "配置选择",
            "是否自动选择 HDF5 转换配置？\n\n选择“是”：自动使用 hdf5_configs/ 下第一个配置或默认配置。\n选择“否”：手动选择一个 JSON 配置文件。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        config_file: Optional[str] = None
        if use_auto_cfg == QMessageBox.StandardButton.No:
            # 默认指向 hdf5_configs/
            init_dir = str(Path(__file__).resolve().parent.parent / "hdf5_configs")
            picked_cfg, _ = QFileDialog.getOpenFileName(
                self,
                "选择 HDF5 转换配置文件",
                init_dir if Path(init_dir).exists() else str(Path.home()),
                "JSON Files (*.json);;All Files (*)",
            )
            config_file = picked_cfg.strip() or None

        # 3) 确认并启动转换
        reply = QMessageBox.question(
            self,
            "确认",
            f"确定要将数据集 '{repo_id}' 转换为HDF5格式吗？\n\n"
            f"- 数据集目录: {ds_path}\n"
            f"- 数据集根目录: {dataset_root}\n"
            f"- 配置文件: {config_file or '自动'}\n\n"
            f"这可能需要一些时间...",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            logger.info("开始转换数据集：%s（目录：%s, config=%s）", repo_id, dataset_root, config_file or "auto")
            self._start_hdf5_conversion(repo_id, dataset_root, config_file)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"转换失败: {e}")
            logger.error("转换失败：%s", e)
    
    def _on_convert_done(self, repo_id: str, success: bool):
        """转换完成回调"""
        if success:
            QMessageBox.information(self, "完成", f"数据集 '{repo_id}' 已成功转换为HDF5格式")
            logger.info("数据集 %s 已转换为 HDF5", repo_id)
            # 刷新列表
            self._refresh_datasets()
        else:
            QMessageBox.critical(self, "失败", f"数据集 '{repo_id}' 转换失败")
            logger.error("数据集 %s 转换失败", repo_id)

    def _start_hdf5_conversion(self, repo_id: str, dataset_root: str, config_file: Optional[str]) -> None:
        """统一入口：启动 HDF5 转换 + 进度对话框。"""
        # 避免并发启动多个转换
        if self._hdf5_progress_dialog is not None:
            QMessageBox.warning(self, "提示", "已有一个 HDF5 转换任务正在进行，请等待完成。")
            return

        import threading

        self._hdf5_cancelled = False
        try:
            cancel_event = threading.Event()
        except Exception:
            cancel_event = None
        self._hdf5_cancel_event = cancel_event

        dlg = QDialog(self)
        dlg.setWindowTitle("HDF5 转换进度")
        dlg.setModal(True)
        layout = QVBoxLayout(dlg)

        title = QLabel(f"正在转换：{repo_id}")
        title.setFont(self.header_font)
        layout.addWidget(title)

        status = QLabel("准备开始…")
        status.setFont(self.body_font)
        layout.addWidget(status)

        bar = QProgressBar()
        bar.setMinimum(0)
        bar.setMaximum(0)  # 未知总数前显示忙碌
        bar.setValue(0)
        layout.addWidget(bar)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("取消转换")

        def _cancel():
            self._hdf5_cancelled = True
            try:
                status.setText("正在取消…（将在当前文件结束后停止）")
            except Exception:
                pass
            try:
                cancel_btn.setEnabled(False)
            except Exception:
                pass
            if cancel_event is not None:
                cancel_event.set()

        cancel_btn.clicked.connect(_cancel)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._hdf5_progress_dialog = dlg
        self._hdf5_progress_bar = bar
        self._hdf5_progress_label = status

        def progress_cb(current: int, total: int, message: str) -> None:
            # 通过信号回到主线程更新 UI
            self.hdf5_progress_signal.emit(int(current), int(total), str(message))

        def should_cancel() -> bool:
            return bool(cancel_event.is_set()) if cancel_event is not None else False

        def worker():
            from .dataset_manager import convert_parquet_to_hdf5
            ok = convert_parquet_to_hdf5(
                repo_id,
                dataset_root,
                config_file=config_file,
                progress_callback=progress_cb,
                should_cancel=should_cancel,
            )
            self.hdf5_done_signal.emit(repo_id, bool(ok), bool(self._hdf5_cancelled))

        threading.Thread(target=worker, daemon=True).start()

        # 非阻塞显示：让 UI 还能响应
        dlg.show()

    def _on_hdf5_progress(self, current: int, total: int, message: str) -> None:
        if self._hdf5_progress_label is not None:
            self._hdf5_progress_label.setText(message)
        if self._hdf5_progress_bar is None:
            return
        if total and total > 0:
            self._hdf5_progress_bar.setMaximum(int(total))
            self._hdf5_progress_bar.setValue(int(current))
        else:
            # 还不知道总数
            self._hdf5_progress_bar.setMaximum(0)

        # 同步更新“转换HDF5数据”页的当前进度条（如果存在且当前在批量模式中）
        try:
            if hasattr(self, "hdf5_convert_current") and self._batch_convert_running:
                if total and total > 0:
                    self.hdf5_convert_current.setMaximum(int(total))
                    self.hdf5_convert_current.setValue(int(current))
                    self.hdf5_convert_current.setFormat(message)
                else:
                    self.hdf5_convert_current.setMaximum(0)
                    self.hdf5_convert_current.setFormat(message)
        except Exception:
            pass

    def _on_hdf5_item_progress(self, repo_id: str, current: int, total: int, message: str) -> None:
        """更新“转换HDF5数据”表格中每一行的进度/状态。"""
        row = self._hdf5_convert_row_map.get(repo_id)
        if row is None:
            return

        # 进度条：total>0 视为 parquet 级；否则无法计算
        pb = self.hdf5_convert_table.cellWidget(row, 4)
        if isinstance(pb, QProgressBar) and total and total > 0:
            pct = int(current / total * 100)
            pct = max(0, min(100, pct))
            pb.setValue(pct)
            pb.setFormat(f"{pct}%")

        st_item = self.hdf5_convert_table.item(row, 5)
        if st_item is not None:
            # 让用户能直接看到输出目录/阶段信息
            st_item.setText(message or "转换中…")

    def _on_hdf5_done(self, repo_id: str, success: bool, cancelled: bool) -> None:
        # 关闭进度对话框
        if self._hdf5_progress_dialog is not None:
            try:
                self._hdf5_progress_dialog.close()
            except Exception:
                pass
        self._hdf5_progress_dialog = None
        self._hdf5_progress_bar = None
        self._hdf5_progress_label = None
        self._hdf5_cancel_event = None

        if cancelled and not success:
            QMessageBox.information(self, "已取消", f"数据集 '{repo_id}' 的 HDF5 转换已取消")
            return

        self._on_convert_done(repo_id, success)

    def _upload_selected_dataset_info(self):
        """将选中的数据集按 uuid 颗粒度以 DataInfo 形式上报到平台。

        说明：
        - 按 api.md 要求，调用 POST /api/v1/data-info/batch-upload；
        - 现在实现为“一个 uuid 目录 = 一条 DataInfoItem” 上报：
            task_id  来自 meta/task.json；
            uuid     使用每个 episode 的 uuid 目录名；
            resource 为对象存储中该 uuid 前缀路径（例如 /tenant-<tid>/dataset/<uuid>）。
        """
        # 1. 找到当前选中的数据集
        selected = self.dataset_table.currentRow()
        if selected < 0 or selected >= len(self.datasets_info):
            QMessageBox.warning(self, "提示", "请先选择要上报的本地数据集")
            return

        ds = self.datasets_info[selected]
        repo_id = ds.get("repo_id") or ""
        dataset_path = ds.get("path") or ""
        task_id = ds.get("task_id")

        if not task_id:
            QMessageBox.warning(
                self,
                "无法上报",
                "该数据集未关联任务（meta/task.json 中没有 task_id），无法上报数据信息。",
            )
            return

        if not dataset_path:
            QMessageBox.warning(
                self,
                "无法上报",
                "未找到该数据集的本地路径，无法构造 resource 字段。",
            )
            return

        # 2. 确认对话框：包含“上传 + 上报”两个步骤
        msg = (
            f"将执行以下操作：\n\n"
            f"  1) 使用 STS 自动将该数据集中已转换好的每个 uuid 目录上传到对象存储；\n"
            f"  2) 调用 /api/v1/data-info/batch-upload 按 uuid 批量上报数据信息。\n\n"
            f"数据集信息：\n"
            f"  - 数据集ID: {repo_id}\n"
            f"  - 任务ID: {task_id}\n"
            f"  - 本地路径: {dataset_path}\n\n"
            f"是否继续？"
        )
        reply = QMessageBox.question(
            self,
            "确认上报数据信息",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # 3. 在后台线程中执行"上传 + 上报"，避免阻塞 UI
        from .qt_login import UploadProgressDialog
        from .server_api import api_batch_upload_data_info, api_get_session_token, upload_path_to_minio, DataInfoItem

        progress_dialog = UploadProgressDialog(self)
        progress_dialog.show()
        QApplication.processEvents()

        from pathlib import Path
        import threading

        dataset_path_obj = Path(dataset_path).expanduser()

        def worker():
            try:
                # 3.1 获取 STS（任何异常都包装成 RuntimeError 显示给用户）
                try:
                    progress_dialog.set_title("正在获取上传凭证...")
                except Exception:
                    pass
                try:
                    sts = api_get_session_token()
                except Exception as e:
                    raise RuntimeError(f"获取 STS 失败：{e}") from e

                # 3.2 找到已转换好的 uuid 目录（COPASI Logistics 结构）
                import json
                
                logger.info("开始查找转换标记文件...")
                flag_file = dataset_path_obj / "meta" / "hdf5_converted.json"
                if not flag_file.exists():
                    raise RuntimeError("未找到 meta/hdf5_converted.json，请先完成 HDF5 转换后再上报。")

                logger.info("读取转换标记: %s", flag_file)
                with flag_file.open("r", encoding="utf-8") as f_flag:
                    flag_cfg = json.load(f_flag)

                output_root = flag_cfg.get("output_root") or ""
                if not output_root:
                    raise RuntimeError("转换标记中缺少 output_root，无法定位 Logistics 目录。")

                # 获取本次转换生成的 UUID 列表
                generated_uuids = flag_cfg.get("generated_uuids", [])
                logger.info("本次转换生成的 UUID 数量: %d", len(generated_uuids))
                if not generated_uuids:
                    raise RuntimeError("转换标记中没有 generated_uuids，无法确定要上传的 UUID。请重新转换数据集。")

                logistics_root = Path(output_root).expanduser() / "Logistics"
                logger.info("Logistics 根目录: %s", logistics_root)
                if not logistics_root.exists():
                    raise RuntimeError(f"未找到 Logistics 目录: {logistics_root}")

                # 只查找本次转换生成的 UUID 目录
                logger.info("开始扫描 UUID 目录...")
                uuid_dirs: List[Path] = []
                for main_dir in logistics_root.iterdir():
                    if not main_dir.is_dir() or main_dir.name == "task_info":
                        continue
                    for sub_dir in main_dir.iterdir():
                        if not sub_dir.is_dir():
                            continue
                        for action_dir in sub_dir.iterdir():
                            if not action_dir.is_dir():
                                continue
                            for uuid_dir in action_dir.iterdir():
                                if uuid_dir.is_dir() and uuid_dir.name in generated_uuids:
                                    uuid_dirs.append(uuid_dir)
                                    logger.info("找到匹配的 UUID: %s", uuid_dir.name)

                logger.info("扫描完成，找到 %d 个匹配的 UUID 目录", len(uuid_dirs))
                if not uuid_dirs:
                    raise RuntimeError(f"在 {logistics_root} 下未找到本次转换的 UUID 目录（共 {len(generated_uuids)} 个），无法上报。")

                # 3.3 逐个 uuid 目录上传到对象存储（保持原始结构），并构造 DataInfoItem 列表
                total_uuid = len(uuid_dirs)
                items: List[DataInfoItem] = []

                try:
                    progress_dialog.set_title("正在上传数据到对象存储...")
                except Exception:
                    pass

                for idx, uuid_dir in enumerate(sorted(uuid_dirs)):
                    uuid_str = uuid_dir.name
                    
                    # 更新 UUID 进度
                    try:
                        progress_dialog.set_uuid_progress(idx + 1, total_uuid, uuid_str)
                    except Exception:
                        pass
                    
                    # 创建进度回调函数
                    def progress_callback(filename, current_file, total_files, uploaded_bytes, total_bytes):
                        try:
                            progress_dialog.set_file_info(filename, current_file, total_files)
                            if total_bytes > 0:
                                percent = int(uploaded_bytes / total_bytes * 100)
                                progress_dialog.set_file_progress(percent)
                                progress_dialog.set_stats(
                                    uploaded_bytes / (1024 * 1024),
                                    total_bytes / (1024 * 1024)
                                )
                        except Exception:
                            pass
                    
                    # 计算 UUID 在 Logistics 下的相对路径
                    # 例如: Logistics/Logistics-208Mb_1counts_40s/.../uuid
                    try:
                        rel_path = uuid_dir.relative_to(logistics_root.parent)
                        rel_path_str = f"/{rel_path.as_posix()}"
                    except Exception:
                        # 如果无法计算相对路径，使用 uuid 作为相对路径
                        rel_path_str = f"/{uuid_str}"
                    
                    # resource 路径：按参考文件格式（带 tenant 前缀，保持完整 Logistics 目录层级）
                    # 计算 UUID 在 Logistics 父目录下的相对路径（例如: Logistics/Logistics-xxx/Materialtransfer-xxx/Action-xxx/uuid）
                    try:
                        rel_path = uuid_dir.relative_to(logistics_root.parent)
                        rel_path_str = f"/{rel_path.as_posix()}"
                    except Exception:
                        # 如果无法计算相对路径，使用 uuid 作为相对路径
                        rel_path_str = f"/{uuid_str}"
                    
                    # 生成与 MinIO 上传一致的对象前缀（tenant-N 开头，保持 Logistics 目录层级）
                    minio_upload_prefix = f"tenant-{sts.tenant_id}{rel_path_str}".lstrip("/")
                    
                    # 直接使用与本地上传一致的路径上报给平台（前置一个 "/" 作为绝对路径）
                    resource = f"/{minio_upload_prefix}"
                    
                    logger.info("准备上传 UUID: %s, resource=%s", uuid_str, resource)
                    
                    try:
                        # 对象存储实际对象前缀：使用与 resource 一致的 minio_upload_prefix（去掉前导 /）
                        custom_prefix = minio_upload_prefix
                        up_result = upload_path_to_minio(
                            str(uuid_dir), 
                            sts,
                            progress_callback=progress_callback,
                            custom_object_prefix=custom_prefix,
                        )
                    except Exception as e:
                        raise RuntimeError(f"上传 uuid {uuid_str} 到对象存储失败：{e}") from e

                    size_mb = int(up_result.get("size_mb", 0))

                    # 按你的需求：上报 task_id, uuid, resource, size, duration（duration 固定 60）
                    items.append(
                        DataInfoItem(
                            task_id=int(task_id),
                            uuid=uuid_str,
                            resource=resource,
                            size=size_mb,
                            duration=60,
                        )
                    )

                # 3.4 调用平台 API：batch-upload（一次性上报所有 uuid）
                try:
                    progress_dialog.set_title("正在上报数据信息...")
                    progress_dialog.set_uuid_progress(total_uuid, total_uuid, "")
                    progress_dialog.set_file_info("正在调用平台 API...", 0, 0)
                except Exception:
                    pass
                # 预检 task_id：确保在“我的任务”里
                try:
                    from .server_api import api_get_my_tasks as _api_get_my_tasks
                    task_id_int = int(task_id)
                    tasks_resp = _api_get_my_tasks(page=1, page_size=200)
                    task_items = (tasks_resp.get("data") or {}).get("items") or []
                    allowed_ids = {int(t.get("id")) for t in task_items if str(t.get("id", "")).isdigit()}
                    if task_id_int not in allowed_ids:
                        sample = ", ".join(str(x) for x in sorted(list(allowed_ids))[:20])
                        raise RuntimeError(
                            f"任务ID={task_id_int} 不在“我的任务”列表中（可能无权限/不存在）。"
                            f"请从任务列表复制正确的ID。当前可见ID示例: {sample}"
                        )
                except Exception as e:
                    raise RuntimeError(f"任务ID预检失败：{e}") from e
                # batch-upload：当前 resource 已包含 tenant 前缀，无需重试逻辑（保留注释以防后端格式要求变化）
                try:
                    resp = api_batch_upload_data_info(items)
                except Exception as e:
                    # 如果后端仍需要不带 tenant 前缀的格式，可在此添加重试逻辑
                    # 当前 resource 格式：/tenant-{tenant_id}/Logistics/...（符合后端要求）
                    raise

                failed = resp.get("failed_uuids") or []
                logger.info("数据信息上报响应：%s", resp)
                
                # 写入上传标记
                try:
                    meta_dir = dataset_path_obj / "meta"
                    meta_dir.mkdir(exist_ok=True)
                    uploaded_flag = meta_dir / "uploaded.json"
                    upload_record = {
                        "uploaded": True,
                        "uploaded_at": datetime.now().isoformat(),
                        "task_id": int(task_id),
                        "uuids": [item.uuid for item in items],
                        "success_count": len(items) - len(failed),
                        "failed_uuids": failed,
                    }
                    with uploaded_flag.open("w", encoding="utf-8") as f:
                        json.dump(upload_record, f, ensure_ascii=False, indent=2)
                    logger.info("已写入上传标记: %s", uploaded_flag)
                except Exception as e:
                    logger.warning("写入上传标记失败: %s", e)
                
                # 设置结果标志（不要在这里关闭对话框）
                self._upload_result = {
                    "success": True,
                    "failed_uuids": failed,
                    "task_id": int(task_id),
                    "total_items": len(items),
                }

            except Exception as e:
                logger.error("上传/上报失败：%s", e)
                self._upload_result = {
                    "success": False,
                    "error": str(e),
                }

        # 启动 worker 线程
        self._upload_result = None
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        
        # 用定时器检查上传结果
        def check_result():
            if self._upload_result is not None:
                result = self._upload_result
                self._upload_result = None
                result_timer.stop()
                
                # 先关闭进度对话框
                progress_dialog.close()
                
                if result.get("success"):
                    # 上传成功
                    failed = result.get("failed_uuids", [])
                    task_id = result.get("task_id")
                    total = result.get("total_items")
                    success_count = total - len(failed)
                    
                    msg_parts = [
                        f"✅ 数据已成功上传到对象存储并上报！\n",
                        f"- 任务ID: {task_id}",
                        f"- 上传 UUID 数量: {total}",
                        f"- 成功上报: {success_count}",
                    ]
                    if failed:
                        msg_parts.append(f"- 后端标记失败: {len(failed)}")
                        msg_parts.append(f"\n后端标记为失败的 UUID：")
                        msg_parts.extend([f"  · {uuid}" for uuid in failed[:5]])
                        if len(failed) > 5:
                            msg_parts.append(f"  ... 还有 {len(failed) - 5} 个")
                    
                    QMessageBox.information(self, "上传完成", "\n".join(msg_parts))
                    # 刷新数据集列表，更新上传状态
                    self._refresh_datasets()
                else:
                    # 上传失败
                    QMessageBox.critical(self, "上传失败", f"上传或上报过程中出现错误：\n{result.get('error')}")
        
        result_timer = QTimer(self)
        result_timer.timeout.connect(check_result)
        result_timer.start(500)  # 每 500ms 检查一次
    
    @Slot(list, int, int)
    def _on_upload_success(self, failed_uuids, task_id, total_items):
        """处理上传成功（主线程槽函数）"""
        success_count = total_items - len(failed_uuids)
        msg_parts = [
            f"✅ 数据已成功上传到对象存储并上报！\n",
            f"- 任务ID: {task_id}",
            f"- 上传 UUID 数量: {total_items}",
            f"- 成功上报: {success_count}",
        ]
        if failed_uuids:
            msg_parts.append(f"- 后端标记失败: {len(failed_uuids)}")
            msg_parts.append(f"\n后端标记为失败的 UUID（可能需要联系管理员）：")
            msg_parts.extend([f"  · {uuid}" for uuid in failed_uuids[:5]])  # 最多显示5个
            if len(failed_uuids) > 5:
                msg_parts.append(f"  ... 还有 {len(failed_uuids) - 5} 个")
        
        QMessageBox.information(
            self,
            "上传完成",
            "\n".join(msg_parts),
        )
    
    @Slot(str)
    def _on_upload_error(self, error_message):
        """处理上传失败（主线程槽函数）"""
        QMessageBox.critical(
            self,
            "上传失败",
            f"上传或上报过程中出现错误：\n{error_message}",
        )
        
    def _update_status_indicator(self, color):
        """更新状态指示器颜色（已弃用，新界面使用状态点阵）"""
        # 新界面不再使用状态指示器，此方法保留以兼容旧代码
        pass
        
    def _scan_config_files(self):
        """扫描configs目录下的JSON配置文件
        返回字典列表，包含显示名称（文件名，去掉.json）和完整路径
        """
        import os
        from pathlib import Path
        
        config_files = []
        
        # 查找configs目录（项目根目录下的configs）
        try:
            # 获取项目根目录（向上查找）
            current_file = Path(__file__).resolve()
            # qt_gui.py 在 lerobot_data_collector/lerobot_data_collector/ 下
            # 向上两级到项目根目录
            proj_root = current_file.parent.parent
            configs_dir = proj_root / "configs"
            
            if configs_dir.exists() and configs_dir.is_dir():
                # 扫描所有JSON文件
                for json_file in configs_dir.glob("*.json"):
                    full_path = str(json_file)
                    # 显示名称：去掉路径和.json扩展名
                    display_name = json_file.stem  # 文件名（不含扩展名）
                    config_files.append({
                        'display': display_name,
                        'path': full_path
                    })
                
                # 按显示名称排序
                config_files.sort(key=lambda x: x['display'])
                logger.info("发现 %d 个配置文件", len(config_files))
            else:
                logger.warning("配置文件目录不存在：%s", configs_dir)
        except Exception as e:
            logger.warning("扫描配置文件失败：%s", e)
        
        return config_files
    
    def _refresh_config_files(self):
        """刷新配置文件列表"""
        config_files = self._scan_config_files()
        self.cfg_combobox.clear()
        # 添加项目：显示名称作为文本，完整路径作为数据
        for cfg in config_files:
            self.cfg_combobox.addItem(cfg['display'], cfg['path'])
        if config_files:
            # 如果当前选择的文件不在列表中，选择第一个
            current_index = self.cfg_combobox.currentIndex()
            if current_index < 0 or current_index >= len(config_files):
                self.cfg_combobox.setCurrentIndex(0)
            QMessageBox.information(self, "刷新完成", f"找到 {len(config_files)} 个配置文件")
        else:
            QMessageBox.warning(self, "未找到配置文件", "configs目录下没有找到JSON配置文件")
    
    def browse(self):
        """浏览选择其他位置的配置文件"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择配置文件",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            # 如果选择的文件不在下拉列表中，添加到列表
            from pathlib import Path
            display_name = Path(path).stem  # 文件名（不含扩展名）
            current_count = self.cfg_combobox.count()
            found = False
            for i in range(current_count):
                if self.cfg_combobox.itemData(i) == path:
                    found = True
                    self.cfg_combobox.setCurrentIndex(i)
                    break
            if not found:
                self.cfg_combobox.addItem(display_name, path)
                self.cfg_combobox.setCurrentIndex(self.cfg_combobox.count() - 1)
    
    def _scan_action_steps_files(self):
        """扫描action_steps目录下的JSON动作步骤文件
        返回字典列表，包含显示名称（文件名，去掉.json）和完整路径
        """
        import os
        from pathlib import Path
        
        action_steps_files = []
        
        # 查找action_steps目录（项目根目录下的action_steps）
        try:
            current_file = Path(__file__).resolve()
            proj_root = current_file.parent.parent
            action_steps_dir = proj_root / "action_steps"
            
            if action_steps_dir.exists() and action_steps_dir.is_dir():
                # 扫描所有JSON文件
                for json_file in action_steps_dir.glob("*.json"):
                    full_path = str(json_file)
                    # 显示名称：去掉路径和.json扩展名
                    display_name = json_file.stem  # 文件名（不含扩展名）
                    action_steps_files.append({
                        'display': display_name,
                        'path': full_path
                    })
                
                # 按显示名称排序
                action_steps_files.sort(key=lambda x: x['display'])
                logger.info("发现 %d 个动作步骤文件", len(action_steps_files))
            else:
                logger.warning("动作步骤目录不存在：%s", action_steps_dir)
        except Exception as e:
            logger.warning("扫描动作步骤文件失败：%s", e)
        
        return action_steps_files
    
    def _load_action_steps_from_json(self, json_path: str):
        """从JSON文件加载动作步骤
        返回步骤数据列表，包含时间范围和任务目标
        """
        import json
        import os
        
        if not json_path or not os.path.exists(json_path):
            return [], 0.0
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 解析JSON结构：期望是一个数组，每个元素是一个episode
            # 每个episode有label_info.action_config数组
            action_steps = []
            total_duration = 0.0
            
            if isinstance(data, list) and len(data) > 0:
                # 取第一个episode的动作配置
                episode = data[0]
                
                # 获取总时长（如果存在）
                if "duration" in episode:
                    total_duration = float(episode["duration"])
                
                if "label_info" in episode and "action_config" in episode["label_info"]:
                    action_configs = episode["label_info"]["action_config"]
                    
                    # 假设FPS为30（可以从配置文件中获取）
                    fps = 30.0
                    
                    for action in action_configs:
                        start_frame = action.get("start_frame", 0)
                        end_frame = action.get("end_frame", start_frame)
                        action_text = action.get("action_text", "")
                        skill = action.get("skill", "")
                        
                        # 将帧数转换为秒
                        start_time_sec = start_frame / fps
                        end_time_sec = end_frame / fps
                        
                        # 更新总时长
                        if end_time_sec > total_duration:
                            total_duration = end_time_sec
                        
                        action_steps.append({
                            'start_time': start_time_sec,
                            'end_time': end_time_sec,
                            'action_text': action_text,
                            'skill': skill
                        })
            
            return action_steps, total_duration
        except Exception as e:
            logger.error("加载动作步骤文件失败：%s", e)
            QMessageBox.critical(self, "错误", f"加载动作步骤文件失败: {e}")
            return [], 0.0
    
    def _on_action_steps_file_changed(self, display_name: str):
        """当动作步骤文件选择改变时调用"""
        # 从下拉菜单获取完整路径（通过itemData）
        current_index = self.steps_combobox.currentIndex()
        if current_index < 0:
            return
        
        file_path = self.steps_combobox.itemData(current_index)
        if not file_path:
            return
        
        # 加载并显示动作步骤
        action_steps, total_duration = self._load_action_steps_from_json(file_path)
        self.action_steps = action_steps
        self.current_step_index = -1
        
        # 更新时间轴视图
        self.timeline_widget.set_steps(action_steps, total_duration)
    
    def browse_action_steps(self):
        """浏览选择其他位置的动作步骤文件"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择动作步骤文件",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            # 如果选择的文件不在下拉列表中，添加到列表
            from pathlib import Path
            display_name = Path(path).stem  # 文件名（不含扩展名）
            current_count = self.steps_combobox.count()
            found = False
            for i in range(current_count):
                if self.steps_combobox.itemData(i) == path:
                    found = True
                    self.steps_combobox.setCurrentIndex(i)
                    break
            if not found:
                self.steps_combobox.addItem(display_name, path)
                self.steps_combobox.setCurrentIndex(self.steps_combobox.count() - 1)
    
    def _init_action_steps(self):
        """初始化动作步骤列表"""
        # 如果下拉菜单中有文件，加载第一个
        if self.steps_combobox.count() > 0:
            first_file_path = self.steps_combobox.itemData(0)
            if first_file_path:
                self._on_action_steps_file_changed(self.steps_combobox.itemText(0))
        else:
            # 如果没有文件，显示空时间轴
            self.action_steps = []
            self.current_step_index = -1
            self.timeline_widget.set_steps([], 0.0)
    
    def _draw_status_grids(self):
        """初始化状态指示器点阵"""
        # 初始状态：显示5个激活点
        self.left_status_grid.set_active_count(5)
        self.right_status_grid.set_active_count(5)
    
    def _update_countdown(self):
        """更新倒计时和计时器显示"""
        if self.collection_start_time is None or not self.collector:
            return
        
        # 更新采集数量显示
        if self.collector:
            frames_written = getattr(self.collector, '_frames_written', 0)
            # 如果有已有数据，显示总数据量
            if hasattr(self, '_existing_frames') and self._existing_frames > 0:
                total_frames = self._existing_frames + frames_written
            else:
                total_frames = frames_written
            # 目标帧（用于UI X / Y）
            target_frames_ui = getattr(self, "_target_frames_ui", 0) or 0
            if target_frames_ui > 0:
                self.collection_count_display.setText(f"{total_frames} / {target_frames_ui} 帧")
                remaining_frames = max(0, target_frames_ui - frames_written)
                try:
                    if not (
                        bool(getattr(self, "_pico_session_mode", False))
                        and not bool(getattr(self.collector, "_session_active", False))
                    ):
                        self.status_display.setText(f"采集中，剩余 {remaining_frames} 帧")
                except Exception:
                    pass
            else:
                self.collection_count_display.setText(f"{total_frames} 帧")
            # 同步到左侧信息条
            try:
                if target_frames_ui > 0:
                    self.frames_progress_label_left.setText(f"{total_frames} / {target_frames_ui} 帧")
                else:
                    self.frames_progress_label_left.setText(f"{total_frames} 帧")
            except Exception:
                pass
        
        # 检查 collector 是否存在
        if not self.collector:
            return
        
        if self.collector._actual_start_time is not None:
            import time
            elapsed = time.time() - self.collector._actual_start_time
            remaining = max(0, self.collection_duration - elapsed)
            
            # 更新大计时器显示
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            self.timer_display.setText(f"{mins:02d}:{secs:02d}")
            
            # 更新时间轴当前时间指示器
            self.timeline_widget.set_current_time(elapsed)
        else:
            elapsed = 0
            remaining = self.collection_duration
            self.timer_display.setText("00:00")
            self.timeline_widget.set_current_time(0.0)
        
        # 如果主数据采集已经停止（达到目标帧数或被手动终止），普通模式下自动停止相机录制；
        # Pico 模式下由右B统一收尾，避免在主采集先到 1800 时提前截断相机帧数。
        pico_session_mode = bool(getattr(self, "_pico_session_mode", False))
        if self.collector and (not pico_session_mode) and not self.collector._session_active:
            if getattr(self, "_camera_recording", False):
                logger.info("检测到主采集已停止，自动停止右手摄像头录制")
                try:
                    self._stop_right_camera_recording()
                except Exception:
                    pass
            if getattr(self, "_head_camera_recording", False):
                logger.info("检测到主采集已停止，自动停止头部摄像头录制")
                try:
                    self._stop_head_camera_recording()
                except Exception:
                    pass

        if self.collector and pico_session_mode and not self.collector._session_active:
            right_done = (self.right_camera_manager is None) or (not bool(getattr(self.right_camera_manager, "_recording", False)))
            left_done = (self.left_camera_manager is None) or (not bool(getattr(self.left_camera_manager, "_recording", False)))
            head_mgr = self.head_camera_manager if self.head_camera_manager is not None else self.orbbec_head_camera_manager
            head_done = (head_mgr is None) or (not bool(getattr(head_mgr, "_recording", False)))
            all_done = right_done and left_done and head_done
            if all_done:
                if not bool(getattr(self, "_pico_save_ready_announced", False)):
                    logger.info("三路相机均已到目标帧，可以按右B进行保存")
                    self._pico_save_ready_announced = True
                try:
                    self.status_display.setText("三路相机已到目标帧，可以按右B保存")
                except Exception:
                    pass
            else:
                waiting = []
                if not right_done:
                    waiting.append("右手")
                if not left_done:
                    waiting.append("左手")
                if not head_done:
                    waiting.append("头部")
                try:
                    self.status_display.setText(f"主采集完成，等待相机收尾：{','.join(waiting)}")
                except Exception:
                    pass

        if self.collector and remaining > 0 and self.collector._session_active:
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            if self.collector._actual_start_time is not None:
                # 更新状态指示器点阵（根据已采集时间动态显示）
                active_dots = min(10, int((elapsed / self.collection_duration) * 10))
                self._update_status_grids(active_dots)
            else:
                self._update_status_grids(0)
        elif remaining <= 0 and self.collector._session_active:
            # 仅更新时间提示，不直接修改采集核心状态，避免“未到目标帧被UI提前截断”
            try:
                target_frames_ui = int(getattr(self, "_target_frames_ui", 0) or 0)
            except Exception:
                target_frames_ui = 0
            try:
                current_frames_ui = int(getattr(self.collector, "_frames_written", 0) or 0)
            except Exception:
                current_frames_ui = 0
            if target_frames_ui > 0 and current_frames_ui < target_frames_ui:
                remain_frames = target_frames_ui - current_frames_ui
                try:
                    self.status_display.setText(f"达到预计时长，等待目标帧（剩余 {remain_frames} 帧）")
                except Exception:
                    pass
            else:
                try:
                    self.status_display.setText("采集完成")
                except Exception:
                    pass
            self._update_status_grids(10)  # 全部点亮
    
    def _update_status_grids(self, active_count):
        """更新状态指示器点阵的激活数量"""
        self.left_status_grid.set_active_count(active_count)
        self.right_status_grid.set_active_count(active_count)

    # ---------------- 摄像头相关 ----------------
    def _resolve_dataset_home(self, cfg) -> Path:
        """根据配置或环境变量推断数据集根目录"""
        if getattr(cfg, "dataset_root", None):
            return Path(os.path.expanduser(cfg.dataset_root))
        env_path = os.environ.get("LEROBOT_HOME")
        if env_path:
            return Path(os.path.expanduser(env_path))
        # 默认优先：/hf_dataset -> /data/hf_dataset -> /home/dreame/data/hf_dataset
        for p in ("/hf_dataset", "/data/hf_dataset", "/home/dreame/data/hf_dataset"):
            preferred = Path(p).expanduser()
            if preferred.exists():
                return preferred
        return Path(os.path.expanduser("~/.cache/huggingface/lerobot"))

    def _on_external_camera_timestamp(self, timestamp_ns: int) -> None:
        """将外部相机时间戳同步到 Collector，用于统一时间轴。"""
        try:
            collector = getattr(self, "collector", None)
            if collector is None:
                return
            update_fn = getattr(collector, "update_external_camera_timestamp_ns", None)
            if callable(update_fn):
                update_fn(timestamp_ns)
        except Exception:
            pass

    def _init_right_camera(self) -> None:
        """初始化右手 RealSense 相机预览（始终尝试预览，不影响采集主流程）。"""
        global RealSenseRightCameraManager
        if RealSenseRightCameraManager is None:
            try:
                from .warmup_display_realsense_right import RealSenseRightCameraManager as _Mgr
                RealSenseRightCameraManager = _Mgr
            except Exception:
                logger.info("未启用 RealSense 右手摄像头模块（缺少依赖或导入失败）")
                return
        if RealSenseRightCameraManager is None:
            logger.info("未启用 RealSense 右手摄像头模块（缺少依赖或导入失败）")
            return
        if _sn_disabled(REALSENSE_RIGHT_SN):
            logger.info("已禁用 RealSense 右手摄像头预览（REALSENSE_RIGHT_SN 未配置或为 DISABLED）")
            return
        try:
            logger.info("初始化 Qt 右手摄像头管理器...")
            self.right_camera_manager = RealSenseRightCameraManager(
                frame_timestamp_callback=self._on_external_camera_timestamp
            )
            started = self.right_camera_manager.start()
            if not started:
                logger.warning(
                    "Qt 右手摄像头预览不可用：%s",
                    getattr(self.right_camera_manager, "last_error", ""),
                )
                self.right_camera_manager = None
            else:
                logger.info("Qt 右手摄像头预览线程已启动")
        except Exception as e:
            logger.warning("初始化 Qt 右手摄像头失败：%s", e)
            self.right_camera_manager = None

    def _init_left_camera(self) -> None:
        """初始化左手 RealSense 相机预览（始终尝试预览，不影响采集主流程）。"""
        global RealSenseLeftCameraManager
        if RealSenseLeftCameraManager is None:
            try:
                from .warmup_display_realsense_left import RealSenseLeftCameraManager as _Mgr
                RealSenseLeftCameraManager = _Mgr
            except Exception:
                logger.info("未启用 RealSense 左手摄像头模块（缺少依赖或导入失败）")
                return
        if RealSenseLeftCameraManager is None:
            logger.info("未启用 RealSense 左手摄像头模块（缺少依赖或导入失败）")
            return
        if _sn_disabled(REALSENSE_LEFT_SN):
            logger.info("已禁用 RealSense 左手摄像头预览（REALSENSE_LEFT_SN 未配置或为 DISABLED）")
            return
        try:
            logger.info("初始化 Qt 左手摄像头管理器...")
            self.left_camera_manager = RealSenseLeftCameraManager(
                frame_timestamp_callback=self._on_external_camera_timestamp
            )
            started = self.left_camera_manager.start()
            if not started:
                logger.warning(
                    "Qt 左手摄像头预览不可用：%s",
                    getattr(self.left_camera_manager, "last_error", ""),
                )
                self.left_camera_manager = None
            else:
                logger.info("Qt 左手摄像头预览线程已启动")
        except Exception as e:
            logger.warning("初始化 Qt 左手摄像头失败：%s", e)
            self.left_camera_manager = None

    def _init_head_camera(self) -> None:
        """初始化头部摄像头（优先尝试RealSense，失败则使用奥比中光）。"""
        # 优先尝试RealSense头部摄像头
        global RealSenseHeadCameraProcess
        global OrbbecHeadCameraManager
        if RealSenseHeadCameraProcess is None:
            try:
                from .warmup_display_realsense_head import RealSenseHeadCameraProcess as _Proc
                RealSenseHeadCameraProcess = _Proc
            except Exception:
                RealSenseHeadCameraProcess = None

        if OrbbecHeadCameraManager is None:
            try:
                from .orbbec_head_camera_manager import OrbbecHeadCameraManager as _Orbbec
                OrbbecHeadCameraManager = _Orbbec
            except Exception:
                OrbbecHeadCameraManager = None

        if RealSenseHeadCameraProcess is not None and (not _sn_disabled(REALSENSE_HEAD_SN)):
            try:
                logger.info("初始化头部 RealSense 摄像头多进程管理器... SN=%s", REALSENSE_HEAD_SN)
                self.head_camera_manager = RealSenseHeadCameraProcess(serial_number=REALSENSE_HEAD_SN)
                started = self.head_camera_manager.start()
                if started:
                    logger.info("头部 RealSense 摄像头预览进程已启动")
                    return
                else:
                    logger.warning(
                        "头部 RealSense 摄像头预览不可用：%s",
                        getattr(self.head_camera_manager, "last_error", ""),
                    )
                    self.head_camera_manager = None
            except Exception as e:
                logger.warning("初始化头部 RealSense 摄像头失败：%s", e)
                self.head_camera_manager = None
        elif RealSenseHeadCameraProcess is not None and _sn_disabled(REALSENSE_HEAD_SN):
            logger.info("已禁用 RealSense 头部摄像头预览（REALSENSE_HEAD_SN 未配置或为 DISABLED）")
        
        # 如果RealSense不可用，尝试使用奥比中光作为替代
        if OrbbecHeadCameraManager is not None:
            try:
                logger.info("RealSense 头部摄像头不可用，尝试使用奥比中光相机...")
                self.orbbec_head_camera_manager = OrbbecHeadCameraManager()
                started = self.orbbec_head_camera_manager.start()
                if started:
                    logger.info("奥比中光头部摄像头管理器已启动（作为RealSense的替代）")
                else:
                    logger.warning(
                        "奥比中光头部摄像头不可用：%s",
                        getattr(self.orbbec_head_camera_manager, "last_error", ""),
                    )
                    self.orbbec_head_camera_manager = None
            except Exception as e:
                logger.warning("初始化奥比中光头部摄像头失败：%s", e)
                self.orbbec_head_camera_manager = None
        else:
            logger.info("未启用头部摄像头模块（RealSense和奥比中光都不可用）")

    def _start_camera_preview(self) -> None:
        """通过 QTimer 定期刷新右手摄像头画面。"""
        if self._camera_timer is not None:
            return
        self._camera_timer = QTimer(self)
        self._camera_timer.timeout.connect(self._update_camera_preview)
        self._camera_timer.start(66)  # 约15fps

    def _update_camera_preview(self) -> None:
        def render_to_label(frame: np.ndarray, target_label: QLabel) -> None:
            if frame is None:
                return
            try:
                h, w, _ = frame.shape
            except ValueError:
                return
            image = QImage(
                frame.data,
                w,
                h,
                3 * w,
                QImage.Format.Format_RGB888,
            )
            target_size = target_label.size()
            if target_size.width() > 0 and target_size.height() > 0:
                image = image.scaled(
                    target_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            target_label.setPixmap(QPixmap.fromImage(image))

        if self.right_camera_manager:
            try:
                frame = self.right_camera_manager.get_latest_preview()
                if frame is not None:
                    render_to_label(frame, self.right_hand_camera_display)
            except Exception as e:
                logger.debug("刷新右手摄像头预览失败：%s", e)

        if self.left_camera_manager:
            try:
                frame = self.left_camera_manager.get_latest_preview()
                if frame is not None:
                    render_to_label(frame, self.left_hand_camera_display)
            except Exception as e:
                logger.debug("刷新左手摄像头预览失败：%s", e)

        if self.head_camera_manager:
            try:
                frame = self.head_camera_manager.get_latest_preview()
                if frame is not None:
                    render_to_label(frame, self.head_camera_display)
            except Exception as e:
                logger.debug("刷新头部摄像头预览失败：%s", e)
        
        # 奥比中光相机预览（与RealSense相同的方式）
        if self.orbbec_head_camera_manager:
            try:
                frame = self.orbbec_head_camera_manager.get_latest_preview()
                if frame is not None:
                    render_to_label(frame, self.head_camera_display)
            except Exception as e:
                logger.debug("刷新奥比中光头部摄像头预览失败：%s", e)

    def _start_right_camera_recording(self, cfg, target_frames: int) -> None:
        if not self.right_camera_manager:
            return
        try:
            dataset_home = self._resolve_dataset_home(cfg)
            self.right_camera_manager.set_dataset_home(dataset_home)
            paths = self.right_camera_manager.start_recording(
                cfg.repo_id, target_frames=target_frames
            )
            logger.info(
                "右手摄像头开始录制: repo_id=%s, episode_dir=%s",
                cfg.repo_id,
                getattr(paths, "episode_dir", None),
            )
            self._camera_recording = True
            self._pico_save_ready_announced = False
        except Exception as e:
            logger.warning("启动右手摄像头录制失败：%s", e)

    def _stop_right_camera_recording(self) -> None:
        if not self.right_camera_manager:
            return
        try:
            self.right_camera_manager.stop_recording(wait=True)
            logger.info("右手摄像头录制已停止")
            self._camera_recording = False
        except Exception as e:
            logger.debug("停止右手摄像头录制时出错：%s", e)

    def _start_left_camera_recording(self, cfg, target_frames: int) -> None:
        if not self.left_camera_manager:
            return
        try:
            dataset_home = self._resolve_dataset_home(cfg)
            self.left_camera_manager.set_dataset_home(dataset_home)
            paths = self.left_camera_manager.start_recording(
                cfg.repo_id, target_frames=target_frames
            )
            logger.info(
                "左手摄像头开始录制: repo_id=%s, episode_dir=%s",
                cfg.repo_id,
                getattr(paths, "episode_dir", None),
            )
            self._left_camera_recording = True
            self._pico_save_ready_announced = False
        except Exception as e:
            logger.warning("启动左手摄像头录制失败：%s", e)

    def _stop_left_camera_recording(self) -> None:
        if not self.left_camera_manager:
            return
        try:
            self.left_camera_manager.stop_recording(wait=True)
            logger.info("左手摄像头录制已停止")
            self._left_camera_recording = False
        except Exception as e:
            logger.debug("停止左手摄像头录制时出错：%s", e)

    def _start_head_camera_recording(self, cfg, target_frames: int) -> None:
        # 优先使用RealSense头部摄像头
        if self.head_camera_manager:
            try:
                dataset_home = self._resolve_dataset_home(cfg)
                self.head_camera_manager.set_dataset_home(dataset_home)
                self.head_camera_manager.start_recording(
                    cfg.repo_id, target_frames=target_frames
                )
                logger.info("头部 RealSense 摄像头开始录制: repo_id=%s", cfg.repo_id)
                self._head_camera_recording = True
                self._pico_save_ready_announced = False
            except Exception as e:
                logger.warning("启动头部 RealSense 摄像头录制失败：%s", e)
        # 如果没有RealSense，使用奥比中光
        elif self.orbbec_head_camera_manager:
            try:
                dataset_home = self._resolve_dataset_home(cfg)
                self.orbbec_head_camera_manager.set_dataset_home(dataset_home)
                self.orbbec_head_camera_manager.start_recording(
                    cfg.repo_id, target_frames=target_frames
                )
                logger.info("头部奥比中光摄像头开始录制: repo_id=%s", cfg.repo_id)
                self._head_camera_recording = True
                self._pico_save_ready_announced = False
            except Exception as e:
                logger.warning("启动头部奥比中光摄像头录制失败：%s", e)

    def _stop_head_camera_recording(self) -> None:
        if self.head_camera_manager:
            try:
                self.head_camera_manager.stop_recording(wait=True)
                logger.info("头部 RealSense 摄像头录制已停止")
                self._head_camera_recording = False
            except Exception as e:
                logger.debug("停止头部 RealSense 摄像头录制时出错：%s", e)
        elif self.orbbec_head_camera_manager:
            try:
                self.orbbec_head_camera_manager.stop_recording(wait=True)
                logger.info("头部奥比中光摄像头录制已停止")
                self._head_camera_recording = False
            except Exception as e:
                logger.debug("停止头部奥比中光摄像头录制时出错：%s", e)
    
    def start(self):
        """开始采集"""
        import json
        import os
        import time
        
        # 从下拉菜单获取完整路径（通过itemData）
        current_index = self.cfg_combobox.currentIndex()
        if current_index < 0:
            QMessageBox.critical(self, "错误", "请先选择配置文件")
            return
        
        path = self.cfg_combobox.itemData(current_index)
        if not path or not os.path.exists(path):
            QMessageBox.critical(self, "错误", "请先选择有效的配置 JSON 文件")
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cc = self.collector_config_class.from_dict(cfg)

            # dataset_root：按天创建目录（YYYYMMDD_suffix），当天所有采集都放在同一个“当天目录”下；
            # 每次开始采集都会生成一个新的 repo_id 子目录，保证“每一次采集都是独立的文件夹”。
            try:
                import re
                import socket
                # 优先使用 HF_DATASET_ROOT（若设置且可写），否则使用初始化时挑选的可写根目录
                base_root = (os.environ.get("HF_DATASET_ROOT") or getattr(self, "_base_dataset_root", "")).strip()
                if base_root:
                    base_root = os.path.expanduser(base_root)
                if (not base_root) or (not os.path.isdir(base_root)) or (not os.access(base_root, os.W_OK | os.X_OK)):
                    fallback = getattr(self, "_base_dataset_root", "/data/hf_dataset")
                    logger.warning(
                        "HF_DATASET_ROOT 不可用/不可写（%s），将使用可写根目录：%s",
                        base_root,
                        fallback,
                    )
                    base_root = os.path.expanduser(str(fallback))
                day = datetime.now().strftime("%Y%m%d")
                # 后缀优先使用环境变量固定（例如 magiclabs），否则使用 hostname
                suffix = (os.environ.get("HF_DATASET_DAILY_SUFFIX") or socket.gethostname() or "host").strip()
                suffix_safe = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix).strip("-") or "host"
                day_dir = f"{day}_{suffix_safe}"
                daily_root = os.path.join(base_root, day_dir)
                os.makedirs(daily_root, exist_ok=True)
                cc.dataset_root = daily_root
                logger.info("数据集根目录(dataset_root)：%s（base_root=%s, day_dir=%s）", cc.dataset_root, base_root, day_dir)
            except Exception as e:
                logger.warning("设置数据集根目录失败，将使用配置中的 dataset_root：%s", e)
            finally:
                # 兜底：如果上面没有成功设置 dataset_root，则强制回退到 base_root，避免落到未知目录
                if not getattr(cc, "dataset_root", None):
                    cc.dataset_root = os.path.expanduser(
                        os.environ.get("HF_DATASET_ROOT") or getattr(self, "_base_dataset_root", "/data/hf_dataset")
                    )
                    logger.warning("dataset_root 为空，已兜底设置为：%s", cc.dataset_root)

            # 每次开始采集都生成新的 repo_id（独立目录）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_id = (self.current_task or {}).get("id") or (self.current_task or {}).get("task_id")
            try:
                from .conf import TASK_CODE
                task_code = TASK_CODE
            except Exception:
                task_code = "data_collection"
            prefix = f"task{task_id}" if task_id is not None and str(task_id).strip() != "" else task_code
            repo_id = f"{prefix}_{timestamp}"
            # 若同秒重复点击，保证目录不冲突
            try:
                from pathlib import Path
                base_root = os.path.expanduser(cc.dataset_root) if cc.dataset_root else os.path.expanduser("~/.cache/huggingface/lerobot")
                n = 1
                while (Path(base_root) / repo_id).exists():
                    n += 1
                    repo_id = f"{prefix}_{timestamp}_{n}"
            except Exception:
                pass
            cc.repo_id = repo_id
            logger.info("数据集 ID：%s", cc.repo_id)
            try:
                from pathlib import Path
                _ds_dir = Path(os.path.expanduser(cc.dataset_root)) / cc.repo_id if cc.dataset_root else None
                if _ds_dir is not None:
                    logger.info("本次采集写入目录：%s", str(_ds_dir))
            except Exception:
                pass

            # 设置环境变量，让其他进程（相机录制）使用相同的 repo_id
            os.environ["FIXED_REPO_ID"] = cc.repo_id
            
            # 条数进度：每次采集都是独立文件夹，因此按“当天+task_id”在当天目录下扫描已有采集目录数量
            try:
                from pathlib import Path
                today = datetime.now().strftime("%Y%m%d")
                task_id = (self.current_task or {}).get("id") or (self.current_task or {}).get("task_id")
                prefix = f"task{task_id}_{today}" if task_id is not None and str(task_id).strip() != "" else today
                p = Path(os.path.expanduser(cc.dataset_root)) if cc.dataset_root else Path(os.environ.get("LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot")))
                if p.exists():
                    cnt = 0
                    for child in p.iterdir():
                        if child.is_dir() and child.name.startswith(prefix):
                            cnt += 1
                    self._existing_episodes = cnt
                else:
                    self._existing_episodes = 0
            except Exception:
                self._existing_episodes = 0
            
            # 如果已有 Collector 实例，先停止并清理
            if self.collector is not None:
                logger.info("停止旧的采集器实例")
                try:
                    self.collector.stop()
                except Exception as e:
                    logger.warning("停止旧采集器时出错：%s", e)
                # 清理适配器
                try:
                    for adapter in self.collector._adapters:
                        if hasattr(adapter, 'stop'):
                            adapter.stop()
                except Exception as e:
                    logger.warning("清理适配器时出错：%s", e)
                self.collector = None
            
            # 读取任务目标条数（episode）
            self._target_episodes = None
            if getattr(self, "current_task", None):
                try:
                    self._target_episodes = int(self.current_task.get("collection_count") or 0) or None
                except Exception:
                    self._target_episodes = None

            # 读取目标帧数
            try:
                target_frames = int(self.frames_entry.text().strip())
                if target_frames <= 0:
                    target_frames = 1800
            except:
                target_frames = 1800
            # 存储到UI目标帧，用于显示 X / Y 帧
            self._target_frames_ui = target_frames
            
            # 根据FPS计算目标时长（用于时间轴显示）
            fps = cc.fps if hasattr(cc, 'fps') else 30
            target_duration = target_frames / fps
            
            # 每条 episode 都按目标帧数完整采集（不按累计帧做剩余计算）
            
            pico_only = bool(getattr(cc, "pico_only_collection", False))
            if os.getenv("PICO_ONLY_COLLECTION", "").strip().lower() in ("1", "true", "yes", "on"):
                pico_only = True
            try:
                if hasattr(self, "pico_only_collection_cb") and self.pico_only_collection_cb.isChecked():
                    pico_only = True
            except Exception:
                pass

            skip_prep = os.getenv("SKIP_ROBOT_PREP", "").strip().lower() in ("1", "true", "yes", "on")
            want_prep = not skip_prep and (
                bool(getattr(cc, "robot_prep_on_start", False))
                or (hasattr(self, "robot_prep_cb") and self.robot_prep_cb.isChecked())
            )
            if want_prep:
                self._prep_start_ctx = {
                    "cc": cc,
                    "target_frames": target_frames,
                    "target_duration": target_duration,
                    "pico_only_before": pico_only,
                }
                try:
                    self.status_display.setText("准备动作执行中…")
                except Exception:
                    pass
                self.start_btn.setEnabled(False)
                threading.Thread(target=self._robot_prep_worker, daemon=True).start()
                return

            self._execute_collection_start(cc, target_frames, target_duration, pico_only, from_prep=False)
        except Exception as e:
            QMessageBox.critical(self, "启动失败", str(e))
            logger.exception("启动采集失败：%s", e)

    def _robot_prep_worker(self) -> None:
        err = None
        try:
            from lerobot_data_collector.robot_prep_zhunbei import run_zhunbei_prep_once

            run_zhunbei_prep_once()
        except Exception as e:
            err = e
            logger.exception("机器人准备动作失败：%s", e)
        QTimer.singleShot(0, lambda e=err: self._on_robot_prep_done(e))

    def _on_robot_prep_done(self, err: Optional[Exception]) -> None:
        try:
            self.start_btn.setEnabled(True)
        except Exception:
            pass
        ctx = getattr(self, "_prep_start_ctx", None)
        self._prep_start_ctx = None
        if err is not None:
            QMessageBox.critical(self, "准备动作失败", str(err))
            try:
                self.status_display.setText("就绪")
            except Exception:
                pass
            return
        if not ctx:
            return
        try:
            self.status_display.setText("准备完成：请连接 Pico；左 A/B 遥操，右 A/B 采图")
        except Exception:
            pass
        p0 = ctx.get("pico_only_before", True)
        self._execute_collection_start(
            ctx["cc"],
            ctx["target_frames"],
            ctx["target_duration"],
            p0,
            from_prep=True,
        )

    def _execute_collection_start(
        self,
        cc,
        target_frames: int,
        target_duration: float,
        pico_only: bool,
        from_prep: bool = False,
    ) -> None:
        import json
        import time

        if from_prep:
            pico_only = True
            try:
                if hasattr(self, "pico_only_collection_cb"):
                    self.pico_only_collection_cb.setChecked(True)
            except Exception:
                pass
            logger.info("准备动作完成后：强制 Pico 模式（相机由右 A 启动，左 A/B 需遥操作桥）")

        self.collector = self.collector_class(cc)
        self.collector.start()
        self.collector.start_session_immediate(target_frames, target_duration=None)

        self._update_pico_status_display(cc)

        if self._pico_adapter and self.collector:
            from lerobot_data_collector.pico_controller import create_pico_control_handler

            create_pico_control_handler(self.collector)
        self.collection_start_time = time.time()
        self.collection_duration = target_duration

        if not pico_only:
            self._start_right_camera_recording(cc, target_frames)
            self._start_left_camera_recording(cc, target_frames)
            self._start_head_camera_recording(cc, target_frames)
        else:
            logger.info("Pico 专用：界面未启动相机录制，请按 Pico 右 A 开始相机")

        logger.info("开始采集")
        self.timer_display.setText("00:00")
        try:
            if cc.dataset_root:
                local_root = os.path.expanduser(cc.dataset_root)
            else:
                local_root = os.environ.get("LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
            dataset_root_path = Path(local_root) / cc.repo_id
            task_meta_dir = dataset_root_path / "meta"
            task_meta_dir.mkdir(parents=True, exist_ok=True)
            task_meta_file = task_meta_dir / "task.json"
            task_payload = {
                "task_id": (self.current_task or {}).get("id"),
                "task_name": (self.current_task or {}).get("task_name") or (self.current_task or {}).get("name"),
                "target_frames": self._target_frames_ui,
                "target_episodes": self._target_episodes,
                "existing_episodes": self._existing_episodes,
                "created_at": int(time.time()),
            }
            with open(task_meta_file, "w", encoding="utf-8") as f:
                json.dump(task_payload, f, ensure_ascii=False, indent=2)
            logger.info("已写入任务元信息：%s", str(task_meta_file))
        except Exception as e:
            logger.warning("写入任务元信息失败：%s", e)

        if self._target_frames_ui > 0:
            self.collection_count_display.setText(f"0 / {self._target_frames_ui} 帧")
        else:
            self.collection_count_display.setText("0 帧")
        try:
            if self._target_frames_ui > 0:
                self.frames_progress_label_left.setText(f"0 / {self._target_frames_ui} 帧")
            else:
                self.frames_progress_label_left.setText("0 帧")
        except Exception:
            pass
        if self._target_episodes is not None and self._target_episodes > 0:
            text_eps = f"{self._existing_episodes} / {self._target_episodes} 条"
        else:
            text_eps = f"{self._existing_episodes} 条"
        self.episode_progress_label.setText(text_eps)
        try:
            self.episode_progress_label_left.setText(text_eps)
        except Exception:
            pass
        self._update_status_grids(0)
        self.timer.start(1000)

    def _go_back_to_task_manager(self):
        """返回任务管理窗口"""
        # 如果正在采集，先停止采集
        if self.collector and self.collector._session_active:
            reply = QMessageBox.question(self, "确认返回", 
                                       "正在采集数据，返回将停止采集。\n\n确定要返回吗？",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            # 停止采集
            try:
                self.collector._sampling = False
                self.collector._session_active = False
                self.timer.stop()
                
                # 关闭适配器
                for ad in self.collector._adapters:
                    try:
                        ad.stop()
                    except Exception:
                        pass
                self.collector._adapters.clear()
            except Exception as e:
                logger.exception("停止采集时出错：%s", e)
        # 始终停止相机录制（预览线程保留）
        try:
            self._stop_right_camera_recording()
        except Exception:
            pass
        try:
            self._stop_left_camera_recording()
        except Exception:
            pass
        try:
            self._stop_head_camera_recording()
        except Exception:
            pass
        
        # 断开信号，避免触发 on_collector_closed
        if self.task_manager:
            try:
                self.destroyed.disconnect()
            except:
                pass
        
        # 隐藏窗口而不是关闭，避免触发 destroyed 信号
        if self.task_manager:
            self.hide()
            self.task_manager.show()
            self.task_manager.raise_()
            self.task_manager.activateWindow()
        else:
            # 如果没有任务管理窗口，直接关闭
            self.close()
    
    def cancel_collection(self):
        """立即结束采集，不保存数据"""
        if not self.collector:
            QMessageBox.warning(self, "提示", "尚未开始采集")
            return
        
        # 确认是否要取消采集
        reply = QMessageBox.question(self, "确认取消", 
                                   "确定要结束采集吗？\n\n未保存的数据将丢失。",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            # 立即停止采集循环
            self.collector._sampling = False
            self.collector._session_active = False
            
            # 停止定时器
            self.timer.stop()
            self.collection_start_time = None
            
            # 关闭适配器
            for ad in self.collector._adapters:
                try:
                    ad.stop()
                except Exception:
                    pass
            
            # 清空适配器列表
            self.collector._adapters.clear()
            
            # 重置计时器显示
            self.timer_display.setText("00:00")
            frames_written = getattr(self.collector, '_frames_written', 0)
            # 显示最终采集数量（包括已有数据）
            if hasattr(self, '_existing_frames') and self._existing_frames > 0:
                total_frames = self._existing_frames + frames_written
                if getattr(self, "_target_frames_ui", 0):
                    self.collection_count_display.setText(f"{total_frames} / {self._target_frames_ui} 帧")
                else:
                    self.collection_count_display.setText(f"{total_frames} 帧")
            else:
                if getattr(self, "_target_frames_ui", 0):
                    self.collection_count_display.setText(f"{frames_written} / {self._target_frames_ui} 帧")
                else:
                    self.collection_count_display.setText(f"{frames_written} 帧")
            self._update_status_grids(0)
            
            logger.info("采集流程已取消")
            QMessageBox.information(self, "已取消", 
                                  f"采集已结束\n\n已采集 {frames_written} 帧（未保存）")
            
            # 重置collector，允许重新开始
            self.collector = None
            
        except Exception as e:
            QMessageBox.critical(self, "取消失败", str(e))
            logger.exception("取消采集失败：%s", e)
        finally:
            try:
                self._stop_right_camera_recording()
            except Exception:
                pass
            try:
                self._stop_left_camera_recording()
            except Exception:
                pass
            try:
                self._stop_head_camera_recording()
            except Exception:
                pass
    
    def stop(self):
        """结束并保存"""
        if not self.collector:
            QMessageBox.warning(self, "提示", "尚未开始采集，无法保存")
            return
        
        if self.collector._frames_written == 0:
            QMessageBox.warning(self, "保存失败", 
                              "没有采集到任何数据，无法保存。\n\n请先点击'开始采集'并等待数据采集后再保存。")
            return
        
        try:
            self.collection_start_time = None
            self.timer.stop()

            # 停止Pico状态更新定时器
            self._stop_pico_status_timer()

            # 停止独立的Pico适配器
            if self._pico_adapter:
                try:
                    self._pico_adapter.stop()
                    logger.info("独立的Pico适配器已停止")
                except Exception as e:
                    logger.warning(f"停止Pico适配器时出错: {e}")
                self._pico_adapter = None
            try:
                if getattr(self, "_pico_teleop_runtime", None) is not None:
                    self._pico_teleop_runtime.stop()
            except Exception:
                pass
            self._pico_teleop_runtime = None

            # 重置Pico连接按钮状态
            try:
                if hasattr(self, 'pico_connect_btn'):
                    self.pico_connect_btn.setEnabled(True)
                    self.pico_connect_btn.setText("🔗 连接Pico")
                if hasattr(self, 'pico_disconnect_btn'):
                    self.pico_disconnect_btn.setEnabled(False)
                    self.pico_disconnect_btn.setText("🔌 断开Pico")
                if hasattr(self, 'pico_status_label'):
                    self.pico_status_label.setText("未启用")
                    self.pico_status_label.setStyleSheet(f"color: {self.colors['text_tertiary'].name()}; font-weight: 500;")
                if hasattr(self, 'pico_indicator'):
                    self.pico_indicator.setStyleSheet(f"color: {self.colors['accent_red'].name()}; font-size: 16px; font-weight: bold;")
            except Exception:
                pass
            
            logger.info("保存采集数据")
            QApplication.processEvents()  # 立即更新UI
            
            self.collector.stop_and_save()
            
            # 每条 episode 独立：这里只展示“本次采集帧数”，不展示累计帧，避免误解
            logger.info("保存成功，本次采集 %d 帧数据", self.collector._frames_written)
            # 额外提示：显示实际保存目录与最新生成的 parquet 文件，避免“以为没生成”
            save_hint = ""
            try:
                from pathlib import Path
                ds_root = None
                if getattr(self.collector, "cfg", None) and getattr(self.collector.cfg, "dataset_root", None):
                    ds_root = os.path.expanduser(self.collector.cfg.dataset_root)
                repo_id = getattr(getattr(self.collector, "cfg", None), "repo_id", None)
                if ds_root and repo_id:
                    ds_dir = Path(ds_root) / str(repo_id)
                    parquet_files = sorted(ds_dir.rglob("*.parquet"), key=lambda p: p.stat().st_mtime)
                    latest = parquet_files[-1].name if parquet_files else "-"
                    save_hint = f"\n\n保存目录：{ds_dir}\n最新文件：{latest}\nparquet总数：{len(parquet_files)}"
                    logger.info("保存目录=%s, parquet_total=%d, latest=%s", str(ds_dir), len(parquet_files), latest)
            except Exception as e:
                logger.debug("生成保存提示失败：%s", e)

            QMessageBox.information(
                self,
                "完成",
                f"采集已结束并保存为 LeRobot 数据集\n\n本次采集 {self.collector._frames_written} 帧{save_hint}",
            )
            
            # 更新 episode 进度（保存一次视为新增1条）
            try:
                self._existing_episodes = int(self._existing_episodes or 0) + 1
                if self._target_episodes is not None and self._target_episodes > 0:
                    text_eps = f"{self._existing_episodes} / {self._target_episodes} 条"
                else:
                    text_eps = f"{self._existing_episodes} 条"
                self.episode_progress_label.setText(text_eps)
                # 同步左侧信息条
                try:
                    self.episode_progress_label_left.setText(text_eps)
                except Exception:
                    pass
            except Exception:
                pass
            # 关键：结束并保存后不重置“条数进度”；但下一条采集帧数从 0 开始计数
            self._existing_frames = 0
        except ValueError as e:
            error_msg = str(e)
            if "没有采集到任何数据" in error_msg or "没有采集到任何数据帧" in error_msg:
                QMessageBox.warning(self, "保存失败", 
                                  "没有采集到任何数据，无法保存。\n\n请先点击'开始采集'并等待数据采集后再保存。")
            else:
                QMessageBox.critical(self, "保存失败", error_msg)
                logger.error("保存失败：%s", error_msg)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
            logger.exception("保存失败：%s", e)
        finally:
            # 结束并保存后停止相机录制（预览线程保留）
            try:
                self._stop_right_camera_recording()
            except Exception:
                pass
            try:
                self._stop_head_camera_recording()
            except Exception:
                pass
            # 允许继续按顺序采集下一条（同一任务/同一 repo_id）
            self.collector = None

    def closeEvent(self, event):
        try:
            self._stop_right_camera_recording()
        except Exception:
            pass
        try:
            self._stop_head_camera_recording()
        except Exception:
            pass
        if self.head_camera_manager:
            try:
                self.head_camera_manager.shutdown()
            except Exception:
                pass
            self.head_camera_manager = None

        # 清理独立的Pico适配器
        if self._pico_adapter:
            try:
                self._pico_adapter.stop()
                logger.info("程序退出时清理Pico适配器")
            except Exception as e:
                logger.warning(f"程序退出时清理Pico适配器失败: {e}")
            self._pico_adapter = None
        try:
            if getattr(self, "_pico_teleop_runtime", None) is not None:
                self._pico_teleop_runtime.stop()
        except Exception:
            pass
        self._pico_teleop_runtime = None

        super().closeEvent(event)

    def _update_pico_status_display(self, config):
        """更新Pico状态显示"""
        try:
            # Pico连接按钮是独立功能：UI始终显示，这里只更新“是否在配置中启用 Pico”的提示文案
            enabled_in_cfg = bool(getattr(config, "pico_enabled", False))
            if enabled_in_cfg:
                self.pico_status_label.setText("已启用（未连接）")
                self.pico_status_label.setStyleSheet(
                    f"color: {self.colors['accent_orange'].name()}; font-weight: 500;"
                )
                self.pico_indicator.setStyleSheet(
                    f"color: {self.colors['accent_orange'].name()}; font-size: 16px; font-weight: bold;"
                )
                logger.info("Pico在配置中已启用（等待手动连接）")
            else:
                self.pico_status_label.setText("未配置（可手动连接）")
                self.pico_status_label.setStyleSheet(
                    f"color: {self.colors['text_tertiary'].name()}; font-weight: 500;"
                )
                self.pico_indicator.setStyleSheet(
                    f"color: {self.colors['accent_red'].name()}; font-size: 16px; font-weight: bold;"
                )

        except Exception as e:
            logger.error(f"更新Pico状态显示失败: {e}")

    def _update_pico_connection_status(self, is_connected: bool):
        """更新Pico连接状态指示器"""
        try:
            if is_connected:
                self.pico_indicator.setStyleSheet(f"color: {self.colors['accent_green'].name()}; font-size: 16px; font-weight: bold;")
                self.pico_status_label.setText("已连接")
                self.pico_status_label.setStyleSheet(f"color: {self.colors['accent_green'].name()}; font-weight: 500;")
            else:
                self.pico_indicator.setStyleSheet(f"color: {self.colors['accent_orange'].name()}; font-size: 16px; font-weight: bold;")
                self.pico_status_label.setText("连接中...")
                self.pico_status_label.setStyleSheet(f"color: {self.colors['accent_orange'].name()}; font-weight: 500;")
        except Exception as e:
            logger.debug(f"更新Pico连接状态失败: {e}")

    def _start_pico_status_timer(self):
        """启动Pico状态更新定时器"""
        try:
            if not hasattr(self, '_pico_status_timer'):
                self._pico_status_timer = QTimer()
                self._pico_status_timer.timeout.connect(self._update_pico_status)
                self._pico_status_timer.setInterval(1000)  # 每秒更新一次
            if not self._pico_status_timer.isActive():
                self._pico_status_timer.start()
                logger.debug("Pico状态更新定时器已启动")
        except Exception as e:
            logger.error(f"启动Pico状态定时器失败: {e}")

    def _stop_pico_status_timer(self):
        """停止Pico状态更新定时器"""
        try:
            if hasattr(self, '_pico_status_timer') and self._pico_status_timer.isActive():
                self._pico_status_timer.stop()
                logger.debug("Pico状态更新定时器已停止")
        except Exception as e:
            logger.error(f"停止Pico状态定时器失败: {e}")

    def _update_pico_status(self):
        """定期更新Pico连接状态"""
        try:
            if self._pico_adapter:
                is_connected = self._pico_adapter.is_connected
                self._update_pico_connection_status(is_connected)
        except Exception as e:
            logger.debug(f"Pico状态更新失败: {e}")

    def _connect_pico(self):
        """连接Pico遥控器（独立于采集器，类似于robot_control.py中的行为）"""
        teleop_rt = None
        try:
            # 检查Pico适配器是否已存在
            if self._pico_adapter is not None:
                # 如果已经连接，显示状态
                if self._pico_adapter.is_connected:
                    QMessageBox.information(self, "Pico状态", "Pico遥控器已连接并正常工作")
                else:
                    QMessageBox.information(self, "Pico状态", "Pico遥控器适配器已创建，正在尝试连接...")
                return

            # 禁用连接按钮，避免重复点击
            self.pico_connect_btn.setEnabled(False)
            self.pico_connect_btn.setText("🔄 连接中...")
            QApplication.processEvents()  # 立即更新UI

            logger.info("开始连接Pico遥控器...")

            # 创建Pico适配器（类似于robot_control.py中的TeleopData）
            from lerobot_data_collector.pico_controller import PicoControllerAdapter

            # 获取 Pico IP/端口：优先使用 UI 输入框；其次使用采集器配置；最后使用默认值
            pico_ip = None
            pico_port = None
            try:
                if hasattr(self, "pico_ip_input"):
                    pico_ip = (self.pico_ip_input.text() or "").strip()
                if hasattr(self, "pico_port_input"):
                    pico_port = int(self.pico_port_input.value())
            except Exception:
                pico_ip = None
                pico_port = None

            if (not pico_ip) and self.collector and hasattr(self.collector, "cfg"):
                pico_ip = getattr(self.collector.cfg, "pico_ip", None)
            if (not pico_port) and self.collector and hasattr(self.collector, "cfg"):
                pico_port = getattr(self.collector.cfg, "pico_port", None)

            pico_ip = (pico_ip or "192.168.12.110").strip()
            pico_port = int(pico_port or 12345)

            # 运动学等高级参数仍从配置读取（若没有采集器则默认关闭）
            if self.collector and hasattr(self.collector, "cfg"):
                enable_kinematics = getattr(self.collector.cfg, "pico_kinematics_enabled", False)
                urdf_left = getattr(self.collector.cfg, "pico_urdf_left", None)
                urdf_right = getattr(self.collector.cfg, "pico_urdf_right", None)
            else:
                enable_kinematics = False
                urdf_left = None
                urdf_right = None

            # 连接时也强制写入“上次输入”（防止用户没改动但想保存）
            try:
                if getattr(self, "_settings", None) is not None:
                    self._settings.setValue("pico/last_ip", pico_ip)
                    self._settings.setValue("pico/last_port", int(pico_port))
                    self._settings.sync()
            except Exception:
                pass

            # Pico控制统一由Qt处理，确保右A/右B与原项目100%一致
            pico_control_handler = self._on_pico_control

            use_teleop_bridge = False
            if self.collector and hasattr(self.collector, "cfg"):
                use_teleop_bridge = bool(getattr(self.collector.cfg, "pico_teleop_bridge", False))
                if bool(getattr(self.collector.cfg, "robot_prep_on_start", False)):
                    use_teleop_bridge = True
            # 准备动作模式默认需要左A/B控机，自动启用遥操作桥
            try:
                if hasattr(self, "robot_prep_cb") and self.robot_prep_cb.isChecked():
                    use_teleop_bridge = True
            except Exception:
                pass
            if os.getenv("PICO_TELEOP_BRIDGE", "").strip().lower() in ("1", "true", "yes", "on"):
                use_teleop_bridge = True

            if use_teleop_bridge:
                try:
                    from lerobot_data_collector.pico_teleop_bridge import (
                        PicoTeleopRuntime,
                        teleop_bridge_available,
                    )

                    if teleop_bridge_available():
                        teleop_rt = PicoTeleopRuntime()
                        teleop_rt.start()
                        logger.info("Pico 遥操作桥已启用（单路流 + IK + LCM）")
                except Exception as e:
                    logger.warning("Pico 遥操作桥未启动：%s", e)
                    teleop_rt = None

            # 创建独立的Pico适配器
            self._pico_adapter = PicoControllerAdapter(
                server_ip=pico_ip,
                port=pico_port,
                on_control=pico_control_handler,
                on_motion=None,  # 暂时不处理运动数据
                enable_kinematics=enable_kinematics,
                urdf_left_path=urdf_left,
                urdf_right_path=urdf_right,
                teleop_runtime=teleop_rt,
            )

            # 尝试启动连接（类似于mocapActionInfo的初始化）
            if self._pico_adapter.start():
                self._pico_teleop_runtime = teleop_rt
                # 启动状态更新定时器
                self._start_pico_status_timer()

                # 更新UI状态
                self.pico_connect_btn.setText("✅ 已连接")
                self.pico_connect_btn.setEnabled(False)  # 连接成功后禁用连接按钮
                if hasattr(self, 'pico_disconnect_btn'):
                    self.pico_disconnect_btn.setEnabled(True)  # 启用断开按钮
                self.pico_indicator.setStyleSheet(f"color: {self.colors['accent_green'].name()}; font-size: 16px; font-weight: bold;")
                self.pico_status_label.setText("连接成功")
                self.pico_status_label.setStyleSheet(f"color: {self.colors['accent_green'].name()}; font-weight: 500;")

                logger.info("Pico遥控器连接成功")
                QMessageBox.information(self, "成功", "Pico遥控器连接成功！\n\n现在您可以使用Pico控制器进行数据采集控制。")
            else:
                # 连接失败
                if teleop_rt is not None:
                    try:
                        teleop_rt.stop()
                    except Exception:
                        pass
                self._pico_teleop_runtime = None
                self._pico_adapter = None  # 清理失败的适配器
                self.pico_connect_btn.setText("❌ 连接失败")
                self.pico_connect_btn.setEnabled(True)
                if hasattr(self, 'pico_disconnect_btn'):
                    self.pico_disconnect_btn.setEnabled(False)  # 确保断开按钮禁用
                self.pico_indicator.setStyleSheet(f"color: {self.colors['accent_red'].name()}; font-size: 16px; font-weight: bold;")
                self.pico_status_label.setText("连接失败")
                self.pico_status_label.setStyleSheet(f"color: {self.colors['accent_red'].name()}; font-weight: 500;")

                logger.error("Pico遥控器连接失败")
                QMessageBox.warning(self, "连接失败", "无法连接到Pico遥控器。\n\n请检查：\n• Pico设备是否开启\n• IP地址和端口是否正确\n• 网络连接是否正常")

        except Exception as e:
            logger.error(f"Pico连接过程中出错: {e}")
            try:
                if teleop_rt is not None:
                    teleop_rt.stop()
            except Exception:
                pass
            self._pico_teleop_runtime = None
            self._pico_adapter = None  # 清理失败的适配器
            self.pico_connect_btn.setText("❌ 错误")
            self.pico_connect_btn.setEnabled(True)
            if hasattr(self, 'pico_disconnect_btn'):
                self.pico_disconnect_btn.setEnabled(False)  # 确保断开按钮禁用
            QMessageBox.critical(self, "错误", f"Pico连接过程中出现错误：\n\n{str(e)}")
        finally:
            QApplication.processEvents()  # 确保UI更新

    def _disconnect_pico(self):
        """断开Pico遥控器连接"""
        try:
            if self._pico_adapter is None:
                QMessageBox.information(self, "提示", "Pico遥控器未连接")
                return

            # 禁用断开按钮，避免重复点击
            if hasattr(self, 'pico_disconnect_btn'):
                self.pico_disconnect_btn.setEnabled(False)
                self.pico_disconnect_btn.setText("🔄 断开中...")
            QApplication.processEvents()  # 立即更新UI

            logger.info("开始断开Pico遥控器连接...")

            # 停止状态更新定时器
            self._stop_pico_status_timer()

            # 停止Pico适配器
            try:
                self._pico_adapter.stop()
                logger.info("Pico适配器已停止")
            except Exception as e:
                logger.warning(f"停止Pico适配器时出错: {e}")

            # 清理适配器引用
            self._pico_adapter = None
            try:
                if getattr(self, "_pico_teleop_runtime", None) is not None:
                    self._pico_teleop_runtime.stop()
            except Exception as e:
                logger.warning("停止 Pico 遥操作桥时出错: %s", e)
            self._pico_teleop_runtime = None

            # 更新UI状态
            self.pico_connect_btn.setText("🔗 连接Pico")
            self.pico_connect_btn.setEnabled(True)  # 启用连接按钮
            if hasattr(self, 'pico_disconnect_btn'):
                self.pico_disconnect_btn.setText("🔌 断开Pico")
                self.pico_disconnect_btn.setEnabled(False)  # 禁用断开按钮
            self.pico_indicator.setStyleSheet(f"color: {self.colors['accent_red'].name()}; font-size: 16px; font-weight: bold;")
            self.pico_status_label.setText("已断开")
            self.pico_status_label.setStyleSheet(f"color: {self.colors['text_tertiary'].name()}; font-weight: 500;")

            logger.info("Pico遥控器已断开")
            QMessageBox.information(self, "成功", "Pico遥控器已断开连接")

        except Exception as e:
            logger.error(f"断开Pico连接过程中出错: {e}")
            QMessageBox.critical(self, "错误", f"断开Pico连接过程中出现错误：\n\n{str(e)}")
        finally:
            QApplication.processEvents()  # 确保UI更新

    def _on_pico_control(self, cmd: str) -> None:
        try:
            self.pico_control_signal.emit(cmd)
        except Exception as e:
            logger.error("转发Pico命令到UI线程失败：%s", e)

    def _on_pico_control_main(self, cmd: str) -> None:
        """Pico按钮事件入口：保证右A/右B行为与原项目完全一致（start_save_data(0/1)）。"""
        try:
            if cmd in ("pico_toggle_control", "pico_toggle_data_send", "pico_change_task_mode"):
                tr = getattr(self, "_pico_teleop_runtime", None)
                if tr is not None:
                    tr.handle_command(cmd)
                else:
                    logger.warning("左手Pico命令已收到，但未启用遥操作桥（请重连Pico或启用 robot_prep_on_start/pico_teleop_bridge）")
                return
            if cmd == "pico_right_a":
                self._pico_start_save_data_0()
                return
            if cmd == "pico_right_b":
                self._pico_start_save_data_1()
                return

            logger.info("Pico命令：%s", cmd)
        except Exception as e:
            logger.error("处理Pico命令失败：%s", e)

    def _pico_start_save_data_0(self) -> None:
        import json
        import os
        import time
        from datetime import datetime
        from pathlib import Path

        session_id = f"data_sync_{time.time_ns()}"
        try:
            delay_seconds = float(os.environ.get("PICO_START_DELAY_SECONDS", "0"))
        except Exception:
            delay_seconds = 0.0
        if delay_seconds < 0:
            delay_seconds = 0.0
        # 目标帧数：以软件界面“目标帧数”为准；取不到则回退到 2400（与原 ros_node 默认一致）
        try:
            target_frames = int(self.duration_input.value())
            if target_frames <= 0:
                target_frames = 2400
        except Exception:
            target_frames = 2400
        start_timestamp = time.time() + delay_seconds

        # 同步采集器中的 Pico 地址，避免右A创建采集器时回落到旧默认IP
        runtime_pico_ip = None
        runtime_pico_port = None
        try:
            if getattr(self, "_pico_adapter", None) is not None:
                runtime_pico_ip = str(getattr(self._pico_adapter, "server_ip", "") or "").strip()
                runtime_pico_port = int(getattr(self._pico_adapter, "port", 0) or 0)
            if not runtime_pico_ip and hasattr(self, "pico_ip_input"):
                runtime_pico_ip = (self.pico_ip_input.text() or "").strip()
            if (not runtime_pico_port) and hasattr(self, "pico_port_input"):
                runtime_pico_port = int(self.pico_port_input.value())
        except Exception:
            runtime_pico_ip = None
            runtime_pico_port = None

        logger.info("收到 Pico 右A，准备启动采集：session_id=%s", session_id)
        self._pico_session_mode = True
        try:
            self.status_display.setText("Pico触发：准备开始")
        except Exception:
            pass

        # 如果采集器尚未启动：按当前下拉框配置初始化采集器（但不立刻开始 session）
        if self.collector is None:
            current_index = self.cfg_combobox.currentIndex()
            if current_index < 0:
                QMessageBox.critical(self, "错误", "请先选择配置文件")
                return
            cfg_path = self.cfg_combobox.itemData(current_index)
            if not cfg_path or not os.path.exists(cfg_path):
                QMessageBox.critical(self, "错误", "请先选择有效的配置 JSON 文件")
                return
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cc = self.collector_config_class.from_dict(cfg)
            # UI侧已连接独立Pico适配器，这里禁用采集器内部Pico，避免重复监听与重复触发
            cc.pico_enabled = False
            if runtime_pico_ip:
                cc.pico_ip = runtime_pico_ip
            if runtime_pico_port:
                cc.pico_port = int(runtime_pico_port)

            # dataset_root：沿用 start() 的可写根目录策略（当天目录）
            try:
                import re
                import socket
                base_root = (os.environ.get("HF_DATASET_ROOT") or getattr(self, "_base_dataset_root", "")).strip()
                if base_root:
                    base_root = os.path.expanduser(base_root)
                if (not base_root) or (not os.path.isdir(base_root)) or (not os.access(base_root, os.W_OK | os.X_OK)):
                    base_root = os.path.expanduser(str(getattr(self, "_base_dataset_root", "~/.cache/huggingface/lerobot")))
                day = datetime.now().strftime("%Y%m%d")
                suffix = (os.environ.get("HF_DATASET_DAILY_SUFFIX") or socket.gethostname() or "host").strip()
                suffix_safe = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix).strip("-") or "host"
                daily_root = os.path.join(base_root, f"{day}_{suffix_safe}")
                os.makedirs(daily_root, exist_ok=True)
                cc.dataset_root = daily_root
            except Exception:
                pass

            # repo_id：与原项目“session_id”对齐，便于对应目录
            cc.repo_id = session_id
            os.environ["FIXED_REPO_ID"] = cc.repo_id

            self.collector = self.collector_class(cc)
            self.collector.start()

            # 更新UI：让用户看到 Pico 触发的“待开始”
            try:
                self.status_display.setText("Pico触发：即将开始")
                self.timer_display.setText("00:00")
            except Exception:
                pass

            # Pico配置状态提示（不影响连接按钮显示）
            self._update_pico_status_display(cc)
        else:
            cc = getattr(self.collector, "cfg", None)
            try:
                is_active = bool(getattr(self.collector, "_session_active", False))
                start_pending = bool(
                    hasattr(self.collector, "_scheduled_start_time")
                    and (getattr(self.collector, "_scheduled_start_time", None) is not None)
                )
            except Exception:
                is_active = False
                start_pending = False

            # 每次右A都重开新会话：先停旧会话（含相机），再开始新会话
            if is_active or start_pending:
                logger.info("检测到会话运行中，右A触发重开：先停止旧会话")
                try:
                    if hasattr(self, "_pico_pending_start_timer") and self._pico_pending_start_timer:
                        self._pico_pending_start_timer.stop()
                except Exception:
                    pass
                try:
                    if hasattr(self, "_pico_pending_left_timer") and self._pico_pending_left_timer:
                        self._pico_pending_left_timer.stop()
                except Exception:
                    pass
                try:
                    if hasattr(self, "_pico_pending_head_timer") and self._pico_pending_head_timer:
                        self._pico_pending_head_timer.stop()
                except Exception:
                    pass
                try:
                    self._stop_right_camera_recording()
                except Exception:
                    pass
                try:
                    self._stop_left_camera_recording()
                except Exception:
                    pass
                try:
                    self._stop_head_camera_recording()
                except Exception:
                    pass
                try:
                    self.collector._allow_no_action = False
                    self.collector._on_control("abort")
                except Exception as e:
                    logger.warning("右A重开时停止旧会话失败：%s", e)

            if cc is not None:
                cc.pico_enabled = False
                if runtime_pico_ip:
                    cc.pico_ip = runtime_pico_ip
                if runtime_pico_port:
                    cc.pico_port = int(runtime_pico_port)
                cc.repo_id = session_id
                os.environ["FIXED_REPO_ID"] = cc.repo_id
            try:
                inner_pico = getattr(self.collector, "_pico_adapter", None)
                if inner_pico is not None:
                    inner_pico.stop()
                    self.collector._pico_adapter = None
            except Exception:
                pass
            # 关键修复：每次右A都重建 collector，确保 repo_id/dataset_root 生效到新的数据上下文，
            # 避免出现 episode_000001.parquet 这类残留条目或 Missing Dir。
            if cc is not None:
                try:
                    if hasattr(self, "_pico_pending_stop_timer") and self._pico_pending_stop_timer:
                        self._pico_pending_stop_timer.stop()
                except Exception:
                    pass
                try:
                    old = self.collector
                    if old is not None:
                        old._sampling = False
                        t = getattr(old, "_sampler_thread", None)
                        if t is not None and t.is_alive():
                            t.join(timeout=1.0)
                        for ad in getattr(old, "_adapters", []):
                            try:
                                ad.stop()
                            except Exception:
                                pass
                        try:
                            old._adapters.clear()
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning("重建采集器前停止旧实例失败：%s", e)
                try:
                    self.collector = self.collector_class(cc)
                    self.collector.start()
                    logger.info("Pico右A：已重建采集器实例（repo_id=%s）", cc.repo_id)
                except Exception as e:
                    logger.error("Pico右A：重建采集器失败：%s", e)
                    return

        # 1) 采集：按原消息格式走 Collector._on_control（内部按 scheduled_start_time 启动）
        if self.collector is not None:
            try:
                if hasattr(self.collector, "ensure_sampling_running"):
                    self.collector.ensure_sampling_running()
                self.collector._allow_no_action = True
                self.collector._on_control(f"start:{session_id}:{start_timestamp:.6f}:{target_frames}")
            except Exception as e:
                logger.error("Pico start_save_data(0) 下发失败：%s", e)

        # 2) 相机分步启动，避免同一时刻拉起三个相机造成资源竞争
        try:
            delay_ms = max(0, int(delay_seconds * 1000))
            stagger_ms = 400
            if hasattr(self, "_pico_pending_start_timer") and self._pico_pending_start_timer:
                try:
                    self._pico_pending_start_timer.stop()
                except Exception:
                    pass
            if hasattr(self, "_pico_pending_left_timer") and self._pico_pending_left_timer:
                try:
                    self._pico_pending_left_timer.stop()
                except Exception:
                    pass
            if hasattr(self, "_pico_pending_head_timer") and self._pico_pending_head_timer:
                try:
                    self._pico_pending_head_timer.stop()
                except Exception:
                    pass
            self._pico_pending_start_timer = QTimer()
            self._pico_pending_start_timer.setSingleShot(True)
            self._pico_pending_left_timer = QTimer()
            self._pico_pending_left_timer.setSingleShot(True)
            self._pico_pending_head_timer = QTimer()
            self._pico_pending_head_timer.setSingleShot(True)

            def _start_right_cam():
                if cc is None:
                    return
                try:
                    self._start_right_camera_recording(cc, target_frames)
                except Exception as e:
                    logger.warning("Pico启动右手相机录制失败：%s", e)

            def _start_left_cam():
                if cc is None:
                    return
                try:
                    self._start_left_camera_recording(cc, target_frames)
                except Exception as e:
                    logger.warning("Pico启动左手相机录制失败：%s", e)

            def _start_head_cam():
                if cc is None:
                    return
                try:
                    self._start_head_camera_recording(cc, target_frames)
                except Exception as e:
                    logger.warning("Pico启动头部相机录制失败：%s", e)

            self._pico_pending_start_timer.timeout.connect(_start_right_cam)
            self._pico_pending_left_timer.timeout.connect(_start_left_cam)
            self._pico_pending_head_timer.timeout.connect(_start_head_cam)
            self._pico_pending_start_timer.start(delay_ms)
            self._pico_pending_left_timer.start(delay_ms + stagger_ms)
            self._pico_pending_head_timer.start(delay_ms + 2 * stagger_ms)
        except Exception as e:
            logger.warning("Pico延迟启动相机定时器失败：%s", e)

        # 3) 对齐普通“开始采集”路径：启动UI倒计时/帧数进度刷新
        try:
            import time
            frame_rate = 30.0
            try:
                frame_rate = float(os.environ.get("COLLECTION_FRAME_RATE", "30"))
                if frame_rate <= 0:
                    frame_rate = 30.0
            except Exception:
                frame_rate = 30.0

            self.collection_start_time = time.time()
            self.collection_duration = float(target_frames) / frame_rate
            self._target_frames_ui = int(target_frames)
            self.timer_display.setText("00:00")
            self.collection_count_display.setText(f"0 / {self._target_frames_ui} 帧")
            try:
                self.frames_progress_label_left.setText(f"0 / {self._target_frames_ui} 帧")
            except Exception:
                pass
            self._update_status_grids(0)
            self.timer.start(1000)
        except Exception as e:
            logger.warning("Pico启动后初始化倒计时UI失败：%s", e)

        logger.info(
            "Pico触发采集：start_save_data(0), session_id=%s, delay=%.1fs, target_frames=%s",
            session_id,
            delay_seconds,
            target_frames,
        )

    def _pico_start_save_data_1(self) -> None:
        """等价于 robot_kinemic/ros_node.py start_save_data(1)：停止采集并停止三相机。"""
        def _do_stop_and_save() -> None:
            self._pico_session_mode = False
            # 先停相机（与原项目一致是同时停）
            try:
                self._stop_right_camera_recording()
            except Exception:
                pass
            try:
                self._stop_left_camera_recording()
            except Exception:
                pass
            try:
                self._stop_head_camera_recording()
            except Exception:
                pass

            # 取消可能的延迟启动
            try:
                if hasattr(self, "_pico_pending_start_timer") and self._pico_pending_start_timer:
                    self._pico_pending_start_timer.stop()
            except Exception:
                pass
            try:
                if hasattr(self, "_pico_pending_left_timer") and self._pico_pending_left_timer:
                    self._pico_pending_left_timer.stop()
            except Exception:
                pass
            try:
                if hasattr(self, "_pico_pending_head_timer") and self._pico_pending_head_timer:
                    self._pico_pending_head_timer.stop()
            except Exception:
                pass

            # 再停采集（Collector._on_control("stop") 会保存数据）
            if self.collector is not None:
                try:
                    self.collector._allow_no_action = False
                    self.collector._on_control("stop")
                except Exception as e:
                    logger.error("Pico start_save_data(1) 下发失败：%s", e)
                    return
            else:
                logger.info("Pico停止：当前未启动采集器，已执行相机停止（如有）")
                return

            logger.info("Pico停止：已触发 start_save_data(1)（stop）并停止三相机")

        if self.collector is None:
            _do_stop_and_save()
            return

        try:
            current_frames = int(getattr(self.collector, "_frames_written", 0) or 0)
        except Exception:
            current_frames = 0
        try:
            target_frames = int(getattr(self.collector, "_target_frames", 0) or 0)
        except Exception:
            target_frames = 0
        session_active = bool(getattr(self.collector, "_session_active", False))

        # 修复“剩余几十帧时按B导致截断保存”：B先进入挂起，等主采集达到目标帧再统一 stop+save
        if session_active and target_frames > 0 and current_frames < target_frames:
            remaining = target_frames - current_frames
            logger.info(
                "收到 Pico 右B：主采集尚未达到目标帧，当前 %d/%d，剩余 %d 帧；达到目标后自动停止并保存",
                current_frames,
                target_frames,
                remaining,
            )
            try:
                self.status_display.setText(f"等待采集完成后保存（剩余 {remaining} 帧）")
            except Exception:
                pass

            try:
                if hasattr(self, "_pico_pending_stop_timer") and self._pico_pending_stop_timer:
                    self._pico_pending_stop_timer.stop()
            except Exception:
                pass

            self._pico_pending_stop_timer = QTimer()
            self._pico_pending_stop_timer.setSingleShot(False)

            def _check_and_stop() -> None:
                if self.collector is None:
                    try:
                        self._pico_pending_stop_timer.stop()
                    except Exception:
                        pass
                    return
                now_frames = int(getattr(self.collector, "_frames_written", 0) or 0)
                now_target = int(getattr(self.collector, "_target_frames", 0) or 0)
                now_active = bool(getattr(self.collector, "_session_active", False))
                if (not now_active) or (now_target > 0 and now_frames >= now_target):
                    try:
                        self._pico_pending_stop_timer.stop()
                    except Exception:
                        pass
                    _do_stop_and_save()

            self._pico_pending_stop_timer.timeout.connect(_check_and_stop)
            self._pico_pending_stop_timer.start(100)
            return

        _do_stop_and_save()