import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QFileDialog,
        QMessageBox,
        QFrame,
        QProgressBar,
    )
    from PySide6.QtCore import Qt, QTimer, QDateTime
    from PySide6.QtGui import (
        QFont,
        QFontDatabase,
        QColor,
        QPalette,
        QPixmap,
    )
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QApplication,
            QDialog,
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QFileDialog,
            QMessageBox,
            QFrame,
            QProgressBar,
        )
        from PyQt6.QtCore import Qt, QTimer, QDateTime
        from PyQt6.QtGui import (
            QFont,
            QFontDatabase,
            QColor,
            QPalette,
            QPixmap,
        )
    except ImportError:  # pragma: no cover - Qt 不可用时由上层回退到 Tk
        raise


# 全局变量存储用户token、服务器地址和用户信息
USER_TOKEN: Optional[str] = None
SERVER_HOST: Optional[str] = None
USER_INFO: Optional[dict] = None

# 本地管理员账号：无需连接服务器即可进入客户端
LOCAL_ADMIN_USERNAME = "admin"
LOCAL_ADMIN_PASSWORD = "2525"
LOCAL_ADMIN_TOKEN = "local_admin_token"


def is_local_admin(user_identity: str, password: str) -> bool:
    """判断是否为本地管理员账号。"""
    return user_identity == LOCAL_ADMIN_USERNAME and password == LOCAL_ADMIN_PASSWORD


def local_admin_user_info() -> dict:
    """本地管理员登录后的用户信息。"""
    return {
        "user_id": 0,
        "username": LOCAL_ADMIN_USERNAME,
        "is_superuser": True,
        "role": "admin",
        "tenant_id": 0,
    }


def get_auth_headers():
    """获取带认证token的请求头，用于后续API调用。"""
    global USER_TOKEN
    headers = {
        "Content-Type": "application/json",
    }
    if USER_TOKEN:
        headers["x-auth-tkn"] = USER_TOKEN
    return headers


def get_server_host() -> str:
    """获取服务器地址"""
    global SERVER_HOST
    return SERVER_HOST or "http://localhost:8888"


def get_user_info() -> Optional[dict]:
    """获取当前登录用户信息"""
    global USER_INFO
    return USER_INFO


class LoadingDialog(QDialog):
    """加载对话框 - 显示加载状态（苹果风格）"""

    def __init__(self, parent=None, message="正在加载..."):
        super().__init__(parent)
        self.setWindowTitle("")
        self.setModal(True)
        # 使用窗口标志确保对话框显示在最前面
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )

        # 苹果风格配色
        self.colors = {
            "bg_primary": QColor(20, 20, 23),
            "bg_card": QColor(32, 32, 36),
            "text_primary": QColor(255, 255, 255),
            "accent_blue": QColor(10, 132, 255),
        }

        # 设置窗口样式
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {self.colors['bg_card'].name()};
                border-radius: 18px;
                border: 1px solid {self.colors['bg_primary'].name()};
            }}
        """
        )

        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 35, 40, 35)
        layout.setSpacing(25)

        # 加载指示器（进度条，不确定模式）
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不确定模式
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: none;
                border-radius: 2px;
                background-color: {self.colors['bg_primary'].name()};
            }}
            QProgressBar::chunk {{
                background-color: {self.colors['accent_blue'].name()};
                border-radius: 2px;
            }}
        """
        )
        layout.addWidget(self.progress_bar)

        # 消息标签
        self.message_label = QLabel(message)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setStyleSheet(
            f"""
            QLabel {{
                color: {self.colors['text_primary'].name()};
                font-size: 15px;
                font-weight: 500;
                letter-spacing: -0.2px;
            }}
        """
        )
        layout.addWidget(self.message_label)

        # 设置固定大小
        self.setFixedSize(320, 140)

        # 居中显示
        self._center_on_parent(parent)

    def _center_on_parent(self, parent):
        """在父窗口中心显示"""
        if parent:
            parent_rect = parent.geometry()
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2,
            )
        else:
            # 如果没有父窗口，在屏幕中心显示
            app = QApplication.instance()
            if app:
                screen = app.primaryScreen()
                if screen:
                    screen_rect = screen.availableGeometry()
                    self.move(
                        screen_rect.x() + (screen_rect.width() - self.width()) // 2,
                        screen_rect.y() + (screen_rect.height() - self.height()) // 2,
                    )

    def show(self):
        """显示对话框并确保在最前面"""
        super().show()
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()

    def set_message(self, message: str):
        """更新加载消息"""
        self.message_label.setText(message)
        QApplication.processEvents()


try:
    from PySide6.QtCore import Signal, QObject, Slot
except ImportError:
    from PyQt6.QtCore import pyqtSignal as Signal, QObject, pyqtSlot as Slot


class UploadProgressDialog(QDialog):
    """上传进度对话框 - 显示详细的上传信息和进度（苹果风格）"""
    
    # 定义信号用于线程安全的UI更新
    _signal_set_title = Signal(str)
    _signal_set_uuid_progress = Signal(int, int, str)
    _signal_set_file_info = Signal(str, int, int)
    _signal_set_file_progress = Signal(int)
    _signal_set_stats = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 连接信号到槽函数
        self._signal_set_title.connect(self._do_set_title)
        self._signal_set_uuid_progress.connect(self._do_set_uuid_progress)
        self._signal_set_file_info.connect(self._do_set_file_info)
        self._signal_set_file_progress.connect(self._do_set_file_progress)
        self._signal_set_stats.connect(self._do_set_stats)
        self.setWindowTitle("上传进度")
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
        )

        # 苹果风格配色
        self.colors = {
            "bg_primary": QColor(20, 20, 23),
            "bg_card": QColor(32, 32, 36),
            "text_primary": QColor(255, 255, 255),
            "text_secondary": QColor(160, 160, 165),
            "accent_blue": QColor(10, 132, 255),
            "accent_green": QColor(48, 209, 88),
        }

        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {self.colors['bg_card'].name()};
                border-radius: 20px;
                border: 1px solid {self.colors['bg_primary'].name()};
            }}
            QLabel {{
                color: {self.colors['text_primary'].name()};
            }}
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(35, 30, 35, 30)
        layout.setSpacing(20)

        # 标题
        self.title_label = QLabel("正在上传数据...")
        self.title_label.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {self.colors['text_primary'].name()}; letter-spacing: -0.4px;"
        )
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        # UUID 进度（例如：上传 UUID 2/5）
        self.uuid_label = QLabel("准备中...")
        self.uuid_label.setStyleSheet(
            f"font-size: 14px; font-weight: 500; color: {self.colors['text_secondary'].name()};"
        )
        self.uuid_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.uuid_label)

        # UUID 进度条
        self.uuid_progress = QProgressBar()
        self.uuid_progress.setRange(0, 100)
        self.uuid_progress.setValue(0)
        self.uuid_progress.setFixedHeight(6)
        self.uuid_progress.setStyleSheet(
            f"""
            QProgressBar {{
                border: none;
                border-radius: 3px;
                background-color: {self.colors['bg_primary'].name()};
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {self.colors['accent_blue'].name()};
                border-radius: 3px;
            }}
        """
        )
        layout.addWidget(self.uuid_progress)

        # 当前文件信息
        self.file_label = QLabel("")
        self.file_label.setStyleSheet(
            f"font-size: 13px; color: {self.colors['text_secondary'].name()};"
        )
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        # 文件进度条
        self.file_progress = QProgressBar()
        self.file_progress.setRange(0, 100)
        self.file_progress.setValue(0)
        self.file_progress.setFixedHeight(4)
        self.file_progress.setStyleSheet(
            f"""
            QProgressBar {{
                border: none;
                border-radius: 2px;
                background-color: {self.colors['bg_primary'].name()};
            }}
            QProgressBar::chunk {{
                background-color: {self.colors['accent_green'].name()};
                border-radius: 2px;
            }}
        """
        )
        layout.addWidget(self.file_progress)

        # 统计信息（已上传大小/总大小）
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet(
            f"font-size: 12px; font-weight: 500; color: {self.colors['text_secondary'].name()};"
        )
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.stats_label)

        self.setFixedSize(480, 250)
        self._center_on_parent(parent)

    def _center_on_parent(self, parent):
        """在父窗口中心显示"""
        if parent:
            parent_rect = parent.geometry()
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2,
            )

    def show(self):
        super().show()
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()

    def set_title(self, title: str):
        """设置标题（线程安全）"""
        self._signal_set_title.emit(title)

    @Slot(str)
    def _do_set_title(self, title: str):
        """实际更新标题（主线程）"""
        self.title_label.setText(title)

    def set_uuid_progress(self, current: int, total: int, uuid_name: str = ""):
        """设置 UUID 级别的进度（线程安全）"""
        self._signal_set_uuid_progress.emit(current, total, uuid_name)

    @Slot(int, int, str)
    def _do_set_uuid_progress(self, current: int, total: int, uuid_name: str):
        """实际更新 UUID 进度（主线程）"""
        if total > 0:
            percent = int(current / total * 100)
            self.uuid_progress.setValue(percent)
            if uuid_name:
                self.uuid_label.setText(f"上传 UUID: {uuid_name[:8]}... ({current}/{total})")
            else:
                self.uuid_label.setText(f"上传进度: {current}/{total}")

    def set_file_info(self, filename: str, current_file: int = 0, total_files: int = 0):
        """设置当前上传的文件信息（线程安全）"""
        self._signal_set_file_info.emit(filename, current_file, total_files)

    @Slot(str, int, int)
    def _do_set_file_info(self, filename: str, current_file: int, total_files: int):
        """实际更新文件信息（主线程）"""
        if total_files > 0:
            # 截断长文件名
            display_name = filename if len(filename) < 40 else "..." + filename[-37:]
            self.file_label.setText(f"文件 ({current_file}/{total_files}): {display_name}")
        else:
            self.file_label.setText(f"文件: {filename}")

    def set_file_progress(self, percent: int):
        """设置文件上传进度（0-100）（线程安全）"""
        self._signal_set_file_progress.emit(percent)

    @Slot(int)
    def _do_set_file_progress(self, percent: int):
        """实际更新文件进度（主线程）"""
        self.file_progress.setValue(min(100, max(0, percent)))

    def set_stats(self, uploaded_mb: float, total_mb: float):
        """设置统计信息（线程安全）"""
        self._signal_set_stats.emit(uploaded_mb, total_mb)

    @Slot(float, float)
    def _do_set_stats(self, uploaded_mb: float, total_mb: float):
        """实际更新统计信息（主线程）"""
        self.stats_label.setText(f"已上传: {uploaded_mb:.1f} MB / {total_mb:.1f} MB")

    def set_complete(self):
        """设置为完成状态"""
        self.title_label.setText("上传完成")
        self.uuid_progress.setValue(100)
        self.file_progress.setValue(100)
        QApplication.processEvents()


class LoginDialog(QDialog):
    """登录对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.token: Optional[str] = None
        self.user_info: Optional[dict] = None
        self.server_host: Optional[str] = None

        # 设置窗口属性
        self.setWindowTitle("登录 - SRIC数据采集平台")
        self.resize(1350, 850) # 进一步增大初始尺寸
        self.setMinimumSize(1200, 800) # 增大最小尺寸，确保不被挤压
        self.setModal(True)

        # 苹果风格配色（更深邃、更干净）
        self.colors = {
            "bg_primary": QColor(20, 20, 23),      # 极深色背景
            "bg_card": QColor(28, 28, 32),         # 卡片背景
            "bg_secondary": QColor(38, 38, 42),    # 输入框背景
            "text_primary": QColor(255, 255, 255),
            "text_secondary": QColor(170, 170, 175), # 柔和的次要文字
            "text_tertiary": QColor(110, 110, 115),
            "accent_blue": QColor(10, 132, 255),    # 苹果深蓝色
            "accent_green": QColor(48, 209, 88),
            "accent_red": QColor(255, 69, 58),
            "accent_orange": QColor(255, 159, 10),
            "border": QColor(50, 50, 55),
        }

        # 设置窗口样式
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {self.colors['bg_primary'].name()};
                border-radius: 20px;
            }}
        """
        )

        # 创建UI
        self._create_ui()

        # 加载保存的服务器地址（如果有）
        self._load_saved_config()

    def _create_ui(self):
        """创建登录界面"""
        # 使用水平布局：左侧图片，右侧登录表单
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ========== 左侧：图片区域 ==========
        left_image_widget = QWidget()
        left_image_widget.setStyleSheet(
            f"background-color: {self.colors['bg_primary'].name()};"
        )
        left_image_layout = QVBoxLayout(left_image_widget)
        left_image_layout.setContentsMargins(0, 0, 0, 0)

        # 加载并显示图片
        self.image_label = QLabel()
        # 苹果风格：使用 AspectRatioMode.KeepAspectRatioByExpanding 并配合容器裁剪
        self.image_label.setScaledContents(False)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 尝试加载图片
        self._load_login_image()

        left_image_layout.addWidget(self.image_label)
        main_layout.addWidget(left_image_widget, stretch=1)

        # ========== 右侧：登录表单区域 ==========
        right_form_widget = QWidget()
        right_form_widget.setStyleSheet(
            f"background-color: {self.colors['bg_card'].name()};"
        )
        right_form_layout = QVBoxLayout(right_form_widget)
        right_form_layout.setContentsMargins(80, 40, 80, 40)
        right_form_layout.setSpacing(0) # 由具体的 spacing 和 stretch 控制

        # 顶部弹簧，让内容居中
        right_form_layout.addStretch(1)

        # 标题
        title_label = QLabel("SRIC 数据平台")
        title_label.setStyleSheet(
            f"""
            QLabel {{
                color: {self.colors['text_primary'].name()};
                font-size: 32px;
                font-weight: 700;
                letter-spacing: -0.5px;
            }}
        """
        )
        right_form_layout.addWidget(title_label)
        right_form_layout.addSpacing(8)

        # 副标题
        subtitle_label = QLabel("具身智能机器人综合创新平台")
        subtitle_label.setStyleSheet(
            f"color: {self.colors['text_secondary'].name()}; font-size: 16px;"
        )
        right_form_layout.addWidget(subtitle_label)
        
        right_form_layout.addSpacing(40) # 标题与输入框的间距

        # 输入框通用样式
        input_style = f"""
            QLineEdit {{
                background-color: {self.colors['bg_secondary'].name()};
                color: #FFFFFF;
                border: 1px solid {self.colors['border'].name()};
                border-radius: 12px;
                padding: 0px 16px;
                font-size: 15px;
                selection-background-color: {self.colors['accent_blue'].name()};
                selection-color: #FFFFFF;
            }}
            QLineEdit:focus {{
                border: 2px solid {self.colors['accent_blue'].name()};
                background-color: {self.colors['bg_primary'].name()};
                color: #FFFFFF;
            }}
        """

        # 服务器地址
        server_label = QLabel("服务器地址")
        server_label.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-size: 14px; font-weight: 500;")
        right_form_layout.addWidget(server_label)
        right_form_layout.addSpacing(6)
        self.server_input = QLineEdit()
        self.server_input.setPlaceholderText("http://10.204.5.111:9006")
        self.server_input.setStyleSheet(input_style)
        self.server_input.setMinimumHeight(50) # 物理锁定高度
        right_form_layout.addWidget(self.server_input)
        
        right_form_layout.addSpacing(20) # 组间距

        # 账号
        user_label = QLabel("账号")
        user_label.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-size: 14px; font-weight: 500;")
        right_form_layout.addWidget(user_label)
        right_form_layout.addSpacing(6)
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("请输入账号")
        self.user_input.setStyleSheet(input_style)
        self.user_input.setMinimumHeight(50) # 物理锁定高度
        right_form_layout.addWidget(self.user_input)

        right_form_layout.addSpacing(20) # 组间距

        # 密码
        password_label = QLabel("密码")
        password_label.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-size: 14px; font-weight: 500;")
        right_form_layout.addWidget(password_label)
        right_form_layout.addSpacing(6)
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setStyleSheet(input_style)
        self.password_input.setMinimumHeight(50) # 物理锁定高度
        right_form_layout.addWidget(self.password_input)

        right_form_layout.addSpacing(45) # 显著增加密码框与按钮的间距

        # 登录按钮
        self.login_btn = QPushButton("登录")
        self.login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.login_btn.setFixedHeight(55) # 物理锁定按钮高度
        self.login_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['accent_blue'].name()};
                color: white;
                border: none;
                border-radius: 14px;
                padding: 15px;
                font-size: 17px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.colors['accent_blue'].lighter(110).name()};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['accent_blue'].darker(110).name()};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_tertiary'].name()};
            }}
        """
        )
        self.login_btn.clicked.connect(self._do_login)
        right_form_layout.addWidget(self.login_btn)

        # 状态标签
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {self.colors['accent_red'].name()}; font-size: 13px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_form_layout.addSpacing(10)
        right_form_layout.addWidget(self.status_label)

        # 底部弹簧，与顶部弹簧配合实现垂直居中
        right_form_layout.addStretch(1)

        main_layout.addWidget(right_form_widget, stretch=1)

        # 回车键登录
        self.password_input.returnPressed.connect(self._do_login)
        self.user_input.returnPressed.connect(lambda: self.password_input.setFocus())
        self.server_input.returnPressed.connect(lambda: self.user_input.setFocus())

    def _load_login_image(self):
        """加载登录界面左侧图片，确保完整显示原图并居中"""
        import os
        from pathlib import Path

        possible_paths = [
            Path(__file__).parent.parent / "1280X1280.PNG",
            Path(__file__).parent / "1280X1280.PNG",
            Path(__file__).parent.parent.parent / "1280X1280.PNG",
            Path.home() / "1280X1280.PNG",
        ]

        image_path = None
        for path in possible_paths:
            if path.exists() and path.is_file():
                image_path = path
                break

        if image_path:
            try:
                pixmap = QPixmap(str(image_path))
                if pixmap.isNull():
                    logger.warning("无法加载登录图片：%s", image_path)
                    self._set_default_image()
                    return
                
                # 登录窗口 1350x850，左侧区域 675x850 (约一半)
                target_width, target_height = 675, 850
                # 留出一点边距（Padding），让图片看起来更透气，不那么局促
                padding = 50
                content_width = target_width - (padding * 2)
                content_height = target_height - (padding * 2)
                
                # 使用 KeepAspectRatio 确保图片完整显示（Contain 模式）
                scaled_pixmap = pixmap.scaled(
                    content_width,
                    content_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                
                self.image_label.setPixmap(scaled_pixmap)
                # 设置左侧背景色和圆角，保持一致性
                self.image_label.setStyleSheet(
                    f"""
                    QLabel {{ 
                        background-color: {self.colors['bg_primary'].name()}; 
                        border-top-left-radius: 20px; 
                        border-bottom-left-radius: 20px;
                        padding: {padding}px;
                    }}
                    """
                )
                logger.info(
                    "登录图片完整加载：%s (orig %dx%d -> %dx%d)",
                    image_path,
                    pixmap.width(),
                    pixmap.height(),
                    scaled_pixmap.width(),
                    scaled_pixmap.height(),
                )
            except Exception as exc:
                logger.warning("加载登录图片失败：%s", exc)
                self._set_default_image()
        else:
            logger.warning("未找到登录图片 1280X1280.PNG，使用默认背景")
            self._set_default_image()

    def _set_default_image(self):
        """设置默认背景（如果没有找到图片）"""
        self.image_label.setText("")
        self.image_label.setStyleSheet(
            f"""
            QLabel {{
                background-color: {self.colors['bg_primary'].name()};
                background-image: none;
            }}
        """
        )

    def _load_saved_config(self):
        """加载保存的配置"""
        import json
        from pathlib import Path

        config_file = Path.home() / ".lerobot_collector_config.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    if "server_host" in config:
                        self.server_input.setText(config["server_host"])
            except Exception as e:
                logger.warning("加载登录配置失败：%s", e)

    def _save_config(self):
        """保存配置"""
        import json
        from pathlib import Path

        config_file = Path.home() / ".lerobot_collector_config.json"
        try:
            config = {"server_host": self.server_host}
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.warning("保存登录配置失败：%s", e)

    def _do_login(self):
        """执行登录"""
        import requests
        import json

        global USER_TOKEN, SERVER_HOST, USER_INFO

        server_host = self.server_input.text().strip()
        user_identity = self.user_input.text().strip()
        password = self.password_input.text().strip()

        # 规范化服务器地址
        if server_host:
            if not server_host.startswith(("http://", "https://")):
                server_host = "http://" + server_host
                self.server_input.setText(server_host)
            server_host = server_host.rstrip("/")

            try:
                from urllib.parse import urlparse

                parsed = urlparse(server_host)
                if parsed.hostname and ":" not in parsed.netloc and not parsed.port:
                    import re

                    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
                    if re.match(ip_pattern, parsed.hostname):
                        reply = QMessageBox.question(
                            self,
                            "缺少端口号",
                            f"服务器地址 {server_host} 没有指定端口号。\n\n"
                            f"默认将使用端口 80。\n\n"
                            f"如果服务器运行在其他端口（如 8888），请修改地址为：\n"
                            f"http://{parsed.hostname}:8888\n\n"
                            f"是否继续使用默认端口 80？",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No,
                        )
                        if reply == QMessageBox.StandardButton.No:
                            self.login_btn.setEnabled(True)
                            self.login_btn.setText("登录")
                            return
            except Exception as e:
                logger.debug("解析服务器地址失败：%s", e)

        if not user_identity:
            QMessageBox.warning(self, "输入错误", "请输入账号")
            return
        if not password:
            QMessageBox.warning(self, "输入错误", "请输入密码")
            return

        # 本地管理员登录（无需连接服务器）
        if is_local_admin(user_identity, password):
            self.token = LOCAL_ADMIN_TOKEN
            self.user_info = local_admin_user_info()
            self.server_host = server_host if server_host else "local"
            if server_host:
                self._save_config()
            USER_TOKEN = self.token
            SERVER_HOST = self.server_host
            USER_INFO = self.user_info
            logger.info("本地管理员登录成功：%s", LOCAL_ADMIN_USERNAME)
            self.status_label.setText("管理员登录成功！")
            self.status_label.setStyleSheet(
                f"color: {self.colors['accent_green'].name()}; font-size:20px;"
            )
            QTimer.singleShot(500, self.accept)
            return

        if not server_host:
            QMessageBox.warning(self, "输入错误", "请输入服务器地址")
            return

        self.login_btn.setEnabled(False)
        self.login_btn.setText("登录中...")
        self.status_label.setText("正在连接服务器...")
        QApplication.processEvents()

        loading_dialog = LoadingDialog(self, "正在连接服务器...")
        loading_dialog.show()
        QApplication.processEvents()
        QApplication.processEvents()

        try:
            url = f"{server_host.rstrip('/')}/api/v1/user/login"
            logger.info("请求登录接口：%s", url)

            login_data = {
                "user_identity": user_identity,
                "password": password,
            }
            headers = {"Content-Type": "application/json"}

            loading_dialog.set_message("正在验证登录信息...")
            QApplication.processEvents()

            response = requests.post(url, json=login_data, headers=headers, timeout=10)
            result = response.json()

            # 记录一条调试日志，方便排查 code / data 结构（注意只在 DEBUG 级别下完整输出）
            logger.debug("登录响应 JSON: %s", result)

            loading_dialog.close()

            if result.get("code") == 200:
                self.token = result["data"]["token"]
                self.user_info = result["data"]["user_info"]
                self.server_host = server_host
                self._save_config()
                USER_TOKEN = self.token
                SERVER_HOST = self.server_host
                USER_INFO = self.user_info
                logger.info(
                    "登录成功，已设置全局 USER_TOKEN（前10位）=%s..., SERVER_HOST=%s, tenant_id=%s",
                    str(USER_TOKEN)[:10],
                    SERVER_HOST,
                    USER_INFO.get("tenant_id") if USER_INFO else None,
                )
                self.status_label.setText("✅ 登录成功！")
                self.status_label.setStyleSheet(
                    f"color: {self.colors['accent_green'].name()}; font-size: 14px;"
                )
                QTimer.singleShot(500, self.accept)
            else:
                error_msg = result.get("message", "登录失败")
                self.status_label.setText(f"❌ {error_msg}")
                self.status_label.setStyleSheet(
                    f"color: {self.colors['accent_red'].name()}; font-size: 14px;"
                )
                QMessageBox.warning(self, "登录失败", error_msg)
                self.login_btn.setEnabled(True)
                self.login_btn.setText("登录")
        except requests.exceptions.Timeout:
            loading_dialog.close()
            error_msg = (
                f"连接超时（10秒）\n\n请检查：\n1. 服务器地址是否正确：{server_host}\n2. 服务器是否正在运行\n"
                f"3. 网络连接是否正常\n4. 防火墙是否阻止了连接"
            )
            self.status_label.setText("❌ 连接超时")
            self.status_label.setStyleSheet(
                f"color: {self.colors['accent_red'].name()}; font-size: 14px;"
            )
            QMessageBox.critical(self, "连接超时", error_msg)
            self.login_btn.setEnabled(True)
            self.login_btn.setText("登录")
        except requests.exceptions.ConnectionError as e:
            loading_dialog.close()
            error_detail = str(e)
            if "Name or service not known" in error_detail or "nodename nor servname provided" in error_detail:
                error_msg = (
                    f"无法解析服务器地址\n\n请检查：\n1. 服务器地址是否正确：{server_host}\n2. 域名或IP地址是否存在"
                )
            elif "Connection refused" in error_detail or "Errno 111" in error_detail:
                from urllib.parse import urlparse

                parsed = urlparse(server_host)
                if not parsed.port:
                    port_info = (
                        f"\n\n⚠️ 注意：您的地址中没有指定端口号，系统使用了默认端口 80。\n如果服务器运行在其他端口（如 8888），"
                        f"请修改地址为：\nhttp://{parsed.hostname}:8888"
                    )
                else:
                    port_info = f"\n\n当前使用的端口：{parsed.port}"
                error_msg = (
                    f"连接被拒绝\n\n服务器地址：{server_host}{port_info}\n\n请检查：\n1. 服务器是否正在运行\n2. 端口号是否正确"
                    f"（常见端口：80, 8080, 8888, 3000等）\n3. 防火墙是否阻止了连接\n4. 服务器是否监听在该IP地址上"
                )
            else:
                error_msg = (
                    f"无法连接到服务器\n\n服务器地址：{server_host}\n\n请检查：\n1. 服务器是否正在运行\n2. 网络连接是否正常\n"
                    f"3. 防火墙是否阻止了连接\n4. 服务器地址和端口是否正确\n\n错误详情：{error_detail}"
                )
            self.status_label.setText("❌ 连接失败")
            self.status_label.setStyleSheet(
                f"color: {self.colors['accent_red'].name()}; font-size: 14px;"
            )
            QMessageBox.critical(self, "连接错误", error_msg)
            logger.error("连接服务器失败：%s", error_detail)
            self.login_btn.setEnabled(True)
            self.login_btn.setText("登录")
        except requests.exceptions.InvalidURL as e:
            loading_dialog.close()
            error_msg = (
                f"无效的URL地址\n\n服务器地址：{server_host}\n\n请检查地址格式是否正确，例如：\n"
                f"- http://10.204.5.111:8888\n- https://example.com"
            )
            self.status_label.setText("❌ 无效的URL")
            self.status_label.setStyleSheet(
                f"color: {self.colors['accent_red'].name()}; font-size: 14px;"
            )
            QMessageBox.critical(self, "URL错误", error_msg)
            logger.error("登录地址无效：%s", e)
            self.login_btn.setEnabled(True)
            self.login_btn.setText("登录")
        except Exception as e:
            loading_dialog.close()
            error_msg = (
                f"请求出错: {str(e)}\n\n服务器地址：{server_host}\n\n请检查服务器地址和网络连接。"
            )
            self.status_label.setText("❌ 请求出错")
            self.status_label.setStyleSheet(
                f"color: {self.colors['accent_red'].name()}; font-size: 14px;"
            )
            QMessageBox.critical(self, "错误", error_msg)
            logger.exception("登录请求异常：%s", e)
            self.login_btn.setEnabled(True)
            self.login_btn.setText("登录")


