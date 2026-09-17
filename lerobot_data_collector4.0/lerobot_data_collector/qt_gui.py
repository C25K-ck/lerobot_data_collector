# --- Qt GUI (PySide6) - 更好的字体渲染引擎 ---
import logging
import os
import requests
from pathlib import Path
from datetime import datetime
from functools import partial
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
from .qt_collect import QtApp
from .qt_task_manager import TaskCreationDialog

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QFileDialog,
        QMessageBox,
        QFrame,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QComboBox,
        QListWidget,
        QListWidgetItem,
        QDialog,
        QSpinBox,
        QTextEdit,
        QDateTimeEdit,
        QProgressBar,
    )
    from PySide6.QtCore import Qt, QTimer, QRect, QDateTime
    from PySide6.QtGui import (
        QFont,
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
            QLabel,
            QLineEdit,
            QPushButton,
            QFileDialog,
            QMessageBox,
            QFrame,
            QTabWidget,
            QTableWidget,
            QTableWidgetItem,
            QHeaderView,
            QComboBox,
            QListWidget,
            QListWidgetItem,
            QDialog,
            QSpinBox,
            QTextEdit,
            QDateTimeEdit,
            QProgressBar,
        )
        from PyQt6.QtCore import Qt, QTimer, QRect, QDateTime
        from PyQt6.QtGui import (
            QFont,
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

from .qt_login import get_auth_headers, get_server_host, LoadingDialog, LoginDialog
from .qt_task_manager import TaskManagerWindow

try:
    from .warmup_display_realsense_right import RealSenseRightCameraManager
except Exception:
    RealSenseRightCameraManager = None  # 摄像头为可选组件


STATUS_LABELS = {
    0: "待发布",
    1: "待领取",
    2: "待采集",
    3: "采集中",
    4: "任务超时",
    5: "采集完成",
    6: "采集完成-超时",
}

STATUS_COLORS = {
    0: "#8E8E93",  # dark grey
    1: "#5AC8FA",  # light blue
    2: "#FF9500",  # orange
    3: "#0A84FF",  # blue
    4: "#FF3B30",  # red
    5: "#30D158",  # green
    6: "#FF2D55",  # pink
}


def create_status_badge(status: int) -> QLabel:
    text = STATUS_LABELS.get(status, f"状态{status}")
    color_hex = STATUS_COLORS.get(status, "#5E5CE6")
    badge = QLabel(text)
    # 确保文本居中对齐
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setFixedHeight(30)
    # 设置字体，确保中文正确显示
    font = QFont()
    font.setPixelSize(12)
    badge.setFont(font)
    badge.setStyleSheet(f"""
        QLabel {{
            background-color: {color_hex}33;
            color: {color_hex};
            border: 1px solid {color_hex};
            border-radius: 15px;
            padding: 4px 16px;
            font-size: 12px;
            font-weight: 500;
        }}
    """)
    return badge


@dataclass
class TaskAction:
    text: str
    background: str
    hover: str
    callback: Callable[[], None]
    min_width: int = 80
    height: int = 32
    font_size: int = 12


def create_action_button(
    text: str,
    background: str,
    hover: str,
    callback,
    min_width: int = 80,
    height: int = 32,
    font_size: int = 12,
) -> QPushButton:
    """Qt 通用按钮工厂，Task 管理和采集界面复用。"""
    btn = QPushButton(text)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(height)
    btn.setMinimumWidth(min_width)
    font = QFont()
    font.setPixelSize(font_size)
    btn.setFont(font)
    btn.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {background};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 6px 14px;
            font-size: {font_size}px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {hover};
        }}
        QPushButton:pressed {{
            background-color: {hover};
            opacity: 0.85;
        }}
    """
    )
    btn.clicked.connect(callback)
    return btn


class LoadingDialog(QDialog):
    """加载对话框 - 显示加载状态"""
    def __init__(self, parent=None, message="正在加载..."):
        super().__init__(parent)
        self.setWindowTitle("")
        self.setModal(True)
        # 使用窗口标志确保对话框显示在最前面
        self.setWindowFlags(
            Qt.WindowType.Dialog | 
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        
        # 深色主题配色
        self.colors = {
            'bg_primary': QColor(30, 30, 35),
            'bg_card': QColor(40, 40, 45),
            'text_primary': QColor(255, 255, 255),
            'accent_blue': QColor(0, 122, 255),
        }
        
        # 设置窗口样式
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {self.colors['bg_card'].name()};
                border-radius: 12px;
                border: 1px solid {self.colors['bg_primary'].name()};
            }}
        """)
        
        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        
        # 加载指示器（进度条，不确定模式）
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不确定模式
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 3px;
                background-color: {self.colors['bg_primary'].name()};
            }}
            QProgressBar::chunk {{
                background-color: {self.colors['accent_blue'].name()};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)
        
        # 消息标签
        self.message_label = QLabel(message)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setStyleSheet(f"""
            QLabel {{
                color: {self.colors['text_primary'].name()};
                font-size: 14px;
                padding: 10px;
            }}
        """)
        layout.addWidget(self.message_label)
        
        # 设置固定大小
        self.setFixedSize(300, 120)
        
        # 居中显示
        self._center_on_parent(parent)
    
    def _center_on_parent(self, parent):
        """在父窗口中心显示"""
        if parent:
            parent_rect = parent.geometry()
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2
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
                        screen_rect.y() + (screen_rect.height() - self.height()) // 2
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


    def _setup_ui(self):
        self.setWindowTitle("任务管理 - SRIC数据采集平台")
        self.resize(1280, 840)

        central = QWidget()
        central.setStyleSheet(f"background-color: {self.colors['bg_primary'].name()};")
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        # Header
        title = QLabel("采集任务管理")
        title_font = QFont("Microsoft YaHei", 24, QFont.Weight.Bold)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {self.colors['text_primary'].name()};")
        layout.addWidget(title)

        subtitle = QLabel("在开始数据采集前，请先领取并确认您的任务。")
        subtitle.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-size: 13px;")
        layout.addWidget(subtitle)

        layout.addSpacing(12)

        # Filters
        filter_widget = QWidget()
        filter_widget.setStyleSheet(f"background-color: {self.colors['bg_card'].name()}; border-radius: 10px;")
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(16, 16, 16, 16)
        filter_layout.setSpacing(16)

        self.task_name_input = QLineEdit()
        self.task_name_input.setPlaceholderText("任务名称")
        self.task_name_input.setStyleSheet(self._input_style())
        self.task_name_input.setMinimumHeight(44)
        self.task_name_input.setFont(QFont(self.font_family, 14))
        filter_layout.addWidget(self.task_name_input, stretch=1)

        self.status_combo = QComboBox()
        self.status_combo.setStyleSheet(self._combobox_style())
        self.status_combo.addItem("全部状态", None)
        for key in STATUS_LABELS:
            self.status_combo.addItem(STATUS_LABELS[key], key)
        self.status_combo.setMinimumHeight(44)
        self.status_combo.setFont(QFont(self.font_family, 14))
        filter_layout.addWidget(self.status_combo)

        search_btn = create_action_button("查询", "#0A84FF", "#0A84FF", self.apply_filters)
        refresh_btn = create_action_button("刷新", "#5E5CE6", "#5E5CE6", self.fetch_tasks)
        reset_btn = create_action_button("重置", "#8E8E93", "#8E8E93", self.reset_filters)
        create_btn = create_action_button("创建任务", "#30D158", "#30D158", self.create_task)
        for btn in (search_btn, refresh_btn, reset_btn, create_btn):
            filter_layout.addWidget(btn)

        filter_layout.addStretch()
        layout.addWidget(filter_widget)

        # Table
        self.task_table = QTableWidget()
        self.task_table.setColumnCount(9)
        self.task_table.setHorizontalHeaderLabels([
            "任务ID", "任务名称", "场景-子场景", "所属项目", "采集数量",
            "采集员", "采集状态", "开始时间", "操作"
        ])
        self.task_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.task_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        # 为操作列设置固定宽度，确保按钮有足够空间
        self.task_table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeMode.Fixed)
        self.task_table.setColumnWidth(8, 320)  # 操作列（索引8）设置固定宽度320px
        self.task_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.task_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.task_table.setStyleSheet(self._table_style())
        self.task_table.setFont(QFont(self.font_family, 14))
        header_font = QFont(self.font_family, 14, QFont.Weight.Medium)
        self.task_table.horizontalHeader().setFont(header_font)
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.verticalHeader().setDefaultSectionSize(52)
        self.task_table.setAlternatingRowColors(True)
        layout.addWidget(self.task_table, stretch=1)

        # Pagination
        pagination_widget = QWidget()
        pagination_widget.setStyleSheet("background-color: transparent;")
        pagination_layout = QHBoxLayout(pagination_widget)
        pagination_layout.setContentsMargins(0, 0, 0, 0)

        self.prev_btn = create_action_button("上一页", "#2C2C2E", "#3A3A3C", self.prev_page)
        self.next_btn = create_action_button("下一页", "#2C2C2E", "#3A3A3C", self.next_page)
        pagination_layout.addWidget(self.prev_btn)

        self.page_info_label = QLabel("第 1 / 1 页")
        self.page_info_label.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-size: 13px;")
        pagination_layout.addWidget(self.page_info_label)

        pagination_layout.addSpacing(16)

        self.page_size_combo = QComboBox()
        self.page_size_combo.setStyleSheet(self._combobox_style())
        for size in [10, 20, 50]:
            self.page_size_combo.addItem(f"{size} 条/页", size)
        self.page_size_combo.setCurrentIndex(0)
        self.page_size_combo.currentIndexChanged.connect(self.on_page_size_changed)
        pagination_layout.addWidget(self.page_size_combo)

        pagination_layout.addStretch()
        pagination_layout.addWidget(self.next_btn)

        layout.addWidget(pagination_widget)

    def _input_style(self) -> str:
        return f"""
            QLineEdit {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_primary'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border: 1px solid {self.colors['accent_blue'].name()};
            }}
        """

    def _combobox_style(self) -> str:
        return f"""
            QComboBox {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_primary'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 8px;
                padding: 8px 14px;
                font-size: 14px;
            }}
            QComboBox::drop-down {{
                width: 24px;
                border: none;
            }}
        """

    def _table_style(self) -> str:
        return f"""
            QTableWidget {{
                background-color: {self.colors['bg_card'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 10px;
                gridline-color: {self.colors['border'].name()};
                color: {self.colors['text_primary'].name()};
                font-size: 14px;
                selection-background-color: {self.colors['accent_blue'].name()};
                alternate-background-color: {self.colors['bg_secondary'].name()};
            }}
            QHeaderView::section {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_primary'].name()};
                padding: 12px 8px;
                border: none;
                font-size: 14px;
            }}
        """

    # ---------- Data Fetching ----------
    def apply_filters(self):
        self.page = 1
        self.fetch_tasks()

    def reset_filters(self):
        self.task_name_input.clear()
        self.status_combo.setCurrentIndex(0)
        self.page = 1
        self.fetch_tasks()

    def on_page_size_changed(self):
        self.page_size = self.page_size_combo.currentData()
        self.page = 1
        self.fetch_tasks()

    def prev_page(self):
        if self.page > 1:
            self.page -= 1
            self.fetch_tasks()

    def next_page(self):
        max_page = max(1, (self.total + self.page_size - 1) // self.page_size)
        if self.page < max_page:
            self.page += 1
            self.fetch_tasks()

    def fetch_tasks(self):
        server_host = get_server_host()
        headers = get_auth_headers()
        if not headers.get("x-auth-tkn"):
            # 超级用户或未登录，只显示本地任务，不需要加载对话框
            self.populate_table()
            return

        params = {
            "page": self.page,
            "page_size": self.page_size,
        }
        task_name = self.task_name_input.text().strip()
        status = self.status_combo.currentData()
        if task_name:
            params["task_name"] = task_name
        if status is not None:
            params["collection_status"] = status

        # 显示加载对话框
        loading_dialog = LoadingDialog(self, "正在获取任务列表...")
        loading_dialog.show()
        # 确保对话框显示并处理事件
        QApplication.processEvents()
        QApplication.processEvents()  # 再次处理以确保显示

        try:
            resp = requests.get(
                f"{server_host.rstrip('/')}/api/v1/task/my-tasks",
                params=params,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 200:
                raise ValueError(data.get("message", "接口返回异常"))

            payload = data.get("data", {})
            self.server_tasks = payload.get("items", [])
            self.total = payload.get("total", 0)
            self.page = payload.get("page", self.page)
            self.page_size = payload.get("page_size", self.page_size)

            # 关闭加载对话框
            loading_dialog.close()
            self.populate_table()
        except Exception as e:
            # 关闭加载对话框
            loading_dialog.close()
            QMessageBox.warning(self, "获取任务失败", f"无法获取任务列表：{e}\n\n将显示本地任务。")
            logger.warning("获取任务列表失败：%s", e)
            self.server_tasks = []
            self.populate_table()

    # ---------- Table Rendering ----------
    def populate_table(self):
        tasks = self.get_all_tasks()
        self.task_table.setRowCount(0)
        self.task_table.setRowCount(len(tasks))

        for row, task in enumerate(tasks):
            self._set_table_item(row, 0, self._format_task_id(task))
            self._set_table_item(row, 1, task.get("task_name", ""))

            scene = task.get("scene_name") or ""
            sub_scene = task.get("sub_scene_name") or ""
            scene_text = f"{scene}-{sub_scene}" if sub_scene else scene
            self._set_table_item(row, 2, scene_text or "-")

            self._set_table_item(row, 3, task.get("project", "-"))
            self._set_table_item(row, 4, str(task.get("collection_count", "-")))
            self._set_table_item(row, 5, task.get("collector_name") or "-")

            badge = create_status_badge(task.get("collection_status", -1))
            # 创建一个容器widget来确保徽章在单元格中居中
            badge_container = QWidget()
            row_bg = self.colors['bg_secondary'] if (row % 2 == 1) else self.colors['bg_card']
            badge_container.setStyleSheet(f"background-color: {row_bg.name()};")
            badge_layout = QHBoxLayout(badge_container)
            badge_layout.setContentsMargins(0, 0, 0, 0)
            badge_layout.addStretch()
            badge_layout.addWidget(badge)
            badge_layout.addStretch()
            self.task_table.setCellWidget(row, 6, badge_container)

            start_time = task.get("task_start_time")
            start_text = self.format_timestamp(start_time)
            self._set_table_item(row, 7, start_text)

            action_widget = self._build_action_widget(task)
            self.task_table.setCellWidget(row, 8, action_widget)
            self.task_table.setRowHeight(row, 52)

        max_page = max(1, (self.total + self.page_size - 1) // self.page_size)
        total_display = self.total + len(self.local_tasks)
        self.page_info_label.setText(f"第 {self.page} / {max_page} 页，共 {total_display} 个任务（含本地 {len(self.local_tasks)} 个）")

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        combined: List[Dict[str, Any]] = []
        combined.extend(self.local_tasks)  # 保持引用以便更新
        combined.extend(self.server_tasks)
        return combined

    def _format_task_id(self, task: Dict[str, Any]) -> str:
        task_id = task.get("id", "")
        if task.get("is_local"):
            try:
                numeric_id = abs(int(task_id))
            except (ValueError, TypeError):
                numeric_id = 0
            return f"本地-{numeric_id or '新建'}"
        return str(task_id)

    def _set_table_item(self, row: int, column: int, text: str):
        item = QTableWidgetItem(text)
        item.setForeground(self.colors['text_primary'])
        self.task_table.setItem(row, column, item)

    def format_timestamp(self, timestamp) -> str:
        if not timestamp:
            return "-"
        try:
            return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "-"

    def _task_actions(self, task: Dict[str, Any]) -> List[TaskAction]:
        status = task.get("collection_status")
        is_local = task.get("is_local", False)

        actions: List[TaskAction] = [
            TaskAction(
                text="详情",
                background="#5E5CE6",
                hover="#5E5CE6",
                callback=lambda _checked=False, _t=task: self.show_task_detail(_t),
                min_width=60,
                height=28,
                font_size=11,
            )
        ]

        if is_local or status in (0, 1):
            actions.append(
                TaskAction(
                    text="领取",
                    background="#0A84FF",
                    hover="#0A84FF",
                    callback=lambda _checked=False, _t=task: self.claim_task(_t),
                    min_width=60,
                    height=28,
                    font_size=11,
                )
            )

        if is_local or status == 2:
            actions.append(
                TaskAction(
                    text="开始采集",
                    background="#30D158",
                    hover="#30D158",
                    callback=lambda _checked=False, _t=task: self.start_collection(_t),
                    min_width=82,
                    height=28,
                    font_size=11,
                )
            )

        if status in (3, 4, 5, 6):
            actions.append(
                TaskAction(
                    text="继续采集",
                    background="#FF9500",
                    hover="#FF9500",
                    callback=lambda _checked=False, _t=task: self.continue_collection(_t),
                    min_width=82,
                    height=28,
                    font_size=11,
                )
            )
        return actions

    def _build_action_widget(self, task: Dict[str, Any]) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        actions = self._task_actions(task)
        if not actions:
            placeholder = QLabel("暂无操作")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(f"color: {self.colors['text_secondary'].name()}; font-size: 12px;")
            layout.addWidget(placeholder)
            return container

        for action in actions:
            btn = create_action_button(
                action.text,
                action.background,
                action.hover,
                action.callback,
                min_width=action.min_width,
                height=action.height,
                font_size=action.font_size,
            )
            layout.addWidget(btn)

        layout.addStretch()
        return container

    # ---------- Actions ----------
    def show_task_detail(self, task: Dict[str, Any]):
        description = task.get("task_description") or "无任务描述"
        detail_text = (
            f"任务名称：{task.get('task_name', '-')}\n"
            f"所属项目：{task.get('project', '-')}\n"
            f"场景：{task.get('scene_name', '-')} - {task.get('sub_scene_name', '-')}\n"
            f"采集数量：{task.get('collection_count', '-')}\n"
            f"审核通过：{task.get('approved_count', '-')}\n"
            f"采集员：{task.get('collector_name', '-')}\n"
            f"任务开始：{self.format_timestamp(task.get('task_start_time'))}\n"
            f"任务结束：{self.format_timestamp(task.get('task_end_time'))}\n"
            f"连续动作：{task.get('continuous_action', '-')}\n\n"
            f"任务描述：\n{description}"
        )
        QMessageBox.information(self, "任务详情", detail_text)

    def claim_task(self, task: Dict[str, Any]):
        if task.get("is_local"):
            task['collection_status'] = 2
            self.update_local_task(task)
            self.populate_table()
            QMessageBox.information(self, "领取成功", f"本地任务 [{task.get('task_name', '')}] 已标记为“待采集”。")
            return

        if self.operate_task(task['id'], op_type=3, collection_status=2):
            task['collection_status'] = 2
            self.populate_table()
            QMessageBox.information(self, "领取成功", f"任务 [{task.get('task_name', '')}] 已领取。")

    def start_collection(self, task: Dict[str, Any]):
        current_status = task.get("collection_status")
        if current_status in (0, 1):
            QMessageBox.warning(self, "无法开始", "请先领取任务后再开始采集。")
            return

        if current_status == 2:
            if task.get("is_local"):
                task['collection_status'] = 3
                self.update_local_task(task)
            else:
                if not self.operate_task(task['id'], op_type=3, collection_status=3):
                    return
            task['collection_status'] = 3
            self.populate_table()

        # 打开采集窗口
        self.hide()
        self.collector_window = QtApp(self.collector_class, self.collector_config_class, task=task)
        self.collector_window.setWindowTitle("SRIC数据采集平台")
        # 保存任务管理窗口的引用，以便返回
        self.collector_window.task_manager = self
        self.collector_window.destroyed.connect(self.on_collector_closed)
        self.collector_window.show()

    def continue_collection(self, task: Dict[str, Any]):
        """继续采集（断点续采），在已有数据的基础上继续"""
        status = task.get("collection_status")

        # 若任务尚未领取，提示先领取
        if status in (0, 1):
            QMessageBox.warning(self, "无法继续", "该任务尚未领取，请先领取任务后再继续采集。")
            return

        # 本地任务：直接更新状态为采集中
        if task.get("is_local"):
            if status != 3:
                task['collection_status'] = 3
                self.update_local_task(task)
                self.populate_table()
        else:
            # 服务器任务：必要时更新状态为采集中
            if status in (4, 5, 6):
                if not self.operate_task(task['id'], op_type=3, collection_status=3):
                    return
                task['collection_status'] = 3
                self.populate_table()

        # 复用开始采集流程（会自动打开采集窗口并支持断点续采）
        self.start_collection(task)

    def on_collector_closed(self):
        """当采集窗口被关闭时调用（通过 destroyed 信号）"""
        # 只有在窗口真正被销毁时才执行
        if self.collector_window:
            self.collector_window = None
            # 如果当前窗口是隐藏的，说明是通过返回按钮返回的，不需要再显示
            if not self.isVisible():
                self.show()
                self.raise_()
                self.activateWindow()
                self.fetch_tasks()

    def operate_task(self, task_id: int, op_type: int, collection_status: Optional[int] = None,
                     approved_count: Optional[int] = None) -> bool:
        local_task = next((t for t in self.local_tasks if t['id'] == task_id), None)
        if local_task is not None:
            if collection_status is not None:
                local_task['collection_status'] = collection_status
            if approved_count is not None:
                local_task['approved_count'] = approved_count
            self.update_local_task(local_task)
            return True

        server_host = get_server_host()
        headers = get_auth_headers()
        payload: Dict[str, Any] = {"type": op_type}
        if op_type == 3 and collection_status is not None:
            payload["collection_status"] = collection_status
        if op_type == 4 and approved_count is not None:
            payload["approved_count"] = approved_count

        try:
            resp = requests.put(
                f"{server_host.rstrip('/')}/api/v1/task/tasks/{task_id}/operate",
                json=payload,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 200:
                raise ValueError(data.get("message", "接口返回异常"))
            return True
        except Exception as e:
            QMessageBox.critical(self, "操作失败", f"操作任务失败：{e}")
            logger.error("任务操作失败：%s", e)
            return False

    def create_task(self):
        dialog = TaskCreationDialog(self.colors, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        data = dialog.get_task_data()
        data['id'] = self.next_local_id
        self.next_local_id -= 1
        data['is_local'] = True
        data['tenant_id'] = 0
        data['collection_status'] = 0
        data['approved_count'] = 0
        data['created_at'] = int(datetime.now().timestamp())
        data['updated_at'] = data['created_at']
        self.local_tasks.append(data)
        self.populate_table()
        QMessageBox.information(self, "创建成功", f"本地任务 [{data['task_name']}] 已创建。")

    def update_local_task(self, task: Dict[str, Any]):
        for idx, t in enumerate(self.local_tasks):
            if t['id'] == task['id']:
                task['updated_at'] = int(datetime.now().timestamp())
                self.local_tasks[idx] = task
                return

def main_qt(collector_class, collector_config_class):
    """Qt GUI入口函数"""
    if not PYSIDE6_AVAILABLE:
        logger.error("PySide6/PyQt6 未安装，无法启动 Qt GUI")
        logger.info("请安装依赖: pip install PySide6")
        return
    
    app = QApplication([])
    
    # 启用高DPI支持（更好的字体渲染）
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    # 显示登录窗口
    login_dialog = LoginDialog()
    if login_dialog.exec() != QDialog.DialogCode.Accepted:
        # 用户取消登录，退出应用
        return
    
    # 登录成功，显示任务管理窗口
    task_window = TaskManagerWindow(collector_class, collector_config_class)
    task_window.show()
    
    app.exec()

