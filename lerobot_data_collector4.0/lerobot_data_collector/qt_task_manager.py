import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import requests

from .qt_login import get_auth_headers, get_server_host, LoadingDialog

logger = logging.getLogger(__name__)

try:
    from PySide6.QtWidgets import (
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
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
    from PySide6.QtCore import Qt, QDateTime
    from PySide6.QtGui import QFont, QColor
except ImportError:
    from PyQt6.QtWidgets import (
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
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
    from PyQt6.QtCore import Qt, QDateTime
    from PyQt6.QtGui import QFont, QColor


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
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setFixedHeight(30)
    font = QFont()
    font.setPixelSize(12)
    badge.setFont(font)
    badge.setStyleSheet(
        f"""
        QLabel {{
            background-color: {color_hex}33;
            color: {color_hex};
            border: 1px solid {color_hex};
            border-radius: 15px;
            padding: 4px 16px;
            font-size: 12px;
            font-weight: 500;
        }}
    """
    )
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


class TaskCreationDialog(QDialog):
    """新建任务对话框（本地创建）"""

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.colors = colors
        self.setModal(True)
        self.setWindowTitle("创建采集任务")
        self.setFixedSize(620, 720)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {self.colors['bg_primary'].name()};
            }}
            QLabel {{
                color: {self.colors['text_primary'].name()};
                font-size: 13px;
            }}
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        layout.addWidget(self._create_group_label("任务名称"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入任务名称")
        self.name_input.setStyleSheet(self._input_style())
        layout.addWidget(self.name_input)

        layout.addWidget(self._create_group_label("所属项目"))
        self.project_input = QLineEdit()
        self.project_input.setPlaceholderText("例如：项目A")
        self.project_input.setStyleSheet(self._input_style())
        layout.addWidget(self.project_input)

        layout.addWidget(self._create_group_label("场景 / 子场景"))
        scene_container = QWidget()
        scene_layout = QHBoxLayout(scene_container)
        scene_layout.setContentsMargins(0, 0, 0, 0)
        scene_layout.setSpacing(8)
        self.scene_input = QLineEdit()
        self.scene_input.setPlaceholderText("场景")
        self.scene_input.setStyleSheet(self._input_style())
        self.sub_scene_input = QLineEdit()
        self.sub_scene_input.setPlaceholderText("子场景")
        self.sub_scene_input.setStyleSheet(self._input_style())
        scene_layout.addWidget(self.scene_input)
        scene_layout.addWidget(self.sub_scene_input)
        layout.addWidget(scene_container)

        layout.addWidget(self._create_group_label("采集数量"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 1000000)
        self.count_spin.setValue(100)
        self.count_spin.setStyleSheet(self._input_style())
        layout.addWidget(self.count_spin)

        layout.addWidget(self._create_group_label("采集本体 / 方案"))
        body_container = QWidget()
        body_layout = QHBoxLayout(body_container)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)
        self.body_input = QLineEdit()
        self.body_input.setPlaceholderText("例如：GEN1")
        self.body_input.setStyleSheet(self._input_style())
        self.plan_input = QLineEdit()
        self.plan_input.setPlaceholderText("例如：PICO遥操作")
        self.plan_input.setStyleSheet(self._input_style())
        body_layout.addWidget(self.body_input)
        body_layout.addWidget(self.plan_input)
        layout.addWidget(body_container)

        layout.addWidget(self._create_group_label("任务用途 / 连续动作"))
        purpose_container = QWidget()
        purpose_layout = QHBoxLayout(purpose_container)
        purpose_layout.setContentsMargins(0, 0, 0, 0)
        purpose_layout.setSpacing(8)
        self.purpose_input = QLineEdit()
        self.purpose_input.setPlaceholderText("任务用途")
        self.purpose_input.setStyleSheet(self._input_style())
        self.action_input = QLineEdit()
        self.action_input.setPlaceholderText("连续动作")
        self.action_input.setStyleSheet(self._input_style())
        purpose_layout.addWidget(self.purpose_input)
        purpose_layout.addWidget(self.action_input)
        layout.addWidget(purpose_container)

        layout.addWidget(self._create_group_label("任务时间"))
        time_container = QWidget()
        time_layout = QHBoxLayout(time_container)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(8)
        self.start_time_edit = QDateTimeEdit(QDateTime.currentDateTime())
        self.start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.start_time_edit.setStyleSheet(self._input_style())
        self.end_time_edit = QDateTimeEdit(QDateTime.currentDateTime().addSecs(3600))
        self.end_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.end_time_edit.setStyleSheet(self._input_style())
        time_layout.addWidget(self.start_time_edit)
        time_layout.addWidget(self.end_time_edit)
        layout.addWidget(time_container)

        layout.addWidget(self._create_group_label("任务描述"))
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("补充任务背景、操作说明等信息")
        self.desc_input.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_primary'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }}
        """
        )
        layout.addWidget(self.desc_input, stretch=1)

        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(12)
        button_layout.addStretch()
        cancel_btn = create_action_button("取消", "#2C2C2E", "#3A3A3C", self.reject)
        ok_btn = create_action_button("创建", "#0A84FF", "#0A84FF", self._accept_if_valid)
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)
        layout.addWidget(button_container)

    def _create_group_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"color: {self.colors['text_secondary'].name()}; font-size: 12px;"
        )
        return label

    def _input_style(self) -> str:
        return (
            f"""
            QLineEdit, QSpinBox, QDateTimeEdit {{
                background-color: {self.colors['bg_secondary'].name()};
                color: {self.colors['text_primary'].name()};
                border: 1px solid {self.colors['border'].name()};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
            }}
            QDateTimeEdit::up-button, QDateTimeEdit::down-button {{
                width: 18px;
                background: transparent;
            }}
        """
        )

    def _accept_if_valid(self):
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "提示", "任务名称不能为空")
            return
        self.accept()

    def get_task_data(self) -> Dict[str, Any]:
        name = self.name_input.text().strip()
        description = self.desc_input.toPlainText().strip()
        project = self.project_input.text().strip()
        scene = self.scene_input.text().strip()
        sub_scene = self.sub_scene_input.text().strip()
        body_text = self.body_input.text().strip()
        plan = self.plan_input.text().strip()
        purpose = self.purpose_input.text().strip()
        continuous_action = self.action_input.text().strip()

        start_dt = self.start_time_edit.dateTime()
        end_dt = self.end_time_edit.dateTime()
        if end_dt <= start_dt:
            end_dt = start_dt.addSecs(3600)

        return {
            "task_name": name or "未命名任务",
            "task_description": description,
            "project": project,
            "scene_name": scene,
            "sub_scene_name": sub_scene,
            "collection_body_text": body_text,
            "collection_plan": plan,
            "task_purpose": purpose,
            "collection_count": self.count_spin.value(),
            "continuous_action": continuous_action,
            "task_start_time": int(start_dt.toSecsSinceEpoch()),
            "task_end_time": int(end_dt.toSecsSinceEpoch()),
            "collection_body_id": 0,
            "scene_id": 0,
            "collector_id": 0,
            "collector_name": "",
        }


class TaskManagerWindow(QMainWindow):
    """任务管理窗口 - 介于登录与数据采集界面之间"""

    def __init__(
        self,
        collector_class,
        collector_config_class,
        collector_window_cls=None,
        parent=None,
    ):
        super().__init__(parent)
        self.collector_class = collector_class
        self.collector_config_class = collector_config_class
        # 采集窗口类可注入；如果未提供则默认使用 QtApp（避免循环导入，延迟导入）
        if collector_window_cls is None:
            from .qt_collect import QtApp  # 延迟导入，防止循环依赖

            collector_window_cls = QtApp
        self.collector_window_cls = collector_window_cls

        self.colors = {
            "bg_primary": QColor(24, 24, 28),
            "bg_card": QColor(36, 36, 40),
            "bg_secondary": QColor(48, 48, 52),
            "text_primary": QColor(255, 255, 255),
            "text_secondary": QColor(180, 180, 186),
            "border": QColor(50, 50, 55),
            "accent_blue": QColor(10, 132, 255),
            "accent_green": QColor(48, 209, 88),
            "accent_red": QColor(255, 69, 58),
        }
        self.font_family = QFont("Microsoft YaHei").family()

        self.page = 1
        self.page_size = 10
        self.total = 0
        self.server_tasks: List[Dict[str, Any]] = []
        self.local_tasks: List[Dict[str, Any]] = []
        self.next_local_id = -1
        self.collector_window: Optional[QMainWindow] = None

        self._setup_ui()
        self.fetch_tasks()

    # ---------- UI ----------
    def _setup_ui(self):
        self.setWindowTitle("任务管理 - SRIC数据采集平台")
        self.resize(1280, 840)

        central = QWidget()
        central.setStyleSheet(
            f"background-color: {self.colors['bg_primary'].name()};"
        )
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        # Header
        title = QLabel("采集任务管理")
        title_font = QFont(self.font_family, 24, QFont.Weight.Bold)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {self.colors['text_primary'].name()};")
        layout.addWidget(title)

        subtitle = QLabel("在开始数据采集前，请先领取并确认您的任务。")
        subtitle.setStyleSheet(
            f"color: {self.colors['text_secondary'].name()}; font-size: 13px;"
        )
        layout.addWidget(subtitle)

        layout.addSpacing(12)

        # Filters
        filter_widget = QWidget()
        filter_widget.setStyleSheet(
            f"background-color: {self.colors['bg_card'].name()}; border-radius: 10px;"
        )
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
        self.task_table.setHorizontalHeaderLabels(
            [
                "任务ID",
                "任务名称",
                "场景-子场景",
                "所属项目",
                "采集数量",
                "采集员",
                "采集状态",
                "开始时间",
                "操作",
            ]
        )
        self.task_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.task_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.task_table.horizontalHeader().setSectionResizeMode(
            8, QHeaderView.ResizeMode.Fixed
        )
        self.task_table.setColumnWidth(8, 320)
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
        self.page_info_label.setStyleSheet(
            f"color: {self.colors['text_secondary'].name()}; font-size: 13px;"
        )
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
        return (
            f"""
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
        )

    def _combobox_style(self) -> str:
        return (
            f"""
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
        )

    def _table_style(self) -> str:
        return (
            f"""
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
        )

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
            logger.info("未检测到用户 Token，跳过服务端任务获取，只显示本地任务。")
            self.populate_table()
            return

        params = {"page": self.page, "page_size": self.page_size}
        task_name = self.task_name_input.text().strip()
        status = self.status_combo.currentData()
        if task_name:
            params["task_name"] = task_name
        if status is not None:
            params["collection_status"] = status

        logger.info(
            "开始获取任务列表：%s/api/v1/task/my-tasks params=%s",
            server_host.rstrip("/"),
            params,
        )

        loading_dialog = LoadingDialog(self, "正在获取任务列表...")
        loading_dialog.show()
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
            items = payload.get("items", [])
            total = payload.get("total", 0)
            page = payload.get("page", self.page)
            page_size = payload.get("page_size", self.page_size)

            logger.info(
                "任务列表获取成功：code=%s, total=%s, page=%s, page_size=%s, items_len=%s",
                data.get("code"),
                total,
                page,
                page_size,
                len(items),
            )

            self.server_tasks = items
            self.total = total
            self.page = page
            self.page_size = page_size
            loading_dialog.close()
            self.populate_table()
        except Exception as e:
            loading_dialog.close()
            QMessageBox.warning(
                self, "获取任务失败", f"无法获取任务列表：{e}\n\n将显示本地任务。"
            )
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
            badge_container = QWidget()
            row_bg = (
                self.colors["bg_secondary"] if (row % 2 == 1) else self.colors["bg_card"]
            )
            badge_container.setStyleSheet(
                f"background-color: {row_bg.name()};"
            )
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
        self.page_info_label.setText(
            f"第 {self.page} / {max_page} 页，共 {total_display} 个任务（含本地 {len(self.local_tasks)} 个）"
        )

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        combined: List[Dict[str, Any]] = []
        combined.extend(self.local_tasks)
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
        item.setForeground(self.colors["text_primary"])
        self.task_table.setItem(row, column, item)

    def format_timestamp(self, timestamp) -> str:
        if not timestamp:
            return "-"
        try:
            return datetime.fromtimestamp(int(timestamp)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
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
                    callback=lambda _checked=False, _t=task: self.continue_collection(
                        _t
                    ),
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
            placeholder.setStyleSheet(
                f"color: {self.colors['text_secondary'].name()}; font-size: 12px;"
            )
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
            task["collection_status"] = 2
            self.update_local_task(task)
            self.populate_table()
            QMessageBox.information(
                self,
                "领取成功",
                f"本地任务 [{task.get('task_name', '')}] 已标记为“待采集”。",
            )
            return

        if self.operate_task(task["id"], op_type=3, collection_status=2):
            task["collection_status"] = 2
            self.populate_table()
            QMessageBox.information(
                self, "领取成功", f"任务 [{task.get('task_name', '')}] 已领取。"
            )

    def start_collection(self, task: Dict[str, Any]):
        current_status = task.get("collection_status")
        if current_status in (0, 1):
            QMessageBox.warning(self, "无法开始", "请先领取任务后再开始采集。")
            return

        if current_status == 2:
            if task.get("is_local"):
                task["collection_status"] = 3
                self.update_local_task(task)
            else:
                if not self.operate_task(task["id"], op_type=3, collection_status=3):
                    return
            task["collection_status"] = 3
            self.populate_table()

        # 打开采集窗口
        self.hide()
        self.collector_window = self.collector_window_cls(
            self.collector_class, self.collector_config_class, task=task
        )
        self.collector_window.setWindowTitle("SRIC数据采集平台")
        # 保存任务管理窗口的引用，以便返回
        self.collector_window.task_manager = self
        self.collector_window.destroyed.connect(self.on_collector_closed)
        self.collector_window.show()

    def continue_collection(self, task: Dict[str, Any]):
        """继续采集（断点续采），在已有数据的基础上继续"""
        status = task.get("collection_status")

        if status in (0, 1):
            QMessageBox.warning(self, "无法继续", "该任务尚未领取，请先领取任务后再继续采集。")
            return

        if task.get("is_local"):
            if status != 3:
                task["collection_status"] = 3
                self.update_local_task(task)
                self.populate_table()
        else:
            if status in (4, 5, 6):
                if not self.operate_task(task["id"], op_type=3, collection_status=3):
                    return
                task["collection_status"] = 3
                self.populate_table()

        self.start_collection(task)

    def on_collector_closed(self):
        """当采集窗口被关闭时调用（通过 destroyed 信号）"""
        if self.collector_window:
            self.collector_window = None
            if not self.isVisible():
                self.show()
                self.raise_()
                self.activateWindow()
                self.fetch_tasks()

    def operate_task(
        self,
        task_id: int,
        op_type: int,
        collection_status: Optional[int] = None,
        approved_count: Optional[int] = None,
    ) -> bool:
        local_task = next(
            (t for t in self.local_tasks if t["id"] == task_id),
            None,
        )
        if local_task is not None:
            if collection_status is not None:
                local_task["collection_status"] = collection_status
            if approved_count is not None:
                local_task["approved_count"] = approved_count
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
        data["id"] = self.next_local_id
        self.next_local_id -= 1
        data["is_local"] = True
        data["tenant_id"] = 0
        data["collection_status"] = 0
        data["approved_count"] = 0
        data["created_at"] = int(datetime.now().timestamp())
        data["updated_at"] = data["created_at"]
        self.local_tasks.append(data)
        self.populate_table()
        QMessageBox.information(
            self, "创建成功", f"本地任务 [{data['task_name']}] 已创建。"
        )

    def update_local_task(self, task: Dict[str, Any]):
        for idx, t in enumerate(self.local_tasks):
            if t["id"] == task["id"]:
                task["updated_at"] = int(datetime.now().timestamp())
                self.local_tasks[idx] = task
                return


