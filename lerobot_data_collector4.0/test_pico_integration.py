#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pico遥控器集成测试
=================

测试Pico遥控器与数据采集系统的集成功能。
"""

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lerobot_data_collector"))

from lerobot_data_collector.collector_core import CollectorConfig, Collector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pico_config_loading():
    """测试Pico配置文件加载"""
    config_path = Path(__file__).parent / "configs" / "g1_lcm_pico.json"

    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return False

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        config = CollectorConfig.from_dict(config_data)
        logger.info(f"配置加载成功: robot_type={config.robot_type}, lcm_enabled={config.lcm_enabled}, pico_enabled={config.pico_enabled}")

        # 验证Pico相关配置
        assert config.robot_type == "gen1", f"robot_type应该是'gen1', 实际是{config.robot_type}"
        assert config.lcm_enabled == True, f"lcm_enabled应该是True, 实际是{config.lcm_enabled}"
        assert config.pico_enabled == True, f"pico_enabled应该是True, 实际是{config.pico_enabled}"
        assert config.pico_ip == "192.168.12.110", f"pico_ip应该是'192.168.12.110', 实际是{config.pico_ip}"
        assert config.pico_port == 12345, f"pico_port应该是12345, 实际是{config.pico_port}"

        logger.info("Pico配置文件验证通过")
        return True

    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pico_collector_init():
    """测试Pico采集器初始化"""
    import tempfile

    config_path = Path(__file__).parent / "configs" / "g1_lcm_pico.json"

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # 使用临时目录避免权限问题
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data["dataset_root"] = temp_dir
            config = CollectorConfig.from_dict(config_data)

            # 创建采集器
            collector = Collector(config)
            logger.info("Pico采集器初始化成功")

            # 验证配置
            assert collector.cfg.robot_type == "gen1"
            assert collector.cfg.lcm_enabled == True
            assert collector.cfg.pico_enabled == True

            logger.info("Pico采集器配置验证通过")
            return True

    except Exception as e:
        logger.error(f"采集器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pico_adapter_import():
    """测试Pico适配器导入"""
    try:
        from lerobot_data_collector.pico_controller import (
            PicoControllerAdapter,
            create_pico_control_handler
        )
        logger.info("Pico适配器导入成功")

        # 测试Pico可用性
        from lerobot_data_collector.pico_controller import PICO_AVAILABLE
        logger.info(f"Pico库可用性: {PICO_AVAILABLE}")

        # 测试控制处理器创建
        class MockCollector:
            def __init__(self):
                self._session_active = False
                self._target_frames = None
                self._scheduled_start_time = None

            def start_session_immediate(self, target_frames):
                self._session_active = True
                logger.info(f"模拟开始录制: {target_frames}帧")

            def stop_and_save(self):
                self._session_active = False
                logger.info("模拟停止录制并保存")

        mock_collector = MockCollector()
        handler = create_pico_control_handler(mock_collector)

        # 测试命令处理
        handler("pico_start_record")
        assert mock_collector._session_active == True, "开始录制命令未生效"

        handler("pico_stop_record")
        assert mock_collector._session_active == False, "停止录制命令未生效"

        logger.info("Pico控制处理器测试通过")
        return True

    except ImportError as e:
        logger.error(f"Pico适配器导入失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始Pico遥控器集成测试")

    tests = [
        ("配置文件加载测试", test_pico_config_loading),
        ("适配器导入测试", test_pico_adapter_import),
        ("采集器初始化测试", test_pico_collector_init),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n执行测试: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name} 通过")
                passed += 1
            else:
                logger.error(f"❌ {test_name} 失败")
        except Exception as e:
            logger.error(f"❌ {test_name} 异常: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\n测试结果: {passed}/{total} 通过")

    if passed == total:
        logger.info("🎉 所有测试通过！Pico遥控器集成成功")
        logger.info("\nPico控制说明:")
        logger.info("- 按下A/X按钮: 开始数据录制 (30秒, 900帧)")
        logger.info("- 按下B/Y按钮: 停止录制并保存数据")
        logger.info("- 按下扳机: 紧急停止录制")
        return True
    else:
        logger.error("❌ 部分测试失败，请检查集成")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
