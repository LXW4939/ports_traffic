#! /usr/bin/env python
# -*- coding: utf-8 -*-
stand = \
    {
        "app_name": "demo",
        "offline": {
            "class_path": "main.service.stat.stand_dev.Offline",
            "input_config": {
                "input_type": "mongo",
                "database": "ion",
                "table": "portsTraffic"
            },
            "output_config": {
                "output_type": "mongo",
                "database": "ion",
                "table": "portsTrafficCheck"
            },

            "alg_argument": {
                "interval": "5",  # 有效设备采集周期时间
                "effective_detect_time": "30",  # 有效设备检测时长
                "effective_detect_count": "3",  # 有效设备阈值
                "traffic_detect_interval": "50",  # 处理间隔
                "traffic_train_size": "30",  # 训练集长度
                "test_size": 1
            }
        },
        "online": {
            "class_path": "main.service.porttraffic.port.Offline",
            "input_config": {
                "input_type": "mongo",
                "database": "ion",
                "table": "portsTraffic"
            },
            "output_config": {
                "output_type": "mongo",
                "database": "ion",
                "table": "portsTraffic_pred"
            },
        }
    }