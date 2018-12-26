#! /usr/bin/env python
# -*- coding: utf-8 -*-
port_traffic = \
    {
        "app_name": "demo",
        "offline": {
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
            "alg_config": {
                "alg_lib": "port_traffic",
                "alg_name": "port_traffic",
                "alg_params": {
                    "criterion": "gini"
                },
                "model_path": "e:/random_forest/"
            },
            "alg_argument": {
                "interval": "10S",
                "test_size": 1,
                "start_time": "2018-12-07T14:09:19.236+0000",
                "did": {
                        "172.18.3.21": ["Gi0/22","Gi0/23","Gi0/24","Gi0/11","Gi0/12","Gi0/13","Gi0/14","Gi0/15",
                                     "Gi0/16","Gi0/17","Gi0/18","Gi0/1","Gi0/2","Gi0/3"],
                    "172.18.3.12": ["Gi0/22", "Gi0/23", "Gi0/24", "Gi0/10", "Gi0/11", "Gi0/13", "Gi0/14", "Gi0/16",
                                    "Gi0/17",
                                    "Gi0/18", "Gi0/19", "Gi0/2", "Gi0/4", "Gi0/5", "Gi0/6", "Gi0/7", "Gi0/8", "Gi0/9",
                                    "Gi0/21"],
                    "172.18.3.13": ["Gi0/22", "Gi0/24", "Gi0/12", "Gi0/13", "Gi0/15", "Gi0/17", "Gi0/18", "Gi0/19",
                                    "Gi0/1",
                                    "Gi0/2", "Gi0/3", "Gi0/4", "Gi0/5", "Gi0/6", "Gi0/7", "Gi0/20", "Gi0/21"],
                    "172.18.3.14": ["Gi0/22", "Gi0/23", "Gi0/24", "Gi0/10", "Gi0/11", "Gi0/14", "Gi0/15", "Gi0/16",
                                    "Gi0/17",
                                    "Gi0/18", "Gi0/19", "Gi0/1", "Gi0/2", "Gi0/3", "Gi0/4", "Gi0/5", "Gi0/6", "Gi0/7",
                                    "Gi0/8",
                                    "Gi0/9", "Gi0/20", "Gi0/21"
                                    ]
                        }
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
            "alg_config": {
                "alg_lib": "port_traffic",
                "alg_name": "port_traffic",
                "alg_params": {},
                "model_path": "e:/random_forest/model.pkl"
            }
        }
    }