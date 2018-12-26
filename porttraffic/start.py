#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:pengxingxiong@ruijie.com.cn
@time: 2018/10/11 9:37
@desc:
"""
import datetime

from main.common import log_utils, configration

LOG = log_utils.getLogger(module_name=__name__)


def alg_select(alg_config, alg_run_type):
    """
    算法选择器\n
    :param alg_run_type: 算法运行类型：offline/online
    :param alg_config: 算法配置参数，dict结构
    :return: Null
    """
    if alg_run_type == "offline":
        offlineConfig = alg_config["offline"]
        clazz_obj = configration.getClassObj(offlineConfig, offlineConfig["class_path"])
        offlinePipeline(clazz_obj)
    elif alg_run_type == "online":
        onlineConfig = alg_config["online"]
        clazz_obj = configration.getClassObj(onlineConfig, onlineConfig["class_path"])
        onlinePipeline(clazz_obj)
    else:
        errMsg = "algRunType = [%s] is  unrecognized , it should be offline or online" % alg_run_type
        LOG.error(errMsg)
        raise RuntimeError(errMsg)


def offlinePipeline(alg_obj):
    """
     离线算法管道
     :param alg_obj: 离线算法的类实例
     :return:
     """
    # org_df = alg_obj.get_data()  # 获取数据
    # train = alg_obj.pre_processing(org_df)  # 数据预处理
    # train_model = alg_obj.train(train)  # 模型训练
    # alg_obj.save_model(train_model)  # 保存模型
    alg_obj.do()

def onlinePipeline(alg_obj):
    """
    离线算法管道
    :param alg_obj: 在线算法的类实例
    :return:
    """
    org_df = alg_obj.get_data()
    if org_df.empty:
        return
    test = alg_obj.pre_processing(org_df)
    model = alg_obj.load_model()
    predict_result = alg_obj.predict(model, test)
    dict_result = alg_obj.save_processing(predict_result)
    alg_obj.to_db(dict_result)


if __name__ == "__main__":
    print("start time:", datetime.datetime.utcnow())
    # TODO 从入参中读取参数
    jsonType = "single"
    jsonName = "port_traffic"
    algRunType = "offline"
    init_config = configration.getPyConfig(jsonType, jsonName)
    LOG.info(init_config)
    alg_select(init_config, algRunType)
    print("end time:", datetime.datetime.utcnow())
