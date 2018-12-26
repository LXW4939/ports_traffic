import pandas as pd
import numpy as np
import time
import datetime
from main.common.dao.mongodb_utils import SingleMongodb
from main.common import log_utils, exception_utils, configration

LOG = log_utils.getLogger(module_name=__name__)


class Offline:
    def __init__(self, offline_config: dict):
        """
        构造函数，从config解析配置
        :param offline_config: 离线配置
        """
        # 输入配置
        input_properties = exception_utils.nullException(offline_config, "input_config")
        self.inputDatabase = exception_utils.nullException(input_properties, "database")
        self.inputTable = exception_utils.nullException(input_properties, "table")
        # 输出配置
        output_properties = exception_utils.nullException(offline_config, "output_config")
        self.outputDatabase = exception_utils.nullException(output_properties, "database")
        self.outputTable = exception_utils.nullException(output_properties, "table")
        # 参数配置
        alg_argument = exception_utils.nullException(offline_config, "alg_argument")
        self.interval = exception_utils.nullException(alg_argument, "interval")

        self.effective_detect_time = exception_utils.nullException(alg_argument, "effective_detect_time")
        self.effective_detect_count = exception_utils.nullException(alg_argument, "effective_detect_count")
        self.traffic_detect_interval = exception_utils.nullException(alg_argument, "traffic_detect_interval")
        self.traffic_train_size = exception_utils.nullException(alg_argument, "traffic_train_size")
        self.test_size = int(exception_utils.nullException(alg_argument, "test_size"))

    def get_data_ip(self):
        singleMongodb = SingleMongodb()
        client = singleMongodb.getClient()
        collection = singleMongodb.getCollection(self.inputDatabase, self.inputTable, client)
        now = datetime.datetime.utcnow()
        startTs = now - datetime.timedelta(minutes=int(self.traffic_train_size))
        #startTs = datetime.datetime(2018, 12, 10, 1, 1, 0)

        # 获取30s的设备did
        port_interval = now - datetime.timedelta(seconds=int(self.effective_detect_time))
        device_list_query = {'ts': {'$gte': port_interval, '$lte': now}}
        xx = list(collection.find(device_list_query, {'_id': 0, 'did': 1}))
        device_port = pd.DataFrame(list(collection.find(device_list_query, {'_id': 0, 'did': 1})))['did'].value_counts()
        device_list = list(device_port[device_port >= int(self.effective_detect_count)].index)
        port_traffic_query = {'did': {'$in': device_list}, 'ts': {'$gte': pd.to_datetime(startTs)}}
        dataSet = collection.find(port_traffic_query, {'_id': 0}, sort=[('ts', 1)])
        tsList, didList, portList, inOctetsList, outOctetsList = [], [], [], [], []
        for dataDic in dataSet:
            for i in range(len(dataDic['traffic'])):
                didList.append(dataDic['did'])
                tsList.append(dataDic['ts'])
                portList.append(dataDic['traffic'][i]['portId'])
                inOctetsList.append(dataDic['traffic'][i]['inOctets'])
                outOctetsList.append(dataDic['traffic'][i]['outOctets'])
        data = pd.DataFrame({'did': didList, 'ts': tsList, 'port': portList, 'inOctets': inOctetsList, 'outOctets': outOctetsList})
        return data, device_list

    def get_data_port(self, data, ip: str, port: str):
        ip_str = ip
        data_every = data[(data['port'] == port)&(data['did'] == ip_str)]
        if data_every.empty:
            return pd.DataFrame()
        data_every = data_every.sort_values(['ts'])
        data_every = data_every.reset_index(drop=True)

        first = data_every.ts[0]
        second = data_every.ts[1]
        thrid = data_every.ts[2]
        tdiff1 = second - first
        tdiff2 = thrid - second
        time_diff1 = tdiff1.days * 86400 + tdiff1.seconds
        time_diff2 = tdiff2.days * 86400 + tdiff2.seconds
        if time_diff1 > 30 and time_diff2 > 30:
            return pd.DataFrame()
        data_every['inOctets'] = pd.to_numeric(data_every['inOctets'])
        data_every['outOctets'] = pd.to_numeric(data_every['outOctets'])

        data_every_diff = data_every[['inOctets', 'outOctets']]
        data_every_diff = data_every_diff.diff()
        #data_every_diff = data_every_diff.shift(-1)
        data_every[['inOctets', 'outOctets']] = data_every_diff
        index_less_zero = np.where(data_every["inOctets"] < 0)[0].tolist()
        data_every = data_every.drop(data_every.index[index_less_zero], axis=0)
        data_every = data_every.drop([data_every.index[0]], axis=0)
        if len(np.where(data_every['inOctets'] == 0)[0]) > 0.5 * len(data_every):
            return pd.DataFrame()
        data_every['ioRate'] = data_every['outOctets'] / (data_every['inOctets'] + 0.0001)
        data_every = data_every.drop([data_every.index[len(data_every) - 1]], axis=0)

        data_every.index = range(len(data_every))
        ts_diff = []
        for i in range(1, len(data_every)):
            ts_diff.append((data_every.loc[i, 'ts'] - data_every.loc[i - 1, 'ts']).seconds)
        data_every = data_every.drop([data_every.index[0]], axis=0)
        data_every['ts_diff'] = ts_diff

        for oct_type in ['inOctets', 'outOctets', 'ioRate']:
            data_every[oct_type] = data_every[oct_type] / data_every['ts_diff']
        return data_every

    def save_model(self, model):
        singleMongodb = SingleMongodb()
        client = singleMongodb.getClient()
        collection = singleMongodb.getCollection(self.outputDatabase, self.outputTable, client)
        collection.insert_many(model)

    def save_error(self, model):
        singleMongodb = SingleMongodb()
        client = singleMongodb.getClient()
        collection = singleMongodb.getCollection('ion', 'portTraffic_error', client)
        collection.insert_one(model)

    def get_save_data(self, did, port, data_every, test_time, k_sigma, train_time=30):
        save_data=[]
        last_time = data_every.loc[len(data_every)-1, 'ts']
        test_start_time = last_time - datetime.timedelta(minutes=test_time) # 5
        test_df = data_every[data_every['ts'] > test_start_time]
        test_df = test_df.reset_index(drop=True)
        for i in range(len(test_df)):
            cur_time = test_df.loc[i, 'ts']
            cur_result = {'did': did, 'portId': port, 'ts': cur_time, 'pseudo_anomaly':False,'anomaly':False}
            cur_time_test = [test_df.loc[i,'inOctets'],test_df.loc[i,'outOctets'],test_df.loc[i,'ioRate']]
            # half_hour_before = cur_time - datetime.timedelta(minutes=train_time)
            train_df = data_every[(data_every['ts']<cur_time)]
            train_df = train_df.reset_index(drop = True)
            oct_list = ['inOctets','outOctets','ioRate']
            for i, oct in enumerate(oct_list):
                oct_series = train_df[oct]
                oct_des = oct_series.describe()
                # print('=============================================\n')
                # print(oct_des)
                # print('\n=============================================')
                oct_std = oct_des['std']
                oct_mean = oct_des['mean']
                ksigma = oct_mean + k_sigma*oct_std
                oct_dict={'test':cur_time_test[i],'high':ksigma}
                cur_result[oct] = oct_dict
            if cur_result['inOctets']['high'] < cur_result['inOctets']['test'] and cur_result['outOctets']['high'] < cur_result['outOctets']['test'] \
                and cur_result['ioRate']['high'] < cur_result['ioRate']['test']:
                cur_result['pseudo_anomaly'] =True

            if cur_result['pseudo_anomaly']:
                save_data_sorted = sorted(save_data, key=lambda x: x['ts'])
                front_two = save_data[:-2]
                if front_two[0]['pseudo_anomaly'] and front_two[1]['pseudo_anomaly']:
                    cur_result['anomaly'] = True
                else:
                    cur_result['anomaly'] = False
                    cur_result['inOctets']['high'] = cur_result['inOctets']['test']*1.1
                    cur_result['outOctets']['high'] = cur_result['outOctets']['test'] * 1.1
                    cur_result['ioRate']['high'] = cur_result['ioRate']['test'] * 1.1

            save_data.append(cur_result)
        return save_data

    def do(self):
        start = datetime.datetime.utcnow()
        data, device_list = self.get_data_ip()
        results = []
        for ip in device_list:
            data_ip = data[data['did'] == ip]
            port_ids = list(data_ip['port'].unique())
            for port in port_ids:
                try:
                    data_every = self.get_data_port(data, ip, port)
                    k_sigma = 3
                    save_data = self.get_save_data(ip, port, data_every, int(self.traffic_detect_interval), k_sigma, int(self.traffic_train_size))
                    results.extend(save_data)
                except:
                    continue
        self.save_model(results)
        end = datetime.datetime.utcnow()
        print('total time:', end-start)
