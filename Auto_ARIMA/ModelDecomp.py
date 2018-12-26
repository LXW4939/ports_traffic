#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import datetime
import time
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from datetime import timedelta
import statsmodels.api as sm
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
        #参数配置
        alg_argument = exception_utils.nullException(offline_config, "alg_argument")
        self.interval = exception_utils.nullException(alg_argument, "interval")

        self.startTs = exception_utils.nullException(alg_argument, "start_time")
        self.did = exception_utils.nullException(alg_argument, "did")
        #self.ts = self.get_data().iloc[:1144]
        self.test_size = int(exception_utils.nullException(alg_argument, "test_size"))
        #self.train_size = len(self.ts) - self.test_size


    def get_data(self, ip):
        portId = self.did[ip]
        singleMongodb = SingleMongodb()
        client = singleMongodb.getClient()
        collection = singleMongodb.getCollection(self.inputDatabase, self.inputTable, client)
        data = pd.DataFrame(columns=['ts', 'did', 'port', 'inOctets', 'outOctets'])
        dataSet = collection.find({'did': "ip:"+ip, "ts": {"$gte": pd.to_datetime(self.startTs)}}, {'_id': 0},
                                  sort=[('ts', 1)]).limit(400)
        tsList, didList, portList, inOctetsList, outOctetsList = [], [], [], [], []
        for dataDic in dataSet:
            for i in range(len(dataDic['traffic'])):
                print(i)
                if dataDic['traffic'][i]['portId'] in portId:
                    didList.append(dataDic['did'])
                    tsList.append(dataDic['ts'])
                    portList.append(dataDic['traffic'][i]['portId'])
                    inOctetsList.append(dataDic['traffic'][i]['inOctets'])
                    outOctetsList.append(dataDic['traffic'][i]['outOctets'])
        data = pd.DataFrame(
            {'did': didList, 'ts': tsList, 'port': portList, 'inOctets': inOctetsList, 'outOctets': outOctetsList})
        return data
        # res = {"did": ip}
        # nowTime = datetime.datetime.utcnow()
        # startTime = nowTime - datetime.timedelta(minutes=10)
        # singleMongodb = SingleMongodb()
        # client = singleMongodb.getClient()
        # collection = singleMongodb.getCollection(self.inputDatabase, self.inputTable, client)
        #
        # data = pd.DataFrame(columns=['ts', 'did', 'port', 'inOctets', 'outOctets'])
        # cursor = collection.find({'did': 'ip:'+ip, 'ts': {"$gte": pd.to_datetime(self.startTs)}},
        #                          {'_id': 0}).sort([('ts', 1)]).limit(200) #self.startTs
        #
        # i = 0
        # for document in cursor:
        #     print(i)
        #     i += 1
        #     did = document['did']
        #     ts = document['ts']
        #     for traffic in document['traffic']:
        #         if traffic['portId'] in self.did[ip]:
        #             inOctets = traffic['inOctets']
        #             outOctets = traffic['outOctets']
        #             port = traffic['portId']
        #             lst = [ts, did[3:], port, inOctets, outOctets]
        #             data.loc[data.shape[0]] = lst
        #     if data.empty:
        #         raise Exception("从数据库读取的值为空")
       # return data

        # for port in self.did[ip]:
        #     data_every = data[data['port'] == port]
        #     if data_every.empty:
        #         continue
        #     data_every = data_every.sort_values(['ts'])
        #     data_every['inOctets'] = pd.to_numeric(data_every['inOctets'])
        #     data_every['outOctets'] = pd.to_numeric(data_every['outOctets'])
        #
        #     data_every_diff = data_every[['inOctets', 'outOctets']]
        #     data_every_diff = data_every_diff.diff()
        #     data_every_diff = data_every_diff.shift(-1)
        #     data_every[['inOctets', 'outOctets']] = data_every_diff
        #     index_less_zero = np.where(data_every["inOctets"] < 0)[0].tolist()
        #     data_every = data_every.drop(index_less_zero, axis=0)
        #     data_every = data_every.drop([data_every.index[len(data_every) - 1]], axis=0)
        #     f = lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
        #     data_every['ts'] = data_every['ts'].apply(f)
        #     data_every['ts'] = pd.to_datetime(data_every['ts'])
        #     # length =data_every.shape[0]
        #     data_every.index = data_every['ts']
        #
        #     startTime = data_every.index[0] + timedelta(seconds=2)
        #     endTime = data_every.index[-1] + timedelta(seconds=2)
        #     timeStamp = pd.date_range(startTime, endTime, freq=self.interval)  # 间隔时间
        #     data_everyMinus = data_every.asof(timeStamp)
        #
        #     df = pd.DataFrame(columns=data_everyMinus.columns)
        #     df = df.append(data_everyMinus.iloc[0, :])
        #     for i in range(1, len(data_everyMinus)):
        #         if data_everyMinus.iloc[i]['inOctets'] != df.iloc[-1]['inOctets'] or \
        #                 (data_everyMinus.iloc[i]['outOctets'] != df.iloc[-1]['outOctets']):
        #             df = df.append(data_everyMinus.iloc[i, :])
        #     df.ts = df.index
        #
        #     dff = pd.DataFrame(timeStamp, columns=['ts'])
        #     dff['did'] = ip + '_' + port
        #     tmp = pd.merge(dff, df, how='left', on='ts')
        #     tmp = tmp.drop('did_y', axis=1)
        #     data_everyMinus = tmp.replace(np.nan, 0)
        #     data_everyMinus = data_everyMinus.rename(columns={'did_x': 'did'})
        #
        #     window = 2  # window为7或2比较好
        #     data_everyMinus['rolling_inOctets'] = data_everyMinus.inOctets.rolling(window=window).mean()
        #     data_everyMinus['rolling_outOctets'] = data_everyMinus.outOctets.rolling(window=window).mean()
        #     data_everyMinus.index = range(data_everyMinus.shape[0])
        #     lst = np.where(data_everyMinus.inOctets == 0)[0]
        #     for i in range(len(lst)):
        #         ind = lst[i]
        #         data_everyMinus.loc[ind, 'inOctets'] = data_everyMinus.iloc[ind]['rolling_inOctets']
        #         data_everyMinus.loc[ind, 'outOctets'] = data_everyMinus.iloc[ind]['rolling_outOctets']
        #     port_dict = {}
        #     port_dict['inOctets'] = data_everyMinus['inOctets'].tolist()
        #     port_dict['outOctets'] = data_everyMinus['outOctets'].tolist()
        #     res[port] = port_dict
        #     res['ts'] = data_everyMinus.ts.apply(f).tolist()
        # return res

    def pre_processing(self, data, ip, port):
        dip = 'ip:'+ip

        data_every = data[(data['port'] == port) & (data['did'] == dip)]

        if data_every.empty:
            return
        data_every = data_every.reset_index(drop=True)
        data_every = data_every.sort_values(['ts'])
        data_every['inOctets'] = pd.to_numeric(data_every['inOctets'])
        data_every['outOctets'] = pd.to_numeric(data_every['outOctets'])

        data_every_diff = data_every[['inOctets', 'outOctets']]
        data_every_diff = data_every_diff.diff()
        data_every_diff = data_every_diff.shift(-1)
        data_every[['inOctets', 'outOctets']] = data_every_diff
        index_less_zero = np.where(data_every["inOctets"] < 0)[0].tolist()
        data_every = data_every.drop(data_every.index[index_less_zero], axis=0)
        data_every = data_every.drop([data_every.index[len(data_every) - 1]], axis=0)

        f = lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
        data_every['ts'] = data_every['ts'].apply(f)
        data_every['ts'] = pd.to_datetime(data_every['ts'])
        # length =data_every.shape[0]
        data_every.index = data_every['ts']

        startTime = data_every.index[0] + timedelta(seconds=2)
        endTime = data_every.index[-1] + timedelta(seconds=2)
        timeStamp = pd.date_range(startTime, endTime, freq=self.interval)  # 间隔时间
        data_everyMinus = data_every.asof(timeStamp)

        df = pd.DataFrame(columns=data_everyMinus.columns)
        df = df.append(data_everyMinus.iloc[0, :])
        for i in range(1, len(data_everyMinus)):
            if data_everyMinus.iloc[i]['inOctets'] != df.iloc[-1]['inOctets'] or \
                    (data_everyMinus.iloc[i]['outOctets'] != df.iloc[-1]['outOctets']):
                df = df.append(data_everyMinus.iloc[i, :])
        df.ts = df.index
        
        dff = pd.DataFrame(timeStamp, columns=['ts'])
        dff['did'] = ip + '_' + port
        tmp = pd.merge(dff, df, how='left', on='ts')
        tmp = tmp.drop('did_y', axis=1)
        data_everyMinus = tmp.replace(np.nan, 0)
        data_everyMinus = data_everyMinus.rename(columns={'did_x': 'did'})

        window = 2  # window为7或2比较好
        data_everyMinus['rolling_inOctets'] = data_everyMinus.inOctets.rolling(window=window).mean()
        data_everyMinus['rolling_outOctets'] = data_everyMinus.outOctets.rolling(window=window).mean()
        data_everyMinus.index = range(data_everyMinus.shape[0])
        lst = np.where(data_everyMinus.inOctets == 0)[0]
        for i in range(len(lst)):
            ind = lst[i]
            data_everyMinus.loc[ind, 'inOctets'] = data_everyMinus.iloc[ind]['rolling_inOctets']
            data_everyMinus.loc[ind, 'outOctets'] = data_everyMinus.iloc[ind]['rolling_outOctets']
        return data_everyMinus

    def get_oct(self, data_everyMinus, oct):
        ts = data_everyMinus[['ts', oct]]
        ts = ts.set_index('ts', drop=True)
        ts = ts[oct]
        dif = ts.diff().dropna()
        td = dif.describe()
        high = td['75%'] + 1.5 * (td['75%'] - td['25%'])
        low = td['25%'] - 1.5 * (td['75%'] - td['25%'])

        forbid_index = dif[(dif > high) | (dif < low)].index
        i = 0
        while i < len(forbid_index) - 1:
            n = 1
            start = forbid_index[i]
            while i + n < len(forbid_index) and forbid_index[i + n] == start + timedelta(
                    seconds=int(self.interval[:-1]) * n):
                n += 1
            i += n - 1

            end = forbid_index[i]
            try:
                value = np.linspace(ts[start - timedelta(seconds=int(self.interval[:-1]) * n)],
                                    ts[end + timedelta(seconds=int(self.interval[:-1]) * n)], n)
                ts[start: end] = value
            except:
                pass
            i += 1
        return ts

    def train(self, order):
        # self.trainsamples索引为时间戳的series
        decomposition = seasonal_decompose(self.trainsamples, freq=int(self.interval[:-1]), two_sided=False)
        self.trend = decomposition.trend.fillna(0)
        self.seasonal = decomposition.seasonal.fillna(0)
        self.residual = decomposition.resid.fillna(0)


        d = self.residual.describe()
        delta = d['75%'] - d['25%']

        self.low_error, self.high_error = (d['25%'] - 1 * delta, d['75%'] + 1 * delta)

        self.trend.dropna(inplace=True)
        #model = auto_arima(self.trainsamples, trace=True, error_action='ignore', suppress_warnings=True)
        #self.trend_model = model.fit(self.trainsamples)

        self.trend_model = ARIMA(self.trainsamples, order).fit(disp=-1, method='css')


        self.train_season = self.seasonal

        n = self.test_size
        self.pred_time_index = pd.date_range(start=self.trainsamples.index[-1], periods=n + 1, freq=self.interval)[1:]
        #self.trend_pred = self.trend_model.predict(n_periods=n)
        self.trend_pred = self.trend_model.forecast(n)[0]
        values = list()
        low_conf_values = list()
        high_conf_values = list()

        for i, t in enumerate(self.pred_time_index):
            trend_part = self.trend_pred[i]

            # 相同时间的数据均值
            season_part = self.train_season[
                self.train_season.index.time == t.time()].mean()
            if math.isnan(season_part):
                season_part = 0
            # 趋势+周期+误差界限
            predict = trend_part + season_part
            low_bound = trend_part + season_part + self.low_error
            high_bound = trend_part + season_part + self.high_error

            values.append(predict)
            low_conf_values.append(low_bound)
            high_conf_values.append(high_bound)

        test_list = list(self.test.values)
        array_test = np.array(test_list)
        array_high_conf = np.array(high_conf_values)
        index = np.where(array_test < array_high_conf)[0].tolist()
        length = len(index)
        print('Accuracy:', length/len(array_test))

        return {'values': values, 'highV': high_conf_values, 'test': list(self.test.values),
                'ts': self.test.index.tolist(), 'type': self.trainsamples.name, 'index': index,
                'length': length}

    def save_model(self, model):
        singleMongodb = SingleMongodb()
        client = singleMongodb.getClient()
        collection = singleMongodb.getCollection(self.outputDatabase, self.outputTable, client)
        collection.insert_one(model)

    def ecdf(self, dataRaw, efp=0.95):
        type = dataRaw.name
        dataRaw = pd.DataFrame({"ts": dataRaw.index, "value": dataRaw.values})
        dataRawNum = dataRaw['value'].count()  # 计算数据量
        dataNew = pd.DataFrame(dataRaw['value'].value_counts())  # 统计每个值出现次数
        dataNew = dataNew.reset_index()
        dataNew = dataNew.sort_values(by='index', ascending=True)
        dataNew['cumCount'] = dataNew['value'].cumsum()
        dataNew['ecdf'] = dataNew['cumCount'] / dataRawNum
        result = dataNew[['index', 'ecdf']]
        result.columns = ['value', 'ecdf']
        thresholdData = result.loc[result['ecdf'] >= efp]
        threshold = float(thresholdData['value'].head(1))
        return {'test': list(self.test.values), 'pred': threshold,'ts':self.test.index.tolist(),
                'type': type}

    def process(self):
        total_start = time.time()
        get_data_time = 0
        process_time = 0
        train_time =0
        save_time =0
        for ip in self.did.keys():
            get_data_start = time.time()
            data= self.get_data(ip)
            get_data_end = time.time()
            get_data_time += (get_data_end - get_data_start)

            for port in self.did[ip]:
                process_start_time = time.time()
                data_port = self.pre_processing(data, ip, port)
                if data_port.empty:
                    continue
                process_end_time = time.time()
                process_time += (process_end_time - process_start_time)
                for oct in ['inOctets','outOctets']:
                    process_start_time = time.time()
                    data_oct = self.get_oct(data_port, oct)
                    data_oct.index = pd.to_datetime(data_oct.index)
                    start = data_oct.index[0]
                    end = data_oct.index[1]
                    tdiff = end -start
                    time_diff = tdiff.days*86400 + tdiff.seconds
                    if time_diff > 30:
                        continue
                    # df = data_oct.set_index('ts', drop=True)
                    # df = df[oct]
                    df = data_oct
                    #df.name = ip+'_'+port+'_'+oct
                    self.trainsamples = df[:len(df) - self.test_size]
                    self.test = df[-self.test_size:]
                    self.test.index = self.test.index.astype(str)
                    self.trainsamples.name = ip+'_'+port+'_'+oct

                    process_end_time = time.time()
                    process_time += (process_end_time - process_start_time)
                    train_time_start = time.time()
                    #p, q = sm.tsa.arma_order_select_ic(self.trainsamples, max_ar=5, max_ma=5, ic='aic')['aic_min_order']
                    try:
                        train_model = self.train(order=(1, 1, 3))
                    except:
                        continue
                    #train_model = self.ecdf(self.trainsamples)
                    train_time_end = time.time()
                    train_time += (train_time_end - train_time_start)

                    save_time_start = time.time()
                    self.save_model(train_model)
                    save_time_end = time.time()
                    save_time += (save_time_end - save_time_start)
        total_end = time.time()
        print("get_data_time:", get_data_time)
        print("train_time:", train_time)
        print("process_time:", process_time)
        print("save_time:", save_time)
        print('total time:', total_end-total_start)


