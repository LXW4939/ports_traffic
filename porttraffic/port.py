import pandas as pd
import numpy as np
from datetime import timedelta
import time
import math
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from main.common.dao.mongodb_utils import SingleMongodb
from main.common import log_utils, exception_utils, configration
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
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
        self.test_size = int(exception_utils.nullException(alg_argument, "test_size"))

    def get_data_ip(self, ip: str):
        portId = self.did[ip]
        singleMongodb = SingleMongodb()
        client = singleMongodb.getClient()
        collection = singleMongodb.getCollection(self.inputDatabase, self.inputTable, client)
        did_str = "ip:" + ip
        dataSet = collection.find({'did': did_str, "ts": {"$gte": pd.to_datetime(self.startTs)}}, {'_id': 0},
                                       sort=[('ts', 1)]).limit(200)
        tsList, didList, portList, inOctetsList, outOctetsList = [], [], [], [], []
        for dataDic in dataSet:
            for i in range(len(dataDic['traffic'])):
                if dataDic['traffic'][i]['portId'] in portId:
                    didList.append(dataDic['did'])
                    tsList.append(dataDic['ts'])
                    portList.append(dataDic['traffic'][i]['portId'])
                    inOctetsList.append(dataDic['traffic'][i]['inOctets'])
                    outOctetsList.append(dataDic['traffic'][i]['outOctets'])
        data = pd.DataFrame(
            {'did': didList, 'ts': tsList, 'port': portList, 'inOctets': inOctetsList, 'outOctets': outOctetsList})
        return data

    def get_data_port(self, data, ip: str, port: str):
        ip_str = 'ip:' + ip
        data_every = data[(data['port'] == port)&(data['did'] == ip_str)]
        if data_every.empty or len(np.where(data_every['inOctets'] == 0)[0]) > 0.5*len(data_every):
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
        data_every_diff = data_every_diff.shift(-1)
        data_every[['inOctets', 'outOctets']] = data_every_diff
        index_less_zero = np.where(data_every["inOctets"] < 0)[0].tolist()
        data_every = data_every.drop(data_every.index[index_less_zero], axis=0)
        data_every = data_every.drop([data_every.index[len(data_every) - 1]], axis=0)

        f = lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
        data_every['ts'] = data_every['ts'].apply(f)
        data_every['ts'] = pd.to_datetime(data_every['ts'])
        data_every.index = data_every['ts']

        startTime = data_every.index[0] + timedelta(seconds=2)
        endTime = data_every.index[-1] + timedelta(seconds=2)
        timeStamp = pd.date_range(startTime, endTime, freq=self.interval)  # 间隔时间
        data_everyMinus = data_every.asof(timeStamp)
        #return data_everyMinus

        window = 7
        df = pd.DataFrame(columns=['inOctets', 'outOctets'])
        df['inOctets'] = data_everyMinus.inOctets.rolling(window=window).mean()/int(self.interval[:-1])
        df['outOctets'] = data_everyMinus.outOctets.rolling(window=window).mean()/int(self.interval[:-1])
        return df

        # df = pd.DataFrame(columns=data_everyMinus.columns)
        # df = df.append(data_everyMinus.iloc[0, :])
        # for i in range(1, len(data_everyMinus)):
        #     if data_everyMinus.iloc[i]['inOctets'] != df.iloc[-1]['inOctets'] or \
        #             (data_everyMinus.iloc[i]['outOctets'] != df.iloc[-1]['outOctets']):
        #         df = df.append(data_everyMinus.iloc[i, :])
        # df.ts = df.index
        #
        # dff = pd.DataFrame(timeStamp, columns=['ts'])
        # dff['did'] = ip + '_' + port
        # tmp = pd.merge(dff, df, how='left', on='ts')
        # tmp = tmp.drop('did_y', axis=1)
        # data_everyMinus = tmp.replace(np.nan, 0)
        # data_everyMinus = data_everyMinus.rename(columns={'did_x': 'did'})
        #
        # window = 2  # window为7或2比较好
        # data_everyMinus['rolling_inOctets'] = data_everyMinus.inOctets.rolling(window=window).mean()
        # data_everyMinus['rolling_outOctets'] = data_everyMinus.outOctets.rolling(window=window).mean()
        # data_everyMinus.index = range(data_everyMinus.shape[0])
        # lst = np.where(data_everyMinus.inOctets == 0)[0]
        # for i in range(len(lst)):
        #     ind = lst[i]
        #     data_everyMinus.loc[ind, 'inOctets'] = data_everyMinus.iloc[ind]['rolling_inOctets']
        #     data_everyMinus.loc[ind, 'outOctets'] = data_everyMinus.iloc[ind]['rolling_outOctets']
        # return data_everyMinus

    def get_oct(self, data_everyMinus, oct: str):
        ts = data_everyMinus[oct]
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
    def train_arima(self, trainsamples, test):
        # trainsamples索引为时间戳的series
        trainsamples.dropna(inplace=True)
        decomposition = seasonal_decompose(trainsamples, freq=int(self.interval[:-1]), two_sided=False)
        # trend = decomposition.trend.fillna(0)
        # seasonal = decomposition.seasonal.fillna(0)
        residual = decomposition.resid.fillna(0)
        d = residual.describe()
        delta = d['75%'] - d['25%']

        #model = auto_arima(trainsamples, trace=True, error_action='ignore', suppress_warnings=True)

        p, q = sm.tsa.arma_order_select_ic(trainsamples,max_ar=5,max_ma=5,ic='aic')['aic_min_order']
        model = ARIMA(trainsamples, order=(p, 1, q)).fit(disp=-1, method='css')
        # trend_model = model.fit(trainsamples)
        # threshold= trend_model.predict(n_periods=1)[0]
        threshold = model.forecast(1)[0]
        if threshold < 0:
            threshold = -threshold
        tst = list(test.values)[0]
        if tst < 0:
            tst = -tst
        if tst < 0.1:
            return {}
        return {'threshold': threshold, 'test': tst,'type': trainsamples.name,
                'high':threshold+trainsamples.describe()['std']*0.5,'ts': test.index.tolist()[0]}

    def train(self, train_sample, test_sample, efp=0.95):
        type = train_sample.name
        dataRaw = pd.DataFrame({"ts": train_sample.index, "value": train_sample.values})
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
        if threshold < 0:
            threshold = -threshold
        test = list(test_sample.values)[0]
        if test < 0:
            test = -test
        if test < 0.1:
            return {}
        # if threshold < test:
        #     print('False')
        # else:
        #     print('True')
        return {'test': list(test_sample.values)[0], 'threshold': threshold, 'ts': test_sample.index.tolist()[0],
                'type': type}

    def save_model(self, model):
        singleMongodb = SingleMongodb()
        client = singleMongodb.getClient()
        collection = singleMongodb.getCollection(self.outputDatabase, self.outputTable, client)
        collection.insert_one(model)

    def do(self):
        y_true =[]
        y_pred =[]
        start = time.time()
        get_data_time = 0
        process_data_time = 0
        train_time = 0
        save_model_time = 0
        for ip in self.did.keys():
            get_data_start = time.time()
            data = self.get_data_ip(ip)
            get_data_end = time.time()
            get_data_time += get_data_end - get_data_start
    
            for port in self.did[ip]:
                process_time_start = time.time()
                data_everyMinus = self.get_data_port(data, ip, port)
                if data_everyMinus.empty:
                    print('获取数据为空')
                    continue
                process_time_end = time.time()
                process_data_time += process_time_end - process_time_start
    
                for oct in ['inOctets', 'outOctets']:
                    process_time_start = time.time()
                    #data_oct = self.get_oct(data_everyMinus, oct)
                    data_oct = data_everyMinus[oct]
                    train_sample = data_oct[:len(data_oct) - self.test_size]
                    test_sample = data_oct[-self.test_size:]
                    train_sample.name = ip + '_' + port + '_' + oct
                    test_sample.index = test_sample.index.astype(str)
                    process_time_end = time.time()
                    process_data_time += process_time_end - process_time_start
    
                    train_time_start = time.time()
                    #train_model = self.train(train_sample, test_sample, efp=0.95)
                    # try:
                    #     train_model = self.train_arima(train_sample, test_sample)
                    # except:
                    #     train_model = self.train(train_sample, test_sample, efp=0.95)
                    train_model = self.train_arima(train_sample, test_sample)
                    if train_model:
                        y_pred.append(train_model['threshold'])
                        y_true.append(train_model['test'])
                        train_time_end = time.time()
                        train_time += train_time_end - train_time_start
                        save_model_time_start = time.time()
                        #self.save_model(train_model)
                        save_model_time_end = time.time()
                        save_model_time += save_model_time_end - save_model_time_start
        end = time.time()
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        r2score = r2_score(y_true, y_pred)
        pct = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        print('保存数据完成,使用时间：', end - start)
        print('读取数据时间：', get_data_time)
        print('处理数据时间：', process_data_time)
        print('训练模型时间：', train_time)
        print('保存模型时间：', save_model_time)
        print('rmse = ', rmse)
        print('误差百分比：', pct)
        print('r2_score=',r2score)