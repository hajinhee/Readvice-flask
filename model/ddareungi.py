import tensorflow.compat.v1 as tf
from icecream import ic
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns




class Bicycle:
    def __init__(self) -> None:
        self.basedir = os.path.join(basedir, 'model')
        self.train = None
        self.test = None
        self.submission = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None

    def preprocessing(self):
        # hour_bef_temperature(1시간 전 기온), hour_bef_precipitation(1시간 전 비 정보), hour_bef_windspeed(1시간 전 풍속), 
        # hour_bef_humidity(1시간 전 습도), hour_bef_visibility(1시간 전 가시성), hour_bef_ozone(1시간 전 오존), hour_bef_pm10(1시간 전 미세먼지), hour_bef_pm2.5(1시간 전 미세먼지), count(시간에 따른 따릉이 대여 수)
        
        path='data/train.csv'
        self.train = pd.read_csv(path, encoding='UTF-8', thousands=',')

        # 상관계수 확인 
        # hour, hour_bef_temperature, hour_bef_windspeed, hour_bef_humidity
        # print(self.train.corr()) 
        '''
                                    id      hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  ...  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5     count
        id                      1.000000 -0.013904              0.008044               -0.050635            0.001187  ...             0.014717        0.057916      -0.019327        0.003907 -0.001272       
        hour                   -0.013904  1.000000              0.400910                0.022076            0.459260  ...             0.195596        0.387982      -0.047936       -0.062396  0.620985       
        hour_bef_temperature    0.008044  0.400910              1.000000               -0.095280            0.369624  ...             0.200940        0.528050      -0.036909       -0.078767  0.610444       
        hour_bef_precipitation -0.050635  0.022076             -0.095280                1.000000            0.012133  ...            -0.196578       -0.063887      -0.050290        0.004096 -0.159449       
        hour_bef_windspeed      0.001187  0.459260              0.369624                0.012133            1.000000  ...             0.260716        0.515415      -0.004542       -0.197861  0.458083       
        hour_bef_humidity      -0.021193 -0.324225             -0.483889                0.264732           -0.427355  ...            -0.594662       -0.406968      -0.090507        0.165890 -0.459149       
        hour_bef_visibility     0.014717  0.195596              0.200940               -0.196578            0.260716  ...             1.000000        0.096771      -0.417851       -0.643252  0.308597       
        hour_bef_ozone          0.057916  0.387982              0.528050               -0.063887            0.515415  ...             0.096771        1.000000       0.089405        0.016552  0.468639       
        hour_bef_pm10          -0.019327 -0.047936             -0.036909               -0.050290           -0.004542  ...            -0.417851        0.089405       1.000000        0.487626 -0.137321       
        hour_bef_pm2.5          0.003907 -0.062396             -0.078767                0.004096           -0.197861  ...            -0.643252        0.016552       0.487626        1.000000 -0.136345       
        count                  -0.001272  0.620985              0.610444               -0.159449            0.458083  ...             0.308597        0.468639      -0.137321       -0.136345  1.000000 
        '''

        # 결측치 확인
        # print(self.train.info())
        # print(self.train.isnull().sum())
        '''
        id                          0
        hour                        0
        hour_bef_temperature        2
        hour_bef_precipitation      2
        hour_bef_windspeed          9
        hour_bef_humidity           2
        hour_bef_visibility         2
        hour_bef_ozone             76
        hour_bef_pm10              90
        hour_bef_pm2.5            117
        count                       0
        '''
        
        # 결측치 있는 부분 인덱싱
        # print(self.train[self.train['hour_bef_temperature'].isna()])
        # print(self.train[self.train['hour_bef_windspeed'].isna()])
        # print(self.train[self.train['hour_bef_humidity'].isna()])
        '''
                id  hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5  count
        934   1420     0                   NaN                     NaN                 NaN                NaN                  NaN             NaN            NaN             NaN   39.0
        1035  1553    18                   NaN                     NaN                 NaN                NaN                  NaN             NaN            NaN             NaN    1.0
        '''

        # 시간별 한시간 전 온도
        # print(self.train.groupby('hour').mean()['hour_bef_temperature'])
        # print(self.train.groupby('hour').mean()['hour_bef_windspeed'])
        # print(self.train.groupby('hour').mean()['hour_bef_humidity'])
        '''
                hour
        0     14.788136
        1     14.155738
        2     13.747541
        3     13.357377
        4     13.001639
        5     12.673770
        6     12.377049
        7     12.191803
        8     12.600000
        9     14.318033
        10    16.242623
        11    18.019672
        12    19.457377
        13    20.648333
        14    21.234426
        15    21.744262
        16    22.015000
        17    21.603333
        18    20.926667
        19    19.704918
        20    18.191803
        21    16.978333
        22    16.063934
        23    15.418033
        '''

        # 결측치를 채우고자 하는 컬럼명과 대신하여 넣고자 하는 값 명시
        self.train['hour_bef_temperature'].fillna({934:14.788136, 1035:20.926667}, inplace=True)
        self.train['hour_bef_windspeed'].fillna({18:3.281356, 244:1.836667, 260:1.620000, 376:1.965517, 780:3.278333, 934:1.965517, 1035:3.838333,
                                                1138:2.766667, 1229:1.633333}, inplace=True)
        self.train['hour_bef_humidity'].fillna({934:58.169492, 1035:40.450000}, inplace=True)

        
        # test 결측치 확인 및 대체 값 투입
        path='data/test.csv'
        self.test = pd.read_csv(path, encoding='UTF-8', thousands=',')
        # print(self.test.isnull().sum())      
        '''
        id                         0
        hour                       0
        hour_bef_temperature       1
        hour_bef_precipitation     1
        hour_bef_windspeed         1
        hour_bef_humidity          1
        hour_bef_visibility        1
        hour_bef_ozone            35
        hour_bef_pm10             37
        hour_bef_pm2.5            36
        '''
        # print(self.test[self.test['hour_bef_temperature'].isna()])
        # print(self.test[self.test['hour_bef_windspeed'].isna()])
        # print(self.test[self.test['hour_bef_humidity'].isna()])
        '''
               id  hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5
        653  1943    19                   NaN                     NaN                 NaN                NaN                  NaN             NaN            NaN             NaN
        '''

        # print(self.train.groupby('hour').mean()['hour_bef_temperature'])
        # print(self.train.groupby('hour').mean()['hour_bef_windspeed'])
        # print(self.train.groupby('hour').mean()['hour_bef_humidity'])

        self.test['hour_bef_temperature'].fillna(19.704918, inplace=True)
        self.test['hour_bef_windspeed'].fillna(3.595082, inplace=True)
        self.test['hour_bef_humidity'].fillna(43.573770, inplace=True)

        features = ['hour', 'hour_bef_temperature', 'hour_bef_windspeed', 'hour_bef_humidity']
        self.x_train = self.train[features]
        self.y_train = self.train['count']
        self.x_test = self.test[features]
        # print(self.x_train.shape) # (1459, 4)
        # print(self.y_train.shape) # (1459,)
        # print(self.x_test.shape) # (715, 4)

    def create_model(self): 
        self.preprocessing()
        model = self.model
        model = RandomForestRegressor(n_estimators=100, random_state=0) # 100개의 의사결정 나무를 학습시켜 각각의 나무들에게 예측값을 산출시켜 종합적인 하나의 값 출력
        # 다양한 옵션으로 적절한 형태로 튜닝
        # model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0) 
        # model = RandomForestRegressor(n_estimators=200)
        
        # 모델학습
        model.fit(self.x_train, self.y_train)

        # 모델예측
        pred = model.predict(self.x_test)

        # 제출 
        path='data/submission.csv'
        self.submission = pd.read_csv(path, encoding='UTF-8', thousands=',')
        self.submission['count'] = pred
        self.submission.to_csv('result.csv', index=False)


        
if __name__=='__main__':
    tf.disable_v2_behavior()
    Bicycle().create_model()