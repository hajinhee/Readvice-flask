import tensorflow.compat.v1 as tf
from icecream import ic
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
from sklearn.ensemble import RandomForestRegressor


class Bicycle:
    def __init__(self) -> None:
        self.basedir = os.path.join(basedir, 'model')
        self.train = None
        self.test = None
        self.submission = None
        self.x_data = None
        self.y_data = None
        self.model = None

    def preprocessing(self):
        path='data/train.csv'
        train = self.train 
        train = pd.read_csv(path, encoding='UTF-8', thousands=',')
        # ic(train.info())
        # ic(train.isnull().sum())
        train.fillna(0, inplace=True)
        xy = np.array(train, dtype=np.float32)
        # ic(type(xy)) # <class 'numpy.ndarray'>
        # ic(xy.ndim) # xy.ndim: 2
        # ic(xy.shape) # xy.shape: (1459, 11)
        # hour_bef_temperature(1시간 전 기온), hour_bef_precipitation(1시간 전 비 정보), hour_bef_windspeed(1시간 전 풍속), 
        # hour_bef_humidity(1시간 전 습도), hour_bef_visibility(1시간 전 가시성), hour_bef_ozone(1시간 전 오존), hour_bef_pm10(1시간 전 미세먼지), hour_bef_pm2.5(1시간 전 미세먼지), count(시간에 따른 따릉이 대여 수)
        self.x_data = train.drop(['count'], axis=1)
        self.y_data = train['count']    
        # ic(self.x_data.info())
        # ic(self.y_data.info())

    def submit(self): 
        self.preprocessing()
        self.model = RandomForestRegressor(n_estimators=100)
        # x_data = train, y_data = label
        self.model.fit(self.x_data, self.y_data)

        path='data/test.csv'
        self.test = pd.read_csv(path, encoding='UTF-8', thousands=',')
        pred = self.model.predict(self.test)

        path='data/submission.csv'
        self.submission = pd.read_csv(path, encoding='UTF-8', thousands=',')
        self.submission['count'] = pred
        self.submission.to_csv('베이스라인.csv', index=False)
        
    def create_k_fold(self): return KFold(n_splits=10, shuffle=True, random_state=0)

    def get_accuracy(self, k_fold):
        score = cross_val_score(RandomForestClassifier(), self.x_data, self.y_data, cv=k_fold, n_jobs=1, scoring='accuracy')
        return round(np.mean(score)*100, 2)  # *백분률, 평균

        
if __name__=='__main__':
    tf.disable_v2_behavior()
    Bicycle().submit()
    k_fold = self.create_k_fold()
    accuracy = self.get_accuracy(this, k_fold)
    ic(accuracy)