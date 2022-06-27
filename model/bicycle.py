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



class Bicycle:
    def __init__(self) -> None:
        self.basedir = os.path.join(basedir, 'model')
        self.train = None
        self.test = None
        self.submission = None
        self.x_data = None
        self.y_data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def preprocessing(self):
        # hour_bef_temperature(1시간 전 기온), hour_bef_precipitation(1시간 전 비 정보), hour_bef_windspeed(1시간 전 풍속), 
        # hour_bef_humidity(1시간 전 습도), hour_bef_visibility(1시간 전 가시성), hour_bef_ozone(1시간 전 오존), hour_bef_pm10(1시간 전 미세먼지), hour_bef_pm2.5(1시간 전 미세먼지), count(시간에 따른 따릉이 대여 수)
        
        path='data/train.csv'
        self.train = pd.read_csv(path, encoding='UTF-8', thousands=',')
        self.train.dropna(inplace=True)
        # ic(self.train.info())
        # ic(self.train.isnull().sum())
        self.x_data = self.train.drop(['count'], axis=1)
        self.y_data = self.train['count']

        path='data/test.csv'
        self.test = pd.read_csv(path, encoding='UTF-8', thousands=',')
        self.test.fillna(0, inplace=True)
        # ic(self.test.info())
        # ic(self.test.isnull().sum())      
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data, test_size=0.2, shuffle=True, random_state=34)

    def create_model(self): 
        self.preprocessing()
        self.model = RandomForestRegressor(n_estimators=100)

    #    # 선형식(가설)제작 y = Wx+b
    #     X = tf.placeholder(tf.float32, shape=[None, 4])
    #     Y = tf.placeholder(tf.float32, shape=[None, 1])
    #     W = tf.Variable(tf.random_normal([4, 1]), name="weight")
    #     b = tf.Variable(tf.random_normal([1]), name="bias")
    #     hypothesis = tf.matmul(X,W) + b 
    #     # 손실함수
    #     cost = tf.reduce_mean(tf.square(hypothesis - Y))
    #     # 최적화알고리즘
    #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
    #     train = optimizer.minimize(cost)
    #     # 세션생성
    #     sess = tf.Session()
    #     sess.run(tf.global_variables_initializer())
    #     # 트레이닝
    #     for step in range(100000):
    #         cost_, hypo_, _ = sess.run([cost, hypothesis, train],
    #                                     feed_dict={X: self.x_data, Y: self.y_data})
    #         if step % 500 == 0:
    #             print('# %d 손실비용: %d'%(step, cost_))
    #             print('- 배추가격: %d '%(hypo_[0]))   
    #     # 모델저장
    #     saver = tf.train.Saver()
    #     saver.save(sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt'), global_step=1000)
    #     print('저장완료')

        pred = self.model.predict(self.x_test)
        

        path='data/submission.csv'
        self.submission = pd.read_csv(path, encoding='UTF-8', thousands=',')
        self.submission['count'] = pred
        self.submission.to_csv('베이스라인.csv', index=False)


        
if __name__=='__main__':
    tf.disable_v2_behavior()
    Bicycle().create_model()