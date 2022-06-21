import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
from icecream import ic
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir


class Cabbage:
    def __init__(self):
        self.basedir = os.path.join(basedir, 'model') 
        self.df = None
       
    def preprocessing(self):
        path='data/price_data.csv'   
        self.df = pd.read_csv(path, encoding='UTF-8', thousands=',')
        # ic(self.df)     
        # year,avgTemp,minTemp,maxTemp,rainFall,avgPrice
        xy = np.array(self.df, dtype=np.float32) # csv파일을 배열로 변환
        ic(type(xy)) # <class 'numpy.ndarray'>
        ic(xy.ndim)  # 차원 # 2
        ic(xy.shape) # 행렬의 갯수 # (2922, 6)
        x_date = xy[:, 1:-1] # 행 전체, 열(2번째 열:마지막 열)
        y_date = xy[:, [-1]] # 행 전체, 열(마지막 열) avgPrice
        ic(x_date.ndim) 
        ic(y_date.ndim)

        # self.create_model()

    def create_model(self): # 모델생성
        # 텐서모델 초기화(모델템플릿 생성)
        model = tf.global_variables_initializer()
        # 확률변수 데이터
        self.preprocessing()
        # 선형식(가설, hypothesis) 제작 y = Wx + b
        X = tf.placeholder(tf.float32, shape=[None, 4])  # placeholder() 외부에서 들어오는 값 # shape=[None, 4] None-> 특정 값을 정하지 않는다.
        Y = tf.placeholder(tf.float32, shape=[None, 1]) 
        W = tf.Variable(tf.random_normal([4, 1]), name='weight') # 4개가 투입되고 하나가 나온다.
        b = tf.Variable(tf.random_normal([1]), name='weight')
        hypothesis = tf.matmul(W, X) + b # hypothesis = 가설 -> 선형식 # Wx+b 와 같다.
        # 손실 함수
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        # 최적화 알고리즘
        optimizer = tf.train.GradientDescenOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        # 세션생성
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # 트레이닝
        for step in range(10000):
            cost_, hypo_, = sess.run([cost, hypothesis, train],
                                    feed_dict={X: self.x_data, Y:y_data})
            if step % 500 == 0:
                print('# %d 손실비용: %d'%(step, cost_))
                print('- 배추가격: %d' %(hypo_[0]))
        # 모델저장
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt'), global_step=1000)
        ic('저장완료')
    
    def load_model(self, avgTemp, minTemp,maxTemp, rainFall, avgPrice): # 모델로드
        X = tf.placeholder(tf.float32, shape=[None, 4])  # placeholder() 외부에서 들어오는 값 # shape=[None, 4] None-> 특정 값을 정하지 않는다.
    
        W = tf.Variable(tf.random_normal([4, 1]), name='weight') # 4개가 투입되고 하나가 나온다.
        b = tf.Variable(tf.random_normal([1]), name='weight')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(self.basedir,'cabbage', 'cabbage.ckpt'))
            data = [[avgTemp, minTemp,maxTemp, rainFall, avgPrice]]
            ar = np.array(data, dtype = np.float32)
            dict = sess.run(tf.matmul(X,W) + b, {X: arr[0:4]})
            print(dict)
        return dict

      
if __name__=='__main__':
    tf.disable_v2_behavior()
    Cabbage().preprocessing()

  
