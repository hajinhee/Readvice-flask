import tensorflow.compat.v1 as tf
from icecream import ic
import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import true
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
tf.disable_v2_behavior()


# 데이터 불러오기
path='data/train.csv'
train = pd.read_csv(path, encoding='UTF-8', thousands=',', index_col=0)

# ic(train)
# ic(train.shape)  # (1459, 11)

path='data/test.csv'
test = pd.read_csv(path, encoding='UTF-8', thousands=',', index_col=0) # 0번째 column을 index로 사용

# ic(test)
# ic(test.shape)  # (715, 10)

ic(train.columns)
'''
['id', 'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count']
'''
# ic(train.info())
'''
id                      1459 non-null   int64
 1   hour                    1459 non-null   int64
 2   hour_bef_temperature    1457 non-null   float64
 3   hour_bef_precipitation  1457 non-null   float64
 4   hour_bef_windspeed      1450 non-null   float64
 5   hour_bef_humidity       1457 non-null   float64
 6   hour_bef_visibility     1457 non-null   float64
 7   hour_bef_ozone          1383 non-null   float64
 8   hour_bef_pm10           1369 non-null   float64
 9   hour_bef_pm2.5          1342 non-null   float64
 10  count  
'''

# 결측치 제거
# ic(train.isnull().sum())
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
train.dropna(inplace=True) # null 값이 하나라도 있으면 drop

ic(test.isnull().sum())
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
dtype: int64
'''

test.fillna(0, inplace=True) 

x = train.drop(['count'], axis=1)  # 정답을 제거한 train_set 문제지
y = train['count'] # train_set 정답지

print(x.shape) 
print(y.shape) 

# train_set을 훈련과 평가 데이터로 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.99, shuffle=True, random_state=750)

# 모델구성
model = Sequential()
model.add(Dense(20, input_dim = 9))  # 입력 뉴련의 수
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(7))
model.add(Dense(1))

# 컴파일
model.compile(loss='mse', optimizer="adam")

# 훈련
model.fit(x_train, y_train, epochs=300, batch_size=10)

# 평가
loss = model.evaluate(x_test, y_test)
ic('loss:', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
ic("RMSE", rmse)

# 제출할 파일에 예측한 값 추가
y_summit = model.predict(test)
path = 'data/submission.csv'
result = pd.read_csv(path, index_col=0) # count 열이 비어져있는 상태
result['count'] = y_summit # count 열에 예측값 넣기

# 파일 저장
path = 'save/submission.csv'
result.to_csv(path, index=True)