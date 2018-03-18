#-*- coding: utf-8 -*-
import numpy as np
ab=[3, 76]  # 기울기 a와 y_real 절편 b
# x,y의 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data];  y_real = [i[1] for i in data]
# y_hat = ax + b에 a,b 값 대입하여 결과를 출력하는 함수
def predict(x):
   return ab[0] * x + ab[1]

# RMSE 함수    y_real 실제값,  p : predict값
# 실제값에서 예측값을 빼고 제곱하고 평균을 구한다. 이값에 제곱근을 취한다
def rmse(p, y_real):
   return np.sqrt(((y_real - p) ** 2).mean())  # sqrt : Return the positive square-root

# RMSE 함수를 각 y값에 대입하여 최종 값을 구하는 함수
# predict_result >> <class 'list'>: [82, 88, 94, 100]
#         y_real >> <class 'list'>: [81, 93, 91, 97]
def rmse_val(predict_result, y_real):
   # print('np.array(predict_result)', np.array(predict_result))  # [82  88  94  100]
   return rmse(np.array(predict_result), np.array(y_real))

predict_result = [] # 예측값이 들어갈 빈 리스트

# 모든 x값을 한 번씩 대입하여 predict_result 리스트완성.
for i in range(len(x)):
   predict_result.append(predict(x[i]))
   print("공부시간=%.f, 실제점수=%.f, 예측점수=%.f" % (x[i], y_real[i], predict(x[i])))

# 최종 RMSE 출력
print("rmse 최종값: " + str(rmse_val(predict_result, y_real)))

