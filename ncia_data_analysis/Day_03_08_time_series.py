# Day_03_08_time_series.py
# 시간에 대한 정보가 반드시 있어야 함.
import numpy as np
import pandas as pd
import datetime

dates = pd.date_range(start = '2018.1.1',
                      periods=10, freq='D') # period는 개수, freq : 반복주기
print(dates)

dates = pd.date_range(start='1/1/2018',
                       end ='1/7/2018')
print(dates)


dates = pd.date_range(start='20171230',
                       end ='2018-01-05')
print(dates)

# 문제
# 처음 날짜와 마지막 날짜 3개를 출력해보세요.
dates = pd.date_range(start='20171230',
                      end ='2018-01-05')
print(dates[0])
print(dates[-3:])

# ts = pd.Series(np.arange(1, 24, 2),
#                index=dates)       # error. 더 많으면 안됨.
ts = pd.Series(np.arange(1, 14, 2),
               index=dates)
print(ts)

# 문제
# 1번째 데이터를 출력하는 3가지 코드를 찾아주세요. '2017-12-31     3' 출력하는 문제
print(type(ts))
print(ts[1])
print(ts['2017-12-31'])
print(ts['20171231'])
print(ts[datetime.datetime(2017, 12, 31)])


# 문제
# 2018년 데이터만 출력해보세요 (2가지 방법)
print(ts[2:])       # index이 큰거
print(ts['20180101':])
print(ts['20180101': '20180103'])
print(ts['2018':])
print('-' * 50)
print(ts[datetime.datetime(2018,1,1):])
# print(ts[datetime.datetime(2018,01,01):])     # error. 01 이란 문법을 지원 하지 않음
periods = ['2018.1.1', '2018.1.3', '2018.1.7' ]

ts = pd.Series(np.arange(1, 7, 2),
               index=periods)

print(ts)
print(ts.index)
# print(ts['20180101'])

ts.index = pd.to_datetime(periods,
                          format='%Y-%m-%d')
print(ts['20180101'])
print('-' * 50)

rs = ts.resample('D')
print(rs)
print(rs.sum())     # 중간에 빈 데이터가 매꿔짐
print(rs.size())
print(rs.bfill())   # b : back
print(rs.ffill())   # f : fore
print('-' * 50)

index = pd.date_range('20180101',
                      periods=20,
                      freq='60s')
print(index)

ts = pd.Series(range(20),
               index=index)
print(ts)
print('-' * 50)


print(ts.resample('3T'))        # 3분으로 변경. 아무것도 출력은 안된다.
print(ts.resample('3T').sum())
print(ts.resample('3T').mean())



























