# Day_02_04_pandas.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib
#
#
# pandas를 쓰는 이유 다양한 값을 쓰고 여러가지 function을 제공함, numpy는 숫자만 쓰기에 좋음
df = pd.read_csv('Data/scores.csv')
print(df)
print(df.index)
print(df.columns)
print(df.values)
print('-' * 50)

# 해당 columns을 가지고 오는 방법
print(df['kor'])
print(df.kor)       # error가 날때가 있음.

# 인덱스 배열
print(df[ ['kor', 'eng'] ])
subjects = ['kor', 'eng', 'mat', 'bio']
print(df[subjects])     # data frame

# 문제
# 과목 합계를 구해주세요.
# 과목 평균을 구해주세요.
#print(np.sum(a, axis=0))  # 수직(열) 합
#print(np.sum(a, axis=1))  # 수평(행) 합
# print(np.sum.df[subjects])
#강사코드
values = df[subjects].values
print(type(values))
print(np.sum(values))
print(np.sum(values, axis=0))        # 수직 합
# print(np.sum(values, axis=1)
print(np.mean(values, axis=0))      # 평균
# numpy를 쓰면서 하지 않아도 된다. pandas자체 제공 함수 있다.
print(df[subjects].sum(axis=0))
print(df[subjects].mean(axis=0))

df['sum'] = df[subjects].sum(axis=1)    # 학생 합계
df['avg'] = df[subjects].mean(axis=1)
# df.avg = df[subjects].mean(axis=1)      # error. 컬럼 생성 안됨. dot 표현법은 안됨.
print(df)
print('-' * 50)

    # df['avg'].plot()
    # df['avg'].plot(kind='bar')
    # plt.show()          # pandas도 matplot 기반이므로 show를 해야 함.
    #

df_avg = df.sort_values('avg', ascending=False)     # False로 해야 내림차순
print(df_avg)

df_avg.index = df_avg.name      # index를 교체

del df_avg['name']
print(df_avg)
print('-' * 50)

# df_avg['avg'].plot(kind='bar')
# plt.show()

print(df['class'] == 1)     # 관계연산자에 대한 boardcasting  Ture/False로 반환 해줌
c1 = df[ df['class'] == 1 ]
c2 = df[ df['class'] == 2 ]
print(c1)

# df[subjects].boxplot()
plt.subplot(1, 2, 1)
c1[subjects]. boxplot()

plt.subplot(1, 2, 2)
c2[subjects]. boxplot()

plt.figure()
df.plot(kind='scatter', x='mat', y='kor')   # 상관성을 확인 할때 linear하게 나오면 상관성 있음

plt.show()













