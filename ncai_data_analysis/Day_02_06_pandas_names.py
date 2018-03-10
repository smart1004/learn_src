# Day_02_06_pandas_names.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

babies = pd.read_csv('Data/yob1880.txt',
                     header=None,
                     names=['Name', 'Gender', 'Births'])
print(babies.head())       # 출석을 줄이고자 head()를 붙임


# 문제1
# 남자와 여자 아이의 이름 갯수를 알려 주세요.
# fe_babies = babies.sort_values(by='Gender',
#                                values='F')
# print(fe_babies.sum())
# mal_babies = babies.sort_values(by='Gender',
#                                 values='M')
# print(mal_babies.sum())
# 강사코드
print(babies.Gender.count())
print((babies.Gender == 'F').sum())
print((babies.Gender == 'M').sum())
# 위 윗처럼 할 ㅅ도 있다.

by_gender = babies.groupby(by='Gender').size()
print(by_gender)

# 문제 2
# 가장 많이 출생한 아기 이름의 횟수를 top5만
# 막대 그래프로 그려주세요
# 여성분은 여자아기, 남성분은 남자아기
# sort_babies_name = babies.sort_values(by='Births',
#                                     ascending=False)
# sort_mal_babies=sort_babies_name.groupby(by='Gender')
# print(sort_mal_babies)

# 강사 코드
male_only = babies[babies.Gender == 'M']
print(male_only)

top_male = male_only.sort_values(by='Births',
                                 ascending=False)
print(top_male.head())
print(top_male[:5])

top_male.index = top_male.Name
del top_male['Name']

top_male[:5].plot(kind = 'bar')
plt.show()

# 문제 3
# 남자와 여자 이름에 공통으로 사용된 이름을 알려주세요.
# 남자 여자 데이터를 분리 하고, 각 이름을 비교, 같은 것이 있다면 그 이름을 저장
# male_only = babies[babies.Gender == 'M']
# fem_only = babies[babies.Gender == 'F']
# print(male_only, fem_only)
# for i in male_only.Name:
#     if male_only.Name == fem_only.Name:
#         print(male_only.name)
#
# 강사 코드
print('-' * 50)
by_count = babies.groupby(by='Name').size()     # 이름이 데이터 내에 들어 있는 횟수를 출력
print(by_count)     # 2이상이면 famael, male에 있다는 뜻

over_1 = by_count[by_count >1 ]
print(over_1)
print(over_1.index)     # 그 이름을 다 출력
print('-' * 50)


by_names = babies.pivot_table(values='Births',
                              index='Name',
                              columns='Gender')
print(by_names.ix[over_1.index])
















