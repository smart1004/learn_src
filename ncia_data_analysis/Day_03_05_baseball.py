# Day_03_05_baseball.py
import numpy as np
import  pandas as pd
from sklearn import preprocessing


# xlsx을 열수는 없는데 읽어서 올수 있음
path = 'world-series/MLB World Series Champions_ 1903-2016.xlsx'
champs = pd.read_excel(path)
print(champs)
print('-' * 50)
#        Year, Champion, Wins, Losses, Ties, WinRatio

# 문제
# Year 컬럼을 인덱스로 교체해주세요.
champs.index = champs.Year
del champs['Year']

print(champs.head())

def quiz_1():
    # 문제 1
    # 우승한 팀의 갯수를 알려주세요. (2가지)
    # chaption을 뽑고 그 수를 count하면 됨.
    # champs_count = champs.groupby(by='Champion').size()
    # print(champs_count)
    # print(champs_count.sum())       #  승의 합을 한다?? 원하는 바가 아님
    # 강사 코드
    # 1
    print(champs.Champion.unique())
    # 2
    by_team = champs.groupby(by='Champion').size()
    print(by_team.index)
    # 3
    le = preprocessing.LabelEncoder().fit(champs.Champion)
    print(le.classes_)

def quiz_2():
    # 문제 2
    # 정규 시즌에서 100승 이상한 팀들만 알려주세요.
    # by_team = champs.groupby(by='Wins').size()
    # print(by_team.index)
    over_100 = (champs.Wins >= 100)
    print(over_100)
    print(champs[over_100])         # booln은 True만 꺼내줌. 인덱스 배열을 통해서 꺼내면 된다.

def quiz_3():
    # 문제 3
    # 가장 많이 우승한 Top5를 찾아 주세요.
    # Champion의 size를 구해서
    by_team = champs.groupby(by='Champion').size()
    top_teams = by_team.sort_values(ascending=False)
    print(top_teams)
    print(type(top_teams))
    wins_count = top_teams[4]
    print(wins_count)

    print(top_teams[top_teams >= wins_count])

def quiz_4():
    # 문제 4
    # 월드시리즈가 개최되지 않은 년도는??

    start = champs.index[0]
    end = champs.index[-1]
    print(start, end)

    #set으로 형변환




