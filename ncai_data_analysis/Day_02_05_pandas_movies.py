# Day_02_05_pandas_movies.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pandas에서는 피봇 테이블, 그룹 바이(groupby) 를 잘 하면 된다.

pd.set_option('display.width', 1000)

def get_data():
    # UserID::MovieID::Rating::Timestamp            # UserID::MovieID  == pole in Key
    # UserID::Gender::Age::Occupation::Zip-code     # UserID == PK(구분의 기준이 되는 것)
    # MovieID::Title::Genres

    ratings_columns = 'UserID::MovieID::Rating::Timestamp'.split('::')
    user_columns = 'UserID::Gender::Age::Occupation::Zip-code'.split('::')
    movie_columns = 'MovieID::Title::Genres'.split('::')

    # print(ratings_columns)
    # header 없는 data는 None, separtor는 1글자가 기본
    users = pd.read_csv('ml-1m/users.dat',
                        header=None,
                        sep='::',
                        engine='python',
                        names=user_columns)
    movies = pd.read_csv('ml-1m/movies.dat',
                        header=None,
                        sep='::',
                        engine='python',
                        names=movie_columns)
    ratings = pd.read_csv('ml-1m/ratings.dat',
                        header=None,
                        sep='::',
                        engine='python',
                        names=ratings_columns)
    # print(users)

    # data = pd.merge(ratings, users, left_on='UserID', right_on='UserID')
    # data = pd.merge(ratings, users, on='UserID')

    # merge : option없으면 양 data의 같은 index를 기준으로 합침
    data = pd.merge(pd.merge(ratings, users), movies)
    # print(data.head())      # 5개만 출력
    return data



def pivot_basic():
    data = get_data()

    by_1 = data.pivot_table(values='Rating',
                            index='Age')
    print(by_1)
    print(type(by_1))       # <class 'pandas.core.frame.DataFrame'>


    by_2 = data.pivot_table(values='Rating',
                            columns='Gender')
    print(by_2)

    by_3 = data.pivot_table(values='Rating',
                            index='Age',
                            columns='Gender')
    print(by_3)
    # index를 교체
    by_3.index = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    print(by_3)


    # by_3.plot(kind='bar')
    # plt.show()

    by_4 = data.pivot_table(values='Rating',
                            index='Occupation',
                            columns=['Age','Gender'])       # multi_column가능
    print(by_4)
    print(by_4.stack().head())               # index <- columns
    # stack: stack을 index로 가져 오는 것
    print(by_4.stack().unstack().head())     # index -> columns

    by_5 = data.pivot_table(values='Rating',
                            index='Occupation',
                            columns=['Age','Gender'],
                            fill_value=0)                   # NaN에 값을 넣고자 함, 그러나 data에 0을 넣는 것은 올바르지 않음
    # pivot에는 계산 하는 기능은 없음. 임의 값을 넣고자 하면 넣고자 하는 값을 알고 있어야함
    print(by_5.head())


    by_6 = data.pivot_table(values='Rating',
                            index='Age',
                            columns='Gender',
                            aggfunc='sum')
    # 집계를 바로 구할 수 있음.
    print(by_6)

    by_7 = data.pivot_table(values='Rating',
                            index='Age',
                            columns='Gender',
                            aggfunc=[np.mean,np.sum])     # error. ['mean','sum']
    # 두 데이터를 동시에 가능. 그러나 np.XX 로 써야 함.
    print(by_7)

    by_8_1 = data.pivot_table(values='Rating',
                            index='Age',
                            columns='Gender',
                            aggfunc=np.mean)
    print(by_8_1)

    by_8_2 = data.pivot_table(values='Rating',
                            index='Age',
                            columns='Gender',
                            aggfunc=np.sum)
    print(by_8_2)
    # 8_1, 8_2를 merge 할 수 있나?? 없다. because of 같은 데이터니까.
    # 이때는
    print(pd.concat([by_8_1, by_8_2], axis=0))
    print(pd.concat([by_8_1, by_8_2], axis=1))

def get_index500(data):
    by_title = data.groupby(by='Title')
    print(by_title)                 # <pandas.core.groupby.DataFrameGroupBy object at 0x000000000B37E9E8>
    print(by_title.size())          # 영화가 등장한 횟수, 사람이 평가한 횟수
    print(by_title.size().sum())    # 1000209

    by_title = data.groupby(by='Title').size()
    print(by_title.index)
    print(type(by_title))       # <class 'pandas.core.series.Series'>

    bools = (by_title >= 500)
    print(bools)
    title_500 = by_title[bools]
    print(title_500)
    print(title_500.index)

    return title_500.index




def show_favorite(rateing_500):
    top_female = rating_500.sort_values(by='F',
                                        ascending=False)    # 내림차순
    print(top_female.head())        # 여성들이 좋아하는 top 5
    rating_500['Diff'] = (rating_500.F - rating_500.M)
    print(rating_500.head())

    female_better = rating_500.sort_values(by='Diff',
                                           ascending=False)
    print(female_better.head())

    # 문제
    # 성별에 따른 호불호가 갈리지 않는 영화 top5를 출력해보세요.
        # fe_Mal_like = rating_500.sort_values(by='Diff',
    # )
    rating_500['Dist'] = (rating_500.F - rating_500.M).abs()        # 절대값을 쓰는 것
    print(rating_500.head())

    far_off = rating_500.sort_values(by='Dist')
    print(far_off)



data = get_data()
index_500 = get_index500(data)
# 평점의 개수가 500개 이상인 것만 뽑고 싶을 때

by_gender = pd.DataFrame.pivot_table(data,
                                     values='Rating',
                                     columns='Gender',
                                     index='Title')
print(by_gender)
print('-' * 50)

rating_500 = by_gender.ix[index_500]
print(rating_500)
print('-' * 50)

show_favorite(rating_500)































