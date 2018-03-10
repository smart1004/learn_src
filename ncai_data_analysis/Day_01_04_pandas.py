# Day_01_04_pandas.py
import pandas as pd
# 데이터를 분석하는 라이브러리, 상용 tool이 많은데, customize하기 위해서는 파이썬을 씀
# 데이터 frame이 중요함. 이것을 하려면 Series

def series():
    s = pd.Series([5, 1, 2, 9])         # list --> series로 변환
    print(s)
    print(s.index)
    print(s.values)
    print(type(s.values))           # <class 'numpy.ndarray'>


    s = pd.Series([5, 1, 2, 9],
                  index=['a', 'b', 'c', 'd'])
    print(s)
    print(s.index)
    print(s[0], s[-1])          # 정수 index, 1st 방법
    print(s['a'], s['d'])       # index 이름, 2nd 방법, 윗 표현은 헷갈리니까
    print('-'*50)

    s = pd.Series({'a':5, 'b':1, 'c':2, 'd':9})
    print(s)

def dataframe():
    df = pd.DataFrame({'city': ['seoul', 'seoul', 'seoul', 'pusan', 'pusan', 'pusan'],
                       'population': [1.5, 1.7, 2.3, 3.4, 2.8, 3.1],
                       'year': [2001, 2002, 2003, 2001, 2002, 2003]})
    print(df)  # index , columns, valus     column은 피쳐라고 행보다 더 중요
    print(df.index)
    print(df.columns)
    print(df.values)
    print(type(df.values))      # <class 'numpy.ndarray'>
    print(df.values.dtype)      # object
    print('-'*50)

    # 데이터를 앞에서 찾는 것
    print(df.head())
    print(df.head(2))
    print('-'*50)
    # 데이터를 뒤에서 찾는 것
    print(df.tail())
    print(df.tail(2))
    print('-'*50)

    # column에 접근 하는 방법
    print(df['city'])
    print(type(df['city']))       # <class 'pandas.core.series.Series'>
    print(type(df['city'][0]))    # <class 'str'>
    print(df['city'][0])            # seoul
    # # print(type(df['city'][-1])) # error

    # column에 접근 하는 방법
    print(df.index)
    print(df.city)      # column에 접근 하는 방법
    print('-'*50)

    # index를 바꾸는 방법
    df.index = ['one', 'two', 'three',
                'four', 'five','six']
    print(df)

    # row에 접근하는 3가지 방법
    # 1
    print(df.iloc[0])       # iloc라는 속성을통해서 접근
    # 2
    print(df.loc['two'])
    # 3
    print(df.ix[0])
    print(df.ix['two'])

    print(df.iloc[1:3])     # slicing 가능함
    print(df.loc['two': 'four'])        # four   pusan         3.4  2001 //지정 영역 출력 됨



dataframe()