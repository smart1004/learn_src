# Day_03_02_preprocessing.py

from sklearn import preprocessing
import numpy as np
import pandas as pd
from io import StringIO

def add_dummy_feature():
    x = [ [0,1],
          [2, 3]]
    print(preprocessing.add_dummy_feature(x))
    # processing은 머신러닝을 하는 단계. 그 이전의 모든 작업을 preprocessing.
    # 데이터가 적을 때 여러 비슷한 데이터를 만들어 내는 것도 preprocessing.
    # [[1. 0. 1.]
    # [1. 2. 3.]], biasing. y 절편에 해당 되는 기준값. 딥러닝 시에도 꼭 쓰임
    print(x)    # 원본은 훼손되지 않았다.

    x = preprocessing.add_dummy_feature(x)
    x = preprocessing.add_dummy_feature(x)
    print(x)
    # featurem는 column을 말한다.

    # 문제
    # 첫 번째 행에 dummy feature를 추가해주세요.
    print(type(x))      # <class 'numpy.ndarray'>
    x = preprocessing.add_dummy_feature(x.T)
    print(x)

    x = x.T
    print(x)


def Binarizer():
    x = [ [1., -1., 2.],
          [2., 0., 0.],
          [0., 1., -1.]]
    print(x)
    bin = preprocessing.Binarizer() # Binarizer(copy=True, threshold=0.0)
    # preprocessing은 기본적으로 원하는 Class를 하나 만들어야 함
    print(bin)
    bin.fit(x)      # x 데이터를 갖고 공부를 함
    # print(bin.fit(x))       # Binarizer(copy=True, threshold=0.0) 가은 값을 알려줌
    bin = preprocessing.Binarizer().fit(x)
    print(bin.transform(x))     # 공부하고 데이터를 변환함

    bin = preprocessing.Binarizer(threshold=1).fit(x)
    print(bin.transform(x))

    x = [-1, 1, 0, -1, 1]         # Error
    print(x)
    # x = [ [-1, 1, 0, -1, 1] ]     # ok
    x = np.array(x).reshape(-1, 1)  # numpy를 써서 이렇게 수정 해야 한다.
    print(x)
    bin = preprocessing.Binarizer().fit(x)
    print(bin.transform(x))


def nan():
    text = '''a,b,c,d
    1,,3,4
    5,,7,8
    9,10,11,
    13,14,15,16'''

    df = pd.read_csv(StringIO(text))
    print(df)

    print(df.isnull())
    print(df.isnull().sum())

    # NaN을 처리 하는 방법을 다룰 것이다.

    print(df.dropna())
    # NaN이 있는 행을 다 없애겠다.
    print(df.dropna(axis=0))        # 열
    print(df.dropna(axis=1))        # 행

def imputer():
    # scikit_learn을 활용한 NaN을 처리 하는 방법을 다룰 것이다.
    imp = preprocessing.Imputer(strategy='mean')
    # allowed_strategies = ["mean", "median", "most_frequent"]

    # (1+7) / 2 = 4  : 공부 하는 방법, 열을 통해서 공부
    # (2 + 4 + 9) /3 =5
    # 아래의 데이터에서 4, 5 가 들어간 이유이다.
    # 앞의 pandas에서는 fill을 통해서 임의의 데이터를 집어 넣는 방법도 학습했다.
    x = [ [1,2],
          [np.nan, 4],
          [7, 9] ]
    imp.fit(x)
    print(imp.transform(x))

    x = [ [np.nan,2],
          [6, np.nan],
          [7, 9] ]
    # imp.fit(x)
    print(imp.transform(x))     # 앞의 fit으로 공부 해서 반영 한다.
    # [[1. 2.]
    #  [4. 4.]
    #  [7. 9.]]
    # [[4. 2.]
    #  [6. 5.]
    #  [7. 9.]]

    # 속성을 출력
    print(imp.strategy)         # mean
    print(imp.missing_values)   # NaN
    print(imp.statistics_)      # [4. 5.]

    # 학습한 배열과 같은 형태 배열이 어야 한다.
    # x = [ [np.nan,2, 3],
    #       [6, np.nan, 1],
    #       [7, 9, 5] ]
    # print(imp.transform(x))   # error


def LabelBinarizer():
    x = [1, 2, 6, 2, 4]

    # sparse는 딥러닝에서 중요하다.
    lb = preprocessing.LabelBinarizer(sparse_output=True)   # 메모리를 아낄 수 있다.
    lb.fit(x)
    print(lb.transform(x))

    # LabelBinarizer, 딥러닝 등에서 참 많이 쓴다.
    lb = preprocessing.LabelBinarizer(sparse_output=False)      # 이게 기본임.
    lb.fit(x)
    # 원합레이블,
    # [[1 0 0 0]
    #  [0 1 0 0]
    #  [0 0 0 1]
    #  [0 1 0 0]
    #  [0 0 1 0]]
    print(lb.transform(x))
    print(lb.classes_)

    t = lb.transform(x)
    a = np.argmax(t, axis=1)        # [0 1 3 1 2] : 1이 어디에 있는지 찾는다.
    print(a)
    print(lb.classes_[a])               # 원본으로 돌아갈 수 있다.
    print(lb.inverse_transform(t))      # 원본으로 돌아갈 수 있다.
    print('-' * 50)

    lb = preprocessing.LabelBinarizer()
    print(lb.fit_transform( [ 'yes', 'no']))
    # [[1]
    #  [0]]

    print(preprocessing.LabelBinarizer().fit_transform( [ 'yes', 'no', 'cancel']))
    # 문자열을 숫자로 encoding하는 능력이 있다.
    # [[0 0 1]
    #  [0 1 0]
    #  [1 0 0]]

def LabelEncoder():
    x = [2, 1, 2, 6]

    le = preprocessing.LabelEncoder().fit(x)
    print(le.classes_)      # 자동완성이 안된다.

    t = le.transform(x)
    print(t)
    print(le.classes_[t])
    print(le.inverse_transform(t))


    x = ['london', 'cuba', 'pusan', 'cuba']

    le = preprocessing.LabelEncoder()
    le.fit(x)

    print(le.classes_)
    print(le.transform(x))
    #print(le.transform([1, 0, 2, 0]))       # 같은 것

def MinMaxScaler():
    # scaler
    x = [ [1, -1, 5],
          [2, 0, -5],
          [0, 1, -10] ]

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x)
    # (나 - 최소값) / (최대값 - 최소값)
    print(scaler.transform(x))

    x = np.array(x)
    max_values = np.max(x, axis=0)
    min_values = np.min(x, axis=0)

    print(max_values)
    print(min_values)

    range_values = (max_values - min_values)
    print(range_values)

    # scalign의 원리
    print( (x - min_values) / range_values)


    # MaxAbsScaler
def MaxAbsScaler():
    x = [[1, -1, 5],
         [2, 0, -5],
         [0, 1, -10]]

    scaler = preprocessing.MaxAbsScaler()
    scaler.fit(x)

    print(scaler.transform(x))
    # [[ 0.5 -1.   0.5]
    #  [ 1.   0.  -0.5]
    #  [ 0.   1.  -1. ]]
    print(scaler.scale_)        # 가장 큰 절대값을 찾는 속성
    print(x / scaler.scale_)




















