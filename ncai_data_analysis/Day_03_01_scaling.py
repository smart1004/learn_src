# Day_03_01_scaling.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, neighbors, preprocessing

def show_result(x, y):
    data = model_selection.train_test_split(x, y)
    # quality가 y_data를 갖고 있다
    # 수직으로 fancy index를 쓸 수 있는지가 중요함.
    # wine[-1:] : 2차원 배열, wine[-1] : int
    x_train, x_test, y_train, y_test = data
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)
    # (1199, 11) (400, 11)
    # (1199,) (400,)

    # 머신러닝이든 딥러닝이든 비슷한 코딩. 알고리즘 개발이 아닌 코드를 가져다 씀
    # KNeighborsClassifier이 어디에 쓸 수 있는지 알아야 함.
    # 옛날 알고리즘은 데이터에 따라서 어떤 알고리즘을 쓰는지에 따라서 결과가 많이 다름.
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)

    print('train : ', clf.score(x_train, y_train))
    print('test  : ', clf.score(x_test, y_test))


wine = pd.read_csv('Data/winequality-red.csv', sep=';')
# print(wine)

# print(wine.quality)
# print(wine.quality.unique())    # unique()함수 호출하여 그 구성을 확인
# # [5 6 7 4 8 3]


# 문제
# wind 데이터를 train_test_split 함수에 잘 전달해 주세요.
x = wine.values[:, :-1]     # data frame이 필요하므로 .values 가 붙어야함
# y = wine.values[:, -1]
y = (wine.values[:, -1] > 5)
print(x.shape, y.shape)
# 함수 호출
show_result(x,y)

# plt.hist(y)       # histogram을 통해서갖고 있는 데이터를 불러 봄
# wine.hist()       # unit가 다르다. 즉, 중요도가 달라진다. scaling이 필요함. 이건 그 field의 노하우가 작용한다.
# plt.show()

scaled_x = preprocessing.scale(x)
show_result(scaled_x, y )



















