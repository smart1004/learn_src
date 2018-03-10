# Day_02_07_scikit_learn.py

from sklearn import (datasets, svm, preprocessing, model_selection)
#                    데이터,  알고리즘,  전처리,      데이터 분할
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# sklearn : 텐서플로어 나오기 전 파이썬에서 가장 많이 사용한 머신런닝 코드


iris = datasets.load_iris()
print(iris)
print(type(iris))       # <class 'sklearn.utils.Bunch'>, dictionay와 비슷
print(iris.keys())      # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])
print('-' * 50)

print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']
print(iris.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.data)
print(type(iris['data']))      # <class 'numpy.ndarray'>
print('-' * 50)

print(iris['target'])
print(iris.data[-1])
print(iris.target[-1])
print(iris.target_names[2])
print(iris.target_names[iris.target])
print(iris.target_names[[0, 0, 2]])
print('-' * 50)


data = model_selection.train_test_split(iris.data, iris.target)     #75 : 25로 split
# 4개의 data
x_train, x_test, y_train, y_test = data
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# classfier
clf = svm.SVC()
clf.fit(x_train, y_train)

print(clf.score(x_train, y_train))
print(clf.score(x_test, y_test))

pred = clf.predict(x_test)
print(pred)
print(y_test)
print(pred == y_test)
print(np.mean(pred == y_test))      # print(clf.score(x_test, y_test))결과가 같음







