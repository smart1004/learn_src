# Day_03_07_grid_search.py
# 알고리즘을 동작시키는 몇몇 변수들을 잘 찾아야 함. 여러번 해보고 경험치로 해야 함
#
from sklearn.datasets import load_iris
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV)
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# 딥러닝 시 에는 이런 parameter를 꼭 찾을 수 있어야 함. 딥러닝에서는 이 코드가 지원이 안됨.
def simple_grid_search(x_train, x_test, y_train, y_test):
    best_score, best_param = 0, {'gamma': 0, 'C': 0 }
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:    # contraint 를 주는 것
            svm = SVC(gamma=gamma, C=C)
            svm.fit(x_train, y_train)

            score = svm.score(x_test, y_test)
            #print(score)

            if best_score < score:
                best_score = score
                best_param['gamma'] = gamma
                best_param['C'] = C
    print('best score :', best_score)
    print('best param :', best_param)


#
# iris = load_iris()
# data = train_test_split(iris.data, iris.target, random_state=0)
# simple_grid_search(*data)
#



def simple_grid_search_adv(x_values, x_test, y_values, y_test):
    # train 데이터 중 valid를 뽑아서 best_gamma, C 찾아서 결과를 확인 하는 방법
    # gamma, C를 좁게 하는 건 딥러닝 시 많은 연산이 필요 할 수 있다. 주의 해야 함
    # 딥러닝 시 중요함
    # 문제
    # 데이터를 train, valid, test로 나눠서 아래 코드를 화인해 주세요.
    # train을 value와 val_value로 쪼개야 함. 그럼 쪼개서 어디에 입력하여 어떻게 쓸 수 있을까???
    data = train_test_split(x_values, y_values, random_state=0)
    x_train, x_valid, y_train, y_valid = data

    best_score, best_param = 0, {'gamma': 0, 'C': 0 }
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:    # contraint 를 주는 것
            svm = SVC(gamma=gamma, C=C)
            svm.fit(x_train, y_train)

            score = svm.score(x_valid, y_valid)
            #print(score)

            if best_score < score:
                best_score = score
                best_param['gamma'] = gamma
                best_param['C'] = C

    svm = SVC(gamma=best_param['gamma'], C=best_param['C'])
    svm.fit(x_values, y_values)

    test_score = svm.score(x_test, y_test)

    print('test score :', test_score)
    print('best score :', best_score)
    print('best param :', best_param)



def grid_search_cv(x_train, x_test, y_train, y_test):
    # data = train_test_split(x_values, y_values, random_state=0)
    # x_train, x_valid, y_train, y_valid = data

    best_score, best_param = 0, {'gamma': 0, 'C': 0 }
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:    # contraint 를 주는 것
            svm = SVC(gamma=gamma, C=C)
            # 앞의 코드에서는 1번만 하는데 cross_val을 써서 여러번 돌려서 더 좋은 값을 뽑을 수 있다.
            # # hyperparameter를 튜닝하는 방법
            scores = cross_val_score(svm, x_train, y_train, cv=5)       # 180번을돌았음 36*5
            score = scores.mean()

            if best_score < score:
                best_score = score
                best_param['gamma'] = gamma
                best_param['C'] = C

    svm = SVC(gamma=best_param['gamma'], C=best_param['C'])
    svm.fit(x_train, y_train)

    test_score = svm.score(x_test, y_test)

    print('test score :', test_score)
    print('best score :', best_score)
    print('best param :', best_param)


def grid_search_cv_adv(x_train, x_test, y_train, y_test):
    params = {'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
              'C' : [0.001, 0.01, 0.1, 1, 10, 100]}
    # parma을 집어 넣고 돌린다. 앞의 코드를 퉁친다
    gs = GridSearchCV(SVC(), params, cv=5)
    # hyperparameter를 튜닝하는 방법
    gs.fit(x_train, y_train)

    test_score = gs.score(x_test, y_test)

    print('test score :', test_score)
    print('best score :', gs.best_score_)
    print('best param :', gs.best_params_)


iris = load_iris()
data = train_test_split(iris.data, iris.target, random_state=0)
# simple_grid_search(*data)
# simple_grid_search_adv(*data)
# grid_search_cv(*data)
grid_search_cv_adv(*data)






