# Day_03_06_cross_validation.py
from sklearn.datasets import make_blobs, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     KFold, LeaveOneOut, ShuffleSplit)

def basic():
    # make_blob은 데이터를 만들어 줌
    # make_blobs : Generate isotropic Gaussian blobs for clustering.
    x, y = make_blobs(random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

    lr = LogisticRegression().fit(x_train, y_train)
    print('accuracy :', lr.score(x_test, y_test))


# 로직에 대한 신뢰성이 떨어진다. --> cross Validation(교차검증)이 필요함
def cross_validation():
    iris = load_iris()
    lr = LogisticRegression()

    scores = cross_val_score(lr, iris.data, iris.target)
    print('3-fold : {}', format(scores))      # format은 여러가지를 문자열로 돌려 주는
    print('3-fold : {}'.format(scores))
    # [0.96078431 0.92156863 0.95833333]

    # 더 많이 회기 하고 싶어서
    scores = cross_val_score(lr, iris.data, iris.target, cv=5)
    # print('5-fold : { }', format(scores))
    print('5-fold : {}'.format(scores))
    print(' mean  : {}'.format(scores.mean()))
    # 5-fold : [1.         0.96666667 0.93333333 0.9        1.        ]
    #  mean  : 0.9600000000000002
    # 텐서플로어에서는 cross_val을 코딩으로 꾸며야 한다.


def use_KFold():
    iris = load_iris()
    # KFold : 3번을 접는다. 3번을 나눈다.
    sp1 = KFold()
    print(sp1)
    for train_index, test_index in sp1.split(iris.data, iris.target):
        # print(train_index)
        print(train_index.shape, test_index.shape)  # 같은 인덱스를 갖고 오지 않음. 데이터를 보면 다름


    sp2 = KFold(n_splits=5)
    for train_index, test_index in sp2.split(iris.data, iris.target):
        print(train_index.shape, test_index.shape)

    sp3 = KFold(n_splits=10)
    index = list(sp3.split(iris.data, iris.target))
    print('-' * 50)
    print(index)
    print(index[0])
    print('train index :', index[0][0])
    print('test index  :', index[0][1])

    print(iris.data[index[0][1]])

def cv_detail():
    iris = load_iris()
    lr = LogisticRegression()

    print(' n : 3')
    print(cross_val_score(lr, iris.data, iris.target, cv=3))

    print(' n : 3')
    print(cross_val_score(lr, iris.data, iris.target, cv=KFold()))
    # 사용자가 KFold를 써서 임으로 조정가능하다. 그러나 data가 잘 정제된 것이면, shuffle=True 꼭 해야한다.
    print(' n : 5')
    print(cross_val_score(lr, iris.data, iris.target, cv=KFold(n_splits=5)))

    print(' n : 3')
    print(cross_val_score(lr, iris.data, iris.target, cv=KFold(shuffle=True)))

    print(' n : 150')
    print(cross_val_score(lr, iris.data, iris.target, cv=KFold(n_splits=150)))
    print(cross_val_score(lr, iris.data, iris.target, cv=KFold(n_splits=150)).mean())
    # train set 을 149개 써서 1개를 예측하겠다. 데이터가 극단적으로 적을때는 이렇게 사용이 되어야 함
    print(' n : LeaveOneOut')
    print(cross_val_score(lr, iris.data, iris.target, cv=LeaveOneOut()))
    # 위의 것과 같은 결과.

def use_ShuffleSplit():
    iris = load_iris()
    lr = LogisticRegression()

    # sp = ShuffleSplit(train_size=0.7, test_size=0.3, n_splits=5)  # 실수는 %
    sp = ShuffleSplit(train_size=110, test_size=40, n_splits=5)     # 정수는 그대로의 갯수
    for train_index, test_index in sp.split(iris.data, iris.target):
        print(train_index.shape, test_index.shape)


# basic()
# cross_validation(()
# use_KFold()
# use_ShuffleSplit()











