# Day_01_03_matplotlib.py
import numpy as np
import matplotlib.pyplot as plt
import csv
# 시각화 제공 matplotlib이 대표적

def plot_1():
    plt.plot([10, 20, 30, 40, 50])
    plt.show()


def plot_2():
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')     # option은 인터넷으로 검색
    # 하나의그램에 계속 덮어서 그릴 수 있다.
    plt.xlim(0,5)
    plt.ylim(0,20)
    plt.show()

# 문제
# x의 범위가 -10에서 10일 때의 x 제곱 그래프를 그려주세요.
a = np.arange(-10,11)       # range함수를 호출하여 list로 변환하여 쓰려고 하니 a * a하면 error가 남
#a = list(a)
# print(a)
# b = a * a
# print(b)
# plt.plot(a, b)
# plt.show()
# 강사 코드
def plot_3():
    x = np.arange(-10,10)
    plt.plot(x, x ** 2)
    plt.plot(x, x **2, 'rx')
    plt.show()

# 문제
# 로그 곡선 4가지를 그려보세요.
def plot_4():
    x1 = np.arange(0.01, 2, 0.01)
    plt.plot(x1, np.log(x1))
    plt.plot(x1, -np.log(x1))
    # plt.show()

    # x2 = -x1        # error.
    x2 = np.arange(0.01-2, 0, 0.01)
    plt.plot(x2, np.log(-x2))
    plt.plot(x2, -np.log(-x2))
    plt.show()


def plot_5():
    x1 = np.arange(0.01, 2, 0.01)
    plt.subplot(1, 2, 1)                    # 행, 열, 인덱스
    plt.grid()
    plt.plot(x1, np.log(x1))
    plt.plot(x1, -np.log(x1))

    x2 = np.arange(0.01-2, 0, 0.01)
    # plt.subplot(1, 2, 2)
    plt.subplot(122)
    plt.grid()

    plt.plot(x2, np.log(-x2))
    plt.plot(x2, -np.log(-x2))
    plt.show()

    x = np.arange(0.01, 2, 0.01)
    plt.subplot(1, 2, 1)                    # 행, 열, 인덱스
    plt.grid()
    plt.plot(x, np.log(x))
    plt.plot(x, -np.log(x))

    plt.subplot(1, 2, 2)
    plt.grid()
    plt.plot(-x, np.log(x))
    plt.plot(-x, -np.log(x))
    plt.show()


# 문제
# 로그 곡선을 각각의 플랏에 그려주세요.
def not_used():
    x1 = np.arange(0.01, 2, 0.01)
    plt.subplot(2, 2, 4)                    # 행, 열, 인덱스
    plt.grid()
    plt.plot(x1, np.log(x1))
    plt.subplot(2, 2, 2)                    # 행, 열, 인덱스
    plt.grid()
    plt.plot(x1, -np.log(x1))

    x2 = np.arange(0.01-2, 0, 0.01)
    plt.subplot(2, 2, 3)                    # 행, 열, 인덱스
    plt.grid()
    plt.plot(x2, np.log(-x2))
    plt.subplot(2, 2, 1)                    # 행, 열, 인덱스
    plt.grid()
    plt.plot(x2, -np.log(-x2))
    plt.show()
# 강사코드
def plot_6():
    x1 = np.arange(0.01, 2, 0.01)
    plt.subplot(2, 2, 4)                    # 행, 열, 인덱스
    plt.grid()
    plt.plot(x1, np.log(x1))
    plt.subplot(2, 2, 2)                    # 행, 열, 인덱스
    plt.grid()
    plt.plot(x1, -np.log(x1))

    x2 = np.arange(0.01-2, 0, 0.01)
    plt.subplot(2, 2, 3)                    # 행, 열, 인덱스
    plt.grid()
    plt.plot(x2, np.log(-x2))
    plt.subplot(2, 2, 1)                    # 행, 열, 인덱스
    plt.grid()
    plt.plot(x2, -np.log(-x2))
    plt.show()

def plot_7():
    x = np.arange(0.01, 2, 0.01)
    plt.subplot(1, 4, 1)
    plt.grid()
    plt.plot(x, np.log(x))

    plt.subplot(1, 4, 2)
    plt.grid()
    plt.plot(x, -np.log(x))

    plt.figure()
    plt.subplot(1, 4, 3)
    plt.grid()
    plt.plot(-x, np.log(x))

    plt.figure(1)
    plt.subplot(1, 4, 4)
    plt.grid()
    plt.plot(-x, -np.log(x))
    plt.show()


def plot_8():
    men = [20, 35, 30, 32, 29]
    women = [ 21, 28, 26, 33, 30]


    x = np.arange(len(men))      # [0 1 2 3 4]
    # print(x)
    bar_width = 0.45        # men, women 을 2개 그리고도 0.1만큼 남음

    plt.bar(x, men, bar_width, color='b')
    # 문제
    # 여자 데이터도 그려보세요.
    # 내코드
    # plt.bar(x+0.5, women, bar_width)
    # 강사 코드
    plt.bar(x+bar_width, women, bar_width, color='r')        # broadcasting을 쓰면 된다.
    plt.show()







