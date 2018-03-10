# Day_01_02_numpy.py
# 데이터 처리에서 가장 중요한 부분

# numpy, matplotlib, pandas, scickit-learn, sklearn, requests
import numpy as np


def not_used_1():
    import requests

    url = 'http://www.naver.com'
    received = requests.get(url)
    print(received)
    print(received.text)

def slicing():
    a= range(10)
    print(a)

    a = list(a)
    print(a)

    print(a[3:7])       # slicing

    # 문제
    # 앞쪽 절반만 출력해보세요.
    # 뒤쪽 절반만 출력해보세요.

    print(a[0:5])
    print(a[:5])        # 생략하면, 처음부터

    print(a[5:10])
    print(a[5:])        # 생략하면, 마지막까지

    # 문제
    # 짝수 번째만 출력해보세요
    # 홀수 번째만 출력해보세요
    for i in a:
        if i % 2 == 0:
            print(a[i], end=' ')
    print('-'*50)
    for i in a:
        if i % 2 == 1:
            print(a[i], end=' ')
    print('-'*50)
    # 강사 코드
    print(a[::2])
    print(a[1::2])

    print(a[-1], a[-2])     # 음수 인덱스를 활용하여 끝에서 첫번째, 끝에서 두번째
    print(a[len(a)-1], a[len(a)-2])

    # 문제
    # 거꾸로 출력해 보세요.
    print(a)
    # for i in a:
    #     print(a[len(a) - i-1], end=' ')

    print(a[3:4])
    print(a[3:4])       # 데이터가 비어있다.
    print(a[9:0:-1])
    print(a[9:-1:-1])      # 시작과 종료가 같다
    print(a[-1:-1:-1])     # 윗 것과 같은 코드
    print(a[::-1])      # 양수(처음->끝), 음수(끝->처음)

    # a =   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # index  0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    # -index-10,-9,-8,-7,-6,-5,-4,-3,-2,-1

    # 문제
    # 짝수 번째만 거꾸로 출력하세요.
    # 홀수 번째만 거꾸로 출력하세요.
    print(a)
    print(a[::-2])
    print(a[-2::-2])


a = np.arange(5)
print(a)                    # list와의 차이는 ',' 가 없음
print(type(a))              # <class 'numpy.ndarray'>, n dimension array, 데이터를 섞어 쓸 수 없다.
print(a.shape, a.dtype)     # shape : 내부구조를 보여 줌, dtype : 내부 타입을 보여줌


print(a[0], a[1], a[-1])

# range 함수와의 차이점, 쓰는 방법은 같음, class 형이 다름
print(np.arange(5))
print(np.arange(0, 5))
print(np.arange(0, 5, 1))

a1 = np.arange(12)
# a1 = np.arange(12, dtype=np.float16)        # type을 변경하고 싶으면, dtype 꼭 써야함
# def arange(start=None, stop=None, step=None, dtype=None): # known case of numpy.core.multiarray.arange
a2 = np.arange(12).reshape(3, 4)        # 3개가 있는데 각각은 3개 데이터가 있음
a3 = np.arange(12).reshape(2, 2, 3)     # 2개가 있는데 각각은 3개 데이터가 있는 2개 짜리
# def reshape(self, shape, *shapes, order='C'): # known case of numpy.core.multiarray.ndarray.reshape
print(a1.shape, a1.dtype)           # (12,) int32
print(a2.shape, a2.dtype)           # (3, 4) int32
print(a3.shape, a3.dtype)           # (2, 2, 3) int32

print(a1)
print(a2)
print(a3)
print('-'*50)


a = np.array([1, 3, 5])     # list를 numpy array로 만들고 싶으면
print(a)
print(list(a))              # 다시 list로 당연히 호환 가능함
print('-'*50)

# 문제
# 0~5 사이의 정수를 갖는 2행 3열 배열을 만들어 주세요.
# np.array 함수를 사용합니다. (3가지 코드)

print(np.array([[0, 1, 2],
               [3, 4, 5]]))
print(np.array([0, 1, 2, 3, 4, 5]).reshape(2,3))
print(np.array([np.arange(3),
                np.arange(3, 6)]))
print(np.array([range(3),
                range(3, 6)]))

# print(np.arange(6).reshape(2,4))        # error. 딱 맞아야 한다. 정확히 나누어 떨어져야함
                                        # ValueError: cannot reshape array of size 6 into shape (2,4)
print(np.arange(6).reshape(2,3))
print(np.arange(6).reshape(2,-1))       # -1은 내가 계산 하기 싫을때, 많은 씀
print(np.arange(6).reshape(-1,3))       # -1은 내가 계산 하기 싫을때,

# 문제
# 2차원 배열을 1차원으로 reshape 해보세요
a = np.arange(6).reshape(2,3)
print(a.reshape(1,-1))
print(a.reshape(1,6))
# print(a[0] + a[1])
# 강사 코드
# a = a.reshape(6)                # reference(point)를 활용한 연산이라서 크게 연산이 많이 안됨
# print(a)
print(a.reshape(6))
print(a.reshape(a.size))          # size : 개수 몇개 들어있늦지 알려줌
print(a.reshape(-1))
# print(a.reshape(len(a)))        # error. lean은 1차원만 return함
print(a.reshape(len(a) * len(a[0])))
print(a.reshape(a.shape[0] * a.shape[1]))   # shape란 함수는 행,열의 개수를 알려줌
print(a.shape)
print(np.reshape(a, -1))            #  자동완성이 뜸. 이 방법을 쓰는 것이 보기에 유리함.
print('-'*50)


# print(np.zeros(2, 3))     # error. tuple로 묶어야 한다
print(np.zeros([2, 3], dtype=np.int32))

print(np.ones((2,3)))

print(np.linspace(0, 2, 9))     # 0~2까지 9개, 출력 시 가장 긴 것으로 맞춰 줌
print(np.arange(0, 2 , 0.25))   # 마지막 2는 포함 안됨.
print('-'*50)

# 이제 연산
a = np.array([1, 3, 5])
a += 10                     # broadcasting(요소의 개수만큼 덧셈이 이루어짐), 각 요소에 10을 더한 결과
print(a)
# print([1, 3, 5] + 10)       # error.  TypeError: can only concatenate list (not "int") to list
print(a ** 2)
print(a >= 13)
b = (a >= 13)
print(a[b])         # b의 값이 Ture인 값만 뽑을 수도 있다.

print(np.sin(a))    # sin은 universal function
# print(a.sin())    # error. 삼각함수가 'a'가 정의된 ndarray class에 없기 때문에 error

c = np.arange(6).reshape(-1,3)
print(c)
print(c + 10)       # 차원에 상관 없이 다 동작함
print(c ** 2)
print(c >= 2)
print(c[c >= 2])    # 1차로 반환, array는 행,열이 정의 되어야 하는데 이건 그러지 못해서.
print('-'*50)


a1 = np.arange(5)
a2 = np.arange(-3, 7, 2)
print(a1)
print(a2)
print(a1 + a2)      # vector operation, 배열의 모든 데이터가 덧셈에 참여함
print(a1 * a2)
print(a1 > a2)
print('-'*50)

# 차원을 섞어 씀
# 갯수가 똑같거나 ( 백터 연산이 가능하거나)
# 갯수가 1개이거나 (브로드캐스팅이 일어날 수 있거나)
a = np.arange(3)
b = np.arange(6)
c = np.arange(3).reshape(-1,3)
d = np.arange(6).reshape(-1,3)
e = np.arange(3).reshape(3,-1)


print(a)
print(b)
print(c)
print(d)
print(e)
print('-'*50)

# print(a + b)        # error.
print(a + c)
print(a + d)        # vector operation와 broadcasting을 적용 (2, 3)짜리 배열
print(a + e)        # vector operation와 broadcasting * 2을 적용 (3,3)짜리 배열

# print(b + c)      # error.
# print(b + d)      # error.
print(b + e)        # (3, 6)짜리 행열

print(c + d)        # (2,3)
print(c + e)        # (3,3)
# print(d + e)      # error.









































